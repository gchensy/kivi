"""Seeded generator of a synthetic monthly panel of instalment loans."""

import math

import numpy as np
import numpy.typing as npt
import pandas as pd

from .. import _stats, _validation

STATUSES = ("current", "1-29", "30-59", "60-89", "90+")
FIRST_COHORT = pd.Period("2024-01", freq="M")

# Chance that a current loan misses its instalment at the peak of seasoning (month on book 6).
_MISS_PEAK = 0.03
_SEASONING_PEAK = 6
# Monthly chance of paying all arrears (cure) and of paying one instalment (stay) at 1-29,
# 30-59 and 60-89 days past due; a loan that pays nothing rolls to the next bucket.
_CURE = np.array([0.50, 0.25, 0.10])
_STAY = np.array([0.20, 0.20, 0.15])
_MONTHLY_RATE = 0.01
_TERMS = np.array([12, 24, 36])
_TERM_SHARES = [0.2, 0.4, 0.4]
_LOAN_RISK_SIGMA = 0.7
_COHORT_RISK_SIGMA = 0.25


def make_loan_panel(
    *,
    n_loans: int = 5000,
    n_months: int = 24,
    prepayment: float = 0.0,
    correlation: float = 0.0,
    factors: npt.ArrayLike | None = None,
    rng: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    r"""Generate a synthetic monthly panel of instalment loans.

    The loans are originated over `n_months` months and observed at the end
    of every month on book until the month after the last origination month.
    Delinquency rolls forward when a borrower misses instalments and cures
    when the arrears are paid, and a loan that reaches 90 days past due stays
    there. The data is simulated, not drawn from a real portfolio.

    Parameters
    ----------
    n_loans : int, default 5000
        Number of loans, each assigned to an origination month at random.
    n_months : int, default 24
        Number of origination months, starting with January 2024. The oldest
        cohort is observed for `n_months` months on book and the youngest
        for one.
    prepayment : float, default 0.0
        Monthly probability that a current loan repays its whole balance
        before its last instalment is due, from 0 to 1. The default, 0,
        gives no early repayments and the same panel as earlier releases.
    correlation : float, default 0.0
        Asset correlation of a systematic factor that moves the payments of
        all loans together each month, from 0 to below 1, as in
        `kivi.lifetime.conditional_matrix`. The default, 0, gives no
        economic cycle and the same panel as earlier releases.
    factors : array_like, optional
        Value of the standard normal systematic factor in each of the
        `n_months` months, from February 2024 on; positive values stand for
        a better economy than on average. By default, with a positive
        `correlation`, the factors are drawn from `rng`. Only with a
        positive `correlation`.
    rng : {None, int, numpy.random.Generator}, optional
        Seed or generator passed to `numpy.random.default_rng`. The global
        NumPy random state is not used, so the same seed always returns the
        same panel.

    Returns
    -------
    pandas.DataFrame
        One row per loan and month on book, sorted by loan and then by month
        on book, with the columns described in the Notes.

    Raises
    ------
    ValueError
        If `n_loans` or `n_months` is not a positive integer, `prepayment`
        is not a number from 0 to 1, `correlation` is not a number from 0 to
        below 1, `factors` is given without a positive `correlation` or does
        not hold `n_months` finite numbers, or `rng` is not a seed or
        generator that `numpy.random.default_rng` accepts.

    See Also
    --------
    kivi.vintage.vintage_table : Cumulative bad rates by cohort and month on book.
    kivi.vintage.transition_matrix : Movements between delinquency buckets.
    kivi.lifetime.survival_table : PD term structure with early repayments.
    kivi.lifetime.matrix_correlation : The correlation of the factor, from the monthly movements.
    load_bank : Real data with a binary target.

    Notes
    -----
    The columns are:

    - ``loan_id``: integer identifier, increasing with the origination month;
    - ``cohort``: the origination month, a monthly ``pandas.Period``;
    - ``month``: the month of the observation, ``cohort + month_on_book``;
    - ``month_on_book``: months since origination, from 1;
    - ``term``: contractual number of monthly instalments, 12, 24 or 36;
    - ``amount``: amount lent;
    - ``balance``: outstanding principal at the end of the month;
    - ``status``: days past due at the end of the month, an ordered
      categorical with the categories ``current``, ``1-29``, ``30-59``,
      ``60-89`` and ``90+``.

    Each loan is repaid in equal monthly instalments at 1% interest a month.
    Its status counts the months since its oldest unpaid instalment fell
    due: one is ``1-29``, two ``30-59``, three ``60-89`` and four or more
    ``90+``. Within the term this is the number of instalments past due, and
    a loan that is still behind after its term keeps rolling.

    Every month, a current loan misses its instalment with a probability that
    rises over the first six months on book and then declines, scaled by a
    risk factor of the loan and one of its cohort, so cohorts differ in
    quality. A delinquent loan pays all its arrears and becomes current, pays
    one instalment and keeps its status, or pays nothing and rolls to the
    next bucket; the chance of curing falls and that of rolling rises with
    the days past due. A ``90+`` loan makes no further payments, so its
    balance stays the same. A loan leaves the panel after its last instalment
    is paid; loans are not charged off.

    With `prepayment`, every month, a loan that was current at the end of
    the month before, as every loan is at origination, and that has more
    instalments left than the one now due repays its whole balance with
    that probability, whatever its risk. Its balance is 0 in that month and
    it has no rows after it, so a balance of 0 before the term marks an
    early repayment, while a loan repaid on schedule reaches 0 at or after
    its term. The early repayments use random numbers of their own, drawn after
    all the others: with the same `rng`, every loan follows the same path as
    without `prepayment` until it repays early.

    With `correlation`, the payments depend on a systematic factor
    :math:`z` of each month. A loan that is not yet 90 days past due
    misses its instalment, or fails to pay all of its arrears, with
    probability :math:`w`, and rolls to the next bucket, paying nothing,
    with probability :math:`v \le w`. Given the factor, each of them
    becomes

    .. math::

       \Phi\!\left(\frac{\Phi^{-1}(w) - \sqrt{\rho}\,z}{\sqrt{1 - \rho}}\right),

    the shift of `kivi.lifetime.conditional_matrix`, so averaged over the
    factor every loan keeps its probabilities. The factor moves the misses
    of current loans and the cures and rolls of delinquent ones together.
    The factors drawn by default come after all the other random numbers,
    also those of `prepayment`. The monthly movements of the whole panel
    mix loans of different risk and age, so their matrix shifts with the
    factor only approximately as `conditional_matrix` gives.

    Examples
    --------
    >>> from kivi.datasets import make_loan_panel
    >>> panel = make_loan_panel(rng=0)
    >>> panel.shape
    (59295, 8)
    >>> panel.head(3)
       loan_id   cohort    month  month_on_book  term   amount   balance   status
    0        1  2024-01  2024-02              1    24  18300.0  17621.56  current
    1        1  2024-01  2024-03              2    24  18300.0  16936.33  current
    2        1  2024-01  2024-04              3    24  18300.0  16244.25  current
    >>> panel["status"].value_counts(sort=False)
    status
    current    55246
    1-29        1490
    30-59        506
    60-89        320
    90+         1733
    Name: count, dtype: int64

    With a 2% chance a month of repaying early, 962 loans repay before
    their term, and 182 loans reach 90+ instead of 205:

    >>> repaying = make_loan_panel(prepayment=0.02, rng=0)
    >>> last = repaying.groupby("loan_id").tail(1)
    >>> early = (last["balance"] == 0) & (last["month_on_book"] < last["term"])
    >>> int(early.sum()), int((last["status"] == "90+").sum())
    (962, 182)

    In a year of bad months, more loans fall behind than in a year of good
    ones:

    >>> bad = make_loan_panel(n_months=12, correlation=0.05, factors=[-1.5] * 12, rng=0)
    >>> good = make_loan_panel(n_months=12, correlation=0.05, factors=[1.5] * 12, rng=0)
    >>> int((bad["status"] != "current").sum()), int((good["status"] != "current").sum())
    (3790, 488)
    """
    n_loans = _validation.check_count(n_loans, "n_loans")
    n_months = _validation.check_count(n_months, "n_months")
    if (
        isinstance(prepayment, (bool, np.bool_))
        or not _validation.is_real(prepayment)
        or not 0 <= prepayment <= 1
    ):
        raise ValueError(
            "prepayment must be a monthly probability from 0 to 1, got "
            f"{_validation.shown_argument(prepayment)}"
        )
    if (
        isinstance(correlation, (bool, np.bool_))
        or not _validation.is_real(correlation)
        or not 0 <= correlation < 1
    ):
        raise ValueError(
            "correlation must be a number from 0 to below 1, got "
            f"{_validation.shown_argument(correlation)}"
        )
    path = None if factors is None else _factor_path(factors, n_months, correlation)
    rng = _validation.as_generator(rng)

    cohort = np.sort(rng.integers(0, n_months, n_loans))
    amount = np.maximum(np.round(rng.lognormal(np.log(8000), 0.5, n_loans), -2), 1000.0)
    term = rng.choice(_TERMS, size=n_loans, p=_TERM_SHARES)
    loan_risk = rng.lognormal(-(_LOAN_RISK_SIGMA**2) / 2, _LOAN_RISK_SIGMA, n_loans)
    cohort_risk = rng.lognormal(-(_COHORT_RISK_SIGMA**2) / 2, _COHORT_RISK_SIGMA, n_months)
    risk = loan_risk * cohort_risk[cohort]

    paid = np.zeros(n_loans, dtype=np.int64)
    # Months since the oldest unpaid instalment fell due, 0 when nothing is due; past the term
    # it keeps growing while nothing is paid, although no new instalment falls due.
    missed = np.zeros(n_loans, dtype=np.int64)
    closed = np.zeros(n_loans, dtype=bool)
    # One draw per loan and month, row by row as in n_months calls of rng.random(n_loans); the
    # draws for early repayments come after all of them, so that they leave these unchanged.
    draws = rng.random((n_months, n_loans))
    early = rng.random((n_months, n_loans)) if prepayment > 0 else None
    if path is None and correlation > 0:
        path = rng.standard_normal(n_months)
    rows, months, paid_rows, missed_rows = [], [], [], []
    month: int | npt.NDArray[np.int_]
    for month in range(1, n_months + 1):
        draw = draws[month - 1]
        on_book = (cohort < month) & ~closed
        month_on_book = month - cohort
        due = np.minimum(month_on_book, term)
        bucket = np.minimum(missed, 4)
        seasoning = month_on_book / _SEASONING_PEAK * np.exp(1 - month_on_book / _SEASONING_PEAK)
        miss = np.minimum(_MISS_PEAK * risk * seasoning, 0.5)

        late = np.clip(bucket - 1, 0, 2)
        cure = np.where(bucket == 0, 1 - miss, _CURE[late])
        stay = np.where(bucket == 0, 0.0, _STAY[late])
        if path is not None:
            cure, stay = _given_factor(
                cure, stay, on_book & (bucket < 4), path[month - 1], correlation
            )
        pays_all = on_book & (bucket < 4) & (draw < cure)
        pays_one = on_book & (bucket < 4) & ~pays_all & (draw < cure + stay)
        paid = np.where(pays_all, due, np.where(pays_one, np.minimum(paid + 1, due), paid))
        if early is not None:
            # A current loan with instalments left after this month repays all of them.
            repays = on_book & (bucket == 0) & (month_on_book < term)
            paid = np.where(repays & (early[month - 1] < prepayment), term, paid)
        missed = np.where(on_book, np.where(paid < due, month_on_book - paid, 0), missed)

        index = np.flatnonzero(on_book)
        rows.append(index)
        months.append(np.full(len(index), month))
        paid_rows.append(paid[index])
        missed_rows.append(missed[index])
        closed |= on_book & (paid == term)

    loan = np.concatenate(rows)
    month = np.concatenate(months)
    paid_row = np.concatenate(paid_rows)
    order = np.lexsort((month, loan))
    loan, month, paid_row = loan[order], month[order], paid_row[order]
    status = np.minimum(np.concatenate(missed_rows)[order], 4)

    growth = 1 + _MONTHLY_RATE
    loan_term = term[loan]
    balance = amount[loan] * (growth**loan_term - growth**paid_row) / (growth**loan_term - 1)
    periods = pd.period_range(FIRST_COHORT, periods=n_months + 1, freq="M")
    return pd.DataFrame(
        {
            "loan_id": loan + 1,
            "cohort": periods[cohort[loan]],
            "month": periods[month],
            "month_on_book": month - cohort[loan],
            "term": loan_term,
            "amount": amount[loan],
            "balance": np.round(balance, 2),
            "status": pd.Categorical.from_codes(status, categories=STATUSES, ordered=True),  # type: ignore[call-overload]  # categories may be any list-like
        }
    )


def _factor_path(
    factors: npt.ArrayLike, n_months: int, correlation: float
) -> npt.NDArray[np.float64]:
    """Return the factors of the months as a float array."""
    if not correlation > 0:
        raise ValueError(
            "factors must be given only with a positive correlation, got correlation "
            f"{_validation.shown_argument(correlation)}"
        )
    path = _validation.as_real(factors, "factors")
    if path.shape != (n_months,):
        raise ValueError(
            f"factors must hold one number for each of the {n_months} months, got shape "
            f"{path.shape}"
        )
    invalid = ~np.isfinite(path)
    if invalid.any():
        raise ValueError(f"factors must be finite, got {_validation.shown(path[invalid][0])}")
    return path


def _given_factor(
    cure: npt.NDArray[np.float64],
    stay: npt.NDArray[np.float64],
    shifted: npt.NDArray[np.bool_],
    factor: float,
    correlation: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the chances of paying all and of paying one instalment, given the factor."""
    # The chances of not paying all and of paying nothing, the worse ends as in conditional_matrix.
    worse = np.stack([1 - cure, 1 - cure - stay])[:, shifted]
    quantile = np.full(worse.shape, -np.inf)
    possible = worse > 0
    quantile[possible] = _stats.normal_quantile(worse[possible])
    given = _stats.normal_cdf(
        (quantile - math.sqrt(correlation) * factor) / math.sqrt(1 - correlation)
    )
    cure, stay = cure.copy(), stay.copy()
    cure[shifted] = 1 - given[0]
    stay[shifted] = given[0] - given[1]
    return cure, stay
