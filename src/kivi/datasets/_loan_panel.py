"""Seeded generator of a synthetic monthly panel of instalment loans."""

from numbers import Integral

import numpy as np
import pandas as pd

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
    *, n_loans: int = 5000, n_months: int = 24, rng: int | np.random.Generator | None = None
) -> pd.DataFrame:
    """Generate a synthetic monthly panel of instalment loans.

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
        If `n_loans` or `n_months` is not a positive integer.

    See Also
    --------
    kivi.vintage.vintage_table : Cumulative bad rates by cohort and month on book.
    kivi.vintage.transition_matrix : Movements between delinquency buckets.
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
    """
    n_loans = _positive_integer(n_loans, "n_loans")
    n_months = _positive_integer(n_months, "n_months")
    rng = np.random.default_rng(rng)

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
    rows, months, paid_rows, missed_rows = [], [], [], []
    for month in range(1, n_months + 1):
        draw = rng.random(n_loans)
        on_book = (cohort < month) & ~closed
        month_on_book = month - cohort
        due = np.minimum(month_on_book, term)
        bucket = np.minimum(missed, 4)
        seasoning = month_on_book / _SEASONING_PEAK * np.exp(1 - month_on_book / _SEASONING_PEAK)
        miss = np.minimum(_MISS_PEAK * risk * seasoning, 0.5)

        late = np.clip(bucket - 1, 0, 2)
        cure = np.where(bucket == 0, 1 - miss, _CURE[late])
        stay = np.where(bucket == 0, 0.0, _STAY[late])
        pays_all = on_book & (bucket < 4) & (draw < cure)
        pays_one = on_book & (bucket < 4) & ~pays_all & (draw < cure + stay)
        paid = np.where(pays_all, due, np.where(pays_one, np.minimum(paid + 1, due), paid))
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
            "status": pd.Categorical.from_codes(status, categories=STATUSES, ordered=True),
        }
    )


def _positive_integer(value, name: str) -> int:
    """``value`` as an int, or a ValueError naming the argument."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)
