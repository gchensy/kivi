"""Seeded generators of synthetic defaulted loans and of the cash flows of their workouts."""

from collections import abc

import numpy as np
import numpy.typing as npt
import pandas as pd

from .. import _validation
from . import _applications, _transactions

SEGMENTS = ("secured", "unsecured", "revolving")
KINDS = ("recovery", "cost", "cure")
REQUIRED_COLUMNS = (
    "loan_id",
    "segment",
    "default_date",
    "ead",
    "collateral_value",
    "closing_date",
)
CUT_OFF = "2025-06-30"

_CUT_OFF_DAY = int(np.datetime64(CUT_OFF, "D").astype(np.int64))
_FIRST_DEFAULT_DAY = int(np.datetime64("2018-01-01", "D").astype(np.int64))
_LAST_DEFAULT_DAY = int(np.datetime64("2024-12-31", "D").astype(np.int64))
_SECURED, _UNSECURED, _REVOLVING = range(len(SEGMENTS))
_RECOVERY, _COST, _CURE = range(len(KINDS))
_SEGMENT_SHARES = [0.40, 0.35, 0.25]

# A workout lasts at least _MIN_WORKOUT days, the probation of a cure, plus an exponential number
# of days whose mean depends on whether the loan cures or is liquidated. The shares of cures and
# the means are those of each segment.
_MIN_WORKOUT = 91
_CURE_SHARE = np.array([0.30, 0.10, 0.15])
_CURE_MEAN = 120.0
_LIQUIDATION_MEAN = np.array([800.0, 450.0, 365.0])

# Exposures of the term loans, lognormal by segment (secured, unsecured), rounded to cents.
_EAD_MEDIAN = np.array([120_000.0, 12_000.0])
_EAD_SIGMA = np.array([0.5, 0.7])
_MIN_EAD = np.array([5_000.0, 500.0])
# Secured loans: the loan-to-value ratio at default is lognormal, and the collateral value is
# rounded to hundreds.
_LTV_MEDIAN = 0.75
_LTV_SIGMA = 0.3
# Revolving lines 12 months before default: a limit rounded to hundreds and a utilisation; a few
# lines are fully drawn, at or above the limit.
_LIMIT_MEDIAN = 6_000.0
_LIMIT_SIGMA = 0.6
_MIN_LIMIT = 500.0
_UTILISATION = (3.0, 2.0)
_AT_LIMIT = 0.03
_OVER_LIMIT = 0.03
_OVER_LIMIT_MAX = 0.1
# Revolving lines from the reference date to default: most draw a share of the undrawn amount,
# some pay down part of the balance and some draw beyond the limit.
_DRAWN_SHARE = (1.5, 1.2)
_PAID_DOWN = 0.1
_PAID_DOWN_MIN = 0.6
_BEYOND_LIMIT = 0.1
_BEYOND_LIMIT_MAX = 0.3
_FULLY_DRAWN_CHANGE = (0.98, 1.15)
_MIN_LINE_EAD = 1.0
# The annual discount rate: a three-month money-market rate interpolated between these months,
# plus a margin of five percentage points (EBA/GL/2017/16, paragraph 143).
_BASE_RATE_MONTHS = ("2018-01", "2021-12", "2022-12", "2023-10", "2024-12")
_BASE_RATES = (0.0, 0.0, 0.025, 0.04, 0.03)
_RATE_MARGIN = 0.05

# Cures: the borrower pays part of the arrears during the probation, and a few small costs.
_ARREARS = (0.01, 0.05)
_ARREARS_PAYMENTS = 1.0
_CURE_COSTS = 0.5
_CURE_COST = (30.0, 0.001)
# Secured liquidations: a few payments by the borrower, then the sale of the collateral on the
# closing date. The bank recovers the proceeds up to its claim, which grows with the default
# interest, and pays the costs of the sale and legal costs.
_SECURED_PAYMENTS = 1.5
_SECURED_PAYMENT = (0.003, 0.012)
_SALE_MEDIAN = 0.85
_SALE_SIGMA = 0.12
_SALE_COST = 0.03
_DEFAULT_INTEREST = (0.04, 0.10)
_LEGAL_COSTS = 1.0
_LEGAL_COST = (200.0, 0.002)
# Unsecured and revolving liquidations: a beta share of the exposure is recovered in several
# payments, with collection costs.
_RECOVERY_SHARE = np.array([[0.0, 0.0], [1.2, 6.0], [1.3, 4.5]])
_RECOVERY_PAYMENTS = 4.0
_COLLECTION_COSTS = 1.5
_COLLECTION_COST = (50.0, 0.004)
# Costs that do not depend on the outcome: legal costs before default and litigation.
_PRE_DEFAULT = 0.05
_PRE_DEFAULT_DAYS = 90
_PRE_DEFAULT_COST = (100.0, 0.002)
_LITIGATION = 0.03
_LITIGATION_COST = (0.1, 0.4)
_COST_SPREAD = 0.4


def make_defaults(
    *,
    n_defaults: int = 2000,
    rng: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate synthetic defaulted loans with their exposures and workouts.

    Every loan defaulted between 2018 and 2024 and is secured by collateral,
    unsecured or a revolving line of credit. The table holds what a lender
    knows about the loan at default, the limit and drawn amount of the
    revolving lines 12 months earlier, and the date the workout ended, if it
    ended by the data cut-off of 30 June 2025.
    `kivi.datasets.make_recoveries` simulates the cash flows of the
    workouts. The data is simulated, not drawn from a real portfolio.

    Parameters
    ----------
    n_defaults : int, default 2000
        Number of defaulted loans.
    rng : {None, int, numpy.random.Generator}, optional
        Seed or generator passed to `numpy.random.default_rng`. The global
        NumPy random state is not used, so the same seed always returns the
        same loans.

    Returns
    -------
    pandas.DataFrame
        One row per defaulted loan, sorted by default date, with the columns
        described in the Notes and a RangeIndex.

    Raises
    ------
    ValueError
        If `n_defaults` is not a positive integer, or `rng` is not a seed or
        generator that `numpy.random.default_rng` accepts.

    See Also
    --------
    make_recoveries : Synthetic cash flows of the workouts of these loans.
    make_loan_panel : Synthetic monthly history of a loan portfolio.

    Notes
    -----
    The columns are:

    - ``loan_id``: integer identifier from 1, increasing with the default
      date;
    - ``segment``: a categorical with the categories ``secured``,
      ``unsecured`` and ``revolving``, about 40%, 35% and 25% of the loans;
    - ``default_date``: the date of default, from 2018 to 2024;
    - ``ead``: the exposure at default, the amount outstanding on that date,
      rounded to cents;
    - ``limit``: the limit of a revolving line 12 months before default,
      the reference date of the CCF, rounded to hundreds, and missing for
      the other segments;
    - ``drawn``: the amount drawn on a revolving line on the reference date,
      rounded to cents, and missing for the other segments;
    - ``collateral_value``: the value of the collateral of a secured loan at
      default, rounded to hundreds, and 0 for the other segments;
    - ``discount_rate``: the annual rate at which to discount the cash flows
      of the workout to the default date, a simulated three-month
      money-market rate of the month of default plus five percentage points,
      as paragraph 143 of EBA/GL/2017/16 prescribes;
    - ``closing_date``: the date the workout ended, with the cure, the sale
      of the collateral or the write-off of what was not recovered, and
      missing for the workouts still open at the cut-off of 30 June 2025.

    A workout lasts at least three months, the probation of a cure. About
    30% of the secured loans, 10% of the unsecured loans and 15% of the
    revolving lines cure, after seven months on average. A liquidation lasts
    about two and a half years on average for secured loans, a year and a
    half for unsecured loans and 15 months for revolving lines, so many of
    the workouts that started in the last years are still open at the
    cut-off.

    The secured loans have a median exposure of 120,000 and a median
    loan-to-value ratio of 75% at default, and some owe more than their
    collateral is worth; the unsecured loans have a median exposure of
    12,000.
    The revolving lines have a median limit of 6,000 and are about 60% drawn
    on the reference date, while 3% are drawn exactly to their limit and 3%
    beyond it. Most then draw part of the remaining limit before default,
    10% pay down part of their balance and 10% end above their limit, so
    that the realized CCF, ``(ead - drawn) / (limit - drawn)``, mostly lies
    from 0 to 1, with some values below 0 and above 1; it is undefined for
    the fully drawn lines.

    Examples
    --------
    >>> from kivi.datasets import make_defaults
    >>> defaults = make_defaults(rng=0)
    >>> defaults.shape
    (2000, 9)
    >>> defaults["segment"].value_counts(sort=False)
    segment
    secured      846
    unsecured    665
    revolving    489
    Name: count, dtype: int64

    The workouts of 278 loans, 13.9%, are still open at the cut-off:

    >>> int(defaults["closing_date"].isna().sum())
    278
    """
    n_defaults = _validation.check_count(n_defaults, "n_defaults")
    rng = _validation.as_generator(rng)

    default_day = np.sort(rng.integers(_FIRST_DEFAULT_DAY, _LAST_DEFAULT_DAY + 1, n_defaults))
    segment = rng.choice(len(SEGMENTS), size=n_defaults, p=_SEGMENT_SHARES)
    cure = rng.random(n_defaults) < _CURE_SHARE[segment]
    mean = np.where(cure, _CURE_MEAN, _LIQUIDATION_MEAN[segment])
    duration = _MIN_WORKOUT + np.floor(rng.exponential(mean)).astype(np.int64)
    closing_day = default_day + duration

    term = np.minimum(segment, _UNSECURED)
    term_ead = np.maximum(
        _EAD_MEDIAN[term] * np.exp(_EAD_SIGMA[term] * rng.standard_normal(n_defaults)),
        _MIN_EAD[term],
    )
    ltv = _LTV_MEDIAN * np.exp(_LTV_SIGMA * rng.standard_normal(n_defaults))
    limit, drawn, line_ead = _revolving_lines(n_defaults, rng)

    revolving = segment == _REVOLVING
    secured = segment == _SECURED
    ead = np.maximum(np.round(np.where(revolving, line_ead, term_ead), 2), _MIN_LINE_EAD)
    collateral = np.where(secured, np.round(ead / ltv, -2), 0.0)
    dates = default_day.astype("datetime64[D]")
    month = dates.astype("datetime64[M]").astype(np.int64)
    knots = np.array(_BASE_RATE_MONTHS, dtype="datetime64[M]").astype(np.int64)
    base_rate = np.interp(month, knots, _BASE_RATES)
    closing = closing_day.astype("datetime64[D]").astype("datetime64[ns]")
    closing[closing_day > _CUT_OFF_DAY] = np.datetime64("NaT", "ns")
    return pd.DataFrame(
        {
            "loan_id": np.arange(1, n_defaults + 1, dtype=np.int64),
            "segment": pd.Categorical.from_codes(segment, categories=SEGMENTS),  # type: ignore[call-overload]  # categories may be any list-like
            "default_date": dates.astype("datetime64[ns]"),
            "ead": ead,
            "limit": np.where(revolving, limit, np.nan),
            "drawn": np.where(revolving, drawn, np.nan),
            "collateral_value": collateral,
            "discount_rate": np.round(base_rate + _RATE_MARGIN, 4),
            "closing_date": closing,
        }
    )


def _revolving_lines(
    n: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Limit and drawn amount on the reference date, and the exposure at default, of n lines."""
    limit = np.maximum(
        np.round(_LIMIT_MEDIAN * np.exp(_LIMIT_SIGMA * rng.standard_normal(n)), -2), _MIN_LIMIT
    )
    use = rng.beta(*_UTILISATION, n)
    fully_drawn = rng.random(n)
    excess = 1 + _OVER_LIMIT_MAX * rng.random(n)
    use = np.where(
        fully_drawn < _AT_LIMIT,
        1.0,
        np.where(fully_drawn < _AT_LIMIT + _OVER_LIMIT, excess, use),
    )
    drawn = np.round(limit * use, 2)

    path = rng.random(n)
    change = rng.random(n)
    drawn_share = rng.beta(*_DRAWN_SHARE, n)
    low, high = _FULLY_DRAWN_CHANGE
    ead = np.where(
        drawn >= limit,
        drawn * (low + (high - low) * change),
        np.where(
            path < _PAID_DOWN,
            drawn * (_PAID_DOWN_MIN + (1 - _PAID_DOWN_MIN) * change),
            np.where(
                path < _PAID_DOWN + _BEYOND_LIMIT,
                limit * (1 + _BEYOND_LIMIT_MAX * change),
                drawn + drawn_share * (limit - drawn),
            ),
        ),
    )
    return limit, drawn, ead


def make_recoveries(
    defaults: pd.DataFrame,
    *,
    rng: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate synthetic cash flows of the workouts of defaulted loans.

    Every loan of `defaults` gets the recoveries, costs and cure of its
    workout, from its default date up to its closing date or, for a workout
    still open, up to the data cut-off of 30 June 2025. Secured loans
    mostly recover their exposure from the sale of the collateral, unsecured
    loans and revolving lines little of it, and cured loans all of it, so
    the realized LGDs are bimodal. The data is simulated, not drawn from a
    real portfolio.

    Parameters
    ----------
    defaults : pandas.DataFrame
        One row per defaulted loan with the columns ``loan_id``,
        ``segment``, ``default_date``, ``ead``, ``collateral_value`` and
        ``closing_date`` of `kivi.datasets.make_defaults`. The dates must
        have a datetime64 dtype; a time of day is ignored. Other columns are
        ignored.
    rng : {None, int, numpy.random.Generator}, optional
        Seed or generator passed to `numpy.random.default_rng`. The global
        NumPy random state is not used, so the same seed always returns the
        same cash flows.

    Returns
    -------
    pandas.DataFrame
        One row per cash flow, sorted by loan, in the order of `defaults`,
        and then by date, with the columns described in the Notes and a
        RangeIndex.

    Raises
    ------
    ValueError
        If `defaults` is not a DataFrame, lacks a required column or has two
        columns of that name, has a missing or repeated ``loan_id``, a
        ``segment`` other than ``secured``, ``unsecured`` or ``revolving``,
        an ``ead`` that is not finite and positive, a ``collateral_value``
        that is not finite and non-negative, dates without a datetime64
        dtype, a missing ``default_date``, a ``default_date`` before
        1678 or after the cut-off, or a ``closing_date`` before the
        ``default_date`` or after the cut-off. If `rng` is not a seed or
        generator that `numpy.random.default_rng` accepts.

    See Also
    --------
    make_defaults : Synthetic defaulted loans with their exposures and workouts.
    kivi.calibration.backtest_table : Back-test LGD estimates against realized LGDs.

    Notes
    -----
    The columns of the result are:

    - ``loan_id``: the ``loan_id`` of the loan in `defaults`;
    - ``date``: the date of the cash flow;
    - ``amount``: the amount, rounded to cents, positive for recoveries and
      cures and negative for costs;
    - ``kind``: a categorical with the categories ``recovery``, ``cost`` and
      ``cure``.

    A ``cure`` is the amount outstanding on the closing date of a cured
    loan, which paragraph 135 of EBA/GL/2017/16 treats as if the borrower
    had repaid it on that date. The realized LGD of a loan is
    ``(ead + PV(costs) - PV(recoveries)) / ead``, where recoveries include
    the cure and the present values discount the cash flows to the default
    date. Discounted with the ``discount_rate`` of `make_defaults`, it is
    close to 0 for most secured loans and for cured loans, close to 1 for
    unsecured loans and revolving lines that do not cure, below 0 for a few
    secured loans whose collateral covered the claim including the default
    interest, and above 1 for a few loans whose costs exceeded their
    recoveries.

    A workout does not depend on the dates in `defaults` other than through
    its length, and short workouts are more likely to be cures:

    - a cured loan pays part of its arrears during the workout, sometimes
      with a small cost, and the cure follows on the closing date;
    - a secured loan that does not cure pays a little during the workout,
      and the collateral is sold on the closing date. The bank recovers the
      proceeds up to its claim, the exposure plus default interest at 4% to
      10% a year, and pays 3% of the proceeds as costs of the sale and a
      few legal costs;
    - an unsecured loan or a revolving line that does not cure recovers
      about 17% or 22% of its exposure on average, in several payments,
      with a few collection costs.

    About 5% of the loans have a legal cost up to 90 days before default,
    which paragraph 144 of EBA/GL/2017/16 counts in the economic loss, and
    3% of the liquidations have litigation costs of 10% to 40% of the
    exposure. A workout still open at the cut-off has the cash flows up to
    the cut-off and none after it, so its LGD so far overstates the LGD it
    will realize.

    Examples
    --------
    >>> from kivi.datasets import make_defaults, make_recoveries
    >>> defaults = make_defaults(rng=0)
    >>> flows = make_recoveries(defaults, rng=0)
    >>> flows.head(7)
       loan_id       date     amount      kind
    0        1 2018-04-10    -808.71      cost
    1        1 2018-11-01   -1543.90      cost
    2        1 2019-05-14    -552.91      cost
    3        1 2019-07-11    3595.22  recovery
    4        1 2019-08-02    2012.10  recovery
    5        1 2019-08-17  233970.35  recovery
    6        1 2019-08-17   -7019.11      cost
    >>> flows["kind"].value_counts(sort=False)
    kind
    recovery    6806
    cost        4278
    cure         413
    Name: count, dtype: int64

    The first loan is secured: after a few legal costs and two payments by
    the borrower, the collateral was sold on the closing date, 17 August
    2019, at a cost of 3% of the proceeds.
    """
    rng = _validation.as_generator(rng)
    loan_id, segment, default_day, closing_day, ead, collateral = _workouts(defaults)
    n = len(defaults)
    is_open = closing_day < 0
    length = np.where(is_open, _CUT_OFF_DAY - default_day, closing_day - default_day)

    # Whether the workout is a cure, given its length, or for an open workout given that it lasts
    # longer than the time to the cut-off: the posterior of the mixture that make_defaults draws.
    cure_mean = _CURE_MEAN
    liquidation_mean = _LIQUIDATION_MEAN[segment]
    beyond = np.maximum(length - _MIN_WORKOUT, 0)
    log_odds = (
        np.log(_CURE_SHARE / (1 - _CURE_SHARE))[segment]
        - beyond / cure_mean
        + beyond / liquidation_mean
        + np.where(is_open, 0.0, np.log(liquidation_mean / cure_mean))
    )
    cure = rng.random(n) < _applications._logistic(log_odds)
    # An open workout ends after the cut-off, so it lasts longer than the time to the cut-off.
    remaining = rng.exponential(np.where(cure, cure_mean, liquidation_mean))
    latent = np.maximum(length, _MIN_WORKOUT) + np.floor(remaining).astype(np.int64)
    duration = np.where(is_open, np.maximum(latent, length + 1), length)
    end_day = default_day + duration

    flows = _Flows(rng, default_day, duration)
    pre_default = rng.random(n) < _PRE_DEFAULT
    flows.add(
        np.flatnonzero(pre_default),
        default_day[pre_default] - rng.integers(1, _PRE_DEFAULT_DAYS + 1, pre_default.sum()),
        -_cost(_PRE_DEFAULT_COST, ead[pre_default], rng),
        _COST,
    )

    cured = np.flatnonzero(cure)
    arrears = ead[cured] * rng.uniform(*_ARREARS, len(cured))
    flows.spread(cured, 1 + rng.poisson(_ARREARS_PAYMENTS, len(cured)), arrears, _RECOVERY)
    flows.add(cured, end_day[cured], ead[cured] - arrears, _CURE)
    owner = np.repeat(cured, rng.poisson(_CURE_COSTS, len(cured)))
    flows.within(owner, -_cost(_CURE_COST, ead[owner], rng), _COST)

    sold = np.flatnonzero(~cure & (segment == _SECURED))
    payments = np.repeat(sold, rng.poisson(_SECURED_PAYMENTS, len(sold)))
    payment = ead[payments] * rng.uniform(*_SECURED_PAYMENT, len(payments))
    flows.within(payments, payment, _RECOVERY)
    proceeds = (
        collateral[sold] * _SALE_MEDIAN * np.exp(_SALE_SIGMA * rng.standard_normal(len(sold)))
    )
    interest = rng.uniform(*_DEFAULT_INTEREST, len(sold))
    claim = ead[sold] * (1 + interest * duration[sold] / 365)
    paid = np.bincount(np.searchsorted(sold, payments), weights=payment, minlength=len(sold))
    flows.add(sold, end_day[sold], np.maximum(np.minimum(proceeds, claim - paid), 0.0), _RECOVERY)
    flows.add(sold, end_day[sold], -_SALE_COST * proceeds, _COST)
    owner = np.repeat(sold, 1 + rng.poisson(_LEGAL_COSTS, len(sold)))
    flows.within(owner, -_cost(_LEGAL_COST, ead[owner], rng), _COST)

    written_off = np.flatnonzero(~cure & (segment != _SECURED))
    shapes = _RECOVERY_SHARE[segment[written_off]]
    recovered = ead[written_off] * rng.beta(shapes[:, 0], shapes[:, 1])
    count = 1 + rng.poisson(_RECOVERY_PAYMENTS, len(written_off))
    flows.spread(written_off, count, recovered, _RECOVERY)
    owner = np.repeat(written_off, 1 + rng.poisson(_COLLECTION_COSTS, len(written_off)))
    flows.within(owner, -_cost(_COLLECTION_COST, ead[owner], rng), _COST)

    liquidated = np.flatnonzero(~cure)
    litigated = liquidated[rng.random(len(liquidated)) < _LITIGATION]
    flows.within(litigated, -ead[litigated] * rng.uniform(*_LITIGATION_COST, len(litigated)), _COST)
    return flows.frame(loan_id)


def _cost(
    cost: tuple[float, float], ead: npt.NDArray[np.float64], rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Positive costs of a fixed part plus a share of the exposure, with lognormal noise."""
    fixed, share = cost
    return (fixed + share * ead) * np.exp(_COST_SPREAD * rng.standard_normal(len(ead)))


class _Flows:
    """Cash flows collected by loan position, day, amount and kind."""

    def __init__(
        self,
        rng: np.random.Generator,
        default_day: npt.NDArray[np.int64],
        duration: npt.NDArray[np.int64],
    ) -> None:
        self.rng = rng
        self.default_day = default_day
        self.duration = duration
        self.parts: list[tuple[npt.NDArray[np.int64], ...]] = []
        self.amounts: list[npt.NDArray[np.float64]] = []

    def add(
        self,
        loans: npt.NDArray[np.int64],
        days: npt.NDArray[np.int64],
        amounts: npt.NDArray[np.float64],
        kind: int,
    ) -> None:
        """Add one cash flow of each loan at the given position, on the given day."""
        self.parts.append((loans, days, np.full(len(loans), kind)))
        self.amounts.append(amounts)

    def within(
        self, loans: npt.NDArray[np.int64], amounts: npt.NDArray[np.float64], kind: int
    ) -> None:
        """Add cash flows on days drawn uniformly from the default day to before the end."""
        days = self.default_day[loans] + np.floor(
            self.rng.random(len(loans)) * self.duration[loans]
        ).astype(np.int64)
        self.add(loans, days, amounts, kind)

    def spread(
        self,
        loans: npt.NDArray[np.int64],
        count: npt.NDArray[np.int64],
        total: npt.NDArray[np.float64],
        kind: int,
    ) -> npt.NDArray[np.int64]:
        """Split each total into `count` payments of random size within the workout.

        Returns the position in `loans` of each payment.
        """
        owner = np.repeat(np.arange(len(loans)), count)
        weight = self.rng.exponential(1.0, len(owner))
        share = weight / np.bincount(owner, weights=weight, minlength=len(loans))[owner]
        self.within(loans[owner], total[owner] * share, kind)
        return owner

    def frame(self, loan_id: pd.Index) -> pd.DataFrame:
        """Return the cash flows up to the cut-off with a non-zero amount, by loan and day."""
        loans, days, kinds = (np.concatenate(column) for column in zip(*self.parts, strict=True))
        amounts = np.round(np.concatenate(self.amounts), 2)
        kept = (days <= _CUT_OFF_DAY) & (amounts != 0)
        loans, days, kinds, amounts = loans[kept], days[kept], kinds[kept], amounts[kept]
        order = np.lexsort((kinds, days, loans))
        return pd.DataFrame(
            {
                "loan_id": loan_id.take(loans[order]),
                "date": days[order].astype("datetime64[D]").astype("datetime64[ns]"),
                "amount": amounts[order],
                "kind": pd.Categorical.from_codes(kinds[order], categories=KINDS),  # type: ignore[call-overload]  # categories may be any list-like
            }
        )


def _workouts(
    defaults: pd.DataFrame,
) -> tuple[
    pd.Index,
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Loan ids, segment codes, default and closing days (-1 if open), exposures and collateral."""
    if not isinstance(defaults, pd.DataFrame):
        raise ValueError(f"defaults must be a DataFrame, got {type(defaults).__name__}")
    absent = [name for name in REQUIRED_COLUMNS if name not in defaults.columns]
    if absent:
        raise ValueError(
            f"defaults must have the columns {_validation.shown_list(REQUIRED_COLUMNS)}, "
            f"got no column {_validation.shown_list(absent)}"
        )
    loan_id = pd.Index(_column(defaults, "loan_id"))
    if loan_id.hasnans:
        raise ValueError(
            f"column 'loan_id' must not contain missing values, got {loan_id.isna().sum()}"
        )
    if loan_id.has_duplicates:
        raise ValueError(
            "column 'loan_id' must hold unique values, got "
            f"{_validation.shown(loan_id[loan_id.duplicated()])} twice"
        )
    values = _column(defaults, "segment")
    segment = pd.Index(SEGMENTS).get_indexer(pd.Index(values)).astype(np.int64)
    unknown = segment < 0
    if unknown.any():
        raise ValueError(
            f"column 'segment' must hold {_validation.shown_list(SEGMENTS)}, got "
            f"{_validation.shown(values[unknown])}"
        )
    default_day, missing = _days(defaults, "default_date")
    if missing.any():
        raise ValueError(
            f"column 'default_date' must not contain missing values, got {missing.sum()}"
        )
    outside = (default_day < _transactions._FIRST_DAY) | (default_day > _CUT_OFF_DAY)
    if outside.any():
        raise ValueError(
            f"column 'default_date' must lie between 1678-01-01 and the cut-off {CUT_OFF}, "
            f"got {_validation.shown(defaults['default_date'][outside])}"
        )
    closing_day, is_open = _days(defaults, "closing_date")
    early = ~is_open & (closing_day < default_day)
    if early.any():
        raise ValueError(
            "column 'closing_date' must not lie before the default date, got "
            f"{_validation.shown(defaults['closing_date'][early])}"
        )
    late = ~is_open & (closing_day > _CUT_OFF_DAY)
    if late.any():
        raise ValueError(
            f"column 'closing_date' must not lie after the cut-off {CUT_OFF}, got "
            f"{_validation.shown(defaults['closing_date'][late])}"
        )
    ead = _real(defaults, "ead", lambda v: np.isfinite(v) & (v > 0), "finite and positive")
    collateral = _real(
        defaults,
        "collateral_value",
        lambda v: np.isfinite(v) & (v >= 0),
        "finite and non-negative",
    )
    return loan_id, segment, default_day, np.where(is_open, -1, closing_day), ead, collateral


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return the column ``name`` of the frame, which must be the only one of that name."""
    if (frame.columns == name).sum() > 1:
        raise ValueError(f"defaults has more than one column named {_validation.shown(name)}")
    return frame[name]


def _days(frame: pd.DataFrame, name: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """Days since 1970-01-01 of a datetime64 column, without the time of day, and where missing."""
    values = _column(frame, name)
    if not pd.api.types.is_datetime64_dtype(values.dtype):
        raise ValueError(
            f"column {_validation.shown(name)} must have a datetime64 dtype, got dtype "
            f"{values.dtype}"
        )
    missing = values.isna().to_numpy(dtype=bool)
    days = values.to_numpy().astype("datetime64[D]").astype(np.int64)
    return np.where(missing, 0, days), missing


def _real(
    frame: pd.DataFrame,
    name: str,
    valid: abc.Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool_]],
    requirement: str,
) -> npt.NDArray[np.float64]:
    """Values of a numeric column, each of which ``valid`` accepts."""
    label = f"column {_validation.shown(name)}"
    values = _validation.as_real(_column(frame, name), label)
    invalid = ~valid(values)
    if invalid.any():
        raise ValueError(
            f"{label} must be {requirement}, got {_validation.shown(values[invalid][0])}"
        )
    return values
