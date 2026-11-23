"""Seeded generator of synthetic account transactions before the application dates."""

import datetime
import typing
from collections import abc

import numpy as np
import numpy.typing as npt
import pandas as pd

from .. import _validation
from . import _applications

KINDS = ("purchase", "cash_withdrawal", "repayment")
MERCHANT_CATEGORIES = (
    "grocery",
    "fuel",
    "dining",
    "travel",
    "retail",
    "online",
    "health",
    "utilities",
)
REQUIRED_COLUMNS = ("income", "debt_to_income", "incumbent_pd", "target")
FIRST_DEFAULT_DATE = pd.Timestamp("2024-01-01")

_SECONDS_PER_DAY = 86_400
# The dates that pandas holds as nanosecond timestamps, with a margin, as days and months since
# 1970.
_FIRST_DAY = int(np.datetime64("1678-01-01", "D").astype(np.int64))
_LAST_DAY = int(np.datetime64("2261-12-31", "D").astype(np.int64))
_FIRST_MONTH = int(np.datetime64("1678-01", "M").astype(np.int64))
_DAYS_IN_MONTH_2024 = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# Purchases: a Poisson count scaled by income, by a trait of the applicant and by the season,
# a little lower under stress.
_PURCHASE_RATE = 2.0
_PURCHASE_INCOME_EXPONENT = 0.3
_PURCHASE_TRAIT = 0.3
_PURCHASE_STRESS = 0.15
_PURCHASE_SHARE = 0.045
_PURCHASE_SPREAD = 0.7
# Multipliers of the purchase rate by the calendar month in which a month window starts,
# January first.
_SEASON = np.array([0.90, 0.88, 0.95, 0.98, 1.00, 1.00, 1.02, 1.00, 0.98, 1.00, 1.12, 1.37])
_MERCHANT_SHARES = np.array([0.26, 0.14, 0.14, 0.04, 0.16, 0.18, 0.04, 0.04])
# Cash withdrawals: a Poisson count that rises with the stress and with the incumbent's view of
# the applicant's risk.
_CASH_RATE = 0.4
_CASH_INCOME_EXPONENT = 0.2
_CASH_TRAIT = 0.5
_CASH_VIEW = 0.25
_CASH_STRESS = 0.9
_CASH_SHARE = 0.03
_CASH_SPREAD = 0.3
_CASH_AMOUNT_STRESS = 0.25
# Repayments: one instalment of income x debt_to_income a month, missed with a probability that
# rises with the stress and paid short under stress.
_MISS_LOGIT = float(np.log(0.03 / (1 - 0.03)))
_MISS_VIEW = 0.3
_MISS_STRESS = 1.8
_SHORT_MAX = 0.1
_INSTALMENT_SPREAD = 0.02
_INSTALMENT_TRAIT = 0.2
_MIN_DEBT_TO_INCOME = 0.01
# Stress: it builds up over the months before the application; defaulters have it, and a share
# of the other applicants has a milder episode.
_ONSET = (3, 10)
_STRESS_SPREAD = 0.8
_DEFAULTER_STRESS = 0.45
_EPISODE_SHARE = 0.25
_EPISODE_STRESS = 0.35
# The incumbent PD enters through its logit, centred and scaled as make_applications draws it.
_VIEW_CENTRE = -2.13
_VIEW_SCALE = 0.77
_MIN_PD = 1e-4
_MIN_AMOUNT = 1.0


def make_transactions(
    applications: pd.DataFrame,
    *,
    application_date: str | datetime.date | np.datetime64 | npt.ArrayLike | None = None,
    months: int = 12,
    rng: int | np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic account transactions before the application dates.

    Every applicant has transactions on an account in each of the `months`
    months before the application: card purchases, cash withdrawals and
    instalment repayments. Defaulters withdraw more cash as the application
    nears and repay less, and the volumes scale with income, so features of
    the transactions in time windows carry a clear but imperfect signal about
    ``target``. The data is simulated, not drawn from a real portfolio.

    Parameters
    ----------
    applications : pandas.DataFrame
        One row per applicant with the columns ``income``,
        ``debt_to_income``, ``incumbent_pd`` and ``target`` of
        `kivi.datasets.make_applications` and unique index labels. Other
        columns are ignored.
    application_date : {None, str, date, array_like}, optional
        The date of each application. A string that names a column of
        `applications` reads the dates from that column. Any other string, a
        `datetime.date`, a `numpy.datetime64` or a `pandas.Timestamp` is
        one date for every application. A Series with the index of
        `applications`, or an array_like with one date per row, gives each
        application its own. A time of day is ignored and a time zone is
        dropped, as in `kivi.vintage.delinquency_panel`. By default the
        applications are spread evenly over the twelve months of 2024 in the
        order of their rows: the first twelfth of the rows in January, the
        next in February and so on, and within a month evenly over its days.
    months : int, default 12
        Number of months before the application date that the transactions
        cover. The transactions of an applicant lie from `months` calendar
        months before the application date, counted as `pandas.DateOffset`
        counts them, up to but not including the application date.
    rng : {None, int, numpy.random.Generator}, optional
        Seed or generator passed to `numpy.random.default_rng`. The global
        NumPy random state is not used, so the same seed always returns the
        same transactions.

    Returns
    -------
    transactions : pandas.DataFrame
        One row per transaction, sorted by applicant and then by time, with
        the columns described in the Notes and a RangeIndex.
    observations : pandas.DataFrame
        One row per application, in the order and with the index of
        `applications`, with the columns ``applicant_id`` (the index label)
        and ``application_date`` (a datetime at midnight). It is the table of
        observations for the time windows, and its index joins the features
        back to `applications`.

    Raises
    ------
    ValueError
        If `applications` is not a DataFrame, lacks a required column, has
        repeated or non-flat index labels, or holds a missing, non-numeric or
        out-of-range value in a required column. If `application_date` is
        not a column name, a date or dates with one value per application,
        holds a missing value, or is a Series with an index different from
        that of `applications`. If `months` is not a positive integer or
        reaches back before the year 1678, or `rng` is not a seed or
        generator that `numpy.random.default_rng` accepts.

    See Also
    --------
    make_applications : Synthetic loan applications with an approval decision.
    make_loan_panel : Synthetic monthly history of a loan portfolio.

    Notes
    -----
    The columns of `transactions` are:

    - ``applicant_id``: the index label of the applicant in `applications`;
    - ``timestamp``: the time of the transaction, to the second, on a day
      before the application date;
    - ``amount``: the amount, positive and rounded to cents;
    - ``kind``: a categorical with the categories ``purchase``,
      ``cash_withdrawal`` and ``repayment``;
    - ``merchant_category``: a categorical with eight categories such as
      ``grocery`` and ``travel`` for a purchase, missing for the other kinds.

    The transactions of each month are drawn independently. In a year an
    applicant has on average about 26 purchases, 6 cash withdrawals and 12
    repayments, 43 transactions in all. A purchase costs about 4.5% of the
    monthly income, a cash withdrawal about 3%, and a repayment is the
    instalment that ``income * debt_to_income`` implies, which varies by
    about 20% between applicants. Applicants with a higher ``income`` make
    a few more purchases and withdrawals, and purchases are more frequent in
    the months that start in November and December. An applicant whom the
    incumbent scorecard rates as riskier withdraws more cash and misses
    more repayments.

    The outcome drives a stress that builds up before the application. For a
    defaulter it starts between three and nine months before it and is
    strongest in the last month, and 25% of the other applicants have an
    episode of the same shape that is a little milder. Under stress, the
    cash withdrawals are more frequent and larger, the purchases a little
    rarer, and a repayment is more likely to be missed, or paid up to a tenth
    short. The number and sum of the cash withdrawals of the last months,
    the number of repayments and the change of their sum from the months
    before therefore carry a signal about ``target``. Applicants differ in
    their baseline and in the size of their stress, so no feature predicts
    the outcome well.

    The transactions are drawn from the outcome, which a lender learns only
    after the transactions. Use them to learn how window features work, not
    to estimate how well behaviour predicts default.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.datasets import make_applications, make_transactions
    >>> applications = make_applications(n_applicants=1000, rng=0)
    >>> transactions, observations = make_transactions(applications, rng=0)
    >>> list(transactions.columns)  # doctest: +NORMALIZE_WHITESPACE
    ['applicant_id', 'timestamp', 'amount', 'kind', 'merchant_category']
    >>> observations.head(2)
       applicant_id application_date
    0             0       2024-01-01
    1             1       2024-01-01

    The dates of the applications that the data holds are passed as the name
    of a column, as one date or as a Series:

    >>> applications["decided"] = pd.Timestamp("2024-06-30")
    >>> transactions, observations = make_transactions(
    ...     applications, application_date="decided", months=6, rng=0
    ... )
    >>> transactions["timestamp"].max() < pd.Timestamp("2024-06-30")
    True
    """
    months = _validation.check_count(months, "months")
    rng = _validation.as_generator(rng)
    income, debt_to_income, incumbent_pd, target = _drivers(applications)
    n = len(applications)
    days = _application_days(application_date, applications)
    bounds = _month_bounds(days, months)

    # Month j before the application (0 for the last) holds the days [bounds[:, j + 1],
    # bounds[:, j]).
    stress = _stress(target, months, rng)
    pd_clipped = np.clip(incumbent_pd, _MIN_PD, 1 - _MIN_PD)
    view = (np.log(pd_clipped / (1 - pd_clipped)) - _VIEW_CENTRE) / _VIEW_SCALE
    relative_income = income / _applications._MEDIAN_INCOME
    season = _SEASON[_calendar_month(bounds[:, 1:])]
    purchase_rate = (
        _PURCHASE_RATE
        * relative_income**_PURCHASE_INCOME_EXPONENT
        * np.exp(_PURCHASE_TRAIT * rng.standard_normal(n))
    )[:, None] * (season * np.exp(-_PURCHASE_STRESS * stress))
    cash_rate = (
        _CASH_RATE
        * relative_income**_CASH_INCOME_EXPONENT
        * np.exp(_CASH_TRAIT * rng.standard_normal(n) + _CASH_VIEW * view)
    )[:, None] * np.exp(_CASH_STRESS * stress)
    miss = _applications._logistic(_MISS_LOGIT + _MISS_VIEW * view[:, None] + _MISS_STRESS * stress)
    shortfall = _SHORT_MAX * np.minimum(stress, 1.0) * rng.random((n, months))
    instalment = (
        income
        * np.maximum(debt_to_income, _MIN_DEBT_TO_INCOME)
        * np.exp(_INSTALMENT_TRAIT * rng.standard_normal(n))
    )

    counts = np.stack(
        [
            rng.poisson(purchase_rate),
            rng.poisson(cash_rate),
            (rng.random((n, months)) >= miss).astype(np.int64),
        ],
        axis=-1,
    )
    # One row per transaction: its kind, the month before the application and the applicant.
    cell = np.repeat(np.arange(counts.size), counts.ravel())
    kind = cell % len(KINDS)
    month = (cell // len(KINDS)) % months
    applicant = cell // (len(KINDS) * months)

    amount = np.empty(len(cell))
    purchase = kind == 0
    amount[purchase] = (
        income[applicant[purchase]]
        * _PURCHASE_SHARE
        * rng.lognormal(0.0, _PURCHASE_SPREAD, purchase.sum())
    )
    cash = kind == 1
    amount[cash] = (
        income[applicant[cash]]
        * _CASH_SHARE
        * np.exp(_CASH_AMOUNT_STRESS * stress[applicant[cash], month[cash]])
        * rng.lognormal(0.0, _CASH_SPREAD, cash.sum())
    )
    repayment = kind == 2
    amount[repayment] = (
        instalment[applicant[repayment]]
        * (1 - shortfall[applicant[repayment], month[repayment]])
        * rng.lognormal(0.0, _INSTALMENT_SPREAD, repayment.sum())
    )
    merchant = np.full(len(cell), -1, dtype=np.int64)
    merchant[purchase] = rng.choice(
        len(MERCHANT_CATEGORIES), size=purchase.sum(), p=_MERCHANT_SHARES
    )

    start = bounds[applicant, month + 1]
    seconds = start * _SECONDS_PER_DAY + rng.integers(
        0, (bounds[applicant, month] - start) * _SECONDS_PER_DAY
    )
    order = np.lexsort((seconds, applicant))
    labels = applications.index.take(applicant[order])
    transactions = pd.DataFrame(
        {
            "applicant_id": labels,
            "timestamp": seconds[order].astype("datetime64[s]").astype("datetime64[ns]"),
            "amount": np.maximum(np.round(amount[order], 2), _MIN_AMOUNT),
            "kind": pd.Categorical.from_codes(kind[order], categories=KINDS),  # type: ignore[call-overload]  # categories may be any list-like
            "merchant_category": pd.Categorical.from_codes(
                merchant[order],
                categories=MERCHANT_CATEGORIES,  # type: ignore[call-overload]  # categories may be any list-like
            ),
        }
    )
    observations = pd.DataFrame(
        {
            "applicant_id": applications.index,
            "application_date": days.astype("datetime64[D]").astype("datetime64[ns]"),
        },
        index=applications.index.rename(None),
    )
    return transactions, observations


def _stress(
    target: npt.NDArray[np.intp], months: int, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Stress of each applicant in each month before the application, shape (n, months)."""
    n = len(target)
    onset = rng.integers(_ONSET[0], _ONSET[1], n)
    size = rng.lognormal(0.0, _STRESS_SPREAD, n)
    episode = rng.random(n) < _EPISODE_SHARE
    level = np.where(target == 1, _DEFAULTER_STRESS, np.where(episode, _EPISODE_STRESS, 0.0))
    # Full stress in the last month, none from `onset` months before it.
    build_up = np.maximum(0.0, 1.0 - np.arange(months) / onset[:, None])
    return (level * size)[:, None] * build_up


def _calendar_month(days: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Calendar month, 0 for January, of days since 1970-01-01."""
    return days.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) % 12


def _drivers(
    applications: pd.DataFrame,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.intp]
]:
    """Income, debt-to-income ratio, incumbent PD and target of the applications, checked."""
    if not isinstance(applications, pd.DataFrame):
        raise ValueError(f"applications must be a DataFrame, got {type(applications).__name__}")
    absent = [name for name in REQUIRED_COLUMNS if name not in applications.columns]
    if absent:
        raise ValueError(
            f"applications must have the columns {_validation.shown_list(REQUIRED_COLUMNS)}, "
            f"got no column {_validation.shown_list(absent)}"
        )
    if isinstance(applications.index, pd.MultiIndex):
        raise ValueError("applications must have a flat index, got a MultiIndex")
    if applications.index.has_duplicates:
        repeated = applications.index[applications.index.duplicated()]
        raise ValueError(
            f"applications must have unique index labels, got {_validation.shown(repeated)} twice"
        )
    income = _real_column(
        applications, "income", lambda v: np.isfinite(v) & (v > 0), "finite and positive"
    )
    debt_to_income = _real_column(
        applications,
        "debt_to_income",
        lambda v: np.isfinite(v) & (v >= 0),
        "finite and non-negative",
    )
    incumbent_pd = _real_column(
        applications, "incumbent_pd", lambda v: (v >= 0) & (v <= 1), "between 0 and 1"
    )
    target = _validation.binary_labels(_column(applications, "target"), both_classes=False)
    return income, debt_to_income, incumbent_pd, target


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return the column ``name`` of the frame, which must be the only one of that name."""
    if (frame.columns == name).sum() > 1:
        raise ValueError(f"applications has more than one column named {_validation.shown(name)}")
    return frame[name]


def _real_column(
    frame: pd.DataFrame,
    name: str,
    valid: abc.Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool_]],
    requirement: str,
) -> npt.NDArray[np.float64]:
    """Values of a numeric column without missing values, each of which ``valid`` accepts."""
    label = f"column {_validation.shown(name)}"
    values = _validation.as_real(_column(frame, name), label)
    missing = np.isnan(values)
    if missing.any():
        raise ValueError(f"{label} must not contain missing values, got {missing.sum()}")
    invalid = ~valid(values)
    if invalid.any():
        raise ValueError(
            f"{label} must be {requirement}, got {_validation.shown(values[invalid][0])}"
        )
    return values


def _application_days(
    application_date: typing.Any, applications: pd.DataFrame
) -> npt.NDArray[np.int64]:
    """Days since 1970-01-01 of the application dates, one per row of the applications."""
    n = len(applications)
    if application_date is None:
        return _default_days(n)
    if isinstance(application_date, str) and application_date in applications.columns:
        return _days(_column(applications, application_date))
    if np.ndim(application_date) == 0:
        try:
            return np.full(n, _days(pd.Series([application_date]))[0])
        except ValueError:
            if isinstance(application_date, str):
                raise ValueError(
                    "application_date must be a column of applications or a date, got "
                    f"{_validation.shown(application_date)}"
                ) from None
            raise
    dates = _validation.as_series(typing.cast(npt.ArrayLike, application_date), "application_date")
    if len(dates) != n:
        raise ValueError(
            f"application_date must have one value per row of applications, got {len(dates)} "
            f"for {n}"
        )
    # Only a Series is matched by its index; other sequences are matched by position.
    _validation.shared_index(
        {"applications": applications, "application_date": application_date}, frames=True
    )
    return _days(dates)


def _days(values: pd.Series) -> npt.NDArray[np.int64]:
    """Days since 1970-01-01 of dates or strings of dates, without the time of day."""
    if not len(values):
        return np.empty(0, dtype=np.int64)
    dtype = values.dtype
    if (
        pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or isinstance(dtype, pd.PeriodDtype)
    ):
        # pandas.to_datetime reads numbers as nanoseconds since 1970.
        raise ValueError(f"application_date must hold dates, got dtype {dtype}")
    missing = values.isna().to_numpy(dtype=bool)
    if missing.any():
        raise ValueError(f"application_date must not contain missing values, got {missing.sum()}")
    try:
        dates = pd.to_datetime(values)
    except (TypeError, ValueError):
        failed = pd.to_datetime(values, errors="coerce").isna()
        value = values[failed] if failed.any() else values
        raise ValueError(
            f"application_date must hold dates, got {_validation.shown(value)}"
        ) from None
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    days = dates.to_numpy().astype("datetime64[D]").astype(np.int64)
    outside = (days < _FIRST_DAY) | (days > _LAST_DAY)
    if outside.any():
        raise ValueError(
            "application_date must lie between the years 1678 and 2261, got "
            f"{_validation.shown(values[outside])}"
        )
    return days


def _default_days(n: int) -> npt.NDArray[np.int64]:
    """Days of the default dates: the rows spread evenly over the months of 2024."""
    row = np.arange(n)
    month = row * 12 // n if n else row
    # The first row of a month is the first with row * 12 >= month * n.
    first = (month * n + 11) // 12
    size = ((month + 1) * n + 11) // 12 - first
    day = (row - first) * _DAYS_IN_MONTH_2024[month] // np.maximum(size, 1)
    month_start = np.concatenate([[0], np.cumsum(_DAYS_IN_MONTH_2024)])[month]
    first_day = FIRST_DEFAULT_DATE.to_datetime64().astype("datetime64[D]").astype(np.int64)
    return (first_day + month_start + day).astype(np.int64)


def _month_bounds(days: npt.NDArray[np.int64], months: int) -> npt.NDArray[np.int64]:
    """Days of the dates 0 to `months` months before each date, shape (n, months + 1).

    A month before a date is the same day of the previous month, or its last day when that
    month is shorter, as in ``pandas.DateOffset(months=...)``.
    """
    month = days.astype("datetime64[D]").astype("datetime64[M]")
    if len(days) and month.astype(np.int64).min() - months < _FIRST_MONTH:
        raise ValueError(f"months must not reach back before the year 1678, got {months}")
    day_of_month = days - month.astype("datetime64[D]").astype(np.int64)
    earlier = month.astype(np.int64)[:, None] - np.arange(months + 1)
    first = earlier.astype("datetime64[M]").astype("datetime64[D]").astype(np.int64)
    following = (earlier + 1).astype("datetime64[M]").astype("datetime64[D]").astype(np.int64)
    return (first + np.minimum(day_of_month[:, None], following - first - 1)).astype(np.int64)
