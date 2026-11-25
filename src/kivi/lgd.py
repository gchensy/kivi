"""Loss given default and credit conversion factors: realized values, tables and models.

``workout_lgd`` discounts the recoveries and costs of defaulted loans to
their default dates and gives the realized LGD of each, and
``conversion_factor`` gives the realized credit conversion factor of a
credit line. ``lgd_table`` summarizes realized LGDs by group, and
``fractional_regression`` fits the fractional logit of an LGD between 0 and
1 on its drivers.

Realized LGDs are not clipped: recoveries above the exposure give a negative
LGD, and costs can take it above 1. ``lgd_table`` floors them at 0 before
averaging, as the regulations ask for long-run averages.
"""

import math
import typing
import warnings
from collections import abc

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import _counts, _logit, _validation

__all__ = ["workout_lgd", "conversion_factor", "lgd_table", "fractional_regression"]

# Days of the year in the ACT/365 day count of the discounting.
_YEAR = 365


def workout_lgd(
    loan: npt.ArrayLike,
    default_date: npt.ArrayLike,
    ead: npt.ArrayLike,
    flow_loan: npt.ArrayLike,
    flow_date: npt.ArrayLike,
    flow_amount: npt.ArrayLike,
    *,
    rate: npt.ArrayLike,
) -> pd.DataFrame:
    r"""Compute the realized LGD of defaulted loans from their workout cash flows.

    The economic loss of a defaulted loan is its exposure at default plus
    the costs of the workout, minus the recoveries, all discounted to the
    date of default. Its ratio to the exposure at default is the realized
    loss given default of the workout approach.

    Parameters
    ----------
    loan : array_like
        Defaulted loans, each once. The result has one row per loan, in this
        order.
    default_date : array_like
        Date of default of each loan, as dates or as strings that
        `pandas.to_datetime` reads. A time of day is ignored.
    ead : array_like
        Exposure at default of each loan, positive and finite: the amount
        outstanding at default, including accrued interest and fees, plus
        any amount written off at or before default.
    flow_loan : array_like
        Loan of each cash flow of the workouts. Every one must be a loan of
        `loan`, and a loan without flows has recovered nothing. The flows
        need not be sorted, and their Series may have any index.
    flow_date : array_like
        Date of each cash flow, read like `default_date`.
    flow_amount : array_like
        Amount of each cash flow, finite: positive for a recovery, such as a
        repayment, the proceeds of collateral or a cure, and negative for a
        cost of the workout or a drawing after default. Costs may come
        before the date of default; recoveries may not.
    rate : float or array_like
        Annual discount rate, greater than -1, as one rate for every loan or
        one per loan, matched to `loan` like `ead`, such as the 3-month
        interbank rate at the date of default plus 5 percentage points.

    Returns
    -------
    pandas.DataFrame
        One row per loan, indexed by `loan` in an index named ``loan``, with
        ``ead``; ``recovery``, the present value of the positive flows;
        ``cost``, the present value of the negative flows as a positive
        amount; and ``lgd``, which is ``(ead + cost - recovery) / ead``.

    Raises
    ------
    ValueError
        If the inputs of the loans, or those of the flows, are not
        one-dimensional, have different lengths or are Series with different
        indices, or if `loan` is empty. Also if a loan or a date is missing
        or repeats a loan, a date cannot be read, `ead` is missing or not
        positive and finite, an amount is missing, infinite or not a number,
        a flow belongs to a loan that is not in `loan`, a recovery comes
        before the date of default of its loan, or `rate` is missing, not
        finite, at most -1 or not one rate or one per loan. Booleans are not
        amounts or rates.

    See Also
    --------
    lgd_table : Long-run average LGDs by group.
    fractional_regression : Model the LGD on its drivers.
    kivi.calibration.backtest_table : Back-test LGD estimates against realized LGDs.
    kivi.vintage.delinquency_panel : Reads dates in the same way.

    Notes
    -----
    A flow of amount :math:`a_j` on the date :math:`t_j` of a loan that
    defaulted on the date :math:`t_D` with the annual rate :math:`r` has the
    present value

    .. math::

       \mathrm{PV}_j = a_j \, (1 + r)^{-(t_j - t_D) / 365},

    with the days counted ACT/365. A cost before the date of default has a
    negative exponent and is compounded to the date of default. With the
    sums :math:`R` of the present values of the positive flows and
    :math:`C` of those of the negative flows, as a positive amount,

    .. math::

       \mathrm{LGD} = \frac{\mathrm{EAD} + C - R}{\mathrm{EAD}}.

    This is the economic loss of paragraphs 131 to 146 of [1]_ and of
    CRE36.76 of the Basel framework. The LGD is not clipped to [0, 1]: it is
    negative when the recoveries exceed the exposure and the costs, and
    above 1 when the costs exceed the recoveries. `lgd_table` floors it at 0
    for long-run averages (paragraph 160 of [1]_).

    The function does not know whether a workout has ended, and gives the
    loss of the flows so far: a loan without flows has an LGD of 1. A
    realized LGD exists only for a closed workout (paragraph 154 of [1]_;
    section 2.6.1 of [2]_), so select the loans whose recovery process has
    closed, by the bank's definition of its end, before averaging or
    back-testing.

    The flows are the caller's to choose:

    - A drawing after default is a negative flow, like a cost, so that it
      counts in the loss (paragraph 142 of [1]_). Where the conversion
      factor includes the drawings, paragraph 140 of [1]_ counts them in the
      exposure as well: add their present value to `ead` instead of passing
      them as flows, since `ead` counts in the loss too. The ``cost`` of a
      call with the drawings alone, negated, is that present value.
    - An amount written off at or before default belongs in `ead`, so that
      it counts in the loss and in the exposure (paragraph 134 of [1]_).
    - A cure is a recovery of the amount outstanding on the date of the
      cure (paragraph 135 of [1]_).
    - Interest and fees capitalized after default are not flows
      (paragraphs 137 and 138 of [1]_); their recoveries are.

    References
    ----------
    .. [1] European Banking Authority, "Guidelines on PD estimation, LGD
       estimation and the treatment of defaulted exposures",
       EBA/GL/2017/16, 20 November 2017, paragraphs 131 to 163.
    .. [2] European Central Bank, "Instructions for reporting the
       validation results of internal models: IRB Pillar I models for
       credit risk", February 2019, section 2.6.1.

    Examples
    --------
    Three loans defaulted on 1 January 2023. Loan A recovers 55 and costs
    11 a year later, loan B recovers its exposure on the day of default
    after a legal cost of 5 a year before, and loan C has no flows yet:

    >>> from kivi.lgd import workout_lgd
    >>> workout_lgd(
    ...     ["A", "B", "C"],
    ...     ["2023-01-01"] * 3,
    ...     [100.0, 200.0, 50.0],
    ...     ["A", "A", "B", "B"],
    ...     ["2024-01-01", "2024-01-01", "2022-01-01", "2023-01-01"],
    ...     [55.0, -11.0, -5.0, 200.0],
    ...     rate=0.1,
    ... ).round(4)
            ead  recovery  cost     lgd
    loan
    A     100.0      50.0  10.0  0.6000
    B     200.0     200.0   5.5  0.0275
    C      50.0       0.0   0.0  1.0000

    At 10% a year, the flows of A a year after default are worth 55 / 1.1 =
    50 and 11 / 1.1 = 10, and the cost of B a year before default is worth
    5 × 1.1 = 5.5 at default.
    """
    defaults = {"loan": loan, "default_date": default_date, "ead": ead}
    flows = {"flow_loan": flow_loan, "flow_date": flow_date, "flow_amount": flow_amount}
    _validation.shared_index({**defaults, "rate": rate})
    _validation.shared_index(flows)
    loans = {name: _validation.as_series(values, name) for name, values in defaults.items()}
    paid = {name: _validation.as_series(values, name) for name, values in flows.items()}
    for group in (loans, paid):
        _check_lengths(group)
    if not len(loans["loan"]):
        raise ValueError("loan, default_date and ead must not be empty")
    labels = _loan_index(loans["loan"])
    default_day = _days(loans["default_date"], "default_date")
    exposure = _amounts(loans["ead"], "ead")
    if (exposure <= 0).any():
        raise ValueError(
            f"ead must be positive, got {_validation.shown(exposure[exposure <= 0][0])}"
        )
    rates = _rates(rate, len(labels))

    flow_loans = paid["flow_loan"]
    missing = flow_loans.isna().to_numpy(dtype=bool)
    if missing.any():
        raise ValueError(f"flow_loan must not contain missing values, got {missing.sum()}")
    code = labels.get_indexer(flow_loans)  # type: ignore[arg-type]  # takes any array_like
    unknown = code < 0
    if unknown.any():
        shown = _validation.shown(flow_loans[unknown])
        raise ValueError(f"flow_loan must contain only loans of loan, got {shown}")
    flow_day = _days(paid["flow_date"], "flow_date")
    amount = _amounts(paid["flow_amount"], "flow_amount")
    elapsed = flow_day - default_day[code]
    early = (amount > 0) & (elapsed < 0)
    if early.any():
        row = np.flatnonzero(early)[0]
        raise ValueError(
            "flow_amount must not be positive before the date of default: loan "
            f"{_validation.shown(flow_loans.iloc[row])} recovers "
            f"{_validation.shown(amount[row])} on {_date(flow_day[row])}, before its default on "
            f"{_date(default_day[code[row]])}"
        )

    # (1 + r)^(-t) as exp(-t ln(1 + r)), which keeps the precision of small rates.
    present = amount * np.exp(-(elapsed / _YEAR) * np.log1p(rates[code]))
    positive, negative = amount > 0, amount < 0
    # bincount gives integers when it has nothing to add, also with weights.
    recovery = np.bincount(code[positive], weights=present[positive], minlength=len(labels))
    cost = np.bincount(code[negative], weights=-present[negative], minlength=len(labels))
    recovery, cost = recovery.astype(np.float64), cost.astype(np.float64)
    return pd.DataFrame(
        {
            "ead": exposure,
            "recovery": recovery,
            "cost": cost,
            "lgd": (exposure + cost - recovery) / exposure,
        },
        index=labels,
    )


@typing.overload
def conversion_factor(ead: pd.Series, drawn: npt.ArrayLike, limit: npt.ArrayLike) -> pd.Series: ...
@typing.overload
def conversion_factor(ead: npt.ArrayLike, drawn: pd.Series, limit: npt.ArrayLike) -> pd.Series: ...
@typing.overload
def conversion_factor(ead: npt.ArrayLike, drawn: npt.ArrayLike, limit: pd.Series) -> pd.Series: ...
@typing.overload
def conversion_factor(ead: float, drawn: float, limit: float) -> float: ...
@typing.overload
def conversion_factor(
    ead: _validation.ListLike,
    drawn: float | _validation.ListLike,
    limit: float | _validation.ListLike,
) -> npt.NDArray[np.float64]: ...
@typing.overload
def conversion_factor(
    ead: npt.ArrayLike, drawn: npt.ArrayLike, limit: npt.ArrayLike
) -> float | pd.Series | npt.NDArray[np.float64]: ...
def conversion_factor(
    ead: npt.ArrayLike, drawn: npt.ArrayLike, limit: npt.ArrayLike
) -> float | pd.Series | npt.NDArray[np.float64]:
    r"""Compute the realized credit conversion factor of each credit line.

    The credit conversion factor (CCF) is the share of the undrawn part of
    a credit line at a reference date that the borrower has drawn by the
    time of default.

    Parameters
    ----------
    ead : array_like
        Amount drawn at default, the realized exposure at default.
    drawn : array_like
        Amount drawn at the reference date.
    limit : array_like
        Limit of the credit line at the reference date. Series inputs must
        have identical indices; otherwise the inputs are broadcast together.

    Returns
    -------
    float, pandas.Series or numpy.ndarray
        Realized CCFs, with the broadcast shape of the inputs: NaN where
        the line was fully drawn at the reference date (`limit` at most
        `drawn`) or an input is missing. If an input is a Series, the result
        is a Series with its index, named ``ccf``; scalar inputs give a
        float.

    Raises
    ------
    ValueError
        If an input is not real and numeric or has an infinite value, the
        inputs cannot be broadcast together, or Series inputs have different
        indices. Booleans are not amounts.

    See Also
    --------
    workout_lgd : Realized LGDs of the same defaults.
    lgd_table : Summarize realized CCFs by group, with ``floor=None``.
    kivi.calibration.backtest_table : Back-test CCF estimates against realized CCFs.

    Notes
    -----
    With the amount :math:`E` drawn at default, and the amount :math:`D`
    drawn and the limit :math:`L` at the reference date,

    .. math::

       \mathrm{CCF} = \frac{E - D}{L - D}, \qquad L > D.

    This is the realized CCF of CRE36.95 of the Basel framework and of
    paragraph 61(a) of the EBA's consultation on CCF estimation [1]_. The
    reference date of the fixed-horizon approach lies 12 months before
    default (CRE36.93; article 182(1)(g) of the CRR3), and the limit is the
    larger of the advised and the unadvised limit at that date: a later
    increase of the limit stays out of the denominator, so the CCF can
    exceed 1.

    A line that is fully drawn or overdrawn at the reference date has no
    undrawn part, and its CCF is NaN instead of the sign-reversed ratio of
    an overdrawn line. The consultation measures such lines by
    ``ead / limit`` (paragraph 61(b)), and the ECB's validation report
    back-tests their EAD instead of a CCF.

    The CCFs are not clipped. Negative values, from lines repaid after the
    reference date, still rank the risk, but the quantification sets them
    to 0 (article 182(1) of the CRR3), as ``numpy.maximum(ccf, 0)`` does;
    values above 1 are kept, since the CCF must not be capped
    (CRE36.95-96).

    References
    ----------
    .. [1] European Banking Authority, "Consultation paper on draft
       guidelines on CCF estimation", EBA/CP/2025/10, 2 July 2025.

    Examples
    --------
    >>> from kivi.lgd import conversion_factor
    >>> conversion_factor([600.0, 800.0, 300.0, 900.0], [400.0, 400.0, 500.0, 1000.0], 1000.0)
    array([ 0.33333333,  0.66666667, -0.4       ,         nan])

    The first line drew 200 of its 600 undrawn; the third repaid 200 of
    its 500 undrawn, and the fourth was fully drawn at the reference date.
    """
    arrays, template = _broadcast({"ead": ead, "drawn": drawn, "limit": limit})
    undrawn = arrays["limit"] - arrays["drawn"]
    # The division is evaluated for every line, also those that the NaN replaces.
    with np.errstate(divide="ignore", invalid="ignore"):
        ccf = np.where(undrawn > 0, (arrays["ead"] - arrays["drawn"]) / undrawn, np.nan)
    result = _validation.like_input(template, np.asarray(ccf, dtype=float))
    if isinstance(result, pd.Series):
        result.name = "ccf"
    return result


def lgd_table(
    groups: npt.ArrayLike,
    lgd: npt.ArrayLike,
    *,
    floor: float | None = 0.0,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Summarize realized LGDs by group.

    The table gives the number of defaults, the average LGD, its spread and
    the shares of defaults without loss and with a total loss in each
    group, such as a segment, a facility grade or a bin of a driver: the
    long-run average LGD of each group and the bimodal shape of the LGDs
    within it.

    Parameters
    ----------
    groups : array_like
        Group labels, such as facility grades or the bins of a driver from
        `kivi.binning.cut`. Missing values form a group.
    lgd : array_like
        Realized LGD of each default, such as the ``lgd`` column of
        `workout_lgd`, finite and without missing values. If `groups` and
        `lgd` are Series, they must have identical indices; otherwise they
        are matched by position.
    floor : float or None, default 0.0
        Lower bound applied to the LGDs before anything is computed:
        negative LGDs count as 0 by default, as the regulations ask for
        long-run averages. None keeps the LGDs as they are, which suits an
        analysis of the drivers.
    weights : array_like, optional
        Non-negative frequency weight of each default, matched like `lgd`:
        integer weights give the same table as repeating rows, and groups
        whose weights sum to zero are left out. See the Notes on weighting
        by exposure.

    Returns
    -------
    pandas.DataFrame
        One row per group, in an index named ``bin``, with the columns
        described in the Notes.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, are empty, or their indices
        or lengths do not match, `lgd` is not real and numeric or has a
        missing or infinite value, or `floor` is neither None nor a finite
        number. Also if `weights` is not numeric, does not give each default
        one finite, non-negative weight with a positive total, or has an
        index other than that of `groups`.

    See Also
    --------
    workout_lgd : Realized LGDs from workout cash flows.
    fractional_regression : Model the LGD on several drivers at once.
    kivi.woe.woe_table : The table of a binary target.
    kivi.calibration.backtest_table : Back-test LGD estimates by group.

    Notes
    -----
    For the defaults :math:`i` of a group, with weights :math:`w_i` (1
    without `weights`) and LGDs :math:`x_i = \max(\mathrm{LGD}_i, f)` at
    the floor :math:`f`, the columns are:

    - ``total``: the number of defaults, :math:`n = \sum_i w_i`;
    - ``total_share``: ``total`` divided by the number of all defaults;
    - ``mean``: the average LGD, :math:`\bar x = \sum_i w_i x_i / n`;
    - ``std``: the sample standard deviation,
      :math:`\sqrt{\sum_i w_i (x_i - \bar x)^2 / (n - 1)}`, NaN for a group
      with a total weight of 1 or less;
    - ``min`` and ``max``: the smallest and largest LGD with a positive
      weight;
    - ``zero_share``: the share of defaults with :math:`x_i \le 0`, without
      loss;
    - ``full_share``: the share of defaults with :math:`x_i \ge 1`, a total
      loss.

    The average counts every default once, the default-weighted average of
    paragraph 150 of [1]_, CRE36.83 of the Basel framework and part VI(4)
    of annex 5 of China's 2023 capital rules, which also set negative LGDs
    to 0 (paragraph 160 of [1]_; part VI(4) 15 of the annex). Weighting by
    the exposure at default gives an exposure-weighted average instead,
    which is not the long-run average of the regulations; use weights that
    count defaults, such as the inverse of a sampling rate.

    Groups are counted as in `kivi.woe.woe_table`: categorical groups, such
    as the result of ``pandas.cut``, follow the order of their categories,
    with missing values last, and other groups the order in which they
    first appear. Categories that are never observed are omitted.

    References
    ----------
    .. [1] European Banking Authority, "Guidelines on PD estimation, LGD
       estimation and the treatment of defaulted exposures",
       EBA/GL/2017/16, 20 November 2017, paragraphs 150 to 162.

    Examples
    --------
    >>> from kivi.lgd import lgd_table
    >>> grade = ["F1"] * 4 + ["F2"] * 3
    >>> lgd = [-0.05, 0.0, 0.10, 0.35, 0.80, 1.0, 1.2]
    >>> lgd_table(grade, lgd).round(4)
         total  total_share    mean     std  min   max  zero_share  full_share
    bin
    F1       4       0.5714  0.1125  0.1652  0.0  0.35         0.5      0.0000
    F2       3       0.4286  1.0000  0.2000  0.8  1.20         0.0      0.6667

    The LGD of -0.05 in grade F1, from recoveries above the exposure, counts
    as 0, so half of the defaults of F1 lost nothing, and two of the three
    defaults of F2 lost all of their exposure or more.
    """
    inputs = {"groups": groups, "lgd": lgd}
    _validation.shared_index(inputs)
    raw_groups = groups
    groups, values = (_validation.as_series(values, name) for name, values in inputs.items())
    if len(groups) != len(values):
        raise ValueError(
            f"groups and lgd must have the same length, got {len(groups)} and {len(values)}"
        )
    if groups.empty:
        raise ValueError("groups and lgd must contain observations")
    losses = _amounts(values, "lgd")
    bound = _checked_floor(floor)
    if bound is not None:
        losses = np.maximum(losses, bound)
    mass = _validation.as_weights(weights, raw_groups, of="groups", others={"lgd": inputs["lgd"]})
    full = losses >= 1
    counts = _counts.group_counts(groups, (losses <= 0).astype(np.intp), mass)
    rows = _counts.group_rows(groups, counts.index)
    total = counts.to_numpy().sum(axis=1)
    weight = np.ones(len(losses)) if mass is None else mass
    # Only an observation of zero weight lies in a group that the counts leave out.
    kept = (rows >= 0) & (weight > 0)
    rows, losses, weight, full = rows[kept], losses[kept], weight[kept], full[kept]
    mean, deviation = _group_means(losses, rows, weight, total)
    squares = np.bincount(rows, weights=weight * deviation**2, minlength=len(total))
    with np.errstate(divide="ignore", invalid="ignore"):
        std = np.where(total > 1, np.sqrt(squares / (total - 1)), np.nan)
    order = np.lexsort((losses, rows))
    starts = np.flatnonzero(np.r_[True, np.diff(rows[order]) != 0])
    ends = np.r_[starts[1:], len(order)] - 1
    return pd.DataFrame(
        {
            "total": total,
            "total_share": total / total.sum(),
            "mean": mean,
            "std": std,
            "min": losses[order[starts]],
            "max": losses[order[ends]],
            "zero_share": counts.to_numpy()[:, 1] / total,
            "full_share": np.bincount(rows, weights=weight * full, minlength=len(total)) / total,
        },
        index=counts.index,
    )


def fractional_regression(
    X: pd.DataFrame | npt.ArrayLike,
    response: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Fit a fractional logit regression of a response between 0 and 1.

    The model of Papke and Wooldridge [1]_ gives the mean of a response
    between 0 and 1, such as a realized LGD or a recovery rate, as the
    logistic function of a linear combination of the columns of `X`. Its
    coefficients maximize the Bernoulli quasi-likelihood, and their
    standard errors are robust, so the model needs no assumption about the
    distribution of the response around its mean.

    Parameters
    ----------
    X : pandas.DataFrame or array_like
        Two-dimensional numeric values with one column per driver, such as
        the loan-to-value ratio or a segment dummy.
    response : array_like
        Response of each row, from 0 to 1, with exact 0 and 1 allowed, such
        as realized LGDs floored at 0 and capped at 1. If `X` is a DataFrame
        and `response` a Series, they must have identical indices; otherwise
        they are matched by position.
    weights : array_like, optional
        Non-negative frequency weight of each row, matched like `response`:
        integer weights give the same estimates, standard errors and
        quasi-log-likelihood as repeating rows. Rows with zero weight are
        left out of the fit.

    Returns
    -------
    pandas.DataFrame
        One row for the intercept, labelled ``intercept``, and one for each
        column of `X`, with the coefficient ``coef``, its robust standard
        error ``se``, the z statistic ``z`` and its two-sided p-value
        ``p_value``, in an index named ``variable``, as
        `kivi.selection.logistic_regression` gives them.
        ``attrs["log_likelihood"]`` holds the maximized quasi-log-likelihood
        and ``attrs["converged"]`` whether the fit converged.

    Raises
    ------
    kivi.selection.SeparationError
        If the columns separate the responses of 0 from those of 1, while no
        response between them stops the coefficients from growing without
        bound, so that the estimates do not exist.
    ValueError
        If `X` is not two-dimensional, has fewer than 2 rows or no columns,
        repeats a column label, has a column labelled ``intercept``, or has a
        column that is not numeric, contains missing or infinite values, is
        constant or is a linear combination of the intercept and the columns
        before it, or so close to one that its coefficient cannot be
        estimated; if `response` is not one-dimensional, its index or length
        does not match `X`, it is not real and numeric, has a missing value
        or a value outside [0, 1], or is 0 for every row or 1 for every row;
        or if `weights` is not numeric or does not give each row one finite,
        non-negative weight with a positive total. With `weights`, the
        checks for constant columns, linear combinations and a response of
        0 or 1 for every row are limited to the rows with positive weight.

    Warns
    -----
    RuntimeWarning
        If the fit does not converge for a reason other than separation.

    See Also
    --------
    kivi.selection.logistic_regression : The same model for a binary target.
    lgd_table : Realized LGDs by the groups of one driver.
    workout_lgd : Realized LGDs from workout cash flows.

    Notes
    -----
    The model is

    .. math::

       E(y \mid x) = \frac{1}{1 + e^{-\eta}}, \qquad
       \eta = \beta_0 + \sum_j \beta_j x_j,

    and the estimates maximize the quasi-log-likelihood

    .. math::

       \ell(\beta) = \sum_i w_i \left[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i)
       \right],

    which is the log-likelihood of `kivi.selection.logistic_regression` for
    a binary response, so a response of only 0 and 1 gives its
    coefficients. The fit uses the same iteratively reweighted least
    squares. The mean is consistently estimated whenever the model of the
    mean is right, but the Fisher information understates the variance of
    a fractional response, so the covariance is the HC0 sandwich

    .. math::

       \hat V = A^{-1} B A^{-1}, \qquad
       A = \sum_i w_i p_i (1 - p_i) x_i x_i^\top, \qquad
       B = \sum_i w_i (y_i - p_i)^2 x_i x_i^\top,

    with :math:`x_i` including the 1 of the intercept, as statsmodels'
    ``GLM(family=Binomial(), freq_weights=w).fit(cov_type="HC0")`` gives
    it. The z statistic is :math:`\hat\beta / \mathrm{se}` and the p-value
    :math:`P(|Z| > |z|)` for a standard normal :math:`Z`. For a binary
    response, the robust standard errors differ from the model-based ones
    of `kivi.selection.logistic_regression`.

    A fractional response bounds the linear predictor in both directions,
    so the estimates exist unless a combination of the columns separates
    the responses of exactly 0 from those of exactly 1 and leaves the
    linear predictor of every response between them unchanged.

    References
    ----------
    .. [1] L. E. Papke and J. M. Wooldridge, "Econometric methods for
       fractional response variables with an application to 401(k) plan
       participation rates", Journal of Applied Econometrics, 11(6),
       pp. 619-632, 1996.

    Examples
    --------
    With one dummy, the fitted means are the mean responses of its two
    groups, 0.2 for the secured and 0.65 for the unsecured loans:

    >>> import pandas as pd
    >>> from kivi.lgd import fractional_regression
    >>> X = pd.DataFrame({"unsecured": [0, 0, 0, 0, 1, 1, 1, 1]})
    >>> lgd = pd.Series([0.0, 0.1, 0.5, 0.2, 0.3, 0.9, 0.4, 1.0])
    >>> fractional_regression(X, lgd).round(4)
                 coef      se       z  p_value
    variable
    intercept -1.3863  0.5846 -2.3712   0.0177
    unsecured  2.0053  0.8880  2.2582   0.0239

    The intercept is :math:`\ln(0.2 / 0.8)`, the log odds of the secured
    mean, and its robust variance is
    :math:`\sum_i (y_i - 0.2)^2 / (4 \cdot 0.2 \cdot 0.8)^2 = 0.14 / 0.4096`
    over the four secured rows.
    """
    frame, values, responses, mass = _model_inputs(X, response, weights)
    weighted = mass is not None
    _check_independent(frame.columns, values, weighted=weighted)
    design = np.column_stack([np.ones(len(values)), values])
    try:
        fitted = _logit.fit(design, responses, weights=mass, robust=True)
    except _logit.SeparationError:
        raise _logit.SeparationError(
            "X separates the responses of 0 from those of 1: some fitted means tend to 0 or 1 "
            "while no response between them bounds the coefficients, so the estimates do not "
            "exist; merge the groups or drop the variables that predict response without error"
        ) from None
    except _logit.SingularError as error:
        label = "intercept" if error.column == 0 else frame.columns[error.column - 1]
        raise ValueError(
            f"X column {_validation.shown(label)} is so close to a linear combination of the "
            "intercept and the other columns that its coefficient cannot be estimated"
        ) from None
    if not fitted.converged:
        warnings.warn(
            f"fractional regression did not converge after {fitted.n_iter} iterations",
            RuntimeWarning,
            stacklevel=2,
        )
    table = pd.DataFrame(
        {"coef": fitted.coef, "se": fitted.se, "z": fitted.z, "p_value": fitted.p_value},
        index=pd.Index(["intercept"]).append(frame.columns).rename("variable"),
    )
    table.attrs = {"log_likelihood": fitted.log_likelihood, "converged": fitted.converged}
    return table


def _check_lengths(group: abc.Mapping[str, pd.Series]) -> None:
    """Require the inputs of one table, the loans or the flows, to have the same length."""
    lengths = [len(values) for values in group.values()]
    if len(set(lengths)) > 1:
        names = list(group)
        raise ValueError(
            f"{', '.join(names[:-1])} and {names[-1]} must have the same length, got "
            f"{', '.join(map(str, lengths[:-1]))} and {lengths[-1]}"
        )


def _loan_index(loans: pd.Series) -> pd.Index:
    """Return the loans as an Index named ``loan``; missing and repeated loans raise."""
    missing = loans.isna().to_numpy(dtype=bool)
    if missing.any():
        raise ValueError(f"loan must not contain missing values, got {missing.sum()}")
    labels = pd.Index(loans, name="loan")
    if labels.has_duplicates:
        repeated = labels[labels.duplicated()][0]
        raise ValueError(
            f"loan must contain each loan once, got {_validation.shown(repeated)} more than once"
        )
    return labels


def _days(values: pd.Series, name: str) -> npt.NDArray[np.int64]:
    """Days since 1 January 1970 of dates or strings of dates, without the time of day.

    Dates are read as `kivi.vintage.delinquency_panel` reads them.
    """
    if not len(values):
        return np.empty(0, dtype=np.int64)
    dtype = values.dtype
    if (
        pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or isinstance(dtype, pd.PeriodDtype)
    ):
        # pandas.to_datetime reads numbers as nanoseconds since 1970.
        raise ValueError(f"{name} must hold dates, got dtype {dtype}")
    missing = values.isna().to_numpy(dtype=bool)
    if missing.any():
        raise ValueError(f"{name} must not contain missing values, got {missing.sum()}")
    try:
        dates = pd.to_datetime(values)
    except (TypeError, ValueError):
        failed = pd.to_datetime(values, errors="coerce").isna()
        value = values[failed] if failed.any() else values
        raise ValueError(f"{name} must hold dates, got {_validation.shown(value)}") from None
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    return dates.to_numpy().astype("datetime64[D]").astype(np.int64)


def _date(day: np.int64) -> str:
    """Return the date of a day since 1 January 1970, such as ``2024-01-31``."""
    return str(np.datetime64(int(day), "D"))


def _amounts(values: pd.Series, name: str) -> npt.NDArray[np.float64]:
    """Finite real values as floats; missing values, infinities and booleans raise."""
    array = _validation.as_real(values, name)
    missing = np.isnan(array)
    if missing.any():
        raise ValueError(f"{name} must not contain missing values, got {missing.sum()}")
    infinite = np.isinf(array)
    if infinite.any():
        raise ValueError(f"{name} must be finite, got {_validation.shown(array[infinite][0])}")
    return array


def _rates(rate: npt.ArrayLike, count: int) -> npt.NDArray[np.float64]:
    """One finite discount rate above -1 per loan, from one rate or one per loan."""
    rates = _validation.as_real(rate, "rate")
    if rates.ndim > 1 or (rates.ndim == 1 and len(rates) != count):
        raise ValueError(
            f"rate must be one rate or one per loan, got shape {rates.shape} for {count} loans"
        )
    invalid = ~(np.isfinite(rates) & (rates > -1))
    if invalid.any():
        raise ValueError(
            "rate must be finite and greater than -1, got "
            f"{_validation.shown(np.atleast_1d(rates)[np.atleast_1d(invalid)][0])}"
        )
    return np.broadcast_to(rates, (count,))


def _broadcast(
    inputs: abc.Mapping[str, npt.ArrayLike],
) -> tuple[dict[str, npt.NDArray[np.float64]], npt.ArrayLike]:
    """Broadcast amounts; also return the input whose shape the result takes.

    Missing values stay NaN, and infinities and booleans raise. A Series among the inputs sets
    the index of the result.
    """
    _validation.shared_index(inputs)
    arrays = {name: _validation.as_real(values, name) for name, values in inputs.items()}
    for name, array in arrays.items():
        infinite = np.isinf(array)
        if infinite.any():
            raise ValueError(
                f"{name} must be finite or missing, got {_validation.shown(array[infinite][0])}"
            )
    try:
        shape = np.broadcast_shapes(*(array.shape for array in arrays.values()))
    except ValueError:
        shapes = ", ".join(f"{name} {array.shape}" for name, array in arrays.items())
        raise ValueError(f"{', '.join(arrays)} cannot be broadcast together: {shapes}") from None
    series = [values for values in inputs.values() if isinstance(values, pd.Series)]
    template = series[0] if series else next(iter(inputs.values()))
    if series and shape != (len(series[0]),):
        raise ValueError(
            f"{', '.join(arrays)} must broadcast to the length {len(series[0])} of their Series "
            f"index, got shape {shape}"
        )
    return {name: np.broadcast_to(array, shape) for name, array in arrays.items()}, template


def _checked_floor(floor: object) -> float | None:
    """Return the floor of `lgd_table` as a float, or None."""
    if floor is None:
        return None
    if (
        isinstance(floor, (bool, np.bool_))
        or not _validation.is_real(floor)
        or not math.isfinite(floor)
    ):
        raise ValueError(
            f"floor must be None or a finite number, got {_validation.shown_argument(floor)}"
        )
    return float(floor)


def _group_means(
    values: npt.NDArray[np.float64],
    rows: npt.NDArray[np.intp],
    weight: npt.NDArray[np.float64],
    total: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Weighted mean of each group, and the deviation of each observation from its group's mean.

    Every observation has a positive weight, and every group has an observation. The values are
    shifted by the first value of their group, so that a group of equal values has that value as
    its mean and deviations of exactly 0.
    """
    _, first = np.unique(rows, return_index=True)
    shift = np.empty(len(total))
    shift[rows[first]] = values[first]
    shifted = values - shift[rows]
    mean_shift = np.bincount(rows, weights=weight * shifted, minlength=len(total)) / total
    return shift + mean_shift, shifted - mean_shift[rows]


def _model_inputs(
    X: pd.DataFrame | npt.ArrayLike,
    response: npt.ArrayLike,
    weights: npt.ArrayLike | None,
) -> tuple[
    pd.DataFrame, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64] | None
]:
    """Validate the inputs of `fractional_regression`.

    Return the frame, and the values, responses and weights of the rows with positive weight;
    the weights are None without `weights`.
    """
    for name, values in (("response", response), ("weights", weights)):
        if isinstance(X, pd.DataFrame) and isinstance(values, pd.Series):
            if not X.index.equals(values.index):
                raise ValueError(f"X and {name} must have identical indices")
    frame, values = _design(X)
    raw_response = response
    series = _validation.as_series(response, "response")
    if len(series) != len(frame):
        raise ValueError(
            f"X and response must have the same number of rows, got {len(frame)} and {len(series)}"
        )
    responses = _validation.as_real(series, "response")
    missing = np.isnan(responses)
    if missing.any():
        raise ValueError(f"response must not contain missing values, got {missing.sum()}")
    outside = (responses < 0) | (responses > 1)
    if outside.any():
        raise ValueError(
            f"response must be between 0 and 1, got {_validation.shown(responses[outside][0])}"
        )
    mass = None
    if weights is not None:
        if np.ndim(weights) == 1 and len(weights) != len(frame):  # type: ignore[arg-type]  # one-dimensional
            raise ValueError(
                f"weights must have one weight per row of X, got {len(weights)} for {len(frame)}"  # type: ignore[arg-type]  # one-dimensional
            )
        mass = typing.cast(
            npt.NDArray[np.float64],
            _validation.as_weights(weights, frame, of="X", others={"response": raw_response}),
        )
        # Rows without weight must not enter the centring and scaling of the fit.
        used = mass > 0
        values, responses, mass = values[used], responses[used], mass[used]
    where = "" if mass is None else " on the rows with positive weight"
    for bound in (0, 1):
        if (responses == bound).all():
            raise ValueError(f"response must not be {bound} for every row{where}")
    return frame, values, responses, mass


def _design(X: pd.DataFrame | npt.ArrayLike) -> tuple[pd.DataFrame, npt.NDArray[np.float64]]:
    """Return `X` as a DataFrame, without copying one, and its float values."""
    if isinstance(X, pd.DataFrame):
        frame = X
    else:
        if np.ndim(X) != 2:
            raise ValueError(f"X must be two-dimensional, got {np.ndim(X)} dimensions")
        frame = pd.DataFrame(X)  # type: ignore[arg-type]  # a two-dimensional array_like
    if len(frame) < 2:
        raise ValueError(f"X must have at least 2 rows, got {len(frame)}")
    if not frame.shape[1]:
        raise ValueError("X must have at least one column")
    if frame.columns.has_duplicates:
        label: abc.Hashable = frame.columns[frame.columns.duplicated()][0]
        raise ValueError(
            f"X must have unique column labels, got {_validation.shown(label)} more than once"
        )
    if "intercept" in frame.columns:
        raise ValueError("X must not have a column labelled 'intercept'")
    for label, dtype in frame.dtypes.items():
        if not pd.api.types.is_numeric_dtype(dtype):
            shown = _validation.shown(label)
            raise ValueError(f"X column {shown} must be numeric, got dtype {dtype}")
    values = frame.to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(values).all(axis=0)
    if not finite.all():
        label = frame.columns[np.argmin(finite)]  # type: ignore[call-overload]  # a NumPy integer selects one label
        raise ValueError(
            f"X column {_validation.shown(label)} must not contain missing or infinite values"
        )
    return frame, values


def _check_independent(
    columns: pd.Index, values: npt.NDArray[np.float64], *, weighted: bool
) -> None:
    """Require the columns of `values` to vary and to be independent of those before them."""
    where = " on the rows with positive weight" if weighted else ""
    constant = np.ptp(values, axis=0) == 0
    if constant.any():
        label = columns[int(np.argmax(constant))]
        raise ValueError(
            f"X column {_validation.shown(label)} is constant{where}; it cannot be told apart "
            "from the intercept"
        )
    dependent = _logit.dependent_column(np.column_stack([np.ones(len(values)), values]))
    if dependent is not None:
        raise ValueError(
            f"X column {_validation.shown(columns[dependent - 1])} is a linear combination "
            f"of the intercept and the columns before it{where}"
        )
