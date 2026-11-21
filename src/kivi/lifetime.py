"""Lifetime PD term structures and IFRS 9 expected credit losses.

``convert_pd`` converts a PD term structure, such as the cumulative bad
rates of ``kivi.vintage.vintage_table``, between cumulative, marginal and
conditional PDs, and ``transition_pd`` projects cumulative PDs from a
transition matrix. ``expected_credit_loss`` turns marginal PDs, LGDs and
exposures into the discounted 12-month or lifetime expected credit loss of
each account. ``significant_increase`` and ``assign_stages`` allocate the
accounts to the three stages of IFRS 9, and ``stage_table`` summarizes the
exposure and the loss allowance of each stage.

Period 1 is the first period after the starting point of a term structure,
such as the origination or the reporting date. The PD thresholds and the
30-day backstop are exceeded strictly, while an exposure 90 days past due is
already in stage 3, as in the presumptions of IFRS 9.
"""

import math
from collections.abc import Hashable
from typing import TypeGuard

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._counts import ratio
from ._validation import (
    as_flags,
    as_real,
    as_series,
    as_weights,
    check_real_dtype,
    first_value,
    is_integral,
    is_real,
    is_real_dtype,
    shared_index,
)

__all__ = [
    "convert_pd",
    "transition_pd",
    "expected_credit_loss",
    "significant_increase",
    "assign_stages",
    "stage_table",
]

_KINDS = ("cumulative", "marginal", "conditional")
# Rounding allowed in sums of probabilities that must not exceed 1, or must equal 1.
_TOLERANCE = 1e-9


def convert_pd(
    probability: npt.ArrayLike | pd.DataFrame, *, kind: str, to: str
) -> pd.DataFrame | pd.Series | npt.NDArray[np.float64]:
    r"""Convert a PD term structure between cumulative, marginal and conditional PDs.

    A term structure gives the probability of default in each period after
    a starting point, such as the origination of a loan. The cumulative PD
    of period :math:`k` is the probability of a default in periods 1 to
    :math:`k`, the marginal PD the probability of a default in period
    :math:`k`, and the conditional PD, or hazard rate, the probability of a
    default in period :math:`k` given that there was none before.

    Parameters
    ----------
    probability : array_like or pandas.DataFrame
        PDs from 0 to 1, as one curve, one-dimensional, or one curve per
        row, two-dimensional, such as the cumulative bad rates of
        `kivi.vintage.vintage_table` with one row per cohort and one column
        per month on book. The periods are the positions along the curve:
        the first is period 1. Missing values mark the periods that are not
        observed, such as the months on book that a cohort has not reached.
        Booleans count as 0 and 1.
    kind : {"cumulative", "marginal", "conditional"}
        Kind of the PDs in `probability`.
    to : {"cumulative", "marginal", "conditional"}
        Kind of the PDs to return.

    Returns
    -------
    pandas.DataFrame, pandas.Series or numpy.ndarray
        PDs of kind `to` in the shape of `probability`, with its index,
        columns and name. Missing values stay missing, and so do the PDs
        that depend on them, as described in the Notes.

    Raises
    ------
    ValueError
        If `kind` or `to` is not one of the three kinds, `probability` is
        not real and numeric, is not one- or two-dimensional, has no period
        or has a value outside [0, 1]. Also if cumulative PDs decrease along
        a curve or the marginal PDs of a curve add up to more than 1.

    See Also
    --------
    transition_pd : Cumulative PDs from a transition matrix.
    expected_credit_loss : Expected credit loss from marginal PDs.
    kivi.vintage.vintage_table : Cumulative bad rates by cohort and month on book.

    Notes
    -----
    With the cumulative PDs :math:`C_k`, the marginal PDs :math:`m_k` and
    the conditional PDs :math:`h_k` of periods :math:`k = 1, 2, \ldots`, and
    :math:`C_0 = 0`,

    .. math::

       m_k = C_k - C_{k-1}, \qquad
       h_k = \frac{C_k - C_{k-1}}{1 - C_{k-1}}, \qquad
       C_k = \sum_{j \le k} m_j = 1 - \prod_{j \le k} (1 - h_j).

    The conversions are inverses of each other, up to rounding. The labels
    of the periods are not used: the columns of a vintage table that start
    at month on book 1 give monthly PDs, and columns of quarterly snapshots
    give the PDs of the quarters between them.

    Missing values are not filled. The marginal and the conditional PD of a
    period need the cumulative PDs at its end and at the end of the period
    before, and the cumulative PD of a period needs the marginal or
    conditional PDs of every period up to it. The months on book that a
    cohort has not reached therefore stay missing in every kind, also after
    a round trip. Missing values at the start of a curve, which
    `kivi.vintage.vintage_table` gives when the panel starts after
    origination, leave the marginal and conditional PDs known from the
    second observed period on, but converting those back cannot restore the
    cumulative PDs, because the cumulative PD before the panel starts is not
    known.

    Once the cumulative PD of a curve reaches 1, no exposure is left to
    default, and the conditional PDs of the later periods, 0 / 0, are 0, so
    that the conversions stay inverses.

    The cumulative bad rates of `kivi.vintage.vintage_table` count every
    loan of a cohort, also the loans that were repaid early and can no
    longer default. Its marginal PDs therefore already allow for early
    repayment, and its conditional PDs are conditional on no default, not
    on the loan still being on book.

    Examples
    --------
    >>> from kivi.lifetime import convert_pd
    >>> cumulative = [0.02, 0.05, 0.08]
    >>> convert_pd(cumulative, kind="cumulative", to="marginal")
    array([0.02, 0.03, 0.03])
    >>> conditional = convert_pd(cumulative, kind="cumulative", to="conditional")
    >>> conditional.round(4)
    array([0.02  , 0.0306, 0.0316])
    >>> convert_pd(conditional, kind="conditional", to="cumulative")
    array([0.02, 0.05, 0.08])

    A vintage table has one curve per cohort, and the months on book that
    a cohort has not reached stay missing:

    >>> from kivi.vintage import vintage_table
    >>> loan = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    >>> cohort = ["2024-01"] * 6 + ["2024-02"] * 4
    >>> month_on_book = [1, 2, 3, 1, 2, 3, 1, 2, 1, 2]
    >>> bad = [0, 1, 1, 0, 0, 0, 1, 1, 0, 0]
    >>> table = vintage_table(loan, cohort, month_on_book, bad)
    >>> convert_pd(table, kind="cumulative", to="conditional")
    month_on_book    1    2    3
    cohort
    2024-01        0.0  0.5  0.0
    2024-02        0.5  0.0  NaN
    """
    for name, value in (("kind", kind), ("to", to)):
        if not (isinstance(value, str) and value in _KINDS):  # type: ignore[redundant-expr]  # rejects non-str values
            raise ValueError(
                f"{name} must be 'cumulative', 'marginal' or 'conditional', got "
                f"{first_value(value)!r}"
            )
    values = _curves(probability, "probability")
    _check_curves(values, kind)
    if kind == to:
        values = values.copy()
    else:
        values = _from_cumulative(_to_cumulative(values, kind), to)
    if isinstance(probability, pd.DataFrame):
        return pd.DataFrame(values, index=probability.index, columns=probability.columns)
    if isinstance(probability, pd.Series):
        return pd.Series(values[0], index=probability.index, name=probability.name)
    return values[0] if np.ndim(probability) == 1 else values


def transition_pd(
    matrix: npt.ArrayLike | pd.DataFrame,
    *,
    horizons: int | npt.ArrayLike,
    default_state: Hashable | None = None,
) -> pd.DataFrame:
    r"""Project the cumulative PD of each state from a transition matrix.

    Row :math:`i` of a transition matrix holds the probabilities of moving
    from state :math:`i`, such as a rating grade or a delinquency bucket, to
    each state over one period. If the movements follow a time-homogeneous
    Markov chain, the :math:`h`-th power of the matrix holds the
    probabilities over :math:`h` periods, and its default column the
    cumulative PD of each starting state.

    Parameters
    ----------
    matrix : array_like or pandas.DataFrame
        Square matrix of the probabilities of moving between the states over
        one period, such as a year or a month, for example the result of
        `kivi.vintage.transition_matrix` with ``normalize=True``. Each row
        holds probabilities from 0 to 1 that add up to 1, except that the
        row of the default state is not used. The states of a DataFrame are
        its index, which must equal its columns; the states of an array are
        the positions 0, 1, and so on.
    horizons : int or array_like
        Numbers of periods after which to report the cumulative PD: a
        positive integer :math:`n` for every horizon from 1 to :math:`n`,
        or increasing positive integers, such as ``[12, 24, 36]``.
    default_state : optional
        Label of the default state. By default the last state, the worst
        one when the states are ordered from the best to the worst, as for
        `kivi.vintage.transition_matrix`.

    Returns
    -------
    pandas.DataFrame
        Cumulative PD with one row per starting state, indexed by ``state``
        or, for states with several levels, by the levels of the index of
        `matrix`, and one column per horizon, labelled ``horizon``. The row
        of the default state is 1.

    Raises
    ------
    ValueError
        If `matrix` is not a square, two-dimensional matrix of real numbers,
        a DataFrame's index and columns differ or have duplicates,
        `default_state` is not one of the states, or a row of another state
        has a missing value or a value outside [0, 1] or does not add up to
        1. Also if `horizons` is not a positive integer or a sequence of
        increasing positive integers.

    See Also
    --------
    kivi.vintage.transition_matrix : Transition matrix of observed movements.
    convert_pd : Marginal and conditional PDs from the cumulative PDs.
    expected_credit_loss : Expected credit loss from marginal PDs.

    Notes
    -----
    The default state is made absorbing: its row is replaced by one that
    stays in default with probability 1. Exposures that cure from default in
    the data therefore still count as defaulted, as in the ever-bad rates of
    `kivi.vintage.vintage_table`. With this matrix :math:`P` and the default
    state :math:`d`, the cumulative PD of state :math:`i` at horizon
    :math:`h` is

    .. math::

       \mathrm{PD}_i(h) = (P^h)_{id}.

    It never decreases with the horizon and never exceeds 1. Because
    :math:`P^{a+b} = P^a P^b`, the PD at :math:`a + b` periods is the PD at
    :math:`b` periods of the states reached after :math:`a` periods,
    weighted by the probabilities of reaching them.

    The projection assumes that the probabilities of moving between states
    are the same in every period and do not depend on how long an exposure
    has been in its state or on book. A matrix estimated over one year gives
    annual PDs, and a monthly one monthly PDs. Rows must add up to 1 within
    1e-9; divide a published matrix that was rounded by its row sums first.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.lifetime import transition_pd
    >>> states = ["A", "B", "D"]
    >>> matrix = pd.DataFrame(
    ...     [[0.90, 0.08, 0.02], [0.10, 0.70, 0.20], [0.00, 0.00, 1.00]],
    ...     index=states,
    ...     columns=states,
    ... )
    >>> transition_pd(matrix, horizons=3)
    horizon     1      2        3
    state
    A        0.02  0.054  0.09596
    B        0.20  0.342  0.44480
    D        1.00  1.000  1.00000

    After two periods, state A has defaulted with probability
    0.90 × 0.02 + 0.08 × 0.20 + 0.02 = 0.054.
    """
    probabilities, states = _transition_probabilities(matrix)
    steps = _horizons(horizons)
    if default_state is None:
        default = len(states) - 1
    else:
        try:
            default = states.get_loc(default_state)  # type: ignore[assignment]  # an int when found
            # get_loc also finds the interval that holds a number; the label itself must match.
            found = is_integral(default) and bool(states[default] == default_state)
        except (KeyError, TypeError, ValueError, pd.errors.InvalidIndexError):
            found = False
        if not found:
            raise ValueError(
                f"default_state must be one of the states, got {first_value(default_state)!r}"
            )
    _check_transition_rows(probabilities, states, default)

    absorbing = probabilities.copy()
    absorbing[default] = 0.0
    absorbing[default, default] = 1.0
    result = np.empty((len(states), len(steps)))
    power = np.eye(len(states))
    previous = 0
    for position, step in enumerate(steps):
        # P^h = P^g P^(h - g) for the previous horizon g. The default row of every power is
        # exactly the unit row, so the default column gains P^g[i, d] times 1 plus non-negative
        # terms and does not decrease, also after rounding.
        power = power @ np.linalg.matrix_power(absorbing, int(step - previous))
        result[:, position] = power[:, default]
        previous = step
    # Rounding can move an entry by a unit in the last place; keep the PDs monotone and at most 1.
    result = np.minimum(np.maximum.accumulate(result, axis=1), 1.0)
    index = states.rename("state") if states.nlevels == 1 else states
    return pd.DataFrame(result, index=index, columns=pd.Index(steps, name="horizon"))


def expected_credit_loss(
    probability: npt.ArrayLike | pd.DataFrame,
    lgd: float | npt.ArrayLike | pd.DataFrame,
    ead: float | npt.ArrayLike | pd.DataFrame,
    *,
    rate: float | npt.ArrayLike,
    horizon: int | None = None,
) -> pd.Series | npt.NDArray[np.float64]:
    r"""Compute the 12-month or lifetime expected credit loss of each account.

    The expected credit loss of IFRS 9 adds up, over the periods after the
    reporting date, the probability of a default in each period times the
    loss given default and the exposure at default in that period,
    discounted to the reporting date at the effective interest rate. Pass
    the number of periods in 12 months as `horizon` for the 12-month
    expected credit loss of stage 1, and leave it out for the lifetime
    expected credit loss of stages 2 and 3.

    Parameters
    ----------
    probability : array_like or pandas.DataFrame
        Marginal PDs with one row per account and one column per period
        after the reporting date: column :math:`t` holds the probability,
        seen at the reporting date, that the account defaults in period
        :math:`t`. Each row adds up to at most 1. `convert_pd` gives the
        marginal PDs of a cumulative or conditional term structure. Booleans
        count as 0 and 1.
    lgd : float, array_like or pandas.DataFrame
        Loss given default as a share of the exposure at default, from 0 to
        1: one value for all accounts and periods, a one-dimensional
        array_like with one value per account, or a two-dimensional schedule
        with one value per account and period. A two-dimensional array may
        also have one row for all accounts or one column for all periods.
    ead : float, array_like or pandas.DataFrame
        Exposure at default, finite and non-negative, in the same forms as
        `lgd`: one value, one value per account, such as the balance at the
        reporting date, or a schedule, such as the balance expected in each
        period. Give the periods after the maturity of an account an
        exposure of 0. Series and DataFrames with one row per account must
        have identical indices, and DataFrames with one column per period
        identical columns. A single row or column is broadcast, so with one
        account or one period only the labels of `probability` are used.
    rate : float or array_like
        Effective interest rate per period at which the losses are
        discounted, greater than -1, as one value or one value per account.
        For monthly periods and an annual rate :math:`r`, pass
        :math:`(1 + r)^{1/12} - 1`. Pass 0 to leave the losses undiscounted.
    horizon : int, optional
        Number of periods to add up, starting from period 1: 12 for monthly
        periods, or 1 for annual ones, gives the 12-month expected credit
        loss. By default all the periods of `probability`, which give the
        lifetime expected credit loss.

    Returns
    -------
    pandas.Series or numpy.ndarray
        Expected credit loss of each account, in the units of `ead`. If
        `probability` is a DataFrame, or another input with one row per
        account of several is a pandas object, a Series named
        ``expected_credit_loss`` with its index; otherwise an array.

    Raises
    ------
    ValueError
        If `probability` is not two-dimensional with at least one account
        and one period, an input is not real and numeric or does not have
        one of the shapes described, or pandas inputs have different indices
        or columns. Also if `probability` or `lgd` has a value outside
        [0, 1], a row of `probability` adds up to more than 1, `ead` has a
        negative or infinite value, `rate` is infinite, not greater than -1
        or makes a discount factor overflow, or `horizon` is not a positive
        integer up to the number of periods.

    See Also
    --------
    convert_pd : Marginal PDs from cumulative or conditional ones.
    transition_pd : Cumulative PDs from a transition matrix.
    assign_stages : Stage of each account, which decides the horizon.
    stage_table : Exposure and expected credit loss by stage.
    kivi.strategy.expected_loss : Expected loss over a single horizon.

    Notes
    -----
    For an account with the marginal PD :math:`m_t`, the loss given default
    :math:`\mathrm{LGD}_t` and the exposure at default
    :math:`\mathrm{EAD}_t` of period :math:`t`, and the effective interest
    rate :math:`r` per period,

    .. math::

       \mathrm{ECL} = \sum_{t=1}^{H} m_t \, \mathrm{LGD}_t \,
                      \mathrm{EAD}_t \, (1 + r)^{-t},

    where :math:`H` is `horizon` or the number of periods. The conventions
    are:

    - Period 1 starts at the reporting date and ends one period later;
      period :math:`t` ends :math:`t` periods after the reporting date.
    - A default in period :math:`t` is taken to occur at the end of the
      period, so its loss is discounted over :math:`t` periods.
    - The marginal PDs are not conditional on survival: they are seen from
      the reporting date and add up to the cumulative PD. For an account
      that has not defaulted by the reporting date, convert the conditional
      PDs of its term structure from its age at the reporting date on, with
      ``kind="conditional"`` and ``to="marginal"``.
    - The columns must cover the remaining life of the longest account.
      An account that matures within the horizon has an exposure of 0 after
      maturity, so its 12-month and lifetime expected credit losses are
      equal.

    An exposure already in default at the reporting date has defaulted at
    that date, not at the end of a period. Give it a marginal PD of 1 in
    period 1 and a rate of 0, so that its expected credit loss is its LGD
    times its EAD: the LGD of a defaulted exposure usually discounts the
    recoveries already.

    A missing PD, LGD or EAD in a period that is added up, or a missing
    rate, gives the account a missing expected credit loss, as in
    `kivi.strategy.expected_loss`. Missing values in the periods after
    `horizon` are not used.

    Examples
    --------
    >>> from kivi.lifetime import expected_credit_loss
    >>> marginal = [[0.02, 0.03, 0.03], [0.10, 0.08, 0.06]]
    >>> schedule = [[1000.0, 700.0, 400.0], [500.0, 250.0, 0.0]]
    >>> expected_credit_loss(marginal, 0.5, schedule, rate=0.05).round(2)
    array([24.23, 32.88])
    >>> expected_credit_loss(marginal, 0.5, schedule, rate=0.05, horizon=1).round(2)
    array([ 9.52, 23.81])

    The first account loses 0.02 × 0.5 × 1000 / 1.05 = 9.52 in period 1,
    0.03 × 0.5 × 700 / 1.05² = 9.52 in period 2 and 0.03 × 0.5 × 400 /
    1.05³ = 5.18 in period 3.
    """
    if np.ndim(probability) != 2:
        raise ValueError(
            "probability must be two-dimensional, with one row per account and one column per "
            f"period, got {np.ndim(probability)} dimensions"
        )
    marginal = _curves(probability, "probability")
    accounts, periods = marginal.shape
    if not accounts:
        raise ValueError("probability must have at least one account")
    total = np.nansum(marginal, axis=1)
    excess = total > 1 + _TOLERANCE
    if excess.any():
        raise ValueError(
            f"probability must add up to at most 1 in each row, got {float(total[excess][0])!r}"
        )
    loss_given_default = _schedule(lgd, "lgd", marginal.shape)
    outside = (loss_given_default < 0) | (loss_given_default > 1)
    if outside.any():
        raise ValueError(
            f"lgd must be between 0 and 1, got {float(loss_given_default[outside][0])!r}"
        )
    exposure = _schedule(ead, "ead", marginal.shape)
    invalid = np.isinf(exposure) | (exposure < 0)
    if invalid.any():
        raise ValueError(
            f"ead must be finite and non-negative, got {float(exposure[invalid][0])!r}"
        )
    rates = _per_account(rate, "rate", accounts)
    invalid = np.isinf(rates) | (rates <= -1)
    if invalid.any():
        raise ValueError(
            f"rate must be finite and greater than -1, got {float(rates[invalid][0])!r}"
        )
    if horizon is None:
        horizon = periods
    elif isinstance(horizon, bool) or not is_integral(horizon) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {first_value(horizon)!r}")
    elif horizon > periods:
        raise ValueError(
            f"horizon must be at most the number of periods of probability, {periods}, "
            f"got {first_value(horizon)}"
        )
    index = _account_index(
        {"probability": probability, "lgd": lgd, "ead": ead, "rate": rate}, accounts, periods
    )

    horizon = int(horizon)
    with np.errstate(over="ignore"):
        discount = (1.0 + rates[:, None]) ** -np.arange(1.0, horizon + 1.0)
    overflow = np.isinf(discount).any(axis=1)
    if overflow.any():
        raise ValueError(
            f"rate must keep the discount factors finite over {horizon} periods, got "
            f"{float(rates[overflow][0])!r}"
        )
    terms = (
        marginal[:, :horizon] * loss_given_default[:, :horizon] * exposure[:, :horizon] * discount
    )
    loss = terms.sum(axis=1)
    if index is None:
        return loss
    return pd.Series(loss, index=index, name="expected_credit_loss")


def significant_increase(
    probability: npt.ArrayLike,
    origination_probability: npt.ArrayLike,
    *,
    relative: float | None,
    absolute: float | None,
) -> pd.Series | npt.NDArray[np.bool_]:
    r"""Flag a significant increase in credit risk since initial recognition.

    IFRS 9 moves an exposure to stage 2 when its credit risk has increased
    significantly since initial recognition. A common quantitative test
    compares the PD at the reporting date with the PD expected at
    origination, as a ratio, a difference or both. The thresholds are the
    policy of the reporting entity, not values set by the standard, so they
    have no defaults.

    Parameters
    ----------
    probability : array_like
        PD of each exposure at the reporting date, from 0 to 1.
    origination_probability : array_like
        PD of each exposure expected at initial recognition for the same
        period, from 0 to 1. If both inputs are Series, they must have
        identical indices; otherwise they are matched by position.
    relative : float or None
        The increase is significant if `probability` is more than
        `relative` times `origination_probability`, so 2 flags the PDs that
        more than doubled. It must be greater than 1. None leaves out this
        test.
    absolute : float or None
        The increase is significant if `probability` exceeds
        `origination_probability` by more than `absolute`, in units of PD,
        so 0.01 flags the PDs that rose by more than one percentage point.
        It must be between 0 and 1. None leaves out this test.

    Returns
    -------
    pandas.Series or numpy.ndarray
        Booleans that are True for a significant increase. If an input is a
        Series, a Series named ``significant_increase`` with its index;
        otherwise an array.

    Raises
    ------
    ValueError
        If the inputs are empty, not one-dimensional, of different lengths,
        Series with different indices, not real and numeric, or have a
        missing value or a value outside [0, 1]. Also if `relative` is not
        None or a number greater than 1, `absolute` is not None or a number
        between 0 and 1, or both are None.

    See Also
    --------
    assign_stages : Stages from the flags and the days past due.
    convert_pd : Cumulative PDs over the remaining life from a term structure.
    transition_pd : Cumulative PDs from a transition matrix.

    Notes
    -----
    With the PD :math:`p` at the reporting date and the PD :math:`p_0`
    expected at origination, the tests are

    .. math::

       p > \mathrm{relative} \cdot p_0 \qquad \text{and} \qquad
       p - p_0 > \mathrm{absolute}.

    With both thresholds, an increase is significant only if it passes both
    tests, so that the absolute threshold keeps small changes of low PDs
    from counting. For a rule under which either test suffices, combine two
    calls with ``|``, each with one threshold. Qualitative criteria, such as
    a watch list or forbearance, combine with the flags in the same way.

    IFRS 9 assesses the change in the risk of a default over the expected
    life of the exposure, and allows the risk over the next 12 months where
    it gives a reasonable approximation. Compare PDs of the same horizon:
    for a lifetime PD, the PD over the same remaining life that was
    expected at origination.

    Examples
    --------
    >>> from kivi.lifetime import significant_increase
    >>> probability = [0.004, 0.012, 0.03, 0.09]
    >>> origination_probability = [0.001, 0.010, 0.01, 0.05]
    >>> significant_increase(
    ...     probability, origination_probability, relative=2.0, absolute=None
    ... )
    array([ True, False,  True, False])
    >>> significant_increase(
    ...     probability, origination_probability, relative=2.0, absolute=0.01
    ... )
    array([False, False,  True, False])

    The first PD quadrupled but rose by only 0.3 percentage points, and the
    last one rose by 4 points without doubling.
    """
    names = ("probability", "origination_probability")
    pairs = {"probability": probability, "origination_probability": origination_probability}
    index = shared_index(pairs)
    current, origination = (_vector(pairs[name], name, booleans=True) for name in names)
    _same_length(current, origination, names)
    for values, name in ((current, "probability"), (origination, "origination_probability")):
        _check_missing(values, name)
        outside = (values < 0) | (values > 1)
        if outside.any():
            raise ValueError(f"{name} must be between 0 and 1, got {float(values[outside][0])!r}")
    if relative is None and absolute is None:
        raise ValueError("relative and absolute must not both be None; pass at least one threshold")
    flag = np.ones(len(current), dtype=bool)
    if relative is not None:
        if not _is_number(relative) or not 1 < relative < math.inf:
            raise ValueError(
                f"relative must be None or a ratio greater than 1, such as 2 for a PD that "
                f"doubled, got {first_value(relative)!r}"
            )
        flag &= current > relative * origination
    if absolute is not None:
        if not _is_number(absolute) or not 0 < absolute < 1:
            raise ValueError(
                f"absolute must be None or a number between 0 and 1, such as 0.01 for one "
                f"percentage point, got {first_value(absolute)!r}"
            )
        flag &= current - origination > absolute
    if index is None:
        return flag
    return pd.Series(flag, index=index, name="significant_increase")


def assign_stages(
    significant_increase: npt.ArrayLike,
    *,
    days_past_due: npt.ArrayLike | None = None,
    defaulted: npt.ArrayLike | None = None,
    stage_2_days: float = 30,
    stage_3_days: float = 90,
) -> pd.Series | npt.NDArray[np.int64]:
    """Allocate exposures to the three stages of IFRS 9.

    Stage 1 holds the exposures whose credit risk has not increased
    significantly since initial recognition, stage 2 those whose credit risk
    has, and stage 3 the credit-impaired exposures. The days past due act as
    backstops: an exposure more than `stage_2_days` past due is at least in
    stage 2, and one at least `stage_3_days` past due, or flagged as
    defaulted, is in stage 3.

    Parameters
    ----------
    significant_increase : array_like
        Whether the credit risk of each exposure has increased significantly
        since initial recognition, as booleans or 0 and 1, such as the
        result of the function `significant_increase`, possibly combined
        with qualitative criteria.
    days_past_due : array_like, optional
        Days past due of each exposure at the reporting date, finite and
        non-negative. Without it, the backstops do not apply.
    defaulted : array_like, optional
        Whether each exposure is credit-impaired, usually because it is in
        default under the entity's definition of default, as booleans or 0
        and 1. These exposures are in stage 3.
    stage_2_days : float, default 30
        Exposures more than this many days past due are at least in stage 2.
        The default follows the rebuttable presumption of IFRS 9, paragraph
        5.5.11, that credit risk has increased significantly when contractual
        payments are more than 30 days past due.
    stage_3_days : float, default 90
        Exposures this many days past due or more are in stage 3. The default
        follows the rebuttable presumption of IFRS 9, paragraph B5.5.37, that
        default occurs no later than when a financial asset is 90 days past
        due. It must be greater than `stage_2_days`.

    Returns
    -------
    pandas.Series or numpy.ndarray
        Stage of each exposure, 1, 2 or 3, as integers. If an input is a
        Series, a Series named ``stage`` with its index; otherwise an array.

    Raises
    ------
    ValueError
        If the inputs are empty, not one-dimensional, of different lengths
        or Series with different indices, `significant_increase` or
        `defaulted` has a value other than 0 and 1, or `days_past_due` is not
        real and numeric or
        has a missing, negative or infinite value. Also if `stage_2_days` or
        `stage_3_days` is not a finite, non-negative number, or
        `stage_3_days` is not greater than `stage_2_days`.

    See Also
    --------
    significant_increase : Quantitative test of a significant increase.
    stage_table : Exposure and expected credit loss by stage.
    expected_credit_loss : 12-month and lifetime expected credit losses.

    Notes
    -----
    The stage of an exposure is 3 if it is defaulted or at least
    `stage_3_days` past due, otherwise 2 if `significant_increase` is true
    or it is more than `stage_2_days` past due, and otherwise 1. With the
    defaults, an exposure exactly 30 days past due therefore stays in stage
    1, and one exactly 90 days past due is in stage 3, as the wording of the
    two presumptions says.

    IFRS 9 defines credit-impaired exposures by events that hurt their
    expected cash flows, such as a default or the significant financial
    difficulty of the borrower. In practice, entities usually align them
    with their definition of default. The thresholds, the definition of
    default and the treatment of cures are the policy of the reporting
    entity. Probation periods, during which an exposure that no longer meets
    a criterion stays in its stage, need the history of each exposure and
    are not applied.

    Examples
    --------
    >>> from kivi.lifetime import assign_stages
    >>> increase = [False, True, False, False, False]
    >>> days_past_due = [0, 0, 45, 90, 0]
    >>> defaulted = [False, False, False, False, True]
    >>> assign_stages(increase, days_past_due=days_past_due, defaulted=defaulted)
    array([1, 2, 2, 3, 3])
    """
    name = "significant_increase"
    inputs = {name: significant_increase, "days_past_due": days_past_due, "defaulted": defaulted}
    index = shared_index(inputs)
    flags = as_series(significant_increase, name)
    if not len(flags):
        raise ValueError(f"{name} must not be empty")
    increased = as_flags(flags, name)
    days = None
    if days_past_due is not None:
        days = _vector(days_past_due, "days_past_due")
        _same_length(increased, days, (name, "days_past_due"))
        invalid = ~np.isfinite(days) | (days < 0)
        if invalid.any():
            raise ValueError(
                f"days_past_due must be finite and non-negative, got {float(days[invalid][0])!r}"
            )
    impaired = np.zeros(len(increased), dtype=bool)
    if defaulted is not None:
        impaired = as_flags(as_series(defaulted, "defaulted"), "defaulted")
        _same_length(increased, impaired, (name, "defaulted"))
    for value, option in ((stage_2_days, "stage_2_days"), (stage_3_days, "stage_3_days")):
        if not _is_number(value) or not 0 <= value < math.inf:
            raise ValueError(
                f"{option} must be a finite, non-negative number, got {first_value(value)!r}"
            )
    if not stage_3_days > stage_2_days:
        raise ValueError(
            f"stage_3_days must be greater than stage_2_days, got "
            f"{first_value(stage_3_days)!r} and {first_value(stage_2_days)!r}"
        )
    if days is not None:
        increased = increased | (days > stage_2_days)
        impaired = impaired | (days >= stage_3_days)
    stage = np.where(impaired, 3, np.where(increased, 2, 1)).astype(np.int64)
    if index is None:
        return stage
    return pd.Series(stage, index=index, name="stage")


def stage_table(
    stage: npt.ArrayLike,
    exposure: npt.ArrayLike,
    allowance: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Summarize the exposure and the expected credit loss of each stage.

    Each row adds up the exposures and the loss allowances of one stage and
    gives their ratio, the coverage ratio that banks report by stage.

    Parameters
    ----------
    stage : array_like
        Stage of each exposure, 1, 2 or 3, such as the result of
        `assign_stages`.
    exposure : array_like
        Exposure of each account at the reporting date, such as its gross
        carrying amount, finite and non-negative.
    allowance : array_like
        Loss allowance of each account, finite and non-negative: its 12-month
        expected credit loss in stage 1 and its lifetime expected credit loss
        in stages 2 and 3. Series inputs must have identical indices;
        otherwise the inputs are matched by position.
    weights : array_like, optional
        Non-negative weight of each exposure, such as the inverse of its
        sampling rate, matched like `exposure`. Counts, exposures and
        allowances become sums of weights, so integer weights give the same
        table as repeating rows.

    Returns
    -------
    pandas.DataFrame
        One row for each of the stages 1, 2 and 3, indexed by ``stage``,
        also for a stage without exposures, with the columns described in
        the Notes.

    Raises
    ------
    ValueError
        If the inputs are empty, not one-dimensional, of different lengths
        or Series with different indices, or not real and numeric. Also if
        `stage` has a value other than 1, 2 and 3, `exposure` or `allowance`
        has a missing, negative or infinite value, or `weights` is not numeric
        or does not give each exposure one finite, non-negative weight with
        a positive total.

    See Also
    --------
    assign_stages : Stage of each exposure.
    expected_credit_loss : 12-month and lifetime expected credit losses.
    kivi.strategy.expected_loss_table : Exposure and expected loss by grade.

    Notes
    -----
    For the exposures :math:`i` of a stage, with weights :math:`w_i`, 1
    without `weights`, the columns are:

    - ``total``: the number of exposures, :math:`\sum_i w_i`;
    - ``exposure``: :math:`\sum_i w_i E_i`, with the exposures :math:`E_i`;
    - ``exposure_share``: the stage's share of the exposure of all stages,
      NaN for every stage when the total exposure is 0;
    - ``expected_credit_loss``: :math:`\sum_i w_i L_i`, with the loss
      allowances :math:`L_i`;
    - ``coverage``: ``expected_credit_loss`` divided by ``exposure``, NaN
      for a stage without exposure.

    The portfolio's exposure and loss allowance are the sums of the
    ``exposure`` and ``expected_credit_loss`` columns, and its coverage
    ratio is the second sum divided by the first.

    Examples
    --------
    >>> from kivi.lifetime import stage_table
    >>> stage = [1, 1, 1, 2, 3]
    >>> exposure = [1000.0, 2000.0, 1000.0, 800.0, 200.0]
    >>> allowance = [5.0, 8.0, 3.0, 40.0, 90.0]
    >>> stage_table(stage, exposure, allowance)
           total  exposure  exposure_share  expected_credit_loss  coverage
    stage
    1          3    4000.0            0.80                  16.0     0.004
    2          1     800.0            0.16                  40.0     0.050
    3          1     200.0            0.04                  90.0     0.450
    """
    inputs = {"stage": stage, "exposure": exposure, "allowance": allowance, "weights": weights}
    shared_index(inputs)
    raw_stage = stage
    stage = as_series(stage, "stage")
    if not len(stage):
        raise ValueError("stage must not be empty")
    if not is_real_dtype(stage.dtype):
        raise ValueError(f"stage must be real and numeric, got dtype {stage.dtype}")
    # For nullable dtypes pandas 2.0 returns an object array, which ~ would negate as integers.
    invalid = ~stage.isin([1, 2, 3]).to_numpy(dtype=bool)
    if invalid.any():
        raise ValueError(f"stage must contain only 1, 2 and 3, got {first_value(stage[invalid])!r}")
    code = stage.to_numpy(dtype=float).astype(np.intp) - 1
    amounts = {}
    for values, name in ((exposure, "exposure"), (allowance, "allowance")):
        array = _vector(values, name)
        _same_length(code, array, ("stage", name))
        invalid = ~np.isfinite(array) | (array < 0)
        if invalid.any():
            raise ValueError(
                f"{name} must be finite and non-negative, got {float(array[invalid][0])!r}"
            )
        amounts[name] = array
    mass = as_weights(weights, raw_stage, of="stage")
    if mass is not None:
        amounts = {name: mass * array for name, array in amounts.items()}
    total = np.bincount(code, weights=mass, minlength=3)
    stage_exposure = np.bincount(code, weights=amounts["exposure"], minlength=3)
    stage_loss = np.bincount(code, weights=amounts["allowance"], minlength=3)
    return pd.DataFrame(
        {
            "total": total,
            "exposure": stage_exposure,
            "exposure_share": ratio(stage_exposure, np.full(3, stage_exposure.sum())),
            "expected_credit_loss": stage_loss,
            "coverage": ratio(stage_loss, stage_exposure),
        },
        index=pd.Index([1, 2, 3], name="stage"),
    )


def _is_number(value: object) -> TypeGuard[float]:
    """Whether a scalar is a real number other than a boolean or NaN."""
    return is_real(value) and not isinstance(value, bool) and not math.isnan(value)


def _missing_as_float(series: pd.Series) -> pd.Series:
    """Read a column of missing values only as floats, as pandas reads [0.1, None]."""
    if pd.api.types.is_object_dtype(series.dtype) and len(series) and series.isna().all():
        return series.astype(float)
    return series


def _float_frame(frame: pd.DataFrame, name: str, *, booleans: bool = False) -> np.ndarray:
    """Real numbers of a DataFrame as a float array; missing values become NaN."""
    if any(pd.api.types.is_object_dtype(dtype) for dtype in frame.dtypes):
        frame = frame.apply(_missing_as_float)
    for dtype in frame.dtypes:
        check_real_dtype(dtype, name, booleans=booleans)
    return frame.to_numpy(dtype=float, na_value=np.nan)


def _curves(values: npt.ArrayLike | pd.DataFrame, name: str) -> npt.NDArray[np.float64]:
    """PD curves as a float array with one row per curve; booleans count as 0 and 1."""
    dimensions = np.ndim(values)
    if dimensions not in (1, 2):
        raise ValueError(f"{name} must be one- or two-dimensional, got {dimensions} dimensions")
    if dimensions == 1:
        series = _missing_as_float(as_series(values, name))
        if not len(series):
            raise ValueError(f"{name} must have at least one period")
        check_real_dtype(series.dtype, name, booleans=True)
        curves = series.to_numpy(dtype=float, na_value=np.nan)[None, :]
    else:
        # pandas reads [[0.1, None]] and nullable arrays as numbers, where NumPy keeps objects.
        frame = values if isinstance(values, pd.DataFrame) else pd.DataFrame(values)  # type: ignore[arg-type]  # takes any 2-D array_like
        if not frame.shape[1]:
            raise ValueError(f"{name} must have at least one period")
        curves = _float_frame(frame, name, booleans=True)
    outside = (curves < 0) | (curves > 1)
    if outside.any():
        raise ValueError(f"{name} must be between 0 and 1, got {float(curves[outside][0])!r}")
    return curves


def _check_curves(curves: np.ndarray, kind: str) -> None:
    """Cumulative PDs must not decrease, and marginal PDs must add up to at most 1."""
    if kind == "cumulative":
        # The running maximum skips missing values, so a decrease across a gap also counts.
        peak = np.fmax.accumulate(curves, axis=1)
        decrease = curves < peak
        if decrease.any():
            raise ValueError(
                "probability must not decrease along a curve when kind is 'cumulative', got "
                f"{float(curves[decrease][0])!r} after {float(peak[decrease][0])!r}"
            )
    elif kind == "marginal":
        total = np.nansum(curves, axis=1)
        excess = total > 1 + _TOLERANCE
        if excess.any():
            raise ValueError(
                "probability must add up to at most 1 along a curve when kind is 'marginal', "
                f"got {float(total[excess][0])!r}"
            )


def _to_cumulative(curves: np.ndarray, kind: str) -> npt.NDArray[np.float64]:
    if kind == "cumulative":
        return curves
    if kind == "marginal":
        # Missing marginal PDs make every later cumulative PD missing.
        return np.minimum(np.cumsum(curves, axis=1), 1.0)
    return 1.0 - np.cumprod(1.0 - curves, axis=1)


def _from_cumulative(cumulative: np.ndarray, to: str) -> npt.NDArray[np.float64]:
    if to == "cumulative":
        return cumulative
    before = np.concatenate([np.zeros((len(cumulative), 1)), cumulative[:, :-1]], axis=1)
    change = cumulative - before
    if to == "marginal":
        return change
    survival = 1.0 - before
    # Where everything has defaulted before, change and survival are both 0 and the PD is 0.
    conditional = np.divide(change, survival, out=np.zeros_like(change), where=survival > 0)
    conditional[np.isnan(change)] = np.nan
    return conditional


def _transition_probabilities(
    matrix: npt.ArrayLike | pd.DataFrame,
) -> tuple[npt.NDArray[np.float64], pd.Index]:
    """Return the matrix as a float array, and its states."""
    if np.ndim(matrix) != 2:
        raise ValueError(f"matrix must be two-dimensional, got {np.ndim(matrix)} dimensions")
    if isinstance(matrix, pd.DataFrame):
        states = matrix.index
        if not states.equals(matrix.columns):
            raise ValueError(
                "matrix must have the same states in its index and its columns, got "
                f"{list(states)} and {list(matrix.columns)}"
            )
        if states.has_duplicates:
            repeated = states[states.duplicated()][:1].tolist()[0]
            raise ValueError(f"matrix must not repeat a state, got {repeated!r} more than once")
        frame = matrix
    else:
        frame = pd.DataFrame(matrix)  # type: ignore[arg-type]  # takes any 2-D array_like
        states = pd.RangeIndex(frame.shape[0])
    if frame.shape[0] != frame.shape[1] or not frame.shape[0]:
        raise ValueError(f"matrix must be square and not empty, got shape {frame.shape}")
    return _float_frame(frame, "matrix"), states


def _check_transition_rows(probabilities: np.ndarray, states: pd.Index, default: int) -> None:
    """Rows other than the default state's hold probabilities that add up to 1."""
    rows = np.delete(np.arange(len(states)), default)
    for row in rows:
        values = probabilities[row]
        # tolist gives Python scalars, also inside the tuples of a MultiIndex.
        state = states[row : row + 1].tolist()[0]
        if np.isnan(values).any():
            raise ValueError(
                f"matrix must not have missing values outside the row of the default state, got "
                f"one in the row of state {state!r}; drop the states without entities"
            )
        outside = (values < 0) | (values > 1)
        if outside.any():
            raise ValueError(
                f"matrix must hold probabilities between 0 and 1, got "
                f"{float(values[outside][0])!r} in the row of state {state!r}"
            )
        total = values.sum()
        if abs(total - 1) > _TOLERANCE:
            raise ValueError(
                f"the rows of matrix must add up to 1, got {float(total)!r} for state {state!r}; "
                f"divide each row by its sum"
            )


def _horizons(horizons: int | npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Increasing positive whole numbers of periods."""
    message = (
        "horizons must be a positive integer or a sequence of increasing positive integers, "
        "got {!r}"
    )
    if isinstance(horizons, pd.Series | pd.Index | np.ndarray):
        shown = horizons.tolist()
    else:
        shown = first_value(horizons) if np.ndim(horizons) == 0 else horizons
    error = ValueError(message.format(shown))
    if np.ndim(horizons) == 0:
        # Whole numbers are accepted as scalars as in sequences, such as 12.0.
        if not (_is_number(horizons) and float(horizons).is_integer() and 1 <= horizons < 2**63):
            raise error
        return np.arange(1, int(horizons) + 1)
    if np.ndim(horizons) != 1:
        raise error
    series = horizons if isinstance(horizons, pd.Series) else pd.Series(horizons)
    if not len(series) or not is_real_dtype(series.dtype):
        raise error
    values = series.to_numpy(dtype=float, na_value=np.nan)
    if (
        not np.isfinite(values).all()
        or (values != np.round(values)).any()
        or values[0] < 1
        or values[-1] >= 2.0**63
        or not (np.diff(values) > 0).all()
    ):
        raise error
    return values.astype(np.int64)


def _account_index(inputs: dict, accounts: int, periods: int) -> pd.Index | None:
    """Return the index of the pandas inputs with one row per account, or None without any.

    Those inputs must share their index, and DataFrames with one column per period their
    columns. A single row or column is broadcast, so its labels are not compared, also when
    there is only one account or one period; the labels of probability are always used.
    """
    rows, columns = [], []
    for name, values in inputs.items():
        # A single row or column is broadcast, unless it is the shape of probability itself.
        own = name == "probability"
        if isinstance(values, pd.Series | pd.DataFrame) and len(values) == accounts:
            if own or accounts > 1:
                rows.append((name, values.index))
        if isinstance(values, pd.DataFrame) and values.shape[1] == periods:
            if own or periods > 1:
                columns.append((name, values.columns))
    for labels, kind in ((rows, "indices"), (columns, "columns")):
        for name, other in labels[1:]:
            if not other.equals(labels[0][1]):
                raise ValueError(f"{labels[0][0]} and {name} must have identical {kind}")
    return rows[0][1] if rows else None


def _vector(values: npt.ArrayLike, name: str, *, booleans: bool = False) -> npt.NDArray[np.float64]:
    """One-dimensional, non-empty real numbers as floats; missing values become NaN.

    Unlike `kivi._validation.as_real` alone, it reads values that are all missing as floats.
    """
    series = _missing_as_float(as_series(values, name))
    if not len(series):
        raise ValueError(f"{name} must not be empty")
    return as_real(series, name, booleans=booleans)


def _same_length(first: np.ndarray, second: np.ndarray, names: tuple[str, str]) -> None:
    if len(first) != len(second):
        raise ValueError(
            f"{names[0]} and {names[1]} must have the same length, got {len(first)} and "
            f"{len(second)}"
        )


def _check_missing(values: np.ndarray, name: str) -> None:
    missing = np.isnan(values)
    if missing.any():
        raise ValueError(f"{name} must not have missing values, got {missing.sum()}")


def _schedule(
    values: float | npt.ArrayLike | pd.DataFrame, name: str, shape: tuple[int, int]
) -> npt.NDArray[np.float64]:
    """One value, one per account or a schedule per account and period, as a 2-D float array."""
    accounts, periods = shape
    dimensions = np.ndim(values)
    message = (
        f"{name} must be one value, one value per account or one per account and period, "
        f"got shape {{}} for {accounts} accounts and {periods} periods"
    )
    if dimensions == 0:
        array = np.asarray(values)
        check_real_dtype(array.dtype, name)
        array = array.astype(float).reshape(1, 1)
    elif dimensions == 1:
        array = _vector(values, name)
        if len(array) not in (1, accounts):
            raise ValueError(message.format(array.shape))
        array = array[:, None]
    elif dimensions == 2:
        frame = values if isinstance(values, pd.DataFrame) else pd.DataFrame(values)  # type: ignore[arg-type]  # takes any 2-D array_like
        array = _float_frame(frame, name)
        if array.shape[0] not in (1, accounts) or array.shape[1] not in (1, periods):
            raise ValueError(message.format(array.shape))
    else:
        raise ValueError(message.format(np.shape(values)))
    return array


def _per_account(
    values: float | npt.ArrayLike, name: str, accounts: int
) -> npt.NDArray[np.float64]:
    """One value or one value per account as a 1-D float array."""
    if np.ndim(values) == 0:
        array = np.asarray(values)
        check_real_dtype(array.dtype, name)
        return array.astype(float).reshape(1)
    if np.ndim(values) != 1:
        raise ValueError(
            f"{name} must be one value or one value per account, got {np.ndim(values)} dimensions"
        )
    array = _vector(values, name)
    if len(array) not in (1, accounts):
        raise ValueError(
            f"{name} must be one value or one value per account, got {len(array)} values for "
            f"{accounts} accounts"
        )
    return array
