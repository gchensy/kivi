"""Lifetime PD term structures and IFRS 9 expected credit losses.

``convert_pd`` converts a PD term structure, such as the cumulative bad
rates of ``kivi.vintage.vintage_table``, between cumulative, marginal and
conditional PDs, ``transition_pd`` projects cumulative PDs from a
transition matrix, and ``survival_table`` estimates them from accounts that
are censored, repaid early or observed only from some age on.
``conditional_matrix`` shifts a transition matrix to the state of the
economy in the one-factor model of Belkin, Suchower and Forest, and
``matrix_factor`` and ``matrix_correlation`` estimate the factor of a
period and the correlation from observed movements.
``expected_credit_loss`` turns marginal PDs, LGDs and exposures into the
discounted 12-month or lifetime expected credit loss of each account.
``significant_increase`` and ``assign_stages`` allocate the
accounts to the three stages of IFRS 9, and ``stage_table`` summarizes the
exposure and the loss allowance of each stage.

Period 1 is the first period after the starting point of a term structure,
such as the origination or the reporting date. The PD thresholds and the
30-day backstop are exceeded strictly, while an exposure 90 days past due is
already in stage 3, as in the presumptions of IFRS 9.
"""

import math
import typing
from collections import abc

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import _counts, _stats, _validation

__all__ = [
    "convert_pd",
    "transition_pd",
    "conditional_matrix",
    "matrix_factor",
    "matrix_correlation",
    "survival_table",
    "expected_credit_loss",
    "significant_increase",
    "assign_stages",
    "stage_table",
]

_KINDS = ("cumulative", "marginal", "conditional")
# Rounding allowed in sums of probabilities that must not exceed 1, or must equal 1.
_TOLERANCE = 1e-9
# The rounding of a PD, of a threshold and of the product or the difference that compares
# them stays below 2 eps times the PD, so a margin of 4 eps keeps ties from counting.
_TIES = 4 * np.finfo(float).eps
_ROOT_TWO_PI = math.sqrt(2 * math.pi)
_HALF_LOG_TWO_PI = 0.5 * math.log(2 * math.pi)
# From here on, the asymptotic series of the Mills ratio to t**-11 is exact to about 3e-15.
_MILLS_SERIES_FROM = 35.0
# matrix_correlation searches the correlations up to the last of this grid.
_CORRELATION_GRID = tuple(step / 100 for step in range(1, 100))


@typing.overload
def convert_pd(probability: pd.DataFrame, *, kind: str, to: str) -> pd.DataFrame: ...
@typing.overload
def convert_pd(probability: pd.Series, *, kind: str, to: str) -> pd.Series: ...
@typing.overload
def convert_pd(
    probability: _validation.ListLike, *, kind: str, to: str
) -> npt.NDArray[np.float64]: ...
@typing.overload
def convert_pd(
    probability: npt.ArrayLike | pd.DataFrame, *, kind: str, to: str
) -> pd.DataFrame | pd.Series | npt.NDArray[np.float64]: ...
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
                f"{_validation.shown_argument(value)}"
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
    default_state: abc.Hashable | None = None,
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
    default = _default_position(states, default_state)
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


def conditional_matrix(
    matrix: npt.ArrayLike | pd.DataFrame,
    *,
    correlation: float,
    factor: float,
    default_state: abc.Hashable | None = None,
) -> pd.DataFrame:
    r"""Shift a transition matrix to the state of the economy in a one-factor model.

    In the one-parameter representation of Belkin, Suchower and Forest
    (1998), the movements from each state follow a standard normal credit
    index, binned by thresholds that reproduce the row of the long-run
    matrix, as in CreditMetrics. The index of every borrower shares one
    systematic factor, so given the factor the whole matrix shifts: in a
    good economy defaults and downgrades become rarer and upgrades more
    frequent, and in a bad one the reverse. Its default column is the
    conditional PD of `kivi.capital.conditional_pd`.

    Parameters
    ----------
    matrix : array_like or pandas.DataFrame
        Square matrix of the long-run probabilities of moving between the
        states over one period, as for `transition_pd`, with the states in
        the order of credit quality from the best to the worst. Each row
        holds probabilities from 0 to 1 that add up to 1, except that the
        row of the default state is not used. The states of a DataFrame
        are its index, which must equal its columns; the states of an
        array are the positions 0, 1, and so on.
    correlation : float
        Asset correlation :math:`\rho` of the one-factor model, from 0 to
        below 1.
    factor : float
        Value :math:`z` of the standard normal systematic factor in the
        period, finite; positive values stand for a better economy than on
        average.
    default_state : optional
        Label of the default state, the worst state whatever its position.
        By default the last state.

    Returns
    -------
    pandas.DataFrame
        The matrix given the factor, with the index and the columns of
        `matrix`, or positions for an array. Each row adds up to 1, and the
        row of the default state stays in default with probability 1.

    Raises
    ------
    ValueError
        If `matrix` or `default_state` is not valid, as for `transition_pd`,
        `correlation` is not a real number from 0 to below 1, or `factor`
        is not a finite real number.

    See Also
    --------
    matrix_factor : Estimate the factor of a period from its movements.
    matrix_correlation : Estimate the correlation from the movements of several periods.
    transition_pd : Cumulative PD from the powers of a transition matrix.
    kivi.capital.conditional_pd : The conditional PD, the default column of the result.

    Notes
    -----
    With the states ordered from the best to the worst and the default
    state last, let :math:`C_{ij}` be the long-run probability of moving
    from state :math:`i` to state :math:`j` or a worse one. The index
    :math:`X = \sqrt{\rho}\,Z + \sqrt{1 - \rho}\,Y` of a borrower in state
    :math:`i` lands in state :math:`j` or a worse one when it falls below
    :math:`b_{ij} = \Phi^{-1}(C_{ij})`, and given :math:`Z = z`

    .. math::

       C_{ij}(z) = \Phi\!\left(\frac{b_{ij} - \sqrt{\rho}\,z}{\sqrt{1 - \rho}}\right).

    Each entry of the result is the difference of two consecutive
    :math:`C_{ij}(z)`, computed from the smaller tails, so entries of 0
    and 1 stay exact and small ones keep their relative accuracy.

    A factor of 0 gives the matrix of the median economy, not `matrix`:
    averaged over the standard normal factor, the result of one period is
    `matrix`. The same holds for a path of independent factors, one per
    period, whose product averages to the power of `matrix`, but not for
    one factor kept over several periods, whose average product differs
    from the power: with two states its PD is lower, and with more states
    it can be higher for some states and lower for others. The factor and
    the correlation belong to the period of `matrix`, such as a month or a
    year.

    The thresholds follow the CreditMetrics Technical Document (Gupton,
    Finger and Bhatia, 1997, section 8.4), and the conditional matrix is
    that of Witzany (2020, eq. 10). Exits that have no place in the order of
    credit quality, such as early repayment, must be left out and the rows
    divided by their sums first, which gives the movements of the loans
    that do not exit.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.lifetime import conditional_matrix, transition_pd
    >>> states = ["A", "B", "D"]
    >>> matrix = pd.DataFrame(
    ...     [[0.90, 0.08, 0.02], [0.10, 0.70, 0.20], [0.00, 0.00, 1.00]],
    ...     index=states,
    ...     columns=states,
    ... )
    >>> shifted = conditional_matrix(matrix, correlation=0.1, factor=-1.0)
    >>> shifted.round(4)
            A       B       D
    A  0.8456  0.1209  0.0335
    B  0.0461  0.6641  0.2899
    D  0.0000  0.0000  1.0000

    With the factor one standard deviation below its median, state A
    defaults with probability 0.0335 instead of 0.02, which is
    ``conditional_pd(0.02, correlation=0.1, factor=-1.0)``, and moves to B
    more often, while B returns to A less often. Kept for three periods,
    the factor gives these cumulative PDs:

    >>> transition_pd(shifted, horizons=3).round(4)
    horizon       1       2       3
    state
    A        0.0335  0.0969  0.1740
    B        0.2899  0.4839  0.6157
    D        1.0000  1.0000  1.0000
    """
    probabilities, states = _transition_probabilities(matrix)
    default = _default_position(states, default_state)
    _check_transition_rows(probabilities, states, default)
    rho = _checked_correlation(correlation, positive=False)
    z = _checked_factor(factor)
    order = _credit_order(len(states), default)
    rows = _credit_rows(probabilities, order)
    # A factor near the largest float shifts the thresholds beyond it, to infinity.
    with np.errstate(over="ignore"):
        bounds = (_thresholds(rows) - math.sqrt(rho) * z) / math.sqrt(1 - rho)
    result = np.zeros((len(states), len(states)))
    result[np.ix_(order[:-1], order)] = _interval_probabilities(bounds[:, :-1], bounds[:, 1:])
    result[default, default] = 1.0
    if isinstance(matrix, pd.DataFrame):
        return pd.DataFrame(result, index=matrix.index, columns=matrix.columns)
    return pd.DataFrame(result)


def matrix_factor(
    counts: npt.ArrayLike | pd.DataFrame,
    matrix: npt.ArrayLike | pd.DataFrame,
    *,
    correlation: float,
    default_state: abc.Hashable | None = None,
) -> float:
    r"""Estimate the systematic factor of a period from its movements between states.

    The factor is the value under which `conditional_matrix` explains the
    movements of the period best: the maximum-likelihood estimate of the
    multinomial distribution of each state's movements. A negative factor
    marks a period worse than on average, with more defaults and
    downgrades than `matrix` gives.

    Parameters
    ----------
    counts : array_like or pandas.DataFrame
        Square matrix of the numbers of entities that moved between the
        states in the period, such as `kivi.vintage.transition_matrix`
        without ``normalize``, or sums of frequency weights. A DataFrame
        must have the index and the columns of `matrix` if that is a
        DataFrame; arrays are matched by position. The counts are
        non-negative and finite, except that the row of the default state
        is not used.
    matrix : array_like or pandas.DataFrame
        Long-run transition matrix, as for `conditional_matrix`, with the
        states ordered from the best to the worst.
    correlation : float
        Asset correlation of the one-factor model, strictly between 0 and 1.
    default_state : optional
        Label of the default state, as for `conditional_matrix`.

    Returns
    -------
    float
        The estimated factor.

    Raises
    ------
    ValueError
        If `matrix`, `correlation` or `default_state` is not valid, as for
        `conditional_matrix`, or `correlation` is 0; if `counts` does not
        match `matrix`, has a negative, missing or infinite count outside
        the row of the default state, or has no movement from a state that
        can move to more than one state; if an entity moved where `matrix`
        gives a probability of 0; or if the factor is not finite, because
        every movement is to the best or every one to the worst state that
        its state can reach.

    See Also
    --------
    conditional_matrix : The transition matrix given the factor.
    matrix_correlation : The correlation under which the factors of several periods vary as Z.
    kivi.capital.systematic_factor : The factor that a default rate implies.

    Notes
    -----
    With :math:`n_{ij}` entities moving from state :math:`i` to state
    :math:`j` and the probabilities :math:`p_{ij}(z)` of
    `conditional_matrix`, the factor maximizes

    .. math::

       \ell(z) = \sum_{i \ne d} \sum_j n_{ij} \log p_{ij}(z)

    over the states :math:`i` other than the default state :math:`d`. Each
    term is concave in :math:`z`, because the normal density is log-concave,
    so the maximum is unique; it is found by bisection on the derivative.
    Counts scaled by a constant give the same factor, and with two states
    the factor is the one `kivi.capital.systematic_factor` gives for the
    default rate.

    Belkin, Suchower and Forest (1998) instead minimize, as restated by
    Shao, Li and Li (2018, eq. 1.5), the squared differences between the
    observed and the conditional probabilities, each weighted by the number
    of entities of its state over the binomial variance
    :math:`p_{ij}(z)(1 - p_{ij}(z))`. That weight grows without bound for a
    rare movement, so that a single entity moving from the best state to
    one it reaches with probability 0.00001 can move the factor by a
    quarter of a standard deviation; the likelihood is not as fragile.

    The factor absorbs every change that moves the rows together, also a
    change in the mix of the portfolio, such as its age, so `counts` and
    `matrix` should describe comparable portfolios.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.lifetime import conditional_matrix, matrix_factor
    >>> states = ["A", "B", "D"]
    >>> matrix = pd.DataFrame(
    ...     [[0.90, 0.08, 0.02], [0.10, 0.70, 0.20], [0.00, 0.00, 1.00]],
    ...     index=states,
    ...     columns=states,
    ... )
    >>> counts = pd.DataFrame(
    ...     [[850, 110, 40], [60, 280, 160], [0, 0, 0]], index=states, columns=states
    ... )
    >>> print(f"{matrix_factor(counts, matrix, correlation=0.1):.4f}")
    -0.8425

    Both states defaulted more often than `matrix` gives, 4% of A against
    2% and 32% of B against 20%, so the period was worse than on average.
    """
    probabilities, states = _transition_probabilities(matrix)
    default = _default_position(states, default_state)
    _check_transition_rows(probabilities, states, default)
    rho = _checked_correlation(correlation, positive=True)
    model = _MatrixModel(probabilities, states, default)
    cells = model.cells([_count_matrix(counts, matrix, states, default, "")], [""])
    return float(cells.factors(rho)[0])


def matrix_correlation(
    counts: abc.Mapping[abc.Hashable, npt.ArrayLike | pd.DataFrame]
    | abc.Sequence[npt.ArrayLike | pd.DataFrame]
    | pd.Series,
    matrix: npt.ArrayLike | pd.DataFrame,
    *,
    default_state: abc.Hashable | None = None,
) -> float:
    r"""Estimate the asset correlation from the movements of several periods.

    Belkin, Suchower and Forest (1998) choose the correlation under which
    the factors of the periods vary as much as the standard normal factor
    of the model: their sample variance is 1. The factor of each period is
    the one of `matrix_factor`.

    Parameters
    ----------
    counts : mapping, sequence or pandas.Series
        Count matrices of the periods, at least two, each as for
        `matrix_factor`: a mapping or a Series from the label of each
        period, such as its month, to its counts, or a list or tuple of
        counts. Errors name the label or the position of the period.
    matrix : array_like or pandas.DataFrame
        Long-run transition matrix, as for `conditional_matrix`, with the
        states ordered from the best to the worst.
    default_state : optional
        Label of the default state, as for `conditional_matrix`.

    Returns
    -------
    float
        The correlation, strictly between 0 and 1.

    Raises
    ------
    ValueError
        If `matrix` or `default_state` is not valid, as for
        `conditional_matrix`; if `counts` is a single matrix or has fewer
        than two periods, or the counts of a period are not valid, as for
        `matrix_factor`; or if no correlation up to 0.99 gives the factors
        a variance of 1, because they are all equal or vary too much.

    See Also
    --------
    matrix_factor : The factor of one period, given the correlation.
    conditional_matrix : The transition matrix given the factor.
    kivi.capital.vasicek_fit : The correlation from a series of default rates.

    Notes
    -----
    The sample variance is taken with ``ddof=1``, as in
    `kivi.capital.vasicek_fit`, and the mean of the factors is not forced
    to 0. The variance falls steeply from infinity as the correlation
    rises from 0 and flattens out near 0.5; at higher correlations it can
    rise and fall again, a shape checked numerically rather than proven.
    The result is the smallest correlation up to 0.99 with a variance of 1:
    a grid in steps of 0.01 finds the first correlation with a variance of
    at most 1, and bisection the correlation between it and the step
    before. Above about 0.5 the unit-variance condition does not identify
    the correlation, and the true one may be a larger root: the expected
    counts of six periods whose factors have a sample variance of 1, under
    the correlations 0.6, 0.7, 0.8 and 0.9 and the matrix of Witzany (2020,
    table 2), give 0.557, 0.503, 0.462 and 0.431.

    Because the variance of the estimated factors is set to 1, the estimate
    is about the true correlation times the sample variance of the true
    factors of the periods, so a few periods give an uncertain estimate.
    The binomial noise of small portfolios adds to the variance of the
    factors and raises the estimate.

    The unit-variance condition is that of Belkin, Suchower and Forest, as
    restated by Shao, Li and Li (2018, section 2); they estimate the factors
    by weighted least squares instead (see `matrix_factor`). Another route
    estimates the correlation from a series of default rates with
    `kivi.capital.vasicek_fit` and the factors with
    `kivi.capital.systematic_factor`.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.lifetime import matrix_correlation, matrix_factor
    >>> states = ["A", "B", "D"]
    >>> matrix = pd.DataFrame(
    ...     [[0.90, 0.08, 0.02], [0.10, 0.70, 0.20], [0.00, 0.00, 1.00]],
    ...     index=states,
    ...     columns=states,
    ... )
    >>> years = {
    ...     2023: [[930, 60, 10], [130, 740, 130], [0, 0, 0]],
    ...     2024: [[850, 110, 40], [60, 280, 160], [0, 0, 0]],
    ...     2025: [[900, 80, 20], [100, 700, 200], [0, 0, 0]],
    ... }
    >>> correlation = matrix_correlation(years, matrix)
    >>> print(f"{correlation:.4f}")
    0.0515
    >>> factors = [matrix_factor(c, matrix, correlation=correlation) for c in years.values()]
    >>> [round(factor, 4) for factor in factors]
    [0.8946, -1.1045, -0.0529]

    The counts are lists matched to the states of `matrix` by position.
    With this correlation, the factors of the three years have a sample
    variance of 1.
    """
    probabilities, states = _transition_probabilities(matrix)
    default = _default_position(states, default_state)
    _check_transition_rows(probabilities, states, default)
    wheres, values = _periods(counts)
    model = _MatrixModel(probabilities, states, default)
    matrices = [
        _count_matrix(value, matrix, states, default, where)
        for value, where in zip(values, wheres, strict=True)
    ]
    # Check the periods in the order given, so that an error names the first invalid one.
    model.cells(matrices, wheres)
    # The factors are computed in one batch. In a canonical order of the periods, the batch is
    # the same whatever the order of counts, and so are the factors to the last bit.
    canonical = sorted(range(len(matrices)), key=lambda period: matrices[period].tobytes())
    cells = model.cells([matrices[k] for k in canonical], [wheres[k] for k in canonical])

    def variance(rho: float) -> float:
        result = _sample_variance(cells.factors(rho))
        if result == 0:
            raise ValueError(
                "counts must differ between the periods; their factors are all equal, so no "
                "correlation gives them a variance of 1"
            )
        return result

    # The variance falls steeply from infinity as the correlation rises from 0, but it need not
    # keep falling: the first step of the grid with a variance of at most 1 bounds the smallest
    # correlation with a variance of 1.
    low = 0.0
    for high in _CORRELATION_GRID:
        if variance(high) <= 1:
            break
        low = high
    else:
        raise ValueError(
            "counts vary too much between the periods: the variance of their factors stays above "
            f"1 for every correlation up to {_CORRELATION_GRID[-1]}"
        )
    # Bisection ends when no number lies between the two ends.
    middle = (low + high) / 2
    while low < middle < high:
        if variance(middle) > 1:
            low = middle
        else:
            high = middle
        middle = (low + high) / 2
    return middle


def survival_table(
    duration: npt.ArrayLike,
    event: npt.ArrayLike,
    *,
    competing: npt.ArrayLike | None = None,
    entry: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Estimate a cumulative PD term structure from censored and truncated accounts.

    Each account, such as a loan, is followed period by period from a
    starting point, such as its origination, until it defaults, leaves for
    another reason, such as an early repayment, or is last observed, when it
    is censored. The table gives the number of accounts at risk in each
    period, the hazard rate of default and, without `competing`, the
    Kaplan–Meier estimate of the cumulative PD or, with it, the
    Aalen–Johansen estimate of the cumulative incidence of default.

    Parameters
    ----------
    duration : array_like
        Period of each account's default, exit or last observation, a
        positive whole number, such as the month on book of its last row in
        a monthly panel.
    event : array_like
        Whether the account defaulted in period `duration`, as booleans or 0
        and 1.
    competing : array_like, optional
        Whether the account left in period `duration` for another reason
        that ends its risk of default, such as an early repayment, as
        booleans or 0 and 1. It must not be 1 on a row where `event` is 1.
        Without it, every account that did not default is censored.
    entry : array_like, optional
        Number of periods that had passed when each account came under
        observation, a whole number from 0 to its `duration` − 1, by default
        0. An account is at risk from period ``entry + 1`` on, so a loan
        first observed at month on book 7 has an entry of 6. This handles
        left truncation, such as a panel that starts after the loans were
        originated.
    weights : array_like, optional
        Non-negative frequency weight of each account. Counts become sums of
        weights, so integer weights give the same table as repeated
        accounts, and accounts with weight 0 are left out. Series inputs
        must have identical indices; otherwise the inputs are matched by
        position.

    Returns
    -------
    pandas.DataFrame
        One row per period from 1 to the largest `duration`, indexed by
        ``period``, with the columns ``at_risk``, ``event``, ``competing``
        (only with `competing`), ``censored``, ``hazard``, ``survival`` and
        ``cumulative_pd`` described in the Notes. The counts are integers
        without `weights` and sums of weights with them.

    Raises
    ------
    ValueError
        If the inputs are empty, not one-dimensional, of different lengths
        or Series with different indices. Also if `duration` is not real and
        numeric or has a value that is not a positive whole number, `event`
        or `competing` has a value other than 0 and 1, both are 1 on a row,
        `entry` is not real and numeric or has a value that is not a whole
        number from 0 to the account's duration − 1, or `weights` is not
        numeric or does not give each account one finite, non-negative
        weight with a positive total.

    See Also
    --------
    convert_pd : Marginal and conditional PDs from ``cumulative_pd``.
    expected_credit_loss : Expected credit loss from marginal PDs.
    kivi.vintage.vintage_table : Cumulative bad rates by cohort and month on book.
    kivi.datasets.make_loan_panel : Synthetic loan panel with early repayments.

    Notes
    -----
    For an account :math:`i` with the duration :math:`T_i`, the entry
    :math:`e_i` and the weight :math:`w_i`, 1 without `weights`, the columns
    of period :math:`t` are:

    - ``at_risk``: :math:`n_t = \sum_i w_i \, [e_i < t \le T_i]`, the
      accounts at risk of default in the period;
    - ``event``: :math:`d_t`, the weight of the accounts that defaulted in
      the period;
    - ``competing``: :math:`c_t`, that of the accounts that left for another
      reason;
    - ``censored``: that of the accounts last observed in the period without
      a default or an exit;
    - ``hazard``: :math:`h_t = d_t / n_t`, the share of the accounts at risk
      that defaulted;
    - ``survival``: :math:`S(t) = \prod_{s \le t} (1 - (d_s + c_s) / n_s)`,
      the probability of neither a default nor an exit by the end of the
      period, with :math:`S(0) = 1`;
    - ``cumulative_pd``: the probability of a default by the end of the
      period.

    Without `competing`, :math:`c_t = 0`, and ``cumulative_pd`` is the
    Kaplan–Meier estimate

    .. math::

       F(t) = 1 - \prod_{s \le t} (1 - h_s) = 1 - S(t).

    With `competing`, it is the Aalen–Johansen estimate of the cumulative
    incidence of default, which adds up the defaults of the accounts that
    have neither defaulted nor left before:

    .. math::

       F(t) = \sum_{s \le t} S(s - 1) \, h_s.

    The accounts censored in a period are at risk in that period: their
    censoring comes after the defaults and the exits of the period. A
    period without accounts at risk has a NaN hazard, and ``survival`` and
    ``cumulative_pd`` are NaN from that period on, because the share of
    accounts that get through it is not known. With `entry` this also
    happens in the first periods if no account is observed in them; the
    hazards of the later periods then give the PDs conditional on reaching
    them.

    An account that leaves early can no longer default. Counting it as
    censored, as 1 − Kaplan–Meier does, estimates the PD of an account that
    stays on book until its default, and overstates the share of the
    accounts that default: pass the early exits as `competing` instead. If
    every account is observed from period 1 until it defaults or leaves,
    or until a last period common to all of them, the cumulative incidence
    is the share of the accounts that defaulted by each period, the
    cumulative bad rate of `kivi.vintage.vintage_table`.

    ``cumulative_pd`` is a cumulative term structure for `convert_pd`.
    Without `competing`, its conditional PDs are the hazards. With
    `competing`, they are conditional on no default before, also counting
    the accounts that left, as in a vintage table, while the hazards are
    conditional on the account being still on book.

    References
    ----------
    .. [1] E. L. Kaplan and P. Meier, "Nonparametric estimation from
       incomplete observations", Journal of the American Statistical
       Association, 53(282), pp. 457-481, 1958.
    .. [2] O. O. Aalen and S. Johansen, "An empirical transition matrix for
       non-homogeneous Markov chains based on censored observations",
       Scandinavian Journal of Statistics, 5(3), pp. 141-150, 1978.

    Examples
    --------
    Eight loans over three months: three default, two repay early and three
    are still on book when the observation ends.

    >>> from kivi.lifetime import survival_table
    >>> duration = [1, 1, 2, 2, 2, 3, 3, 3]
    >>> event = [1, 0, 1, 0, 0, 1, 0, 0]
    >>> repaid = [0, 1, 0, 1, 0, 0, 0, 0]
    >>> survival_table(duration, event, competing=repaid).round(4)
            at_risk  event  competing  censored  hazard  survival  cumulative_pd
    period
    1             8      1          1         0  0.1250    0.7500         0.1250
    2             6      1          1         1  0.1667    0.5000         0.2500
    3             3      1          0         2  0.3333    0.3333         0.4167

    In the third month, half of the loans are still on book, and a third of
    them default: 0.25 + 0.5 × 1/3 = 0.4167. Counting the early repayments
    as censored gives the Kaplan–Meier estimate 1 − (7/8)(5/6)(2/3), which
    is higher:

    >>> survival_table(duration, event)["cumulative_pd"].round(4)
    period
    1    0.1250
    2    0.2708
    3    0.5139
    Name: cumulative_pd, dtype: float64
    """
    inputs = {
        "duration": duration,
        "event": event,
        "competing": competing,
        "entry": entry,
        "weights": weights,
    }
    _validation.shared_index(inputs)
    exit_period = _whole_numbers(duration, "duration", lowest=1)
    defaulted = _flag_vector(event, "event", exit_period)
    left = np.zeros(len(exit_period), dtype=bool)
    if competing is not None:
        left = _flag_vector(competing, "competing", exit_period)
        both = defaulted & left
        if both.any():
            raise ValueError(
                "event and competing must not both be 1 on a row, because an account that "
                f"defaulted did not leave for another reason; got both on {both.sum()} row(s)"
            )
    start = np.zeros(len(exit_period), dtype=np.int64)
    if entry is not None:
        start = _whole_numbers(entry, "entry", lowest=0)
        _same_length(exit_period, start, ("duration", "entry"))
        late = start >= exit_period
        if late.any():
            raise ValueError(
                "entry must be smaller than duration, because an account is at risk from period "
                f"entry + 1 on; got an entry of {start[late][0]} with a duration of "
                f"{exit_period[late][0]}"
            )
    mass = _validation.as_weights(
        weights,
        duration,
        of="duration",
        others={"event": event, "competing": competing, "entry": entry},
    )
    if mass is not None:
        kept = mass > 0
        exit_period, start, mass = exit_period[kept], start[kept], mass[kept]
        defaulted, left = defaulted[kept], left[kept]

    periods = int(exit_period.max())

    def by_period(rows: npt.NDArray[np.bool_]) -> np.ndarray:
        """Count or weight of the given accounts by the period in which they stop."""
        return np.bincount(
            exit_period[rows] - 1,
            weights=None if mass is None else mass[rows],
            minlength=periods,
        )

    at_risk = _at_risk(start + 1, exit_period + 1, mass, periods)
    events = by_period(defaulted)
    exits = by_period(left)
    censored = by_period(~defaulted & ~left)
    # A ratio may round above 1 when all accounts at risk stop; NaN marks an empty risk set.
    hazard = np.minimum(_counts.ratio(events, at_risk), 1.0)
    survival = np.cumprod(np.maximum(1.0 - _counts.ratio(events + exits, at_risk), 0.0))
    if competing is None:
        cumulative = 1.0 - survival
    else:
        before = np.concatenate([[1.0], survival[:-1]])
        cumulative = np.minimum(np.cumsum(before * hazard), 1.0)
    columns = {"at_risk": at_risk, "event": events}
    if competing is not None:
        columns["competing"] = exits
    columns.update(
        {"censored": censored, "hazard": hazard, "survival": survival, "cumulative_pd": cumulative}
    )
    return pd.DataFrame(columns, index=pd.Index(np.arange(1, periods + 1), name="period"))


@typing.overload
def expected_credit_loss(
    probability: pd.DataFrame,
    lgd: float | npt.ArrayLike | pd.DataFrame,
    ead: float | npt.ArrayLike | pd.DataFrame,
    *,
    rate: float | npt.ArrayLike,
    horizon: int | None = ...,
) -> pd.Series: ...
@typing.overload
def expected_credit_loss(
    probability: _validation.ListLike,
    lgd: float | _validation.ListLike,
    ead: float | _validation.ListLike,
    *,
    rate: float | _validation.ListLike,
    horizon: int | None = ...,
) -> npt.NDArray[np.float64]: ...
@typing.overload
def expected_credit_loss(
    probability: npt.ArrayLike | pd.DataFrame,
    lgd: float | npt.ArrayLike | pd.DataFrame,
    ead: float | npt.ArrayLike | pd.DataFrame,
    *,
    rate: float | npt.ArrayLike,
    horizon: int | None = ...,
) -> pd.Series | npt.NDArray[np.float64]: ...
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
    elif isinstance(horizon, bool) or not _validation.is_integral(horizon) or horizon < 1:
        raise ValueError(
            f"horizon must be a positive integer, got {_validation.shown_argument(horizon)}"
        )
    elif horizon > periods:
        raise ValueError(
            f"horizon must be at most the number of periods of probability, {periods}, "
            f"got {_validation.shown_argument(horizon)}"
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


@typing.overload
def significant_increase(
    probability: pd.Series,
    origination_probability: npt.ArrayLike,
    *,
    relative: float | None,
    absolute: float | None,
) -> pd.Series: ...
@typing.overload
def significant_increase(
    probability: npt.ArrayLike,
    origination_probability: pd.Series,
    *,
    relative: float | None,
    absolute: float | None,
) -> pd.Series: ...
@typing.overload
def significant_increase(
    probability: _validation.ListLike,
    origination_probability: _validation.ListLike,
    *,
    relative: float | None,
    absolute: float | None,
) -> npt.NDArray[np.bool_]: ...
@typing.overload
def significant_increase(
    probability: npt.ArrayLike,
    origination_probability: npt.ArrayLike,
    *,
    relative: float | None,
    absolute: float | None,
) -> pd.Series | npt.NDArray[np.bool_]: ...
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

    Both comparisons are strict, and a PD that meets a threshold up to the
    rounding of floating-point numbers, within a few units in the last
    place of the PD, meets it exactly: in floating point, ``0.04 - 0.03``
    exceeds ``0.01`` and ``1.5 * 0.30`` is below ``0.45``, but neither pair
    is flagged.

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
    index = _validation.shared_index(pairs)
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
                f"doubled, got {_validation.shown_argument(relative)}"
            )
        flag &= current > relative * origination + _TIES * current
    if absolute is not None:
        if not _is_number(absolute) or not 0 < absolute < 1:
            raise ValueError(
                f"absolute must be None or a number between 0 and 1, such as 0.01 for one "
                f"percentage point, got {_validation.shown_argument(absolute)}"
            )
        flag &= current - origination > absolute + _TIES * current
    if index is None:
        return flag
    return pd.Series(flag, index=index, name="significant_increase")


@typing.overload
def assign_stages(
    significant_increase: pd.Series,
    days_past_due: npt.ArrayLike | None = ...,
    defaulted: npt.ArrayLike | None = ...,
    *,
    stage_2_days: float = ...,
    stage_3_days: float = ...,
) -> pd.Series: ...
@typing.overload
def assign_stages(
    significant_increase: npt.ArrayLike,
    days_past_due: pd.Series,
    defaulted: npt.ArrayLike | None = ...,
    *,
    stage_2_days: float = ...,
    stage_3_days: float = ...,
) -> pd.Series: ...
@typing.overload
def assign_stages(
    significant_increase: npt.ArrayLike,
    days_past_due: npt.ArrayLike | None,
    defaulted: pd.Series,
    *,
    stage_2_days: float = ...,
    stage_3_days: float = ...,
) -> pd.Series: ...
@typing.overload
def assign_stages(
    significant_increase: _validation.ListLike,
    days_past_due: _validation.ListLike | None = ...,
    defaulted: _validation.ListLike | None = ...,
    *,
    stage_2_days: float = ...,
    stage_3_days: float = ...,
) -> npt.NDArray[np.int64]: ...
@typing.overload
def assign_stages(
    significant_increase: npt.ArrayLike,
    days_past_due: npt.ArrayLike | None = ...,
    defaulted: npt.ArrayLike | None = ...,
    *,
    stage_2_days: float = ...,
    stage_3_days: float = ...,
) -> pd.Series | npt.NDArray[np.int64]: ...
def assign_stages(
    significant_increase: npt.ArrayLike,
    days_past_due: npt.ArrayLike | None = None,
    defaulted: npt.ArrayLike | None = None,
    *,
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
    >>> assign_stages(increase, days_past_due, defaulted)
    array([1, 2, 2, 3, 3])
    """
    name = "significant_increase"
    inputs = {name: significant_increase, "days_past_due": days_past_due, "defaulted": defaulted}
    index = _validation.shared_index(inputs)
    flags = _validation.as_series(significant_increase, name)
    if not len(flags):
        raise ValueError(f"{name} must not be empty")
    increased = _validation.as_flags(flags, name)
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
        impaired = _validation.as_flags(_validation.as_series(defaulted, "defaulted"), "defaulted")
        _same_length(increased, impaired, (name, "defaulted"))
    for value, option in ((stage_2_days, "stage_2_days"), (stage_3_days, "stage_3_days")):
        if not _is_number(value) or not 0 <= value < math.inf:
            raise ValueError(
                f"{option} must be a finite, non-negative number, got "
                f"{_validation.shown_argument(value)}"
            )
    if not stage_3_days > stage_2_days:
        raise ValueError(
            f"stage_3_days must be greater than stage_2_days, got "
            f"{_validation.shown_argument(stage_3_days)} and "
            f"{_validation.shown_argument(stage_2_days)}"
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
    _validation.shared_index(inputs)
    raw_stage = stage
    stage = _validation.as_series(stage, "stage")
    if not len(stage):
        raise ValueError("stage must not be empty")
    if not _validation.is_real_dtype(stage.dtype):
        raise ValueError(f"stage must be real and numeric, got dtype {stage.dtype}")
    # For nullable dtypes pandas 2.0 returns an object array, which ~ would negate as integers.
    invalid = ~stage.isin([1, 2, 3]).to_numpy(dtype=bool)
    if invalid.any():
        raise ValueError(
            f"stage must contain only 1, 2 and 3, got {_validation.shown(stage[invalid])}"
        )
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
    mass = _validation.as_weights(weights, raw_stage, of="stage")
    if mass is not None:
        amounts = {name: mass * array for name, array in amounts.items()}
    total = np.bincount(code, weights=mass, minlength=3)
    stage_exposure = np.bincount(code, weights=amounts["exposure"], minlength=3)
    stage_loss = np.bincount(code, weights=amounts["allowance"], minlength=3)
    return pd.DataFrame(
        {
            "total": total,
            "exposure": stage_exposure,
            "exposure_share": _counts.ratio(stage_exposure, np.full(3, stage_exposure.sum())),
            "expected_credit_loss": stage_loss,
            "coverage": _counts.ratio(stage_loss, stage_exposure),
        },
        index=pd.Index([1, 2, 3], name="stage"),
    )


def _is_number(value: object) -> typing.TypeGuard[float]:
    """Whether a scalar is a real number other than a boolean or NaN."""
    return _validation.is_real(value) and not isinstance(value, bool) and not math.isnan(value)


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
        _validation.check_real_dtype(dtype, name, booleans=booleans)
    return frame.to_numpy(dtype=float, na_value=np.nan)


def _curves(values: npt.ArrayLike | pd.DataFrame, name: str) -> npt.NDArray[np.float64]:
    """PD curves as a float array with one row per curve; booleans count as 0 and 1."""
    dimensions = np.ndim(values)
    if dimensions not in (1, 2):
        raise ValueError(f"{name} must be one- or two-dimensional, got {dimensions} dimensions")
    if dimensions == 1:
        series = _missing_as_float(_validation.as_series(values, name))
        if not len(series):
            raise ValueError(f"{name} must have at least one period")
        _validation.check_real_dtype(series.dtype, name, booleans=True)
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
            raise ValueError(
                f"matrix must not repeat a state, got {_validation.shown(repeated)} more than once"
            )
        frame = matrix
    else:
        frame = pd.DataFrame(matrix)  # type: ignore[arg-type]  # takes any 2-D array_like
        states = pd.RangeIndex(frame.shape[0])
    if frame.shape[0] != frame.shape[1] or not frame.shape[0]:
        raise ValueError(f"matrix must be square and not empty, got shape {frame.shape}")
    return _float_frame(frame, "matrix"), states


def _default_position(states: pd.Index, default_state: abc.Hashable | None) -> int:
    """Return the position of the default state, by default the last one."""
    if default_state is None:
        return len(states) - 1
    try:
        default = states.get_loc(default_state)
        # get_loc also finds the interval that holds a number; the label itself must match.
        found = _validation.is_integral(default) and bool(states[default] == default_state)
    except (KeyError, TypeError, ValueError, pd.errors.InvalidIndexError):
        found = False
    if not found:
        raise ValueError(
            "default_state must be one of the states, got "
            f"{_validation.shown_argument(default_state)}"
        )
    return typing.cast(int, default)


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
                f"one in the row of state {_validation.shown(state)}; drop the states without "
                "entities"
            )
        outside = (values < 0) | (values > 1)
        if outside.any():
            raise ValueError(
                f"matrix must hold probabilities between 0 and 1, got "
                f"{float(values[outside][0])!r} in the row of state {_validation.shown(state)}"
            )
        total = values.sum()
        if abs(total - 1) > _TOLERANCE:
            raise ValueError(
                f"the rows of matrix must add up to 1, got {float(total)!r} for state "
                f"{_validation.shown(state)}; divide each row by its sum"
            )


def _checked_correlation(correlation: object, *, positive: bool) -> float:
    """Return a correlation from 0, or above 0 when positive, to below 1 as a float."""
    if isinstance(correlation, (bool, np.bool_)) or not _validation.is_real(correlation):
        valid = False
    else:
        value = float(correlation)
        valid = 0 < value < 1 if positive else 0 <= value < 1
    if not valid:
        interval = "strictly between 0 and 1" if positive else "from 0 to below 1"
        raise ValueError(
            f"correlation must be a number {interval}, got "
            f"{_validation.shown_argument(correlation)}"
        )
    return float(typing.cast(float, correlation))


def _checked_factor(factor: object) -> float:
    """Return a finite real factor as a float."""
    if (
        isinstance(factor, (bool, np.bool_))
        or not _validation.is_real(factor)
        or not math.isfinite(factor)
    ):
        raise ValueError(
            f"factor must be a finite number, got {_validation.shown_argument(factor)}"
        )
    return float(factor)


def _credit_order(size: int, default: int) -> npt.NDArray[np.intp]:
    """Positions of the states from the best to the worst: the others in order, then default."""
    return np.append(np.delete(np.arange(size), default), default)


def _credit_rows(probabilities: np.ndarray, order: np.ndarray) -> npt.NDArray[np.float64]:
    """Return the rows other than the default state's, in credit order, divided by their sums.

    Rows may add up to 1 within 1e-9 only; thresholds taken from the two ends of such a row
    could cross around a probability smaller than its error.
    """
    rows = probabilities[np.ix_(order[:-1], order)]
    return rows / rows.sum(axis=1, keepdims=True)


def _thresholds(rows: np.ndarray) -> npt.NDArray[np.float64]:
    """Bounds of the credit index for each row of probabilities in the order of credit quality.

    Column j holds the threshold below which the index lands in state j or a worse one, so
    column 0 is +inf and the last column -inf, and state j lies between columns j + 1 and j.
    Each threshold is the quantile of the smaller of its two tails, summed from its own end.
    """
    worse = np.cumsum(rows[:, ::-1], axis=1)[:, ::-1][:, 1:]
    better = np.cumsum(rows, axis=1)[:, :-1]
    lower = worse <= 0.5
    tail = np.where(lower, worse, better)
    quantile = np.full(tail.shape, -np.inf)
    positive = tail > 0
    quantile[positive] = _stats.normal_quantile(tail[positive])
    inner = np.where(lower, quantile, -quantile)
    edge = np.full((len(rows), 1), np.inf)
    return np.concatenate([edge, inner, -edge], axis=1)


def _interval_probabilities(upper: np.ndarray, lower: np.ndarray) -> npt.NDArray[np.float64]:
    """Return Phi(upper) - Phi(lower) for upper >= lower, from the smaller tails.

    Equal bounds give exactly 0 and the whole line exactly 1.
    """
    above = _stats.normal_cdf(-lower) - _stats.normal_cdf(-upper)
    below = _stats.normal_cdf(upper) - _stats.normal_cdf(lower)
    across = 1 - _stats.normal_cdf(-upper) - _stats.normal_cdf(lower)
    return np.where(lower >= 0, above, np.where(upper <= 0, below, across))


def _density(x: np.ndarray) -> npt.NDArray[np.float64]:
    """Return the density of the standard normal distribution."""
    return np.exp(-0.5 * x * x) / _ROOT_TWO_PI


def _mills_ratio(t: np.ndarray) -> npt.NDArray[np.float64]:
    """Return the upper tail over the density, Q(t) / phi(t), for t >= 0; 0 at infinity."""
    result = np.zeros(t.shape)
    near = t < _MILLS_SERIES_FROM
    result[near] = _stats.normal_cdf(-t[near]) / _density(t[near])
    far = ~near & np.isfinite(t)
    # The asymptotic series 1/t (1 - 1/t^2 + 3/t^4 - 15/t^6 + 105/t^8 - 945/t^10).
    inverse = 1 / t[far] ** 2
    series = 1 - inverse * (
        1 - 3 * inverse * (1 - 5 * inverse * (1 - 7 * inverse * (1 - 9 * inverse)))
    )
    result[far] = series / t[far]
    return result


def _log_density_ratio(a: np.ndarray, b: np.ndarray) -> npt.NDArray[np.float64]:
    """Return log((phi(b) - phi(a)) / (Phi(b) - Phi(a))) for a < b and a + b <= 0.

    The ratio is at least 0. Its logarithm keeps the size of a ratio whose densities underflow,
    far out in the tails, and is -inf where the ratio is 0.
    """
    # phi(a) / phi(b) = exp(exponent) is at most 1, because |a| >= |b|.
    exponent = (b - a) * (b + a) / 2
    log_numerator = np.empty(b.shape)
    denominator = np.empty(b.shape)
    across = b > 0
    below = ~across
    with np.errstate(divide="ignore"):
        # Around 0, log(phi(b) - phi(a)) = log(phi(b)) + log(1 - phi(a) / phi(b)).
        log_numerator[across] = (
            -0.5 * b[across] ** 2 - _HALF_LOG_TWO_PI + np.log(-np.expm1(exponent[across]))
        )
        denominator[across] = 1 - _stats.normal_cdf(-b[across]) - _stats.normal_cdf(a[across])
        # Below 0, numerator and denominator are divided by phi(b).
        log_numerator[below] = np.log(-np.expm1(exponent[below]))
        denominator[below] = _mills_ratio(-b[below]) - np.exp(exponent[below]) * _mills_ratio(
            -a[below]
        )
        positive = denominator > 0
        ratio = log_numerator - np.log(np.where(positive, denominator, 1.0))
        # An interval narrower than the rounding of its ends has the slope of the log density at
        # its middle.
        narrow = np.log(np.where(positive, 1.0, -(a + b) / 2))
    return np.where(positive, ratio, narrow)


class _Cells:
    """The movements of each period that carry information on its factor."""

    def __init__(
        self,
        upper: np.ndarray,
        lower: np.ndarray,
        counts: np.ndarray,
        period: np.ndarray,
        periods: int,
    ) -> None:
        self.upper, self.lower, self.counts, self.period = upper, lower, counts, period
        self.periods = periods

    def score(self, z: np.ndarray, rho: float) -> npt.NDArray[np.float64]:
        """Return numbers with the signs of the derivatives of the periods' log-likelihoods.

        The terms of each period are scaled by the largest one, so that terms whose densities
        underflow, at correlations near 1, still decide the sign.
        """
        shift = math.sqrt(rho) * z[self.period]
        scale = math.sqrt(1 - rho)
        upper = (self.upper - shift) / scale
        lower = (self.lower - shift) / scale
        # An interval mostly above 0 is mirrored below it, where its tail cannot underflow.
        mirror = upper + lower > 0
        a = np.where(mirror, -upper, lower)
        b = np.where(mirror, -lower, upper)
        sizes = np.log(self.counts) + _log_density_ratio(a, b)
        largest = np.full(self.periods, -np.inf)
        np.maximum.at(largest, self.period, sizes)
        # A period whose terms are all 0 has a score of 0.
        offset = np.where(np.isfinite(largest), largest, 0.0)[self.period]
        terms = np.where(mirror, 1.0, -1.0) * np.exp(sizes - offset)
        return np.bincount(self.period, weights=terms, minlength=self.periods)

    def factors(self, rho: float) -> npt.NDArray[np.float64]:
        """Return the factor of each period: the root of its score, which falls with z."""
        zero = np.zeros(self.periods)
        start = self.score(zero, rho)
        low = np.where(start < 0, -1.0, 0.0)
        high = np.where(start > 0, 1.0, 0.0)
        while True:
            up = self.score(high, rho) > 0
            down = self.score(low, rho) < 0
            if not (up.any() or down.any()):
                break
            low, high = np.where(up, high, low), np.where(up, 2 * high, high)
            low, high = np.where(down, 2 * low, low), np.where(down, low, high)
        # Bisection ends when no number lies between the two ends.
        middle = (low + high) / 2
        active = (low < middle) & (middle < high)
        while active.any():
            above = self.score(middle, rho) > 0
            low = np.where(active & above, middle, low)
            high = np.where(active & ~above, middle, high)
            middle = (low + high) / 2
            active = (low < middle) & (middle < high)
        return middle


class _MatrixModel:
    """The thresholds of a long-run transition matrix in the order of credit quality."""

    def __init__(self, probabilities: np.ndarray, states: pd.Index, default: int) -> None:
        self.states = states
        self.order = _credit_order(len(states), default)
        self.rows = _credit_rows(probabilities, self.order)
        bounds = _thresholds(self.rows)
        self.upper, self.lower = bounds[:, :-1], bounds[:, 1:]
        # A state that can move to one state only carries no information on the factor.
        self.informative = (self.rows > 0).sum(axis=1) > 1

    def state(self, position: int) -> object:
        """Return the label of a state as a Python scalar, or a tuple of them."""
        return self.states[position : position + 1].tolist()[0]

    def cells(self, matrices: list[np.ndarray], wheres: list[str]) -> _Cells:
        """Check the counts of each period and collect their informative movements."""
        parts: list[tuple[np.ndarray, ...]] = []
        for position, (values, where) in enumerate(zip(matrices, wheres, strict=True)):
            counts = values[np.ix_(self.order[:-1], self.order)]
            if not counts.any():
                raise ValueError(
                    f"counts must have movements outside the row of the default state{where}"
                )
            impossible = (counts > 0) & (self.rows == 0)
            if impossible.any():
                row, column = np.argwhere(impossible)[0]
                raise ValueError(
                    f"counts has {_validation.shown(counts[row, column])} movements from state "
                    f"{_validation.shown(self.state(self.order[row]))} to state "
                    f"{_validation.shown(self.state(self.order[column]))}{where}, which matrix "
                    "gives a probability of 0"
                )
            observed = (counts > 0) & self.informative[:, None]
            if not observed.any():
                raise ValueError(
                    "counts must have movements from a state that matrix lets move to more than "
                    f"one state{where}"
                )
            upper, lower = self.upper[observed], self.lower[observed]
            for bound, end in ((upper, "best"), (lower, "worst")):
                if not np.isfinite(bound).any():
                    raise ValueError(
                        f"the factor is not finite{where}: every movement in counts is to the "
                        f"{end} state that its state can reach"
                    )
            parts.append((upper, lower, counts[observed], np.full(observed.sum(), position)))
        upper, lower, counts, period = (np.concatenate(part) for part in zip(*parts, strict=True))
        return _Cells(upper, lower, counts, period, len(matrices))


def _sample_variance(values: np.ndarray) -> float:
    """Return the sample variance (ddof=1) from correctly rounded sums, the same in any order."""
    mean = math.fsum(values) / len(values)
    return math.fsum((values - mean) ** 2) / (len(values) - 1)


def _count_matrix(
    counts: npt.ArrayLike | pd.DataFrame,
    matrix: npt.ArrayLike | pd.DataFrame,
    states: pd.Index,
    default: int,
    where: str,
) -> npt.NDArray[np.float64]:
    """Return the counts of a period as a float array matched to the states of matrix."""
    if np.ndim(counts) != 2:
        raise ValueError(f"counts must be two-dimensional, got {np.ndim(counts)} dimensions{where}")
    frame = counts if isinstance(counts, pd.DataFrame) else pd.DataFrame(counts)  # type: ignore[arg-type]  # takes any 2-D array_like
    size = len(states)
    if frame.shape != (size, size):
        raise ValueError(
            f"counts must have the shape of matrix, {(size, size)}, got {frame.shape}{where}"
        )
    if isinstance(counts, pd.DataFrame) and isinstance(matrix, pd.DataFrame):
        if not (frame.index.equals(states) and frame.columns.equals(states)):
            raise ValueError(
                "counts must have the states of matrix in its index and its columns, got "
                f"{list(frame.index)} and {list(frame.columns)}{where}"
            )
    try:
        values = _float_frame(frame, "counts")
    except ValueError as error:
        raise ValueError(f"{error}{where}") from None
    for row in np.delete(np.arange(size), default):
        state = states[row : row + 1].tolist()[0]
        if np.isnan(values[row]).any():
            raise ValueError(
                "counts must not have missing values outside the row of the default state, got "
                f"one in the row of state {_validation.shown(state)}{where}"
            )
        invalid = ~np.isfinite(values[row]) | (values[row] < 0)
        if invalid.any():
            raise ValueError(
                f"counts must be non-negative and finite, got {float(values[row][invalid][0])!r} "
                f"in the row of state {_validation.shown(state)}{where}"
            )
    return values


def _periods(
    counts: abc.Mapping[abc.Hashable, npt.ArrayLike | pd.DataFrame]
    | abc.Sequence[npt.ArrayLike | pd.DataFrame]
    | pd.Series,
) -> tuple[list[str], list[typing.Any]]:
    """Return how errors name each period, and the counts of the periods."""
    if isinstance(counts, (abc.Mapping, pd.Series)):
        items = list(counts.items())
        wheres = [f" for period {_validation.shown(label)}" for label, _ in items]
        values = [value for _, value in items]
    elif isinstance(counts, (list, tuple)):
        wheres = [f" at position {position}" for position in range(len(counts))]
        values = list(counts)
    else:
        raise ValueError(
            "counts must be a mapping, a Series, a list or a tuple of count matrices, one per "
            f"period, got {_validation.shown_argument(counts)}"
        )
    if len(values) < 2:
        raise ValueError(f"counts must have at least two periods, got {len(values)}")
    return wheres, values


def _horizons(horizons: int | npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Increasing positive whole numbers of periods."""
    message = (
        "horizons must be a positive integer or a sequence of increasing positive integers, got {}"
    )
    error = ValueError(message.format(_validation.shown_argument(horizons)))
    if np.ndim(horizons) == 0:
        # A number of periods is an integer, as the horizon of expected_credit_loss; a
        # sequence is read like data, so whole numbers such as 12.0 are accepted there.
        if (
            isinstance(horizons, bool)
            or not _validation.is_integral(horizons)
            or not 1 <= horizons < 2**63
        ):
            raise error
        return np.arange(1, int(horizons) + 1)
    if np.ndim(horizons) != 1:
        raise error
    series = horizons if isinstance(horizons, pd.Series) else pd.Series(horizons)
    if not len(series) or not _validation.is_real_dtype(series.dtype):
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
    series = _missing_as_float(_validation.as_series(values, name))
    if not len(series):
        raise ValueError(f"{name} must not be empty")
    return _validation.as_real(series, name, booleans=booleans)


def _whole_numbers(values: npt.ArrayLike, name: str, *, lowest: int) -> npt.NDArray[np.int64]:
    """Whole numbers of periods from `lowest` on as integers; booleans and missing values raise."""
    series = _missing_as_float(_validation.as_series(values, name))
    if not len(series):
        raise ValueError(f"{name} must not be empty")
    array = _validation.as_real(series, name)
    invalid = ~np.isfinite(array) | (array != np.round(array)) | (array < lowest)
    if invalid.any():
        kind = "positive" if lowest == 1 else "non-negative"
        raise ValueError(
            f"{name} must hold {kind} whole numbers of periods, got "
            f"{_validation.shown(series[invalid])}"
        )
    return array.astype(np.int64)


def _flag_vector(
    values: npt.ArrayLike, name: str, durations: npt.NDArray[np.int64]
) -> npt.NDArray[np.bool_]:
    """0/1 or boolean flags with one value per duration, as a bool array."""
    flags = _validation.as_series(values, name)
    if len(flags) != len(durations):
        raise ValueError(
            f"duration and {name} must have the same length, got {len(durations)} and {len(flags)}"
        )
    return _validation.as_flags(flags, name)


def _at_risk(
    first: npt.NDArray[np.int64],
    stop: npt.NDArray[np.int64],
    weights: npt.NDArray[np.float64] | None,
    periods: int,
) -> np.ndarray:
    """Count or weight of the accounts at risk in each period from 1 to `periods`.

    Account i is at risk in the periods ``first[i] <= t < stop[i]``. Its weight goes into the
    aligned blocks of 2**k periods that make up that range, at most two at each k, and every
    period adds up the blocks that hold it. The result is a sum of non-negative weights, so a
    large weight that has left the risk set never rounds away the small weights that remain, as
    a difference of cumulative sums could; without weights the counts are integers.
    """
    period = np.arange(1, periods + 1)
    total = np.zeros(periods, dtype=np.int64 if weights is None else np.float64)
    low, high, mass = first, stop, weights
    level = 0
    while len(low):
        blocks = []
        # A block that starts at an odd position is not part of a block twice as long: count it
        # here. Likewise for one that ends at an odd position. The rest moves up a level.
        start = (low & 1) == 1
        blocks.append((low[start], None if mass is None else mass[start]))
        low = low + start
        end = (high & 1) == 1
        high = high - end
        blocks.append((high[end], None if mass is None else mass[end]))
        size = ((periods + 1) >> level) + 1
        for position, block_weights in blocks:
            sums = np.bincount(position, weights=block_weights, minlength=size)
            total += sums[period >> level]
        low, high = low >> 1, high >> 1
        active = low < high
        low, high = low[active], high[active]
        mass = None if mass is None else mass[active]
        level += 1
    return total


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
        _validation.check_real_dtype(array.dtype, name)
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
        _validation.check_real_dtype(array.dtype, name)
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
