"""Numeric bin boundaries, category groups and their assignment.

``bin_edges`` fits edges from the values alone. ``chimerge_edges``,
``tree_edges``, ``ks_edges``, ``monotonic_edges`` and ``optimal_edges`` also
use a binary target, and ``variance_edges`` a continuous response. All of them return
increasing edges with infinite endpoints for ``pandas.cut``, or for ``cut``, which also
gives each special value its own group.

``category_groups`` pools rare categories and ``chimerge_groups`` merges
categories with similar event rates. Both return a dict from category to
group, which ``group_categories`` applies to any sample.
"""

import math
import typing
from collections import abc

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import _counts, _edges, _special, _validation

__all__ = [
    "bin_edges",
    "cut",
    "chimerge_edges",
    "tree_edges",
    "ks_edges",
    "monotonic_edges",
    "optimal_edges",
    "variance_edges",
    "category_groups",
    "chimerge_groups",
    "group_categories",
]


def bin_edges(
    values: npt.ArrayLike,
    bins: int = 5,
    *,
    method: typing.Literal["quantile", "uniform"] = "quantile",
    special: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Fit numeric bin edges on a reference sample.

    The edges are fitted once and then applied with ``pandas.cut`` to the
    reference and to every sample compared with it.

    Parameters
    ----------
    values : array_like
        One-dimensional numeric reference values, such as a list, a NumPy
        array or a pandas Series. Missing values are omitted.
    bins : int, default 5
        Requested number of bins. Tied quantiles can reduce this number.
    method : {"quantile", "uniform"}, default "quantile"
        Use sample quantiles or equally spaced edges.
    special : array_like, optional
        Special values, such as codes for "not applicable", that are omitted
        when the edges are fitted. Pass the same values to `cut` so that each
        one forms its own group.
    weights : array_like, optional
        Non-negative weight of each value, for example the inverse of its
        sampling rate. Values with zero weight are left out. With weights,
        quantile edges are weighted quantiles by the inverted CDF, so every
        interior edge is an observed value.

    Returns
    -------
    numpy.ndarray
        Increasing edges with infinite endpoints, suitable for pandas.cut.

    Raises
    ------
    ValueError
        If `bins` is not a positive integer, `method` is not recognized,
        `values` is not one-dimensional or not numeric, `special` is not
        one-dimensional or contains a missing value or a boolean, or the
        reference has no observations or contains an infinite value apart from
        special values.
        Also if `weights` is not numeric, does not give each value one finite,
        non-negative weight with a positive total, or has an index other than
        that of `values`.

    Warns
    -----
    UserWarning
        If tied edges leave fewer bins than requested, for example when a
        single value accounts for a large share of the reference.

    See Also
    --------
    cut : Assign values to bins, with a separate group for each special value.
    pandas.cut : Assign values to the bins defined by the edges.
    numpy.quantile : Compute the quantile edges.
    kivi.woe.woe_table : Weight of evidence of the resulting groups.
    kivi.stability.psi_table : Stability of the resulting groups.

    Notes
    -----
    Quantile edges are the sample quantiles at equally spaced probabilities,
    computed with the default linear interpolation of ``numpy.quantile``.
    Uniform edges divide the range from the smallest to the largest value
    into equal widths.

    Repeated edges are merged, and the outer edges are replaced with
    infinities, so every finite value falls into exactly one interval. The
    intervals of ``pandas.cut`` are closed on the right.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.binning import bin_edges
    >>> values = pd.Series([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    >>> edges = bin_edges(values, bins=3)
    >>> edges
    array([-inf,  30.,  60.,  inf])
    >>> pd.cut(pd.Series([-5, 45, 200]), edges, labels=False).tolist()
    [0, 1, 2]

    A special value is left out of the fit:

    >>> bin_edges([-1, -1, -1, 10, 20, 30, 40], bins=2, special=[-1])
    array([-inf,  25.,  inf])
    """
    return _edges.fit_edges(values, bins, method=method, special=special, weights=weights)


@typing.overload
def cut(
    values: pd.Series, edges: npt.ArrayLike, *, special: npt.ArrayLike | None = ...
) -> pd.Series: ...
@typing.overload
def cut(
    values: _validation.ListLike, edges: npt.ArrayLike, *, special: npt.ArrayLike | None = ...
) -> pd.Categorical: ...
@typing.overload
def cut(
    values: npt.ArrayLike, edges: npt.ArrayLike, *, special: npt.ArrayLike | None = ...
) -> pd.Series | pd.Categorical: ...
def cut(
    values: npt.ArrayLike, edges: npt.ArrayLike, *, special: npt.ArrayLike | None = None
) -> pd.Series | pd.Categorical:
    """Assign values to bins, with a separate group for each special value.

    Without `special` the result is the same as ``pandas.cut(values, edges)``
    for every finite value.

    Parameters
    ----------
    values : array_like
        One-dimensional numeric values to assign. Missing values stay missing.
        Infinite values must be special codes.
    edges : array_like
        Increasing bin edges, usually from `bin_edges`.
    special : array_like, optional
        Special values that each form their own group instead of falling into
        an interval. Use the values passed to `bin_edges`.

    Returns
    -------
    pandas.Series or pandas.Categorical
        Categorical groups whose categories are the special values, in the
        order given, followed by the intervals. A Series input gives a Series
        with the same index and name.

    Raises
    ------
    ValueError
        If `values` is not one-dimensional, is not numeric apart from special
        values or has an infinite value that is not a special value, `edges`
        does not increase, or `special` is not one-dimensional or contains a
        missing value or a boolean.

    See Also
    --------
    bin_edges : Fit the edges on a reference sample.
    pandas.cut : Assign values to intervals only.
    kivi.woe.woe_table : Weight of evidence of the resulting groups.
    kivi.stability.psi_table : Stability of the resulting groups.

    Notes
    -----
    A special value is matched by equality, so ``-1`` and ``-1.0`` are the
    same code. Values that are not numbers, such as strings read from a
    file, are also matched after conversion with ``pandas.to_numeric``, so
    the string ``"-1"`` falls into the group of the code ``-1``. Finite values
    outside the edges become missing, as with ``pandas.cut``; the infinite
    outer edges of `bin_edges` avoid this. An infinite value raises an error
    unless it is a special value, because ``pandas.cut`` would leave ``-inf``
    in no interval, so that it would count as a missing value, and put
    ``inf`` in the last interval.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.binning import cut
    >>> values = pd.Series([-1, 15, 35, None, -1])
    >>> cut(values, [-float("inf"), 25, float("inf")], special=[-1])
    0          -1
    1    (-inf, 25.0]
    2     (25.0, inf]
    3             NaN
    4          -1
    dtype: category
    Categories (3, object): [-1, (-inf, 25.0], (25.0, inf]]
    """
    series = _validation.as_series(values, "values")
    special = _special.special_values(special)
    position = _special.special_positions(series, special)
    is_special = position >= 0
    numbers = _validation.as_numbers(series.mask(is_special), "values")
    infinite = np.isinf(numbers.to_numpy(dtype=float, na_value=np.nan))
    if infinite.any():
        raise ValueError(
            "values must be finite or special values, got "
            f"{_validation.shown(numbers[infinite].iloc[0])}; pass infinite codes in special"
        )
    try:
        intervals = pd.cut(
            numbers,
            edges,  # type: ignore[arg-type]  # pandas.cut takes any array_like of edges
        )
    except (TypeError, ValueError) as error:
        if _validation.is_integral(edges) and not isinstance(edges, bool):
            # pandas.cut reads an integer as a number of equal-width bins, which cut does not
            # document; its own message explains why that number could not be used.
            raise
        # pandas names the argument bins and suggests a duplicates keyword that cut lacks.
        raise ValueError(
            f"edges must be increasing numbers, got {_validation.shown_argument(edges)}"
        ) from error
    if special.empty:
        return intervals if isinstance(values, pd.Series) else intervals.array
    codes = intervals.cat.codes.to_numpy()
    codes = np.where(codes >= 0, codes + len(special), -1)
    codes[is_special] = position[is_special]
    groups = pd.Categorical.from_codes(codes, categories=special.append(intervals.cat.categories))
    if isinstance(values, pd.Series):
        return pd.Series(groups, index=series.index, name=series.name)
    return groups


def chimerge_edges(
    values: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    max_bins: int = 5,
    alpha: float | None = 0.05,
    initial_bins: int = 20,
    min_bin_share: float = 0.0,
    special: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    r"""Fit edges by merging adjacent quantile bins with ChiMerge.

    The values are first cut into ``initial_bins`` quantile bins, or into one
    bin per distinct value if there are no more distinct values than that.
    While some bin holds less than ``min_bin_share`` of the fitted values,
    the smallest one is merged with the neighbour whose chi-square statistic
    with it is smaller. The adjacent pair with the smallest chi-square
    statistic, the pair whose event rates are most alike, is then merged, and
    merging is repeated while more than ``max_bins`` bins remain or while
    some adjacent pair is not significantly different at level ``alpha``.

    Parameters
    ----------
    values : array_like
        One-dimensional numeric values, such as a list, a NumPy array or a
        pandas Series. Missing values and their labels are omitted.
    target : array_like
        Binary labels, with 1 indicating the event. If `values` and `target`
        are both pandas Series, they must have identical indices; otherwise
        they are matched by position.
    max_bins : int, default 5
        Maximum number of bins.
    alpha : float or None, default 0.05
        Significance level of the chi-square test with one degree of freedom.
        Adjacent bins are merged until every pair differs at this level. Pass
        None to merge down to ``max_bins`` only.
    initial_bins : int, default 20
        Number of quantile bins to start from. Tied quantiles can reduce it.
    min_bin_share : float, default 0.0
        Minimum share of the fitted values in every bin, from 0 to 0.5, or of
        their total weight when `weights` is given. Missing and special values
        do not count towards it. The default merges no bin for its size.
    special : array_like, optional
        Special values, such as codes for "not applicable", that are omitted
        with their labels when the edges are fitted. Pass the same values to
        `cut` so that each one forms its own group.
    weights : array_like, optional
        Non-negative weight of each observation, for example the inverse of
        its sampling rate, matched like `target`. Observations with zero
        weight are left out, and the counts in each bin are sums of weights.
        The initial quantile bins then use weighted quantiles by the inverted
        CDF, so unit weights can give other edges than no weights when there
        are more distinct values than `initial_bins`.
        The chi-square test treats the weighted counts as counts, so scale
        the weights to sum to the sample size when the p-values matter.

    Returns
    -------
    numpy.ndarray
        Increasing edges with infinite endpoints, suitable for pandas.cut.

    Raises
    ------
    ValueError
        If `max_bins` or `initial_bins` is not a positive integer, `alpha` is
        neither None nor between 0 and 1, `min_bin_share` is not between 0 and
        0.5, `values` and `target` are not one-dimensional or their indices or
        lengths do not match, `special` is not one-dimensional or contains a
        missing value or a boolean, `values` is not numeric or has no finite
        observations apart from missing and special values, or `target` has
        labels other than 0 and 1 or lacks one of them in the rows that are
        fitted.
        Also if `weights` is not numeric, does not give each value one finite,
        non-negative weight with a positive total, or has an index other than
        that of `values`.

    See Also
    --------
    tree_edges : Split the values recursively instead of merging.
    monotonic_edges : Merge until the WOE is monotonic.
    optimal_edges : The binning with the largest IV under constraints.
    cut : Assign values to bins, with a separate group for each special value.

    Notes
    -----
    For an adjacent pair of bins with observed counts :math:`O_{ij}` (bin
    :math:`i`, class :math:`j`) and expected counts
    :math:`E_{ij} = R_i C_j / N` under equal event rates,

    .. math::

       \chi^2 = \sum_{i=1}^{2} \sum_{j=0}^{1} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}

    where cells with :math:`E_{ij} = 0` contribute nothing. A pair stays
    separate when the p-value of :math:`\chi^2` is below ``alpha``.

    Bins smaller than ``min_bin_share`` are merged before any chi-square
    merge, whatever their statistics, so every final bin holds at least that
    share unless a single bin is left. Of several small bins, the smallest is
    merged first, and of bins whose totals agree to a relative 1e-10, the
    first; a small bin merges into its left neighbour when the two
    statistics agree to a relative 1e-10, as in `chimerge_groups`.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.binning import chimerge_edges
    >>> values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    >>> target = pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1])
    >>> chimerge_edges(values, target)
    array([-inf,   4.,  inf])
    >>> chimerge_edges(values, target, max_bins=3, alpha=None)
    array([-inf,   4.,   9.,  inf])

    With four bins, the value 9 forms a bin of its own, 1/12 of the values.
    ``min_bin_share`` applies to the initial bins, before any chi-square
    merge. Here each value is an initial bin of 1/12 of the values, so with
    ``min_bin_share=0.1`` they are first joined until each bin holds at
    least two values, and ChiMerge then merges those bins:

    >>> chimerge_edges(values, target, max_bins=4, alpha=None)
    array([-inf,   4.,   8.,   9.,  inf])
    >>> chimerge_edges(values, target, max_bins=4, alpha=None, min_bin_share=0.1)
    array([-inf,   4.,   8.,  10.,  inf])
    """
    _validation.check_count(max_bins, "max_bins")
    _validation.check_count(initial_bins, "initial_bins")
    if alpha is not None and not (_validation.is_real(alpha) and 0 < alpha < 1):
        raise ValueError(
            f"alpha must be None or between 0 and 1, got {_validation.shown_argument(alpha)}"
        )
    _check_share(min_bin_share, "min_bin_share", 0.5)
    observed, labels, mass = _labeled(values, target, special, weights)
    interior, counts = _merge_small(
        *_fine_bins(observed, labels, initial_bins, mass), min_bin_share
    )
    while len(counts) > 1:
        statistic = _pair_chi2(counts)
        pair = int(np.argmin(statistic))
        significant = alpha is None or math.erfc(math.sqrt(statistic[pair] / 2)) < alpha
        if len(counts) <= max_bins and significant:
            break
        interior, counts = _merge(interior, counts, pair)
    return np.r_[-np.inf, interior, np.inf]


def tree_edges(
    values: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    max_bins: int = 5,
    min_bin_share: float = 0.05,
    criterion: typing.Literal["entropy", "gini"] = "entropy",
    special: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    r"""Fit edges with the splits of a decision tree on one variable.

    The tree grows best first: of all current bins, the one whose best split
    most reduces the total impurity is split next, until there are
    ``max_bins`` bins or no split reduces the impurity while leaving at
    least ``min_bin_share`` of the observations on each side.

    Parameters
    ----------
    values : array_like
        One-dimensional numeric values, such as a list, a NumPy array or a
        pandas Series. Missing values and their labels are omitted.
    target : array_like
        Binary labels, with 1 indicating the event. If `values` and `target`
        are both pandas Series, they must have identical indices; otherwise
        they are matched by position.
    max_bins : int, default 5
        Maximum number of bins, the leaves of the tree.
    min_bin_share : float, default 0.05
        Minimum share of the fitted values in every bin, from 0 to 0.5, or of
        their total weight when `weights` is given. Missing and special values
        do not count towards it.
    criterion : {"entropy", "gini"}, default "entropy"
        Impurity measure: entropy for information gain, or the Gini index.
    special : array_like, optional
        Special values, such as codes for "not applicable", that are omitted
        with their labels when the edges are fitted. Pass the same values to
        `cut` so that each one forms its own group.
    weights : array_like, optional
        Non-negative weight of each observation, for example the inverse of
        its sampling rate, matched like `target`. Observations with zero
        weight are left out, and the counts in each bin are sums of weights.

    Returns
    -------
    numpy.ndarray
        Increasing edges with infinite endpoints, suitable for pandas.cut.
        Every interior edge is an observed value, the largest one in the bin
        to its left.

    Raises
    ------
    ValueError
        If `max_bins` is not a positive integer, `min_bin_share` is not between
        0 and 0.5, `criterion` is not recognized, `values` and `target` are not
        one-dimensional or their indices or lengths do not match, `special` is
        not one-dimensional or contains a missing value or a boolean, `values`
        is not numeric or has no finite observations apart from missing and
        special values, or `target` has labels other than 0 and 1 or lacks one
        of them in the rows that are fitted.
        Also if `weights` is not numeric, does not give each value one finite,
        non-negative weight with a positive total, or has an index other than
        that of `values`.

    See Also
    --------
    chimerge_edges : Merge fine bins instead of splitting.
    monotonic_edges : Merge until the WOE is monotonic.
    cut : Assign values to bins, with a separate group for each special value.

    Notes
    -----
    For a bin with :math:`n` observations (their total weight, with
    `weights`) and event rate :math:`p`, the
    impurity is :math:`H(p) = -p \ln p - (1 - p) \ln(1 - p)` (entropy) or
    :math:`G(p) = 2p(1 - p)` (Gini). A split into :math:`n_L` and
    :math:`n_R` observations reduces the total impurity by
    :math:`n H(p) - n_L H(p_L) - n_R H(p_R)`, and the largest reduction over
    all bins and cut points is chosen.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.binning import tree_edges
    >>> values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    >>> target = pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1])
    >>> tree_edges(values, target, max_bins=2)
    array([-inf,   4.,  inf])
    >>> tree_edges(values, target, max_bins=4)
    array([-inf,   4.,   8.,   9.,  inf])
    >>> tree_edges(values, target, max_bins=4, min_bin_share=0.1)
    array([-inf,   4.,   6.,   9.,  inf])
    """
    _validation.check_count(max_bins, "max_bins")
    if (
        isinstance(min_bin_share, bool)
        or not _validation.is_real(min_bin_share)
        or not 0 <= min_bin_share <= 0.5
    ):
        raise ValueError(
            "min_bin_share must be between 0 and 0.5, got "
            f"{_validation.shown_argument(min_bin_share)}"
        )
    if not isinstance(criterion, str) or criterion not in ("entropy", "gini"):  # type: ignore[redundant-expr]  # rejects non-str values
        raise ValueError(
            f"criterion must be 'entropy' or 'gini', got {_validation.shown_argument(criterion)}"
        )
    observed, labels, mass = _labeled(values, target, special, weights)
    unique, totals, events, _ = _cumulative(observed, labels, mass)
    min_count = min_bin_share * totals[-1]
    impurity = _entropy if criterion == "entropy" else _gini

    def best_split(start: int, stop: int) -> tuple[float, int | None]:
        # Cut points between unique values start..stop-1; left part is start..cut-1.
        cuts = np.arange(start + 1, stop)
        n_left = totals[cuts] - totals[start]
        n_right = totals[stop] - totals[cuts]
        allowed = (n_left >= min_count) & (n_right >= min_count)
        if not allowed.any():
            return 0.0, None
        cuts, n_left, n_right = cuts[allowed], n_left[allowed], n_right[allowed]
        e_left = events[cuts] - events[start]
        e_right = events[stop] - events[cuts]
        n, e = totals[stop] - totals[start], events[stop] - events[start]
        gain = (
            n * impurity(e / n)
            - n_left * impurity(e_left / n_left)
            - n_right * impurity(e_right / n_right)
        )
        best = int(np.argmax(gain))
        if gain[best] <= 1e-10 * n:
            return 0.0, None
        return float(gain[best]), int(cuts[best])

    leaves = [(0, len(unique), *best_split(0, len(unique)))]
    while len(leaves) < max_bins:
        leaf = max(range(len(leaves)), key=lambda i: leaves[i][2])
        start, stop, _, cut = leaves[leaf]
        if cut is None:
            break
        leaves[leaf : leaf + 1] = [
            (start, cut, *best_split(start, cut)),
            (cut, stop, *best_split(cut, stop)),
        ]
    interior = [unique[stop - 1] for _, stop, _, _ in leaves[:-1]]
    return np.r_[-np.inf, interior, np.inf]


def ks_edges(
    values: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    max_bins: int = 5,
    min_bin_share: float = 0.05,
    special: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    r"""Fit edges by splitting the values where the classes separate most (Best-KS).

    Each split cuts a bin at the value where the cumulative shares of its
    events and of its non-events differ most, the KS of the bin, among the
    cuts that leave at least ``min_bin_share`` of the observations on each
    side. The bins are split best first, by the KS of their cut weighted by
    the shares of the events and the non-events that they hold, until there
    are ``max_bins`` bins or no bin has an allowed cut with a positive KS.

    Parameters
    ----------
    values : array_like
        One-dimensional numeric values, such as a list, a NumPy array or a
        pandas Series. Missing values and their labels are omitted.
    target : array_like
        Binary labels, with 1 indicating the event. If `values` and `target`
        are both pandas Series, they must have identical indices; otherwise
        they are matched by position.
    max_bins : int, default 5
        Maximum number of bins.
    min_bin_share : float, default 0.05
        Minimum share of the fitted values in every bin, from 0 to 0.5, or of
        their total weight when `weights` is given. Missing and special values
        do not count towards it.
    special : array_like, optional
        Special values, such as codes for "not applicable", that are omitted
        with their labels when the edges are fitted. Pass the same values to
        `cut` so that each one forms its own group.
    weights : array_like, optional
        Non-negative weight of each observation, for example the inverse of
        its sampling rate, matched like `target`. Observations with zero
        weight are left out, and the counts in each bin are sums of weights.

    Returns
    -------
    numpy.ndarray
        Increasing edges with infinite endpoints, suitable for pandas.cut.
        Every interior edge is an observed value, the largest one in the bin
        to its left.

    Raises
    ------
    ValueError
        If `max_bins` is not a positive integer, `min_bin_share` is not between
        0 and 0.5, `values` and `target` are not one-dimensional or their
        indices or lengths do not match, `special` is not one-dimensional or
        contains a missing value or a boolean, `values` is not numeric or has
        no finite observations apart from missing and special values, or
        `target` has labels other than 0 and 1 or lacks one of them in the rows
        that are fitted.
        Also if `weights` is not numeric, does not give each value one finite,
        non-negative weight with a positive total, or has an index other than
        that of `values`.

    See Also
    --------
    tree_edges : Split where the impurity falls most instead.
    chimerge_edges : Merge fine bins instead of splitting.
    kivi.metrics.ks : The KS statistic of a score.
    cut : Assign values to bins, with a separate group for each special value.

    Notes
    -----
    Let a bin hold :math:`E_b` events and :math:`N_b` non-events (their total
    weights, with `weights`), and let a cut leave :math:`e` events and
    :math:`n` non-events on its left. The bin is cut where

    .. math::

       \mathrm{KS}_b = \left| \frac{e}{E_b} - \frac{n}{N_b} \right|

    is largest; ties go to the smallest value. With :math:`E` events and
    :math:`N` non-events in all the fitted values, the bin split next is the
    one whose cut has the largest

    .. math::

       \frac{E_b}{E} \frac{N_b}{N} \mathrm{KS}_b
       = \frac{\left| e (N_b - n) - (E_b - e) n \right|}{E N}

    The numerator counts the pairs of an event and a non-event that the cut
    puts into different bins, the event in the bin with the higher event
    rate, less the pairs it puts the other way round. The split therefore
    raises the Gini coefficient of the bins, ranked by event rate, by at
    least this amount, and by exactly this amount when no other bin has an
    event rate between those of the two new bins. Ranking the bins by their
    KS alone would split a bin with a handful of customers as readily as
    one with thousands, and ranking them by size would spend splits on large
    bins whose classes barely separate. The first cut is at the KS of the
    whole sample, among the allowed cuts, and two bins have a Gini
    coefficient equal to their KS.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.binning import ks_edges
    >>> values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    >>> target = pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1])
    >>> ks_edges(values, target, max_bins=2)
    array([-inf,   4.,  inf])
    >>> ks_edges(values, target, max_bins=4)
    array([-inf,   4.,   8.,   9.,  inf])
    >>> ks_edges(values, target, min_bin_share=0.5)
    array([-inf,   6.,  inf])
    """
    _validation.check_count(max_bins, "max_bins")
    _check_share(min_bin_share, "min_bin_share", 0.5)
    observed, labels, mass = _labeled(values, target, special, weights)
    unique, totals, events, non_events = _cumulative(observed, labels, mass)
    # A side within rounding of the minimum share reaches it, so that 7 of 100 rows are 7% and
    # scaling the weights leaves the edges unchanged.
    min_count = min_bin_share * totals[-1] * (1 - _TIE)

    def best_cut(start: int, stop: int) -> tuple[float, int | None]:
        # Cut points between unique values start..stop-1; left part is start..cut-1.
        cuts = np.arange(start + 1, stop)
        n_left = totals[cuts] - totals[start]
        n_right = totals[stop] - totals[cuts]
        cuts = cuts[(n_left >= min_count) & (n_right >= min_count)]
        if not len(cuts):
            return 0.0, None
        e, n = events[stop] - events[start], non_events[stop] - non_events[start]
        # |e_left n - e n_left| = e n KS: the pairs that the cut ranks apart, net of the others.
        pairs = np.abs(
            (events[cuts] - events[start]) * n - e * (non_events[cuts] - non_events[start])
        )
        best = _first_max(pairs)
        # A KS within rounding of zero, as in a bin of one class, gives no cut.
        if pairs[best] <= 1e-10 * e * n:
            return 0.0, None
        return float(pairs[best]), int(cuts[best])

    bins = [(0, len(unique), *best_cut(0, len(unique)))]
    while len(bins) < max_bins:
        split = _first_max(np.array([pairs for _, _, pairs, _ in bins]))
        start, stop, _, cut = bins[split]
        if cut is None:
            break
        bins[split : split + 1] = [
            (start, cut, *best_cut(start, cut)),
            (cut, stop, *best_cut(cut, stop)),
        ]
    interior = [unique[stop - 1] for _, stop, _, _ in bins[:-1]]
    return np.r_[-np.inf, interior, np.inf]


def monotonic_edges(
    values: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    max_bins: int | None = 5,
    direction: typing.Literal["auto", "increasing", "decreasing"] = "auto",
    initial_bins: int = 20,
    min_bin_share: float = 0.0,
    smoothing: float = 0.5,
    special: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Fit edges by merging adjacent quantile bins until the WOE is monotonic.

    The values are first cut into ``initial_bins`` quantile bins, or into one
    bin per distinct value if there are no more distinct values than that.
    While some bin holds less than ``min_bin_share`` of the fitted values,
    the smallest one is merged with the neighbour whose chi-square statistic
    with it is smaller, as in :func:`chimerge_edges`. Then, while two
    adjacent bins break the trend, the first such pair is merged and
    the WOE is recomputed (pool adjacent violators), so the WOE of the final
    bins is strictly increasing or strictly decreasing. While more than ``max_bins``
    bins remain, the adjacent pair with the smallest chi-square statistic is
    merged as in :func:`chimerge_edges`, and the trend is checked again.

    Parameters
    ----------
    values : array_like
        One-dimensional numeric values, such as a list, a NumPy array or a
        pandas Series. Missing values and their labels are omitted.
    target : array_like
        Binary labels, with 1 indicating the event. If `values` and `target`
        are both pandas Series, they must have identical indices; otherwise
        they are matched by position.
    max_bins : int or None, default 5
        Maximum number of bins. Pass None to stop as soon as the WOE is
        monotonic.
    direction : {"auto", "increasing", "decreasing"}, default "auto"
        Trend of the WOE as the values grow. ``"auto"`` fits both directions
        and keeps the one with the larger information value, preferring
        ``"increasing"`` on a tie.
    initial_bins : int, default 20
        Number of quantile bins to start from. Tied quantiles can reduce it.
    min_bin_share : float, default 0.0
        Minimum share of the fitted values in every bin, from 0 to 0.5, or of
        their total weight when `weights` is given. Missing and special values
        do not count towards it. The default merges no bin for its size.
    smoothing : float, default 0.5
        Positive replacement for a zero count when calculating the WOE, as in
        :func:`kivi.woe.woe_table`. Use the same value there to see the same
        trend.
    special : array_like, optional
        Special values, such as codes for "not applicable", that are omitted
        with their labels when the edges are fitted. Pass the same values to
        `cut` so that each one forms its own group. The trend applies to the
        intervals only; a special group can have any WOE.
    weights : array_like, optional
        Non-negative weight of each observation, for example the inverse of
        its sampling rate, matched like `target`. Observations with zero
        weight are left out, and the counts in each bin are sums of weights.
        The initial quantile bins then use weighted quantiles by the inverted
        CDF, so unit weights can give other edges than no weights when there
        are more distinct values than `initial_bins`. `smoothing` is then in
        units of weight.

    Returns
    -------
    numpy.ndarray
        Increasing edges with infinite endpoints, suitable for pandas.cut.

    Raises
    ------
    ValueError
        If `max_bins` is neither None nor a positive integer, `initial_bins` is
        not a positive integer, `direction` is not recognized, `min_bin_share`
        is not between 0 and 0.5, `smoothing` is not finite and positive,
        `values` and `target` are not one-dimensional or their indices or
        lengths do not match, `special` is not one-dimensional or contains a
        missing value or a boolean, `values` is not numeric or has no finite
        observations apart from missing and special values, or `target` has
        labels other than 0 and 1 or lacks one of them in the rows that are
        fitted.
        Also if `weights` is not numeric, does not give each value one finite,
        non-negative weight with a positive total, or has an index other than
        that of `values`.

    See Also
    --------
    chimerge_edges : Merge fine bins by a chi-square test.
    tree_edges : Split the values recursively.
    optimal_edges : The monotonic binning with the largest IV.
    cut : Assign values to bins, with a separate group for each special value.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.binning import monotonic_edges
    >>> from kivi.woe import woe_table
    >>> values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    >>> target = pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1])
    >>> edges = monotonic_edges(values, target)
    >>> edges
    array([-inf,   4.,   6.,  inf])
    >>> woe_table(pd.cut(values, edges), target)["woe"].round(3).tolist()
    [-2.159, -0.08, 1.529]
    >>> monotonic_edges(values, target, direction="decreasing")
    array([-inf,  inf])
    """
    if max_bins is not None:
        _validation.check_count(max_bins, "max_bins")
    _validation.check_count(initial_bins, "initial_bins")
    if not isinstance(direction, str) or direction not in ("auto", "increasing", "decreasing"):  # type: ignore[redundant-expr]  # rejects non-str values
        raise ValueError(
            "direction must be 'auto', 'increasing' or 'decreasing', got "
            f"{_validation.shown_argument(direction)}"
        )
    _check_share(min_bin_share, "min_bin_share", 0.5)
    observed, labels, mass = _labeled(values, target, special, weights)
    fine = _merge_small(*_fine_bins(observed, labels, initial_bins, mass), min_bin_share)
    smoothing = _validation.check_smoothing(smoothing)
    limit = len(fine[1]) if max_bins is None else max_bins
    if direction == "auto":
        rising = _monotonic(*fine, 1, limit, smoothing)
        falling = _monotonic(*fine, -1, limit, smoothing)
        interior = (
            rising[0] if _iv(rising[1], smoothing) >= _iv(falling[1], smoothing) else falling[0]
        )
    else:
        interior = _monotonic(*fine, 1 if direction == "increasing" else -1, limit, smoothing)[0]
    return np.r_[-np.inf, interior, np.inf]


def optimal_edges(
    values: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    max_bins: int = 5,
    min_bin_share: float = 0.05,
    min_bin_events: float = 1,
    trend: typing.Literal[
        "auto", "increasing", "decreasing", "u_shaped", "inverted_u", "any"
    ] = "auto",
    initial_bins: int = 20,
    special: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    r"""Fit the edges with the largest information value under constraints.

    The values are first cut into ``initial_bins`` quantile bins, or into one
    bin per distinct value if there are no more distinct values than that,
    as in `chimerge_edges`. Of all the ways to join adjacent ones into at
    most ``max_bins`` bins that hold at least ``min_bin_share`` of the fitted
    values and at least ``min_bin_events`` events and non-events each, and
    whose WOE follows ``trend``, the one with the largest information value
    is returned. Dynamic programming finds it exactly, without a solver.

    Parameters
    ----------
    values : array_like
        One-dimensional numeric values, such as a list, a NumPy array or a
        pandas Series. Missing values and their labels are omitted.
    target : array_like
        Binary labels, with 1 indicating the event. If `values` and `target`
        are both pandas Series, they must have identical indices; otherwise
        they are matched by position.
    max_bins : int, default 5
        Maximum number of bins.
    min_bin_share : float, default 0.05
        Minimum share of the fitted values in every bin, from 0 to 0.5, or of
        their total weight when `weights` is given. Missing and special values
        do not count towards it.
    min_bin_events : float, default 1
        Minimum number of events, and of non-events, in every bin, or their
        total weight when `weights` is given. It must be positive, so that no
        bin has an empty class and its WOE needs no smoothing.
    trend : {"auto", "increasing", "decreasing", "u_shaped", "inverted_u", "any"}, default "auto"
        Shape of the WOE as the values grow, named as by
        :func:`kivi.woe.woe_trend`. ``"increasing"`` and ``"decreasing"``
        require every step from one bin to the next to rise, or to fall.
        ``"u_shaped"`` requires the WOE to fall and then rise, and
        ``"inverted_u"`` to rise and then fall; either part may be empty, so
        a monotonic binning qualifies when its information value is larger.
        ``"any"`` imposes no shape. ``"auto"`` fits both monotonic trends and
        keeps the one with the larger information value, preferring
        ``"increasing"`` on a tie, as :func:`monotonic_edges` does.
    initial_bins : int, default 20
        Number of quantile bins to start from. Tied quantiles can reduce it.
        The fit takes time in proportion to the cube of this number and
        memory in proportion to its square.
    special : array_like, optional
        Special values, such as codes for "not applicable", that are omitted
        with their labels when the edges are fitted. Pass the same values to
        `cut` so that each one forms its own group. The constraints apply to
        the intervals only; a special group can have any size and WOE.
    weights : array_like, optional
        Non-negative weight of each observation, for example the inverse of
        its sampling rate, matched like `target`. Observations with zero
        weight are left out, and the counts in each bin are sums of weights.
        The initial quantile bins then use weighted quantiles by the inverted
        CDF, so unit weights can give other edges than no weights when there
        are more distinct values than `initial_bins`.

    Returns
    -------
    numpy.ndarray
        Increasing edges with infinite endpoints, suitable for pandas.cut.

    Raises
    ------
    ValueError
        If `max_bins` or `initial_bins` is not a positive integer,
        `min_bin_share` is not between 0 and 0.5, `min_bin_events` is not a
        finite, positive number or is larger than the number of events or of
        non-events in the fitted values, `trend` is not recognized, `values`
        and `target` are not one-dimensional or their indices or lengths do
        not match, `special` is not one-dimensional or contains a missing
        value or a boolean, `values` is not numeric or has no finite
        observations apart from missing and special values, or `target` has
        labels other than 0 and 1 or lacks one of them in the rows that are
        fitted.
        Also if `weights` is not numeric, does not give each value one finite,
        non-negative weight with a positive total, or has an index other than
        that of `values`.

    See Also
    --------
    chimerge_edges : Merge fine bins by a chi-square test.
    monotonic_edges : Merge fine bins until the WOE is monotonic.
    kivi.woe.woe_trend : The shape of the WOE of a binning.
    cut : Assign values to bins, with a separate group for each special value.

    Notes
    -----
    With :math:`E_b` events and :math:`N_b` non-events in bin :math:`b`, out
    of :math:`E` and :math:`N` in all the fitted values, the information
    value

    .. math::

       \mathrm{IV} = \sum_b \left( \frac{E_b}{E} - \frac{N_b}{N} \right)
       \ln \frac{E_b / E}{N_b / N}

    is a sum over the bins, and every bin has both classes, so it equals the
    IV of :func:`kivi.woe.woe_table` for the fitted values and no zero count
    is smoothed. A sum over the bins can be maximized one bin at a time:
    for each initial bin that can start a bin, each initial bin that can
    end it and each number of bins, the best binning of the values above it
    is kept, and the shape of the WOE is checked between neighbouring bins.
    This takes time in proportion to :math:`n^3 k` and memory in proportion
    to :math:`n^2 k`, twice that for a U shape, for :math:`n` initial bins
    and :math:`k` bins, the most that the constraints allow up to
    ``max_bins``: a fraction of a second and a few megabytes for 200 initial
    bins, and a few seconds and about 50 megabytes for 1000.

    A step from one bin to the next rises or falls when the two WOE values
    differ by more than a relative 1e-10, as in `kivi.woe.woe_trend`, and
    by more than 1e-10, so that WOE values that rounding leaves near 0 are
    level. The trend of the fitted table is therefore the one requested, or
    a simpler one: a ``"u_shaped"`` fit can be increasing, decreasing or a
    single bin. Two binnings whose information values agree to a relative
    1e-10, or differ by less than 1e-20, count as equal. Of equal binnings,
    the one with fewer bins is returned, and of those with the same number
    of bins, the one whose lowest edge is smallest, then the one whose next
    edge is smallest, and so on. A single bin, with an information value of
    0, meets every constraint, so a result always exists.

    The information value of the fitted sample is the largest that the
    initial bins allow, so it overstates the information value that the
    edges will show on other samples; measure that on a validation sample.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.binning import optimal_edges
    >>> from kivi.woe import woe_table
    >>> values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    >>> target = pd.Series([1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1])
    >>> optimal_edges(values, target)
    array([-inf,   8.,  inf])
    >>> edges = optimal_edges(values, target, trend="u_shaped")
    >>> edges
    array([-inf,   3.,   8.,  inf])
    >>> woe_table(pd.cut(values, edges), target)["woe"].round(3).tolist()
    [0.693, -1.386, 1.099]

    With at least two events and two non-events in every bin, the low
    values cannot keep a bin of their own:

    >>> optimal_edges(values, target, trend="u_shaped", min_bin_events=2)
    array([-inf,   5.,  inf])
    """
    _validation.check_count(max_bins, "max_bins")
    _check_share(min_bin_share, "min_bin_share", 0.5)
    if (
        isinstance(min_bin_events, bool)
        or not _validation.is_real(min_bin_events)
        or not (math.isfinite(min_bin_events) and min_bin_events > 0)
    ):
        raise ValueError(
            "min_bin_events must be a finite, positive number, got "
            f"{_validation.shown_argument(min_bin_events)}"
        )
    _check_trend(trend)
    _validation.check_count(initial_bins, "initial_bins")
    observed, labels, mass = _labeled(values, target, special, weights)
    interior, counts = _fine_bins(observed, labels, initial_bins, mass)
    for side, name in ((1, "events"), (0, "non-events")):
        available = counts[:, side].sum()
        if available < min_bin_events * (1 - _TIE):
            raise ValueError(
                f"min_bin_events must not exceed the {name} of the fitted values, got "
                f"{_validation.shown_argument(min_bin_events)} and "
                f"{_validation.shown(available)} {name}"
            )
    shapes = ["increasing", "decreasing"] if trend == "auto" else [trend]
    fits = [_optimal(counts, shape, max_bins, min_bin_share, min_bin_events) for shape in shapes]
    cuts = _chosen(fits, _GAIN_TIE)
    return np.r_[-np.inf, interior[np.asarray(cuts, dtype=np.intp) - 1], np.inf]


def variance_edges(
    values: npt.ArrayLike,
    response: npt.ArrayLike,
    *,
    max_bins: int = 5,
    min_bin_share: float = 0.05,
    trend: typing.Literal[
        "auto", "increasing", "decreasing", "u_shaped", "inverted_u", "any"
    ] = "auto",
    initial_bins: int = 20,
    special: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    r"""Fit the edges that best separate the means of a continuous response.

    The values are first cut into ``initial_bins`` quantile bins, or into one
    bin per distinct value if there are no more distinct values than that,
    as in `optimal_edges`. Of all the ways to join adjacent ones into at most
    ``max_bins`` bins that hold at least ``min_bin_share`` of the fitted
    values, and whose mean response follows ``trend``, the one with the
    smallest sum of squares of the response within the bins is returned.
    Dynamic programming finds it exactly, as for `optimal_edges`.

    Parameters
    ----------
    values : array_like
        One-dimensional numeric values, such as a list, a NumPy array or a
        pandas Series. Missing values and their responses are omitted.
    response : array_like
        Finite real response of each value, such as a realized LGD or CCF,
        which may lie outside [0, 1]. If `values` and `response` are both
        pandas Series, they must have identical indices; otherwise they are
        matched by position.
    max_bins : int, default 5
        Maximum number of bins.
    min_bin_share : float, default 0.05
        Minimum share of the fitted values in every bin, from 0 to 0.5, or of
        their total weight when `weights` is given. Missing and special values
        do not count towards it.
    trend : {"auto", "increasing", "decreasing", "u_shaped", "inverted_u", "any"}, default "auto"
        Shape of the mean response as the values grow, as in
        `optimal_edges`. ``"increasing"`` and ``"decreasing"`` require every
        step from one bin to the next to rise, or to fall. ``"u_shaped"``
        requires the mean to fall and then rise, and ``"inverted_u"`` to rise
        and then fall; either part may be empty. ``"any"`` imposes no shape.
        ``"auto"`` fits both monotonic trends and keeps the one with the
        smaller sum of squares within the bins, preferring ``"increasing"``
        on a tie.
    initial_bins : int, default 20
        Number of quantile bins to start from. Tied quantiles can reduce it.
        The fit takes time in proportion to the cube of this number and
        memory in proportion to its square.
    special : array_like, optional
        Special values, such as codes for "not applicable", that are omitted
        with their responses when the edges are fitted. Pass the same values
        to `cut` so that each one forms its own group. The constraints apply
        to the intervals only.
    weights : array_like, optional
        Non-negative weight of each observation, matched like `response`.
        Observations with zero weight are left out, and the means and sums of
        squares of the bins are weighted. The initial quantile bins then use
        weighted quantiles by the inverted CDF, so unit weights can give other
        edges than no weights when there are more distinct values than
        `initial_bins`.

    Returns
    -------
    numpy.ndarray
        Increasing edges with infinite endpoints, suitable for pandas.cut.

    Raises
    ------
    ValueError
        If `max_bins` or `initial_bins` is not a positive integer,
        `min_bin_share` is not between 0 and 0.5, `trend` is not recognized,
        `values` and `response` are not one-dimensional or their indices or
        lengths do not match, `special` is not one-dimensional or contains a
        missing value or a boolean, `values` is not numeric or has no finite
        observations apart from missing and special values, or `response` is
        not real and numeric, is boolean, or has a missing or an infinite
        value.
        Also if `weights` is not numeric, does not give each value one finite,
        non-negative weight with a positive total, or has an index other than
        that of `values`.

    See Also
    --------
    optimal_edges : The same search for a binary target, by information value.
    cut : Assign values to bins, with a separate group for each special value.
    kivi.lgd.lgd_table : Summarize the response of each bin.

    Notes
    -----
    With :math:`W_b` the weight of bin :math:`b` and :math:`m_b` its mean
    response, and :math:`m` the mean of all the fitted values, the sum of
    squares within the bins is the total sum of squares less

    .. math::

       B = \sum_b W_b \, (m_b - m)^2,

    so the fit maximizes :math:`B`, the part of the sum of squares of the
    response that the bins explain. Each bin adds :math:`S_b^2 / W_b`, where
    :math:`S_b` is its weighted sum of the response less :math:`m`, so no
    second moment is subtracted from another. Ties follow the rules of
    `optimal_edges`: two means are level when they differ by at most a
    relative 1e-10, or by at most 1e-10 times the root mean square of the
    response, and two binnings tie when their :math:`B` agree to a relative
    1e-10 or differ by less than 1e-20 times the weighted sum of the squared
    responses. A response with the same mean in every bin therefore gives a
    single bin.

    A `kivi.spec.BinningSpec` holds the WOE of a binary target, so
    `kivi.spec.fit_spec` has no method for a continuous response; group the
    values with `cut` and summarize them with :func:`kivi.lgd.lgd_table`.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.binning import variance_edges
    >>> values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    >>> response = pd.Series([0.1, 0.0, 0.2, 0.1, 0.5, 0.4, 0.6, 0.5, 0.9, 1.0, 0.8, 1.1])
    >>> variance_edges(values, response, max_bins=3)
    array([-inf,   4.,   8.,  inf])
    >>> variance_edges(values, response, max_bins=3, trend="decreasing")
    array([-inf,  inf])
    """
    _validation.check_count(max_bins, "max_bins")
    _check_share(min_bin_share, "min_bin_share", 0.5)
    _check_trend(trend)
    _validation.check_count(initial_bins, "initial_bins")
    observed, responses, mass = _responded(values, response, special, weights)
    interior, position, occupied = _fine_positions(observed, initial_bins, mass)
    weight = np.ones(len(observed)) if mass is None else mass
    centre = float(np.sum(weight * responses) / np.sum(weight))
    size = len(interior) + 1
    totals = np.bincount(position, weights=weight, minlength=size)[occupied]
    sums = np.bincount(position, weights=weight * (responses - centre), minlength=size)[occupied]
    squares = float(np.sum(weight * responses**2))
    unit = math.sqrt(squares / float(np.sum(weight)))
    shapes = ["increasing", "decreasing"] if trend == "auto" else [trend]
    fits = [
        _between(totals, sums, centre, shape, max_bins, min_bin_share, unit, _GAIN_TIE * squares)
        for shape in shapes
    ]
    cuts = _chosen(fits, _GAIN_TIE * squares)
    return np.r_[-np.inf, interior[occupied[:-1]][np.asarray(cuts, dtype=np.intp) - 1], np.inf]


def category_groups(
    values: npt.ArrayLike,
    *,
    min_share: float = 0.05,
    min_count: float = 0,
    rare_label: str | float = "other",
    special: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> dict[abc.Hashable, abc.Hashable]:
    """Fit a grouping that pools the rare categories of a reference sample.

    A category is rare when it makes up less than ``min_share`` of the fitted
    values or has fewer than ``min_count`` of them. The rare categories form
    one group labelled ``rare_label``, and every other category is a group of
    its own. Apply the grouping with `group_categories` to the reference and
    to every sample compared with it.

    Parameters
    ----------
    values : array_like
        One-dimensional category labels of the reference sample, such as a
        list, a NumPy array or a pandas Series of strings. Missing values are
        omitted.
    min_share : float, default 0.05
        Share of the fitted values, from 0 to 1, below which a category is
        rare. With `weights`, the share of their total weight.
    min_count : float, default 0
        Number of fitted values, or their total weight with `weights`, below
        which a category is rare. The default leaves the decision to
        `min_share`.
    rare_label : str, int or float, default "other"
        Label of the group of rare categories. It must differ from the
        categories that are not rare and from the special codes.
    special : array_like, optional
        Special codes, such as ``"unknown"``, that are omitted when the
        grouping is fitted. Each one is a group of its own, however rare.
    weights : array_like, optional
        Non-negative weight of each value, for example the inverse of its
        sampling rate. Values with zero weight are left out, and the counts
        of the categories are sums of weights.

    Returns
    -------
    dict
        The group of each category: first the special codes in the order
        given, then the categories that are not rare, each mapped to itself,
        then the rare categories, mapped to `rare_label`. Categories follow
        the order in which they first appear, or the order of the categories
        of a categorical input.

    Raises
    ------
    ValueError
        If `values` is not one-dimensional or has no values apart from
        missing, special and zero-weight values, `min_share` is not between 0
        and 1, `min_count` is not a finite, non-negative number, `rare_label`
        is neither a string nor a number or is the label of a category that
        is not rare or of a special code, or `special` is not
        one-dimensional or contains a missing value or a boolean.
        Also if `weights` is not numeric, does not give each value one finite,
        non-negative weight with a positive total, or has an index other than
        that of `values`.

    See Also
    --------
    chimerge_groups : Merge categories with similar event rates.
    group_categories : Assign values to the fitted groups.
    kivi.woe.woe_table : Weight of evidence of the resulting groups.
    kivi.stability.psi_table : Stability of the resulting groups.

    Notes
    -----
    Rare categories are pooled whatever their event rates, so a rare
    category with a high event rate can share a group with one with a low
    rate. `chimerge_groups` uses the target to avoid this. The group of rare
    categories can itself hold less than `min_share` of the values.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.binning import category_groups, group_categories
    >>> values = pd.Series(["a", "a", "a", "a", "b", "b", "b", "c", "d", None])
    >>> groups = category_groups(values, min_share=0.2)
    >>> groups
    {'a': 'a', 'b': 'b', 'c': 'other', 'd': 'other'}
    >>> group_categories(values, groups).tolist()
    ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'other', 'other', nan]
    """
    _check_share(min_share, "min_share", 1)
    if (
        isinstance(min_count, bool)
        or not _validation.is_real(min_count)
        or not (math.isfinite(min_count) and min_count >= 0)
    ):
        raise ValueError(
            "min_count must be a finite, non-negative number, got "
            f"{_validation.shown_argument(min_count)}"
        )
    if (
        isinstance(rare_label, (bool, np.bool_))
        or not (isinstance(rare_label, str) or _validation.is_real(rare_label))
        or (_validation.is_real(rare_label) and math.isnan(rare_label))
    ):
        raise ValueError(
            f"rare_label must be a string or a number, got {_validation.shown_argument(rare_label)}"
        )
    rare_label = _python(rare_label)
    labels, totals, codes = _category_counts(values, None, special, weights)
    rare = (totals / totals.sum() < min_share) | (totals < min_count)
    kept = [label for label, is_rare in zip(labels, rare, strict=True) if not is_rare]
    if rare.any() and rare_label in {*kept, *codes}:
        raise ValueError(
            f"rare_label {_validation.shown_argument(rare_label)} is the label of a category that "
            "is not rare or of a special code; choose another rare_label"
        )
    groups = {code: code for code in codes}
    groups.update((label, label) for label in kept)
    groups.update(
        (label, rare_label) for label, is_rare in zip(labels, rare, strict=True) if is_rare
    )
    return groups


def chimerge_groups(
    values: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    max_groups: int = 5,
    alpha: float | None = 0.05,
    min_group_share: float = 0.05,
    smoothing: float = 0.5,
    special: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> dict[abc.Hashable, abc.Hashable]:
    """Fit a grouping that merges categories with similar event rates by ChiMerge.

    The categories are sorted by their WOE, with zero counts smoothed as in
    :func:`kivi.woe.woe_table`, and neighbours in that order are merged.
    First, while some group holds less than ``min_group_share`` of the
    fitted values, the smallest one is merged with the neighbour whose
    chi-square statistic with it is smaller. Then, as in `chimerge_edges`,
    the adjacent pair with the smallest chi-square statistic is merged while
    more than ``max_groups`` groups remain or while some adjacent pair is not
    significantly different at level ``alpha``. The groups are sorted by WOE
    again after every merge, so the WOE of the final groups increases from
    the first to the last.

    Parameters
    ----------
    values : array_like
        One-dimensional category labels, such as a list, a NumPy array or a
        pandas Series of strings. Missing values and their labels are omitted.
    target : array_like
        Binary labels, with 1 indicating the event. If `values` and `target`
        are both pandas Series, they must have identical indices; otherwise
        they are matched by position.
    max_groups : int, default 5
        Maximum number of groups, not counting the special codes.
    alpha : float or None, default 0.05
        Significance level of the chi-square test with one degree of freedom.
        Adjacent groups are merged until every pair differs at this level.
        Pass None to merge down to ``max_groups`` only.
    min_group_share : float, default 0.05
        Minimum share of the fitted values in every group, from 0 to 0.5, or
        of their total weight when `weights` is given. Missing and special
        values do not count towards it.
    smoothing : float, default 0.5
        Positive replacement for a zero count when calculating the WOE that
        orders the groups, as in :func:`kivi.woe.woe_table`. Use the same
        value there to see the same order.
    special : array_like, optional
        Special codes, such as ``"unknown"``, that are omitted with their
        labels when the grouping is fitted. Each one is a group of its own.
    weights : array_like, optional
        Non-negative weight of each observation, for example the inverse of
        its sampling rate, matched like `target`. Observations with zero
        weight are left out, and the counts of each group are sums of
        weights. The chi-square test treats the weighted counts as counts, so
        scale the weights to sum to the sample size when the p-values matter.
        `smoothing` is then in units of weight.

    Returns
    -------
    dict
        The group of each category: first the special codes in the order
        given, each mapped to itself, then the categories of each group, in
        increasing order of the group's WOE. A group of one category keeps
        its label; a merged group is labelled with the labels of its
        categories, in increasing order of their own WOE, joined by ``", "``.
        A group of more than five categories shows the first four, followed
        by ``", ... (+n)"`` with the number ``n`` of the others.

    Raises
    ------
    ValueError
        If `max_groups` is not a positive integer, `alpha` is neither None
        nor between 0 and 1, `min_group_share` is not between 0 and 0.5,
        `smoothing` is not finite and positive, `values` and `target` are not
        one-dimensional or their indices or lengths do not match, `special`
        is not one-dimensional or contains a missing value or a boolean,
        `values` has no values apart from missing, special and zero-weight
        values, `target` has labels other than 0 and 1 or lacks one of them
        in the rows that are fitted, or the label of a merged group would be
        the label of another group.
        Also if `weights` is not numeric, does not give each value one finite,
        non-negative weight with a positive total, or has an index other than
        that of `values`.

    See Also
    --------
    category_groups : Pool rare categories without a target.
    chimerge_edges : Merge the bins of a numeric variable in the same way.
    group_categories : Assign values to the fitted groups.

    Notes
    -----
    The chi-square statistic of two adjacent groups is the one described in
    `chimerge_edges`. Sorting by WOE puts categories with similar event rates
    next to each other, so that each merge joins the categories most alike.
    A category that is not merged is a group of its own, so with
    ``min_group_share=0`` and ``alpha=None`` a variable with at most
    ``max_groups`` categories keeps all of them.

    Ties go to the category that appears first, or that comes first among
    the categories of a categorical input: categories with the same
    smoothed WOE keep that order, and of two pairs, or of two small groups,
    whose statistics or totals agree to a relative 1e-10, the first one is
    merged. Each merge takes time in proportion to the number of groups, so
    the fit takes time in proportion to the square of the number of
    categories, a few seconds for 20,000 of them.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.binning import chimerge_groups
    >>> values = pd.Series(["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4)
    >>> target = pd.Series([0, 0, 0, 1] + [1, 1, 1, 0] + [0, 0, 0, 0] + [1, 1, 1, 1])
    >>> chimerge_groups(values, target, max_groups=2, alpha=None)
    {'c': 'c, a', 'a': 'c, a', 'b': 'b, d', 'd': 'b, d'}
    """
    _validation.check_count(max_groups, "max_groups")
    if alpha is not None and not (_validation.is_real(alpha) and 0 < alpha < 1):
        raise ValueError(
            f"alpha must be None or between 0 and 1, got {_validation.shown_argument(alpha)}"
        )
    _check_share(min_group_share, "min_group_share", 0.5)
    _validation.check_smoothing(smoothing)
    labels, counts, codes = _category_counts(values, target, special, weights)
    order = np.argsort(_odds(counts, smoothing), kind="stable")
    labels = [labels[position] for position in order]
    members = _merge_categories(counts[order], max_groups, alpha, min_group_share, smoothing)
    groups = {code: code for code in codes}
    taken = set(codes)
    for ranks in members:
        names = [labels[rank] for rank in sorted(ranks)]
        label = _group_label(names)
        if label in taken:
            raise ValueError(
                f"two groups would have the label {_validation.shown(label)}; rename the "
                "categories whose labels contain ', '"
            )
        taken.add(label)
        groups.update((name, label) for name in names)
    return groups


@typing.overload
def group_categories(
    values: pd.Series,
    groups: abc.Mapping[abc.Hashable, abc.Hashable],
    *,
    unseen: abc.Hashable = ...,
) -> pd.Series: ...
@typing.overload
def group_categories(
    values: _validation.ListLike,
    groups: abc.Mapping[abc.Hashable, abc.Hashable],
    *,
    unseen: abc.Hashable = ...,
) -> pd.Categorical: ...
@typing.overload
def group_categories(
    values: npt.ArrayLike,
    groups: abc.Mapping[abc.Hashable, abc.Hashable],
    *,
    unseen: abc.Hashable = ...,
) -> pd.Series | pd.Categorical: ...
def group_categories(
    values: npt.ArrayLike,
    groups: abc.Mapping[abc.Hashable, abc.Hashable],
    *,
    unseen: abc.Hashable = "raise",
) -> pd.Series | pd.Categorical:
    """Assign category labels to the groups of a fitted grouping.

    Each value is replaced with its group in `groups`, the dict returned by
    `category_groups` or `chimerge_groups`, or one written or edited by
    hand. Missing values stay missing, so :func:`kivi.woe.woe_table` counts
    them as a group of their own.

    Parameters
    ----------
    values : array_like
        One-dimensional category labels to assign.
    groups : mapping
        Group of each category, as a dict from category to group. Neither a
        category nor a group can be missing.
    unseen : default "raise"
        Group of the values whose category is not in `groups`, such as a new
        category: one of the groups in `groups`, for example the group of
        rare categories. By default such values raise an error, so that they
        never become missing values.

    Returns
    -------
    pandas.Series or pandas.Categorical
        Categorical groups whose categories are the groups in the order in
        which they first appear in `groups`. A Series input gives a Series
        with the same index and name.

    Raises
    ------
    ValueError
        If `values` is not one-dimensional, `groups` is not a mapping or has
        a missing or non-scalar category or group, `unseen` is neither
        ``"raise"`` nor one of the groups, or `unseen` is ``"raise"`` and some
        values are categories that `groups` does not map.

    See Also
    --------
    category_groups : Pool rare categories.
    chimerge_groups : Merge categories with similar event rates.
    cut : Assign numeric values to bins.
    kivi.woe.woe_transform : Replace each group with its WOE.

    Notes
    -----
    Categories are matched by equality, as in :func:`kivi.woe.woe_transform`,
    so ``1`` and ``1.0`` are the same category but ``"1"`` is another. The
    string ``"raise"`` always raises, so it cannot name the group of unseen
    categories.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.binning import group_categories
    >>> groups = {"a": "a", "b": "b, c", "c": "b, c"}
    >>> group_categories(pd.Series(["c", "a", None, "b"]), groups)
    0    b, c
    1       a
    2     NaN
    3    b, c
    dtype: category
    Categories (2, object): ['a', 'b, c']

    A category that the grouping does not know raises an error unless
    `unseen` names a group for it:

    >>> group_categories(["d", "a"], groups)
    Traceback (most recent call last):
    ...
    ValueError: values have 1 category that groups does not map, such as 'd'; ...
    >>> group_categories(["d", "a"], groups, unseen="b, c")
    ['b, c', 'a']
    Categories (2, object): ['a', 'b, c']
    """
    mapping = _checked_groups(groups, "groups")
    series = _validation.as_series(values, "values")
    group_of_key, labels = pd.factorize(_validation.object_index(mapping.values()))
    raise_unseen = isinstance(unseen, str) and unseen == "raise"
    if not raise_unseen:
        found = -1
        if pd.api.types.is_scalar(unseen) and not _validation.is_missing(unseen):
            found = int(labels.get_indexer(_validation.object_index([unseen]))[0])
        if found < 0:
            raise ValueError(
                "unseen must be 'raise' or one of the groups, got "
                f"{_validation.shown_argument(unseen)}"
            )
    levels: pd.Index | np.ndarray
    position, levels = pd.factorize(series, sort=False)
    levels = np.asarray(levels, dtype=object)
    key = _validation.object_index(mapping).get_indexer(_validation.object_index(levels))
    level_group = np.r_[group_of_key, -1][key]
    unmapped = key < 0
    if unmapped.any():
        if raise_unseen:
            examples = ", ".join(_validation.shown(level) for level in levels[unmapped][:5])
            count = int(unmapped.sum())
            raise ValueError(
                f"values have {count} {'category' if count == 1 else 'categories'} that groups "
                f"does not map, such as {examples}; pass unseen to give them a group"
            )
        level_group[unmapped] = found
    grouped = pd.Categorical.from_codes(np.r_[level_group, -1][position], categories=labels)
    if isinstance(values, pd.Series):
        return pd.Series(grouped, index=series.index, name=series.name)
    return grouped


def _check_share(value: float, name: str, upper: float) -> None:
    if isinstance(value, bool) or not _validation.is_real(value) or not 0 <= value <= upper:
        raise ValueError(
            f"{name} must be between 0 and {upper:g}, got {_validation.shown_argument(value)}"
        )


def _check_trend(trend: object) -> None:
    if not isinstance(trend, str) or trend not in _TRENDS:
        names = ", ".join(repr(name) for name in _TRENDS)
        raise ValueError(f"trend must be one of {names}, got {_validation.shown_argument(trend)}")


def _python(value: object) -> typing.Any:
    """Return a NumPy number, boolean or string as the Python scalar it holds."""
    if isinstance(value, (np.number, np.bool_, np.str_)):
        return value.item()
    return value


def _checked_groups(groups: object, argument: str) -> dict[abc.Hashable, abc.Hashable]:
    """Return a grouping as a dict of Python labels, rejecting missing and non-scalar labels."""
    if not isinstance(groups, abc.Mapping):
        raise ValueError(
            f"{argument} must be a mapping from category to group, got {type(groups).__name__}"
        )
    checked: dict[abc.Hashable, abc.Hashable] = {}
    for category, group in groups.items():
        for label, role in ((category, "category"), (group, "group")):
            if not pd.api.types.is_scalar(label) or _validation.is_missing(label):
                raise ValueError(
                    f"{argument} must map non-missing scalar categories to non-missing scalar "
                    f"groups, got the {role} {_validation.shown(label)}; missing values always "
                    "form their own group"
                )
        checked[_python(category)] = _python(group)
    return checked


def _category_counts(
    values: npt.ArrayLike,
    target: npt.ArrayLike | None,
    special: npt.ArrayLike | None,
    weights: npt.ArrayLike | None,
) -> tuple[list, np.ndarray, list]:
    """Fitted categories, their counts and the special codes, as Python labels.

    The counts are the number of values of each category without a target,
    or its (non_event, event) counts with one, as sums of weights with
    `weights`. Missing values, special codes and zero-weight values are left
    out together with their labels. Categories follow their first appearance,
    or the order of the categories of a categorical input.
    """
    raw_values, raw_target = values, target
    if target is not None and isinstance(values, pd.Series) and isinstance(target, pd.Series):
        if not values.index.equals(target.index):
            raise ValueError("values and target must have identical indices")
    series = _validation.as_series(values, "values")
    if target is not None:
        target = _validation.as_series(target, "target")
        if len(series) != len(target):
            raise ValueError(
                f"values and target must have the same length, got {len(series)} and {len(target)}"
            )
        _validation.binary_labels(target, both_classes=False)
    codes = _special.special_values(special)
    mass = _validation.as_weights(weights, raw_values, others={"target": raw_target})
    levels: pd.Index | np.ndarray
    position, levels = pd.factorize(series, sort=isinstance(series.dtype, pd.CategoricalDtype))
    levels = np.asarray(levels, dtype=object)
    fitted = position >= 0
    excluded = ["missing"]
    if special is not None:
        found = codes.get_indexer(_validation.object_index(levels))
        is_special = found >= 0
        # True == 1, so a numeric code would match a boolean category, which cut never does.
        flags = [i for i in np.flatnonzero(is_special) if isinstance(levels[i], (bool, np.bool_))]
        if flags:
            raise ValueError(
                "special must not match booleans, got "
                f"{_validation.shown(codes[found[flags[0]]])}, "
                f"which equals the value {_validation.shown(levels[flags[0]])}"
            )
        fitted &= ~np.r_[is_special, False][position]
        excluded.append("special")
    if mass is not None:
        fitted &= mass > 0
        excluded.append("zero-weight")
    if not fitted.any():
        names = ", ".join(excluded[:-1]) + " and " + excluded[-1] if excluded[1:] else "missing"
        raise ValueError(f"values have no categories to fit apart from {names} values")
    # Order the categories among the fitted rows only, so that rows left out, such as those
    # with zero weight, do not change it.
    if isinstance(series.dtype, pd.CategoricalDtype):
        used, category = np.unique(position[fitted], return_inverse=True)
    else:
        category, used = pd.factorize(position[fitted])
    labels = [_python(level) for level in levels[used]]
    weight = None if mass is None else mass[fitted]
    if target is None:
        counts: np.ndarray = np.bincount(category, weights=weight, minlength=len(labels))
    else:
        side = target.to_numpy()[fitted].astype(np.intp)
        if side.min() == side.max():
            rows = "observed and not special" if special is not None else "observed"
            if mass is not None:
                rows += " with a positive weight"
            raise ValueError(f"target must contain both 0 and 1 where values are {rows}")
        counts = np.bincount(2 * category + side, weights=weight, minlength=2 * len(labels))
        counts = counts.reshape(-1, 2)
    return labels, counts, [_python(code) for code in codes]


def _merge_categories(
    counts: np.ndarray, max_groups: int, alpha: float | None, min_share: float, smoothing: float
) -> list[list[int]]:
    """Merge (non_event, event) counts sorted by odds; return the rows of each final group.

    Each merge changes only the merged group, so only its odds and the statistics of its two
    pairs are recomputed. The groups are sorted again, stably, only when the merged group
    falls out of order, which smoothing can cause.
    """
    counts = counts.astype(float)
    members = [[row] for row in range(len(counts))]
    odds = _odds(counts, smoothing)
    statistic = _pair_chi2(counts)
    totals = counts.sum(axis=1)
    while len(counts) > 1:
        small = totals / totals.sum() < min_share
        if small.any():
            candidates = np.flatnonzero(small)
            group = int(candidates[_first_min(totals[candidates])])
            if group == 0:
                pair = 0
            elif group == len(counts) - 1:
                pair = group - 1
            else:
                left, right = statistic[group - 1], statistic[group]
                pair = group if right < left - _TIE * left else group - 1
        else:
            pair = _first_min(statistic)
            significant = alpha is None or math.erfc(math.sqrt(statistic[pair] / 2)) < alpha
            if len(counts) <= max_groups and significant:
                break
        counts[pair] += counts[pair + 1]
        counts = np.delete(counts, pair + 1, axis=0)
        totals[pair] += totals[pair + 1]
        totals = np.delete(totals, pair + 1)
        members[pair].extend(members.pop(pair + 1))
        odds = np.delete(odds, pair + 1)
        odds[pair] = _odds(counts[pair : pair + 1], smoothing)[0]
        last = len(counts) - 1
        if (pair > 0 and odds[pair] < odds[pair - 1]) or (
            pair < last and odds[pair] > odds[pair + 1]
        ):
            order = np.argsort(odds, kind="stable")
            counts, totals, odds = counts[order], totals[order], odds[order]
            members = [members[row] for row in order]
            statistic = _pair_chi2(counts)
        else:
            statistic = np.delete(statistic, pair)
            first, stop = max(pair - 1, 0), min(pair + 2, len(counts))
            statistic[first : stop - 1] = _pair_chi2(counts[first:stop])
    return members


# Relative difference below which two chi-square statistics or totals count as equal, so that
# rounding never decides a tie: ties go to the first group, in the order of first appearance.
_TIE = 1e-10
# A merged group is labelled with at most this many categories.
_LABEL_CATEGORIES = 5


def _first_min(values: np.ndarray) -> int:
    """Position of the first value within a relative `_TIE` of the smallest."""
    smallest = values.min()
    return int(np.flatnonzero(values <= smallest + _TIE * abs(smallest))[0])


def _first_max(values: np.ndarray) -> int:
    """Position of the first value within a relative `_TIE` of the largest."""
    largest = values.max()
    return int(np.flatnonzero(values >= largest - _TIE * abs(largest))[0])


def _odds(counts: np.ndarray, smoothing: float) -> np.ndarray:
    """Smoothed event to non-event ratio of each row, which orders the rows like their WOE.

    One correctly rounded division per row gives proportional counts exactly the same
    value, while their WOE can differ in the last bit.
    """
    adjusted = np.where(counts == 0, smoothing, counts.astype(float))
    return adjusted[:, 1] / adjusted[:, 0]


def _group_label(names: list) -> abc.Hashable:
    """Label of a group: its one category, or up to five of its categories joined by ", "."""
    if len(names) == 1:
        return names[0]
    if len(names) <= _LABEL_CATEGORIES:
        return ", ".join(str(name) for name in names)
    head = ", ".join(str(name) for name in names[: _LABEL_CATEGORIES - 1])
    return f"{head}, ... (+{len(names) - _LABEL_CATEGORIES + 1})"


def _labeled(
    values: npt.ArrayLike,
    target: npt.ArrayLike,
    special: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Observed values and their 0/1 labels, matched by index or position like ``woe_table``.

    Rows whose value is missing, a special value or has zero weight are left
    out together with their labels and weights.
    """
    raw_values, raw_target = values, target
    if isinstance(values, pd.Series) and isinstance(target, pd.Series):
        if not values.index.equals(target.index):
            raise ValueError("values and target must have identical indices")
    values, target = (
        _validation.as_series(values, "values"),
        _validation.as_series(target, "target"),
    )
    if len(values) != len(target):
        raise ValueError(
            f"values and target must have the same length, got {len(values)} and {len(target)}"
        )
    _validation.binary_labels(target, both_classes=False)
    mass = _validation.as_weights(weights, raw_values, others={"target": raw_target})
    fitted = values.notna().to_numpy(dtype=bool, copy=True)
    if special is not None:
        special = _special.special_values(special)
        is_special = _special.special_positions(values, special) >= 0
        fitted &= ~is_special
    else:
        is_special = None
    if mass is not None:
        fitted &= mass > 0
    observed, mass = _edges.observed(values, special, mass, is_special=is_special)
    labels = target.to_numpy()[fitted].astype(np.intp)
    if labels.min() == labels.max():
        rows = "observed and not special" if special is not None else "observed"
        if mass is not None:
            rows += " with a positive weight"
        raise ValueError(f"target must contain both 0 and 1 where values are {rows}")
    return observed, labels, mass


def _responded(
    values: npt.ArrayLike,
    response: npt.ArrayLike,
    special: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> tuple[np.ndarray, npt.NDArray[np.float64], np.ndarray | None]:
    """Observed values and their responses, matched by index or position like `_labeled`.

    The response must be finite and real on every row. Rows whose value is missing, a special
    value or has zero weight are left out together with their responses and weights.
    """
    raw_values, raw_response = values, response
    if isinstance(values, pd.Series) and isinstance(response, pd.Series):
        if not values.index.equals(response.index):
            raise ValueError("values and response must have identical indices")
    values = _validation.as_series(values, "values")
    series = _validation.as_series(response, "response")
    if len(values) != len(series):
        raise ValueError(
            f"values and response must have the same length, got {len(values)} and {len(series)}"
        )
    responses = _validation.as_real(series, "response")
    missing = np.isnan(responses)
    if missing.any():
        raise ValueError(f"response must not contain missing values, got {int(missing.sum())}")
    infinite = np.isinf(responses)
    if infinite.any():
        raise ValueError(
            f"response must be finite, got {_validation.shown(responses[infinite][0])}"
        )
    mass = _validation.as_weights(weights, raw_values, others={"response": raw_response})
    fitted = values.notna().to_numpy(dtype=bool, copy=True)
    is_special = None
    if special is not None:
        special = _special.special_values(special)
        is_special = _special.special_positions(values, special) >= 0
        fitted &= ~is_special
    if mass is not None:
        fitted &= mass > 0
    observed, mass = _edges.observed(values, special, mass, is_special=is_special)
    return observed, responses[fitted], mass


def _cumulative(
    observed: np.ndarray, labels: np.ndarray, mass: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Distinct values, and the counts, events and non-events of the values before each of them.

    ``totals[i]``, ``events[i]`` and ``non_events[i]`` count the rows of the first ``i``
    distinct values, as sums of weights with `mass`, so the values ``start..stop-1`` hold
    ``totals[stop] - totals[start]``. Events and non-events are summed separately rather than
    subtracted from the totals, so a range without one class holds exactly zero of it.
    """
    unique, position = np.unique(observed, return_inverse=True)
    weight = np.ones(len(labels)) if mass is None else mass
    totals = np.cumsum(np.r_[0, np.bincount(position, weights=mass)])
    events = np.cumsum(np.r_[0, np.bincount(position, weights=labels * weight)])
    non_events = np.cumsum(np.r_[0, np.bincount(position, weights=(1 - labels) * weight)])
    return unique, totals, events, non_events


def _fine_positions(
    observed: np.ndarray, bins: int, weights: np.ndarray | None = None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp], npt.NDArray[np.bool_]]:
    """Interior quantile edges, the bin of each value and which bins hold values.

    The values are cut into `bins` quantile bins, or one bin per distinct value when there
    are no more distinct values than that. Interpolated quantiles can leave a bin empty, and
    ``interior[occupied[:-1]]`` joins each empty bin to the next one.
    """
    unique = np.unique(observed)
    if len(unique) <= bins:
        interior = unique[:-1]
    else:
        probabilities = np.linspace(0, 1, bins + 1)
        if weights is None:
            quantiles = np.quantile(observed, probabilities)
        else:
            quantiles = _edges.weighted_quantile(observed, weights, probabilities)
        interior = np.unique(quantiles)[1:-1]
    position = np.searchsorted(interior, observed, side="left")
    occupied = np.bincount(position, minlength=len(interior) + 1) > 0
    return interior, position, occupied


def _fine_bins(
    observed: np.ndarray, labels: np.ndarray, bins: int, weights: np.ndarray | None = None
) -> tuple[npt.NDArray[np.float64], np.ndarray]:
    """Interior quantile edges and (non_event, event) counts of each non-empty bin."""
    interior, position, occupied = _fine_positions(observed, bins, weights)
    counts: np.ndarray = np.bincount(
        2 * position + labels, weights=weights, minlength=2 * (len(interior) + 1)
    )
    return interior[occupied[:-1]], counts.reshape(-1, 2)[occupied]


def _merge(interior: np.ndarray, counts: np.ndarray, pair: int) -> tuple[np.ndarray, np.ndarray]:
    """Merge bin ``pair`` with bin ``pair + 1``."""
    counts = counts.copy()
    counts[pair] += counts[pair + 1]
    return np.delete(interior, pair), np.delete(counts, pair + 1, axis=0)


def _merge_small(
    interior: np.ndarray, counts: np.ndarray, min_share: float
) -> tuple[np.ndarray, np.ndarray]:
    """Merge the smallest bin below `min_share` into its more similar neighbour until none is left.

    The neighbour is the one whose chi-square statistic with the bin is smaller, the left one
    on a tie, and of several smallest bins the first is merged, as in ``chimerge_groups``.
    """
    while len(counts) > 1:
        totals = counts.sum(axis=1)
        small = np.flatnonzero(totals / totals.sum() < min_share)
        if not len(small):
            break
        bin_ = int(small[_first_min(totals[small])])
        if bin_ == 0:
            pair = 0
        elif bin_ == len(counts) - 1:
            pair = bin_ - 1
        else:
            left, right = _pair_chi2(counts[bin_ - 1 : bin_ + 2])
            pair = bin_ if right < left - _TIE * left else bin_ - 1
        interior, counts = _merge(interior, counts, pair)
    return interior, counts


def _pair_chi2(counts: np.ndarray) -> np.ndarray:
    """Chi-square statistic of each adjacent pair of bins."""
    pairs = np.stack([counts[:-1], counts[1:]], axis=1).astype(float)
    rows = pairs.sum(axis=2, keepdims=True)
    columns = pairs.sum(axis=1, keepdims=True)
    expected = rows * columns / rows.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cells = np.where(expected > 0, (pairs - expected) ** 2 / expected, 0.0)
    return cells.sum(axis=(1, 2))


def _entropy(rate: np.ndarray) -> np.ndarray:
    rate = np.asarray(rate, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = -(rate * np.log(rate) + (1 - rate) * np.log(1 - rate))
    return np.nan_to_num(terms, nan=0.0)


def _gini(rate: np.ndarray) -> np.ndarray:
    return 2 * rate * (1 - rate)


def _woe(counts: np.ndarray, smoothing: float) -> np.ndarray:
    return _counts.divergence(counts, smoothing)[1]


def _iv(counts: np.ndarray, smoothing: float) -> float:
    return float(_counts.divergence(counts, smoothing)[2].sum())


def _monotonic(
    interior: np.ndarray, counts: np.ndarray, sign: int, max_bins: int, smoothing: float
) -> tuple[np.ndarray, np.ndarray]:
    while len(counts) > 1:
        breaks = np.flatnonzero(sign * np.diff(_woe(counts, smoothing)) <= 0)
        if len(breaks):
            pair = int(breaks[0])
        elif len(counts) > max_bins:
            pair = int(np.argmin(_pair_chi2(counts)))
        else:
            break
        interior, counts = _merge(interior, counts, pair)
    return interior, counts


# The trends of optimal_edges, in the order of its documentation.
_TRENDS = ("auto", "increasing", "decreasing", "u_shaped", "inverted_u", "any")
# A step from the WOE of one bin to that of the next, as kivi.woe.woe_trend reads it.
_FALL, _FLAT, _RISE = -1, 0, 1
# Gains that differ by less than this share of their scale are equal whatever their size: bins
# with equal event rates or means differ by about 1e-16 instead of 0 when the weights round, and
# splitting them gains about 1e-32 of the scale over a single bin.
_GAIN_TIE = _TIE**2
# The binnings of each trend, read from the last bin down: a single bin is in state 0, and
# (step, state) -> state gives the state of a binning after a bin is added below its lowest bin,
# with that step from the new bin to the old lowest one. Every state is an allowed shape.
_PREPEND: dict[str, dict[tuple[int, int], int]] = {
    "increasing": {(_RISE, 0): 0},
    "decreasing": {(_FALL, 0): 0},
    "any": {(_RISE, 0): 0, (_FLAT, 0): 0, (_FALL, 0): 0},
    # 0: the level only rises; 1: it falls, then rises.
    "u_shaped": {(_RISE, 0): 0, (_FALL, 0): 1, (_FALL, 1): 1},
    # 0: the level only falls; 1: it rises, then falls.
    "inverted_u": {(_FALL, 0): 0, (_RISE, 0): 1, (_RISE, 1): 1},
}


def _steps(before: np.ndarray, after: np.ndarray, unit: float = 1.0) -> np.ndarray:
    """Step from each level in `before` to that in `after`: a rise, a fall or, within a tie, flat.

    A step is flat when it is within a relative `_TIE` of the larger level, as in
    ``kivi.woe.woe_trend``, or within `_TIE` times `unit` of zero, so that two levels of
    rounding size around 0 are not a rise or a fall. The unit is 1 for a WOE and the root mean
    square of the response for a mean.
    """
    change = after - before
    scale = np.maximum(np.maximum(np.abs(before), np.abs(after)), unit)
    return np.where(np.abs(change) > _TIE * scale, np.sign(change), _FLAT).astype(np.intp)


def _floor(largest: float, absolute: float) -> float:
    """Smallest gain that ties with `largest`, within a relative `_TIE` or within `absolute`."""
    return largest - max(_TIE * abs(largest), absolute)


def _chosen(fits: list[tuple[list[int], float]], absolute: float) -> list[int]:
    """Cuts of the first fit whose gain ties with the largest, so "auto" prefers increasing."""
    floor = _floor(max(gain for _, gain in fits), absolute)
    return next(cuts for cuts, gain in fits if gain >= floor)


def _optimal(
    counts: np.ndarray, trend: str, max_bins: int, min_share: float, min_events: float
) -> tuple[list[int], float]:
    """Cut positions among the (non_event, event) rows of the binning with the largest IV."""
    n = len(counts)
    cumulative = np.zeros((n + 1, 2))
    cumulative[1:] = np.cumsum(counts, axis=0)
    # Rows i..j-1 for every start i (axis 0) and stop j (axis 1); only i < j are bins.
    non_event = cumulative[None, :, 0] - cumulative[:, None, 0]
    event = cumulative[None, :, 1] - cumulative[:, None, 1]
    slack = 1 - _TIE
    feasible = (
        np.triu(np.ones((n + 1, n + 1), dtype=bool), 1)
        & (non_event + event >= min_share * cumulative[-1].sum() * slack)
        & (non_event >= min_events * slack)
        & (event >= min_events * slack)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        event_share = event / cumulative[-1, 1]
        non_event_share = non_event / cumulative[-1, 0]
        woe = np.where(feasible, np.log(event_share / non_event_share), 0.0)
    iv = np.where(feasible, (event_share - non_event_share) * woe, -np.inf)
    del non_event, event, event_share, non_event_share, feasible
    return _best_binning(iv, woe, trend, max_bins, unit=1.0, absolute=_GAIN_TIE)


def _between(
    totals: np.ndarray,
    sums: np.ndarray,
    centre: float,
    trend: str,
    max_bins: int,
    min_share: float,
    unit: float,
    absolute: float,
) -> tuple[list[int], float]:
    """Cut positions of the binning with the largest between-bin sum of squares, and that sum.

    `totals` are the weights of the fine bins and `sums` their weighted sums of the response
    less `centre`, its mean, so a bin of weight W and sum S has the mean ``centre + S / W`` and
    adds ``S ** 2 / W`` to the between-bin sum of squares.
    """
    n = len(totals)
    weight = np.r_[0.0, np.cumsum(totals)]
    deviation = np.r_[0.0, np.cumsum(sums)]
    total = weight[None, :] - weight[:, None]
    summed = deviation[None, :] - deviation[:, None]
    feasible = np.triu(np.ones((n + 1, n + 1), dtype=bool), 1) & (
        total >= min_share * weight[-1] * (1 - _TIE)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(feasible, centre + summed / total, 0.0)
        gain = np.where(feasible, summed * (summed / total), -np.inf)
    del total, summed, feasible
    return _best_binning(gain, mean, trend, max_bins, unit=unit, absolute=absolute)


def _best_binning(
    gain: np.ndarray, level: np.ndarray, trend: str, max_bins: int, *, unit: float, absolute: float
) -> tuple[list[int], float]:
    """Cut positions of the binning with the largest sum of `gain` whose `level` follows `trend`.

    ``gain[i, j]`` is what the bin of rows ``i..j-1`` adds, ``-inf`` where the constraints
    rule it out, and ``level[i, j]`` its WOE or mean. ``best[b][state, i, j]`` is the largest
    gain of a binning of rows ``i..n-1`` into ``b`` bins whose lowest bin holds rows ``i..j-1``
    and whose shape is in ``state`` (see `_PREPEND`). The gain of a binning is summed from its
    highest bin down, as the table is filled, so that every binning has one gain, which the
    tie rule of `optimal_edges` compares. Each table takes ``8 * states * (n + 1) ** 2`` bytes,
    and there is one per number of bins that some binning reaches.
    """
    n = gain.shape[0] - 1
    transitions = _PREPEND[trend]
    states = 1 + max(transitions.values())
    # best[b] is allocated only while some binning has b bins, which the constraints limit.
    first = np.full((states, n + 1, n + 1), -np.inf)
    first[0, :, n] = gain[:, n]
    best = [first[:0], first]
    for bins in range(2, min(max_bins, n) + 1):
        table = np.full((states, n + 1, n + 1), -np.inf)
        for stop in range(1, n):
            above = best[bins - 1][:, stop, stop + 1 :]
            if not np.isfinite(above).any():
                continue
            steps = _steps(level[:stop, stop, None], level[None, stop, stop + 1 :], unit)
            for (step, state), new in transitions.items():
                following = np.where(steps == step, above[state], -np.inf).max(axis=1)
                np.maximum(
                    table[new, :stop, stop],
                    gain[:stop, stop] + following,
                    out=table[new, :stop, stop],
                )
        if not np.isfinite(table).any():
            break
        best.append(table)
    by_bins = np.array([table[:, 0, :].max() for table in best[1:]])
    largest = float(by_bins.max())
    floor = _floor(largest, absolute)
    bins = 1 + int(np.flatnonzero(by_bins >= floor)[0])
    # Choose the bins from the lowest up: each time the lowest stop from which a binning of the
    # rest, in a shape that the bins chosen so far allow, still reaches the floor.
    cuts: list[int] = []
    chosen_gain: list[float] = []
    chosen_steps: list[int] = []
    start, previous = 0, None
    for left in range(bins, 0, -1):
        candidates = best[left][:, start, :]
        if previous is not None:
            steps = _steps(previous, level[start], unit)
            allowed = np.zeros(candidates.shape, dtype=bool)
            for (step, state), new in transitions.items():
                if _reaches(new, chosen_steps, transitions):
                    allowed[state] |= steps == step
            candidates = np.where(allowed, candidates, -np.inf)
        total = candidates.max(axis=0)
        for value in reversed(chosen_gain):
            total = value + total
        stop = int(np.flatnonzero(total >= floor)[0])
        if previous is not None:
            chosen_steps.append(int(steps[stop]))
        chosen_gain.append(float(gain[start, stop]))
        previous = level[start, stop]
        cuts.append(stop)
        start = stop
    return cuts[:-1], largest


def _reaches(state: int, steps: list[int], transitions: abc.Mapping[tuple[int, int], int]) -> bool:
    """Whether the bins below a binning in `state` can come with `steps`, read from the top down."""
    for step in reversed(steps):
        if (step, state) not in transitions:
            return False
        state = transitions[step, state]
    return True
