"""Numeric bin boundaries, category groups and their assignment.

``bin_edges`` fits edges from the values alone. ``chimerge_edges``,
``tree_edges`` and ``monotonic_edges`` also use a binary target. All of them
return increasing edges with infinite endpoints for ``pandas.cut``, or for
``cut``, which also gives each special value its own group.

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
    "monotonic_edges",
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
    special: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    r"""Fit edges by merging adjacent quantile bins with ChiMerge.

    The values are first cut into ``initial_bins`` quantile bins, or into one
    bin per distinct value if there are no more distinct values than that.
    The adjacent pair with the smallest chi-square statistic, the pair whose
    event rates are most alike, is then merged, and merging is repeated while
    more than ``max_bins`` bins remain or while some adjacent pair is not
    significantly different at level ``alpha``.

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
        neither None nor between 0 and 1, `values` and `target` are not
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
    tree_edges : Split the values recursively instead of merging.
    monotonic_edges : Merge until the WOE is monotonic.
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
    """
    _validation.check_count(max_bins, "max_bins")
    _validation.check_count(initial_bins, "initial_bins")
    if alpha is not None and not (_validation.is_real(alpha) and 0 < alpha < 1):
        raise ValueError(
            f"alpha must be None or between 0 and 1, got {_validation.shown_argument(alpha)}"
        )
    observed, labels, mass = _labeled(values, target, special, weights)
    interior, counts = _fine_bins(observed, labels, initial_bins, mass)
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
    unique, position = np.unique(observed, return_inverse=True)
    event_mass = labels if mass is None else labels * mass
    totals = np.cumsum(np.r_[0, np.bincount(position, weights=mass)])
    events = np.cumsum(np.r_[0, np.bincount(position, weights=event_mass)])
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


def monotonic_edges(
    values: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    max_bins: int | None = 5,
    direction: typing.Literal["auto", "increasing", "decreasing"] = "auto",
    initial_bins: int = 20,
    smoothing: float = 0.5,
    special: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Fit edges by merging adjacent quantile bins until the WOE is monotonic.

    The values are first cut into ``initial_bins`` quantile bins, or into one
    bin per distinct value if there are no more distinct values than that.
    While two adjacent bins break the trend, the first such pair is merged and
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
        not a positive integer, `direction` is not recognized, `smoothing` is
        not finite and positive, `values` and `target` are not one-dimensional
        or their indices or lengths do not match, `special` is not
        one-dimensional or contains a missing value or a boolean, `values` is
        not numeric or has no finite observations apart from missing and
        special values, or `target` has labels other than 0 and 1 or lacks one
        of them in the rows that are fitted.
        Also if `weights` is not numeric, does not give each value one finite,
        non-negative weight with a positive total, or has an index other than
        that of `values`.

    See Also
    --------
    chimerge_edges : Merge fine bins by a chi-square test.
    tree_edges : Split the values recursively.
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
    observed, labels, mass = _labeled(values, target, special, weights)
    fine = _fine_bins(observed, labels, initial_bins, mass)
    _woe(fine[1], smoothing)
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


def _fine_bins(
    observed: np.ndarray, labels: np.ndarray, bins: int, weights: np.ndarray | None = None
) -> tuple[npt.NDArray[np.float64], np.ndarray]:
    """Interior quantile edges and (non_event, event) counts of each non-empty bin."""
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
    counts: np.ndarray = np.bincount(
        2 * position + labels, weights=weights, minlength=2 * (len(interior) + 1)
    )
    counts = counts.reshape(-1, 2)
    # Interpolated quantiles can leave a bin empty; join it to the next bin.
    occupied = counts.sum(axis=1) > 0
    return interior[occupied[:-1]], counts[occupied]


def _merge(interior: np.ndarray, counts: np.ndarray, pair: int) -> tuple[np.ndarray, np.ndarray]:
    """Merge bin ``pair`` with bin ``pair + 1``."""
    counts = counts.copy()
    counts[pair] += counts[pair + 1]
    return np.delete(interior, pair), np.delete(counts, pair + 1, axis=0)


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
