"""Numeric bin boundaries and bin assignment.

``bin_edges`` fits edges from the values alone. ``chimerge_edges``,
``tree_edges`` and ``monotonic_edges`` also use a binary target. All of them
return increasing edges with infinite endpoints for ``pandas.cut``, or for
``cut``, which also gives each special value its own group.
"""

import math
import warnings
from numbers import Integral, Real
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._counts import shares
from ._validation import as_series

__all__ = ["bin_edges", "cut", "chimerge_edges", "tree_edges", "monotonic_edges"]


def bin_edges(
    values: npt.ArrayLike,
    bins: int = 5,
    method: Literal["quantile", "uniform"] = "quantile",
    *,
    special: npt.ArrayLike | None = None,
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

    Returns
    -------
    numpy.ndarray
        Increasing edges with infinite endpoints, suitable for pandas.cut.

    Raises
    ------
    ValueError
        If `bins` is not a positive integer, `method` is not recognized,
        `values` is not one-dimensional or not numeric, `special` is not
        one-dimensional or contains a missing value, or the reference has no
        observations or contains an infinite value apart from special values.

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
    kivi.evaluate.psi_table : Stability of the resulting groups.

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
    _check_count(bins, "bins")
    if method not in ("quantile", "uniform"):
        raise ValueError(f"method must be 'quantile' or 'uniform', got {method!r}")
    observed = _observed(as_series(values, "values"), _special_values(special))
    if method == "quantile":
        edges = np.quantile(observed, np.linspace(0, 1, bins + 1))
    else:
        edges = np.linspace(observed.min(), observed.max(), bins + 1)
    edges = np.r_[-np.inf, np.unique(edges)[1:-1], np.inf]
    if len(edges) - 1 < bins:
        levels, counts = np.unique(observed, return_counts=True)
        top = counts.argmax()
        warnings.warn(
            f"tied edges leave {len(edges) - 1} of the {bins} requested bins; the value "
            f"{levels[top]:g} makes up {counts[top] / len(observed):.0%} of the fitted values, "
            "consider passing it in special",
            UserWarning,
            stacklevel=2,
        )
    return edges


def cut(
    values: npt.ArrayLike, edges: npt.ArrayLike, *, special: npt.ArrayLike | None = None
) -> pd.Series | pd.Categorical:
    """Assign values to bins, with a separate group for each special value.

    Without `special` the result is the same as ``pandas.cut(values, edges)``.

    Parameters
    ----------
    values : array_like
        One-dimensional numeric values to assign. Missing values stay missing.
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
        If `values` is not one-dimensional or is not numeric apart from
        special values, `edges` does not increase, or `special` is not
        one-dimensional or contains a missing value.

    See Also
    --------
    bin_edges : Fit the edges on a reference sample.
    pandas.cut : Assign values to intervals only.
    kivi.woe.woe_table : Weight of evidence of the resulting groups.
    kivi.evaluate.psi_table : Stability of the resulting groups.

    Notes
    -----
    A special value is matched by equality, so ``-1`` and ``-1.0`` are the
    same code. Values outside the edges become missing, as with
    ``pandas.cut``; the infinite outer edges of `bin_edges` avoid this.

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
    series = as_series(values, "values")
    special = _special_values(special)
    is_special = series.isin(special).to_numpy()
    intervals = pd.cut(pd.to_numeric(series.mask(is_special), errors="raise"), edges)
    if special.empty:
        return intervals if isinstance(values, pd.Series) else intervals.array
    codes = intervals.cat.codes.to_numpy()
    codes = np.where(codes >= 0, codes + len(special), -1)
    codes[is_special] = special.get_indexer(series[is_special])
    groups = pd.Categorical.from_codes(
        codes, categories=special.append(intervals.cat.categories)
    )
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
) -> npt.NDArray[np.float64]:
    """Fit edges by merging adjacent quantile bins with ChiMerge.

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

    Returns
    -------
    numpy.ndarray
        Increasing edges with infinite endpoints, suitable for pandas.cut.

    Raises
    ------
    ValueError
        If `max_bins` or `initial_bins` is not a positive integer, `alpha` is
        neither None nor between 0 and 1, `values` and `target` are not
        one-dimensional or their indices or lengths do not match, `special`
        is not one-dimensional or contains a missing value, `values` is not
        numeric or has no finite observations apart from missing and special
        values, or `target` has labels other than 0 and 1 or lacks one of them
        in the rows that are fitted.

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

       \\chi^2 = \\sum_{i=1}^{2} \\sum_{j=0}^{1} \\frac{(O_{ij} - E_{ij})^2}{E_{ij}}

    where cells with :math:`E_{ij} = 0` contribute nothing. A pair stays
    separate when the p-value of :math:`\\chi^2` is below ``alpha``.

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
    _check_count(max_bins, "max_bins")
    _check_count(initial_bins, "initial_bins")
    if alpha is not None and not (isinstance(alpha, Real) and 0 < alpha < 1):
        raise ValueError(f"alpha must be None or between 0 and 1, got {alpha!r}")
    observed, labels = _labeled(values, target, special)
    interior, counts = _fine_bins(observed, labels, initial_bins)
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
    criterion: Literal["entropy", "gini"] = "entropy",
    special: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Fit edges with the splits of a decision tree on one variable.

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
        Minimum share of the fitted values in every bin, from 0 to 0.5.
        Missing and special values do not count towards it.
    criterion : {"entropy", "gini"}, default "entropy"
        Impurity measure: entropy for information gain, or the Gini index.
    special : array_like, optional
        Special values, such as codes for "not applicable", that are omitted
        with their labels when the edges are fitted. Pass the same values to
        `cut` so that each one forms its own group.

    Returns
    -------
    numpy.ndarray
        Increasing edges with infinite endpoints, suitable for pandas.cut.
        Every interior edge is an observed value, the largest one in the bin
        to its left.

    Raises
    ------
    ValueError
        If `max_bins` is not a positive integer, `min_bin_share` is not
        between 0 and 0.5, `criterion` is not recognized, `values` and
        `target` are not one-dimensional or their indices or lengths do not
        match, `special` is not one-dimensional or contains a missing value,
        `values` is not numeric or has no finite observations apart from
        missing and special values, or `target` has labels other than 0 and 1
        or lacks one of them in the rows that are fitted.

    See Also
    --------
    chimerge_edges : Merge fine bins instead of splitting.
    monotonic_edges : Merge until the WOE is monotonic.
    cut : Assign values to bins, with a separate group for each special value.

    Notes
    -----
    For a bin with :math:`n` observations and event rate :math:`p`, the
    impurity is :math:`H(p) = -p \\ln p - (1 - p) \\ln(1 - p)` (entropy) or
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
    _check_count(max_bins, "max_bins")
    if isinstance(min_bin_share, bool) or not isinstance(min_bin_share, Real) \
            or not 0 <= min_bin_share <= 0.5:
        raise ValueError(f"min_bin_share must be between 0 and 0.5, got {min_bin_share!r}")
    if criterion not in ("entropy", "gini"):
        raise ValueError(f"criterion must be 'entropy' or 'gini', got {criterion!r}")
    observed, labels = _labeled(values, target, special)
    unique, position = np.unique(observed, return_inverse=True)
    totals = np.cumsum(np.r_[0, np.bincount(position)])
    events = np.cumsum(np.r_[0, np.bincount(position, weights=labels)])
    min_count = min_bin_share * len(observed)
    impurity = _entropy if criterion == "entropy" else _gini

    def best_split(start, stop):
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
        gain = (n * impurity(e / n) - n_left * impurity(e_left / n_left)
                - n_right * impurity(e_right / n_right))
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
        leaves[leaf:leaf + 1] = [(start, cut, *best_split(start, cut)),
                                 (cut, stop, *best_split(cut, stop))]
    interior = [unique[stop - 1] for _, stop, _, _ in leaves[:-1]]
    return np.r_[-np.inf, interior, np.inf]


def monotonic_edges(
    values: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    max_bins: int | None = 5,
    direction: Literal["auto", "increasing", "decreasing"] = "auto",
    initial_bins: int = 20,
    smoothing: float = 0.5,
    special: npt.ArrayLike | None = None,
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

    Returns
    -------
    numpy.ndarray
        Increasing edges with infinite endpoints, suitable for pandas.cut.

    Raises
    ------
    ValueError
        If `max_bins` is neither None nor a positive integer, `initial_bins`
        is not a positive integer, `direction` is not recognized, `smoothing`
        is not finite and positive, `values` and `target` are not
        one-dimensional or their indices or lengths do not match, `special`
        is not one-dimensional or contains a missing value, `values` is not
        numeric or has no finite observations apart from missing and special
        values, or `target` has labels other than 0 and 1 or lacks one of them
        in the rows that are fitted.

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
        _check_count(max_bins, "max_bins")
    _check_count(initial_bins, "initial_bins")
    if direction not in ("auto", "increasing", "decreasing"):
        raise ValueError(
            f"direction must be 'auto', 'increasing' or 'decreasing', got {direction!r}"
        )
    observed, labels = _labeled(values, target, special)
    fine = _fine_bins(observed, labels, initial_bins)
    _woe(fine[1], smoothing)
    limit = len(fine[1]) if max_bins is None else max_bins
    if direction == "auto":
        rising = _monotonic(*fine, 1, limit, smoothing)
        falling = _monotonic(*fine, -1, limit, smoothing)
        interior = rising[0] if _iv(rising[1], smoothing) >= _iv(falling[1], smoothing) else falling[0]
    else:
        interior = _monotonic(*fine, 1 if direction == "increasing" else -1, limit, smoothing)[0]
    return np.r_[-np.inf, interior, np.inf]


def _check_count(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _observed(values: pd.Series, special: pd.Index | None = None) -> np.ndarray:
    """Finite numeric values without missing values and, if given, special values."""
    if special is not None:
        values = values[~values.isin(special)]
    observed = pd.to_numeric(values.dropna(), errors="raise").to_numpy(dtype=float)
    excluded = "missing and special values" if special is not None else "missing values"
    if not len(observed):
        raise ValueError(f"reference values have no finite observations apart from {excluded}")
    if not np.isfinite(observed).all():
        hint = "; pass infinite codes in special" if special is not None else ""
        raise ValueError(f"reference values must contain finite observations only{hint}")
    return observed


def _special_values(special: npt.ArrayLike | None) -> pd.Index:
    if special is None:
        return pd.Index([], dtype=object)
    special = pd.Index(pd.unique(as_series(special, "special")))
    if special.hasnans:
        raise ValueError(
            "special must not contain missing values; they always form their own group"
        )
    return special


def _labeled(
    values: npt.ArrayLike, target: npt.ArrayLike, special: npt.ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Observed values and their 0/1 labels, matched by index or position like ``woe_table``.

    Rows whose value is missing or, if ``special`` is given, a special value
    are left out together with their labels.
    """
    if isinstance(values, pd.Series) and isinstance(target, pd.Series):
        if not values.index.equals(target.index):
            raise ValueError("values and target must have identical indices")
    values, target = as_series(values, "values"), as_series(target, "target")
    if len(values) != len(target):
        raise ValueError(
            f"values and target must have the same length, got {len(values)} and {len(target)}"
        )
    if not target.isin([0, 1]).all():
        raise ValueError("target must contain only 0 and 1 and no missing labels")
    fitted = values.notna()
    if special is not None:
        special = _special_values(special)
        fitted = fitted & ~values.isin(special)
    observed = _observed(values, special)
    labels = target.to_numpy()[fitted.to_numpy()].astype(np.intp)
    if labels.min() == labels.max():
        rows = "observed and not special" if special is not None else "observed"
        raise ValueError(f"target must contain both 0 and 1 where values are {rows}")
    return observed, labels


def _fine_bins(observed: np.ndarray, labels: np.ndarray, bins: int):
    """Interior quantile edges and (non_event, event) counts of each non-empty bin."""
    unique = np.unique(observed)
    if len(unique) <= bins:
        interior = unique[:-1]
    else:
        interior = np.unique(np.quantile(observed, np.linspace(0, 1, bins + 1)))[1:-1]
    position = np.searchsorted(interior, observed, side="left")
    counts = np.zeros((len(interior) + 1, 2), dtype=np.int64)
    np.add.at(counts, (position, labels), 1)
    # Interpolated quantiles can leave a bin empty; join it to the next bin.
    occupied = counts.sum(axis=1) > 0
    return interior[occupied[:-1]], counts[occupied]


def _merge(interior: np.ndarray, counts: np.ndarray, pair: int):
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
    proportions = shares(pd.DataFrame(counts), smoothing).to_numpy()
    return np.log(proportions[:, 1] / proportions[:, 0])


def _iv(counts: np.ndarray, smoothing: float) -> float:
    proportions = shares(pd.DataFrame(counts), smoothing).to_numpy()
    difference = proportions[:, 1] - proportions[:, 0]
    return float((difference * np.log(proportions[:, 1] / proportions[:, 0])).sum())


def _monotonic(interior: np.ndarray, counts: np.ndarray, sign: int, max_bins: int, smoothing: float):
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
