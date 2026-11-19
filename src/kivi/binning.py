"""Numeric bin boundaries.

``bin_edges`` fits edges from the values alone. ``chimerge_edges``,
``tree_edges`` and ``monotonic_edges`` also use a binary target. All of them
return increasing edges with infinite endpoints for ``pandas.cut``.
"""

import math
from numbers import Integral, Real
from typing import Literal

import numpy as np
import pandas as pd

from ._counts import shares


def bin_edges(
    values: pd.Series, bins: int = 5, method: Literal["quantile", "uniform"] = "quantile"
) -> np.ndarray:
    """Fit numeric bin edges on a reference sample.

    Parameters
    ----------
    values : pandas.Series
        Numeric reference values. Missing values are omitted.
    bins : int, default 5
        Requested number of bins. Tied quantiles can reduce this number.
    method : {"quantile", "uniform"}, default "quantile"
        Use sample quantiles or equally spaced edges.

    Returns
    -------
    numpy.ndarray
        Increasing edges with infinite endpoints, suitable for pandas.cut.

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
    """
    _check_count(bins, "bins")
    if method not in ("quantile", "uniform"):
        raise ValueError("method must be 'quantile' or 'uniform'")
    observed = _observed(values)
    if method == "quantile":
        edges = np.quantile(observed, np.linspace(0, 1, bins + 1))
    else:
        edges = np.linspace(observed.min(), observed.max(), bins + 1)
    return np.r_[-np.inf, np.unique(edges)[1:-1], np.inf]


def chimerge_edges(
    values: pd.Series,
    target: pd.Series,
    *,
    max_bins: int = 5,
    alpha: float | None = 0.05,
    initial_bins: int = 20,
) -> np.ndarray:
    """Fit edges by merging adjacent quantile bins with ChiMerge.

    The values are first cut into ``initial_bins`` quantile bins, or into one
    bin per distinct value if there are no more distinct values than that.
    The adjacent pair with the smallest chi-square statistic, the pair whose
    event rates are most alike, is then merged, and merging is repeated while
    more than ``max_bins`` bins remain or while some adjacent pair is not
    significantly different at level ``alpha``.

    Parameters
    ----------
    values : pandas.Series
        Numeric values. Missing values and their labels are omitted.
    target : pandas.Series
        Binary labels on the same index, with 1 indicating the event.
    max_bins : int, default 5
        Maximum number of bins.
    alpha : float or None, default 0.05
        Significance level of the chi-square test with one degree of freedom.
        Adjacent bins are merged until every pair differs at this level. Pass
        None to merge down to ``max_bins`` only.
    initial_bins : int, default 20
        Number of quantile bins to start from. Tied quantiles can reduce it.

    Returns
    -------
    numpy.ndarray
        Increasing edges with infinite endpoints, suitable for pandas.cut.

    See Also
    --------
    tree_edges : Split the values recursively instead of merging.
    monotonic_edges : Merge until the WOE is monotonic.

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
        raise ValueError("alpha must be None or between 0 and 1")
    observed, labels = _labeled(values, target)
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
    values: pd.Series,
    target: pd.Series,
    *,
    max_bins: int = 5,
    min_bin_share: float = 0.05,
    criterion: Literal["entropy", "gini"] = "entropy",
) -> np.ndarray:
    """Fit edges with the splits of a decision tree on one variable.

    The tree grows best first: of all current bins, the one whose best split
    most reduces the total impurity is split next, until there are
    ``max_bins`` bins or no split reduces the impurity while leaving at
    least ``min_bin_share`` of the observations on each side.

    Parameters
    ----------
    values : pandas.Series
        Numeric values. Missing values and their labels are omitted.
    target : pandas.Series
        Binary labels on the same index, with 1 indicating the event.
    max_bins : int, default 5
        Maximum number of bins, the leaves of the tree.
    min_bin_share : float, default 0.05
        Minimum share of the observed values in every bin, from 0 to 0.5.
    criterion : {"entropy", "gini"}, default "entropy"
        Impurity measure: entropy for information gain, or the Gini index.

    Returns
    -------
    numpy.ndarray
        Increasing edges with infinite endpoints, suitable for pandas.cut.
        Every interior edge is an observed value, the largest one in the bin
        to its left.

    See Also
    --------
    chimerge_edges : Merge fine bins instead of splitting.
    monotonic_edges : Merge until the WOE is monotonic.

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
        raise ValueError("min_bin_share must be between 0 and 0.5")
    if criterion not in ("entropy", "gini"):
        raise ValueError("criterion must be 'entropy' or 'gini'")
    observed, labels = _labeled(values, target)
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
    values: pd.Series,
    target: pd.Series,
    *,
    max_bins: int | None = 5,
    direction: Literal["auto", "increasing", "decreasing"] = "auto",
    initial_bins: int = 20,
    smoothing: float = 0.5,
) -> np.ndarray:
    """Fit edges by merging adjacent quantile bins until the WOE is monotonic.

    The values are first cut into ``initial_bins`` quantile bins, or into one
    bin per distinct value if there are no more distinct values than that.
    While two adjacent bins break the trend, the first such pair is merged and the WOE
    is recomputed (pool adjacent violators), so the WOE of the final bins is
    strictly increasing or strictly decreasing. While more than ``max_bins``
    bins remain, the adjacent pair with the smallest chi-square statistic is
    merged as in :func:`chimerge_edges`, and the trend is checked again.

    Parameters
    ----------
    values : pandas.Series
        Numeric values. Missing values and their labels are omitted.
    target : pandas.Series
        Binary labels on the same index, with 1 indicating the event.
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

    Returns
    -------
    numpy.ndarray
        Increasing edges with infinite endpoints, suitable for pandas.cut.

    See Also
    --------
    chimerge_edges : Merge fine bins by a chi-square test.
    tree_edges : Split the values recursively.

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
        raise ValueError("direction must be 'auto', 'increasing' or 'decreasing'")
    observed, labels = _labeled(values, target)
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
        raise ValueError(f"{name} must be a positive integer")


def _observed(values: pd.Series) -> np.ndarray:
    observed = pd.to_numeric(values.dropna(), errors="raise").to_numpy(dtype=float)
    if not len(observed) or not np.isfinite(observed).all():
        raise ValueError("reference values must contain finite observations only")
    return observed


def _labeled(values: pd.Series, target: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if not values.index.equals(target.index):
        raise ValueError("values and target must have identical indices")
    if not target.isin([0, 1]).all():
        raise ValueError("target must contain only 0 and 1 and no missing labels")
    observed = _observed(values)
    labels = target[values.notna()].to_numpy(dtype=np.intp)
    if labels.min() == labels.max():
        raise ValueError("target must contain both 0 and 1 where values are observed")
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
