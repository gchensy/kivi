"""Numeric bin boundaries and bin assignment."""

import warnings
from numbers import Integral
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._validation import as_series

__all__ = ["bin_edges", "cut"]


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
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 1:
        raise ValueError(f"bins must be a positive integer, got {bins!r}")
    if method not in ("quantile", "uniform"):
        raise ValueError(f"method must be 'quantile' or 'uniform', got {method!r}")
    reference = as_series(values, "values")
    reference = reference[~reference.isin(_special_values(special))]
    observed = pd.to_numeric(reference.dropna(), errors="raise").to_numpy(dtype=float)
    if not len(observed):
        raise ValueError(
            "reference values have no observations apart from missing and special values"
        )
    if not np.isfinite(observed).all():
        raise ValueError("reference values must be finite; pass infinite codes in special")
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


def _special_values(special: npt.ArrayLike | None) -> pd.Index:
    if special is None:
        return pd.Index([], dtype=object)
    special = pd.Index(pd.unique(as_series(special, "special")))
    if special.hasnans:
        raise ValueError(
            "special must not contain missing values; they always form their own group"
        )
    return special
