"""Numeric bin boundaries."""

import warnings
from numbers import Integral
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._validation import as_series

__all__ = ["bin_edges"]


def bin_edges(
    values: npt.ArrayLike, bins: int = 5, method: Literal["quantile", "uniform"] = "quantile"
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

    Returns
    -------
    numpy.ndarray
        Increasing edges with infinite endpoints, suitable for pandas.cut.

    Raises
    ------
    ValueError
        If `bins` is not a positive integer, `method` is not recognized,
        `values` is not one-dimensional or not numeric, or the reference has
        no observations or contains an infinite value.

    Warns
    -----
    UserWarning
        If tied edges leave fewer bins than requested, for example when a
        single value accounts for a large share of the reference.

    See Also
    --------
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
    """
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 1:
        raise ValueError(f"bins must be a positive integer, got {bins!r}")
    if method not in ("quantile", "uniform"):
        raise ValueError(f"method must be 'quantile' or 'uniform', got {method!r}")
    reference = as_series(values, "values")
    observed = pd.to_numeric(reference.dropna(), errors="raise").to_numpy(dtype=float)
    if not len(observed) or not np.isfinite(observed).all():
        raise ValueError("reference values must contain finite observations only")
    if method == "quantile":
        edges = np.quantile(observed, np.linspace(0, 1, bins + 1))
    else:
        edges = np.linspace(observed.min(), observed.max(), bins + 1)
    edges = np.r_[-np.inf, np.unique(edges)[1:-1], np.inf]
    if len(edges) - 1 < bins:
        warnings.warn(
            f"tied edges leave {len(edges) - 1} of the {bins} requested bins",
            UserWarning,
            stacklevel=2,
        )
    return edges
