"""Numeric bin boundaries."""

from numbers import Integral
from typing import Literal

import numpy as np
import pandas as pd


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
    """
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 1:
        raise ValueError("bins must be a positive integer")
    if method not in ("quantile", "uniform"):
        raise ValueError("method must be 'quantile' or 'uniform'")
    observed = pd.to_numeric(values.dropna(), errors="raise").to_numpy(dtype=float)
    if not len(observed) or not np.isfinite(observed).all():
        raise ValueError("reference values must contain finite observations only")
    if method == "quantile":
        edges = np.quantile(observed, np.linspace(0, 1, bins + 1))
    else:
        edges = np.linspace(observed.min(), observed.max(), bins + 1)
    return np.r_[-np.inf, np.unique(edges)[1:-1], np.inf]
