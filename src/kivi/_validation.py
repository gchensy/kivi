"""Argument checks shared by the public functions."""

import numpy as np
import pandas as pd


def as_series(values, name: str) -> pd.Series:
    """Return a one-dimensional array_like as a Series without copying a Series."""
    if isinstance(values, pd.Series):
        return values
    if np.ndim(values) != 1:
        raise ValueError(f"{name} must be one-dimensional, got {np.ndim(values)} dimensions")
    return pd.Series(values)


def check_smoothing(smoothing: float) -> None:
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError(f"smoothing must be finite and positive, got {smoothing!r}")
