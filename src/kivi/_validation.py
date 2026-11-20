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


def as_count_columns(first, second, names: tuple[str, str]) -> np.ndarray:
    """Stack two one-dimensional count arrays into the columns of a float array."""
    columns = []
    for values, name in zip((first, second), names, strict=True):
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain numeric counts") from error
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got {array.ndim} dimensions")
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError(f"{name} must contain finite, non-negative counts")
        if not array.sum() > 0:
            raise ValueError(f"{name} must contain a positive count")
        columns.append(array)
    if len(columns[0]) != len(columns[1]):
        raise ValueError(
            f"{names[0]} and {names[1]} must have the same length, "
            f"got {len(columns[0])} and {len(columns[1])}"
        )
    return np.column_stack(columns)


def as_weights(weights, values, name: str = "weights") -> np.ndarray | None:
    """Return observation weights as a float array matched to ``values``, or None.

    Two Series must have identical indices; otherwise the weights are matched
    by position, as the target is in ``woe_table``.
    """
    if weights is None:
        return None
    if isinstance(values, pd.Series) and isinstance(weights, pd.Series):
        if not values.index.equals(weights.index):
            raise ValueError(f"{name} must have the same index as the values it weights")
    try:
        array = as_series(weights, name).to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        if "one-dimensional" in str(error):
            raise
        raise ValueError(f"{name} must be numeric") from error
    if len(array) != len(values):
        raise ValueError(
            f"{name} must have one weight per value, got {len(array)} for {len(values)} values"
        )
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    if not array.sum() > 0:
        raise ValueError(f"{name} must have a positive total")
    return array
