"""Argument checks shared by the public functions."""

import numpy as np
import numpy.typing as npt
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


def as_weights(weights, values, name: str = "weights", of: str = "values") -> np.ndarray | None:
    """Return observation weights as a float array matched to ``values``, or None.

    ``values`` is the argument as the caller passed it, after it has been
    validated. Two Series must have identical indices; otherwise the weights
    are matched by position, as the target is in ``woe_table``.
    """
    if weights is None:
        return None
    if isinstance(values, pd.Series) and isinstance(weights, pd.Series):
        if not values.index.equals(weights.index):
            raise ValueError(f"{name} must have the same index as {of}")
    series = as_series(weights, name)
    dtype = series.dtype
    if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
        raise ValueError(f"{name} must be numeric, got dtype {dtype}")
    array = series.to_numpy(dtype=float, na_value=np.nan)
    if len(array) != len(values):
        raise ValueError(
            f"{name} must have one weight per value of {of}, got {len(array)} for {len(values)}"
        )
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    with np.errstate(over="ignore"):
        total = array.sum()
    if not (np.isfinite(total) and total > 0):
        raise ValueError(f"{name} must have a finite, positive total, got {float(total)!r}")
    return array


def scores_and_labels(
    score: npt.ArrayLike,
    target: npt.ArrayLike,
    name: str,
    *,
    both_classes: bool = True,
    weights: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp], npt.NDArray[np.float64] | None]:
    """Finite float scores, 0/1 labels and weights, matched by index or position like ``woe_table``.

    The weights are None without `weights`. With `both_classes`, both classes
    must be present and, with `weights`, have a positive total weight.
    """
    if isinstance(score, pd.Series) and isinstance(target, pd.Series):
        if not score.index.equals(target.index):
            raise ValueError(f"{name} and target must have identical indices")
    raw_score = score
    score, target = as_series(score, name), as_series(target, "target")
    if len(score) != len(target):
        raise ValueError(
            f"{name} and target must have the same length, got {len(score)} and {len(target)}"
        )
    if not len(score):
        raise ValueError(f"{name} and target must not be empty")
    if not pd.api.types.is_numeric_dtype(score.dtype):
        raise ValueError(f"{name} must be numeric, got dtype {score.dtype}")
    if pd.api.types.is_complex_dtype(score.dtype):
        raise ValueError(f"{name} must be real, got dtype {score.dtype}")
    values = score.to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError(
            f"{name} must be finite, got {(~finite).sum()} missing or infinite value(s) "
            f"such as {float(values[~finite][0])!r}"
        )
    # For nullable dtypes pandas 2.0 returns an object array, which ~ would negate as integers.
    invalid = ~target.isin([0, 1]).to_numpy(dtype=bool)
    if invalid.any():
        raise ValueError(f"target must contain only 0 and 1, got {first_value(target[invalid])!r}")
    labels = target.to_numpy(dtype=np.intp)
    if both_classes and labels.sum() in (0, len(labels)):
        raise ValueError(f"target must contain both 0 and 1, got only {labels[0]}")
    mass = as_weights(weights, raw_score, of=name)
    if both_classes and mass is not None:
        totals = np.bincount(labels, weights=mass, minlength=2)
        if not (totals > 0).all():
            raise ValueError(
                "weights must give both target classes a positive total, got "
                f"{float(totals[0])!r} for 0 and {float(totals[1])!r} for 1"
            )
    return values, labels, mass


def first_value(values: pd.Series):
    """First value of ``values`` as a Python scalar, for error messages."""
    value = values.iloc[0]
    return value.item() if isinstance(value, np.generic) else value


def like_input(template, values: np.ndarray) -> float | pd.Series | np.ndarray:
    """Return values shaped like the input: a Series, a float or an array."""
    if isinstance(template, pd.Series):
        return pd.Series(values, index=template.index, name=template.name)
    return float(values) if values.ndim == 0 else values
