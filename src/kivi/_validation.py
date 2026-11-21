"""Argument checks shared by the public functions."""

import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral, Real
from typing import Any, TypeAlias, TypeGuard

import numpy as np
import numpy.typing as npt
import pandas as pd

#: A NumPy or pandas dtype, as ``Series.dtype`` returns it.
DTypeObj: TypeAlias = np.dtype[Any] | pd.api.extensions.ExtensionDtype


def is_real(value: object) -> TypeGuard[float]:
    """Return ``isinstance(value, numbers.Real)``; type checkers read a True result as float.

    mypy does not know that ``int``, ``float`` and the NumPy number types are
    registered as `numbers.Real`, so after ``isinstance(value, Real)`` on a value
    annotated ``float`` it would treat the rest of the function as unreachable
    and leave it unchecked.
    """
    return isinstance(value, Real)


def is_integral(value: object) -> TypeGuard[int]:
    """Return ``isinstance(value, numbers.Integral)``; type checkers read True as int.

    The counterpart of `is_real` for `numbers.Integral`.
    """
    return isinstance(value, Integral)


def as_series(values: npt.ArrayLike, name: str) -> pd.Series:
    """Return a one-dimensional array_like as a Series without copying a Series."""
    if isinstance(values, pd.Series):
        return values
    if np.ndim(values) != 1:
        raise ValueError(f"{name} must be one-dimensional, got {np.ndim(values)} dimensions")
    return pd.Series(values)


def is_real_dtype(dtype: DTypeObj) -> bool:
    """Whether a NumPy or pandas dtype holds real numbers other than booleans."""
    return (
        pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
        and not pd.api.types.is_complex_dtype(dtype)
    )


def check_real_dtype(dtype: DTypeObj, name: str, *, booleans: bool = False) -> None:
    """Require a dtype of real numbers; with ``booleans``, also a boolean dtype."""
    if not (is_real_dtype(dtype) or (booleans and pd.api.types.is_bool_dtype(dtype))):
        raise ValueError(f"{name} must be real and numeric, got dtype {dtype}")


def as_real(values: npt.ArrayLike, name: str, *, booleans: bool = False) -> npt.NDArray[np.float64]:
    """Float array of real numeric values, keeping the shape; missing values become NaN.

    With ``booleans``, True and False count as 1 and 0, as probabilities do in
    `kivi.metrics` and `kivi.calibration`. An empty sequence gives an empty array.
    """
    if np.ndim(values) == 1 and not len(values):  # type: ignore[arg-type]  # one-dimensional
        return np.empty(0)
    if not isinstance(values, pd.Series) and np.ndim(values) == 1:
        # pandas reads [0.1, None] and nullable arrays as numbers, where NumPy keeps objects.
        values = pd.Series(values)
    if isinstance(values, pd.Series):
        check_real_dtype(values.dtype, name, booleans=booleans)
        return values.to_numpy(dtype=float, na_value=np.nan)
    array = np.asarray(values)
    check_real_dtype(array.dtype, name, booleans=booleans)
    return array.astype(float)


def finite_scores(values: pd.Series, name: str) -> npt.NDArray[np.float64]:
    """Finite float scores; booleans are accepted, as in ``kivi.metrics``."""
    dtype = values.dtype
    if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_complex_dtype(dtype):
        raise ValueError(f"{name} must be real and numeric, got dtype {dtype}")
    array = values.to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(array)
    if not finite.all():
        raise ValueError(
            f"{name} must be finite, got {(~finite).sum()} missing or infinite value(s) "
            f"such as {float(array[~finite][0])!r}"
        )
    return array


def as_flags(values: pd.Series, name: str) -> npt.NDArray[np.bool_]:
    """0/1 or boolean values as a bool array."""
    # For nullable dtypes pandas 2.0 returns an object array, which ~ would negate as integers.
    invalid = ~values.isin([0, 1]).to_numpy(dtype=bool)
    if invalid.any():
        raise ValueError(
            f"{name} must contain only booleans or 0 and 1, got {first_value(values[invalid])!r}"
        )
    return values.to_numpy(dtype=bool)


def known_outcomes(
    target: npt.ArrayLike, length: int, of: str
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Boolean masks of the known outcomes and of the events; missing labels are unknown.

    ``target`` needs one label per value of the argument ``of``, which has ``length`` values.
    """
    target = as_series(target, "target")
    if len(target) != length:
        raise ValueError(
            f"target must have one label per value of {of}, got {len(target)} for {length}"
        )
    missing = target.isna().to_numpy(dtype=bool)
    # For nullable dtypes pandas 2.0 returns an object array, which ~ would negate as integers.
    invalid = ~(target.isin([0, 1]).to_numpy(dtype=bool) | missing)
    if invalid.any():
        shown = first_value(target[invalid])
        raise ValueError(f"target must contain only 0, 1 and missing values, got {shown!r}")
    event = np.zeros(length, dtype=bool)
    event[~missing] = target[~missing].to_numpy(dtype=float) == 1
    return ~missing, event


def shared_index(inputs: Mapping[str, object], *, frames: bool = False) -> pd.Index | None:
    """Return the index that the Series among ``inputs`` share, or None without a Series.

    ``inputs`` maps argument names to the arguments that describe the same
    observations. Every Series must have the index of the first one; other
    inputs are matched by position. With ``frames``, DataFrames count too.
    """
    kinds = (pd.Series, pd.DataFrame) if frames else pd.Series
    indexed = [(name, values) for name, values in inputs.items() if isinstance(values, kinds)]
    for name, values in indexed[1:]:
        if not values.index.equals(indexed[0][1].index):
            raise ValueError(f"{indexed[0][0]} and {name} must have identical indices")
    return indexed[0][1].index if indexed else None


def check_smoothing(smoothing: float) -> None:
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError(f"smoothing must be finite and positive, got {smoothing!r}")


def as_count_columns(
    first: npt.ArrayLike, second: npt.ArrayLike, names: tuple[str, str]
) -> npt.NDArray[np.float64]:
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


def as_weights(
    weights: npt.ArrayLike | None,
    values: npt.ArrayLike,
    name: str = "weights",
    of: str = "values",
    *,
    others: Mapping[str, object] | None = None,
) -> npt.NDArray[np.float64] | None:
    """Return observation weights as a float array matched to ``values``, or None.

    ``values`` is the argument as the caller passed it, after it has been
    validated. ``others`` maps the names of the other arguments that describe
    the same observations, such as the target, to the arguments as passed. A
    Series of weights must have the index of every Series among them;
    otherwise the weights are matched by position, as the target is in
    ``woe_table``.
    """
    if weights is None:
        return None
    if isinstance(weights, pd.Series):
        if isinstance(values, pd.Series) and not values.index.equals(weights.index):
            raise ValueError(f"{name} must have the same index as {of}")
        for other, data in (others or {}).items():
            if isinstance(data, pd.Series) and not data.index.equals(weights.index):
                raise ValueError(f"{other} and {name} must have identical indices")
    series = as_series(weights, name)
    dtype = series.dtype
    if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
        raise ValueError(f"{name} must be numeric, got dtype {dtype}")
    array = series.to_numpy(dtype=float, na_value=np.nan)
    count = len(values)  # type: ignore[arg-type]  # one-dimensional: validated by the caller
    if len(array) != count:
        raise ValueError(
            f"{name} must have one weight per value of {of}, got {len(array)} for {count}"
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
    raw_score, raw_target = score, target
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
    mass = as_weights(weights, raw_score, of=name, others={"target": raw_target})
    if both_classes and mass is not None:
        totals = np.bincount(labels, weights=mass, minlength=2)
        if not (totals > 0).all():
            raise ValueError(
                "weights must give both target classes a positive total, got "
                f"{float(totals[0])!r} for 0 and {float(totals[1])!r} for 1"
            )
    return values, labels, mass


def first_value(values: object) -> object:
    """First value of a Series or Index, or ``values`` itself, as a Python scalar.

    For error messages, which show NumPy scalars as plain numbers this way.
    """
    if isinstance(values, pd.Series):
        values = values.iloc[0]
    elif isinstance(values, pd.Index):
        values = values[0]
    return values.item() if isinstance(values, np.generic) else values


def like_input(
    template: npt.ArrayLike, values: npt.NDArray[np.float64]
) -> float | pd.Series | npt.NDArray[np.float64]:
    """Return values shaped like the input: a Series, a float or an array."""
    if isinstance(template, pd.Series):
        return pd.Series(values, index=template.index, name=template.name)
    return float(values) if values.ndim == 0 else values


def woe_values(table: pd.DataFrame | pd.Series) -> pd.Series:
    """Return the WOE by group of a `kivi.woe.woe_table` result or of a Series of WOE values."""
    if isinstance(table, pd.DataFrame):
        if "woe" not in table.columns:
            raise ValueError("table must have a woe column, as returned by woe_table")
        table = table["woe"]
    elif not isinstance(table, pd.Series):
        raise ValueError(
            f"table must be a woe_table result or a Series, got {type(table).__name__}"
        )
    if table.index.has_duplicates or pd.isna(table.index).sum() > 1:
        raise ValueError("table must have one row per group")
    return table


def is_binned_dtype(dtype: DTypeObj) -> bool:
    """Whether a column of this dtype is binned: numeric dtypes other than bool."""
    return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)


def fitted_columns(frame: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    """Return the names of the columns of ``frame`` to use: ``columns``, or every column."""
    if columns is None:
        names = list(frame.columns)
        if not names:
            raise ValueError("frame must have at least one column to fit")
        for name in names:
            if not isinstance(name, str):
                raise ValueError(f"column names must be strings, got {name!r}")
    else:
        names = column_list(columns, "columns")
        absent = [name for name in names if name not in frame.columns]
        if absent:
            raise ValueError(f"columns must be columns of frame, got {shown_list(absent)}")
    repeated = list(dict.fromkeys(name for name in names if (frame.columns == name).sum() > 1))
    if repeated:
        raise ValueError(f"frame has more than one column named {shown_list(repeated)}")
    return names


def column_list(columns: Sequence[str], argument: str) -> list[str]:
    """Return a sequence of distinct column names, the argument ``argument``, as a list."""
    if isinstance(columns, str) or not isinstance(columns, (Sequence, pd.Index, np.ndarray)):
        raise ValueError(
            f"{argument} must be a sequence of column names, got {type(columns).__name__}"
        )
    names = list(columns)
    if not names:
        raise ValueError(f"{argument} must name at least one column")
    for name in names:
        if not isinstance(name, str):
            raise ValueError(f"column names must be strings, got {name!r}")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"{argument} must not repeat a column, got {shown_list(repeated)}")
    return [str(name) for name in names]


def checked_codes(value: npt.ArrayLike, argument: str) -> tuple[float, ...]:
    """Return the special codes of a numeric column as a tuple of distinct Python numbers."""
    if isinstance(value, str):
        raise ValueError(f"{argument} must be a list of codes, got {value!r}")
    codes = []
    for code in pd.unique(as_series(value, argument)):
        # A string never matches a value of a numeric column, and True would match 1 only
        # sometimes, so neither is a valid code.
        if isinstance(code, (bool, np.bool_)) or not isinstance(code, Real) or math.isnan(code):
            raise ValueError(f"{argument} must be non-missing numbers, got {shown(code)}")
        codes.append(int(code) if isinstance(code, Integral) else float(code))
    return tuple(codes)


def check_finite_column(values: pd.Series, codes: tuple[float, ...], name: str) -> None:
    """Require the values of the numeric column ``name`` to be finite, or special codes."""
    # pandas.cut would leave -inf in no interval and count it as missing.
    infinite = np.isinf(values.to_numpy(dtype=float, na_value=np.nan))
    if codes:
        infinite = infinite & ~values.isin(list(codes)).to_numpy(dtype=bool)
    if infinite.any():
        raise ValueError(
            f"column {name!r} has infinite values that are not special codes; pass infinite "
            "codes in special"
        )


def shown(value: object) -> str:
    """Return the repr of a value for an error message, showing NumPy scalars as Python ones."""
    if isinstance(value, (np.generic,)):
        value = value.item()
    return repr(value)


def shown_list(values: Iterable[object]) -> str:
    """Return the values as a list for an error message, each shown like `shown`."""
    return "[" + ", ".join(shown(value) for value in values) + "]"
