"""Special codes: values such as ``-1`` that get a group of their own.

``kivi.binning``, ``kivi.spec`` and ``kivi.univariate`` share these rules:
codes are matched by equality, values that are not numbers also after
``pandas.to_numeric``, and booleans are never codes because ``True == 1``.
"""

import math
import numbers

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import _validation


def special_values(special: npt.ArrayLike | None) -> pd.Index:
    """Return the distinct special codes as an object Index, empty for None."""
    if special is None:
        return pd.Index([], dtype=object)
    codes = _validation.as_series(special, "special")
    # Before pd.unique, which would merge True into 1 and False into 0.
    flags = [value for value in codes if isinstance(value, (bool, np.bool_))]
    if flags or pd.api.types.is_bool_dtype(codes.dtype):
        raise ValueError(
            f"special must not contain booleans, got {_validation.shown((flags or [True])[0])}"
        )
    special = pd.Index(pd.unique(codes), dtype=object)
    if special.hasnans:
        raise ValueError(
            "special must not contain missing values; they always form their own group"
        )
    return special


def special_positions(values: pd.Series, special: pd.Index) -> np.ndarray:
    """Position of each value's code in ``special``, or -1 for other values.

    Values that are not numbers are matched both as given and after
    ``pandas.to_numeric``, so the string ``"-1"`` matches the code ``-1``.
    """
    position = np.full(len(values), -1, dtype=np.intp)
    if special.empty or pd.api.types.is_bool_dtype(values.dtype):
        # Booleans are not numbers here, although True == 1 and False == 0.
        return position
    observed = values.notna().to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(values.dtype):
        # Fast path: look the numbers up among the numeric codes only. Integer
        # values compare with integer codes, so 2**53 + 1 never matches 2.0**53.
        integer = pd.api.types.is_integer_dtype(values.dtype)
        numeric: list[int] | pd.Series
        numeric, codes = [], []
        for i, code in enumerate(special):
            if not isinstance(code, numbers.Real):
                continue
            if integer and not isinstance(code, numbers.Integral):
                if not (math.isfinite(code) and float(code).is_integer()):
                    continue
                code = int(code)  # type: ignore[call-overload]  # float-like codes define __int__
            numeric.append(i)
            codes.append(code)
        if numeric:
            found = pd.Index(codes).get_indexer(values[observed])
            position[observed] = np.where(found >= 0, np.asarray(numeric)[found], -1)
        return position
    candidates = pd.Index(values[observed].to_numpy(dtype=object), dtype=object)
    found = special.get_indexer(candidates)
    unmatched = found < 0
    numeric = pd.to_numeric(pd.Series(candidates[unmatched]), errors="coerce")
    found[unmatched] = special.get_indexer(pd.Index(numeric.to_numpy(dtype=object)))
    position[observed] = found
    return position
