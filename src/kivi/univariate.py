"""Univariate profile of the columns of a data frame.

``profile_table`` summarizes every column in one row: its completeness, its
distribution and, with a binary target, how strongly it separates events
from non-events. It is the first look at the candidates of a model, before
any of them is binned with care.
"""

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import binning, metrics, woe
from ._validation import (
    as_series,
    as_weights,
    check_finite_column,
    checked_codes,
    fitted_columns,
    is_binned_dtype,
    shown_list,
)

__all__ = ["profile_table"]

_QUANTILES = {"q01": 0.01, "q50": 0.5, "q99": 0.99}
_COUNTS = ["count", "missing", "special"]


def profile_table(
    frame: pd.DataFrame,
    target: npt.ArrayLike | None = None,
    *,
    columns: Sequence[str] | None = None,
    bins: int = 5,
    special: Mapping[str, npt.ArrayLike] | None = None,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Summarize each column of a data frame in one row.

    Every column gets its kind, its number of missing values and its most
    frequent value; numeric columns also get their range, quantiles, mean
    and standard deviation. With a binary target, each column also gets the
    event rates of its missing and non-missing rows, the information value
    of a quick binning and, for numeric columns, the Gini coefficient of its
    values.

    Parameters
    ----------
    frame : pandas.DataFrame
        One column per variable. Columns with a numeric dtype other than
        bool are numeric; all others, such as object, string, category or
        bool, are categorical, as in `kivi.spec.fit_spec`.
    target : array_like, optional
        Binary labels, with 1 indicating the event, one per row of `frame`.
        Both classes must be present. A Series must have the same index as
        `frame`; other inputs are matched by position. Without `target` the
        table describes the columns only.
    columns : sequence of str, optional
        Columns to profile, in the order of the rows. By default every
        column of `frame`.
    bins : int, default 5
        Number of quantile bins of the quick binning of numeric columns,
        from `kivi.binning.bin_edges`, for the ``iv`` column.
    special : dict, optional
        Special codes of numeric columns, such as ``{"pdays": [-1]}``, as in
        `kivi.spec.fit_spec`. They count as values, not as missing, but are
        left out of the numeric summary and of the Gini coefficient, and
        each code forms its own group in the quick binning.
    weights : array_like, optional
        Non-negative frequency weight of each row, matched like `target`,
        such as the inverse of its sampling rate. Counts become sums of
        weights and every statistic is weighted, so integer weights give the
        same result as repeating rows with unit weights. Rows with zero
        weight count for nothing, so their values are not checked.

    Returns
    -------
    pandas.DataFrame
        One row per column, indexed by the column names in an index named
        ``variable``. The columns are described in the Notes.

    Raises
    ------
    ValueError
        If `frame` is not a DataFrame or has no rows or no columns,
        `columns` repeats a column or names one that `frame` lacks, a column
        name is not a string, or `bins` is not a positive integer. If
        `target` does not give each row a label 0 or 1, lacks one of the two
        classes or is a column being profiled, or if `weights` is invalid or
        leaves a class without weight. Also if `special` is not a dict by
        column, names a column that is not profiled or is categorical, or has
        a code that is not a number, or if a numeric column has infinite
        values that are not special codes.

    See Also
    --------
    kivi.spec.fit_spec : Bin and encode the columns chosen after profiling.
    kivi.woe.woe_table : The WOE and IV of one grouped column.
    kivi.metrics.gini : The Gini coefficient of a score.
    kivi.binning.bin_edges : The quantile edges of the quick binning.

    Notes
    -----
    Every column gets

    * ``kind``: ``"numeric"`` or ``"categorical"``;
    * ``count`` and ``missing``: the number of rows with and without a
      value, and ``missing_share``, the share of rows without one;
    * ``distinct``: the number of different values;
    * ``top`` and ``top_share``: the most frequent value and its share of
      the rows with a value. Of values that are equally frequent, the first
      in the order of the categories, or else of appearance, is shown;
    * ``special`` and ``special_share``: the number and share of rows with a
      special code, 0 for categorical columns;
    * ``mean``, ``std``, ``min``, ``q01``, ``q50``, ``q99`` and ``max``:
      the mean, the sample standard deviation, the smallest value, the 1%,
      50% and 99% quantiles and the largest value of a numeric column,
      without missing values and special codes. Categorical columns get NaN.

    The quantiles interpolate linearly, as ``numpy.quantile`` does; with
    `weights` they are weighted quantiles by the inverted CDF, as the edges
    of `kivi.binning.bin_edges`. Weights, even unit weights, therefore
    change the quantiles and, through the edges, ``iv`` and ``tied_bins``;
    with integer weights every other column equals that of the repeated
    rows without weights. The
    standard deviation divides by the total weight minus 1, as it divides
    by the number of rows minus 1 without weights, so it is NaN when the
    weights of a column's values sum to 1 or less.

    With `target`, every column also gets

    * ``event_rate``: the event rate of all rows, the same in every row of
      the table;
    * ``missing_event_rate`` and ``non_missing_event_rate``: the event
      rates of the rows without and with a value, NaN when there are no
      such rows;
    * ``missing_lift`` and ``non_missing_lift``: these rates divided by
      ``event_rate``, so a lift above 1 marks riskier rows;
    * ``iv``: the information value of the column, summed over the groups
      of `kivi.woe.woe_table`. Numeric columns are cut into `bins` quantile
      bins with `kivi.binning.bin_edges` and `kivi.binning.cut`, with a group
      for each special code; categorical columns keep their labels. Missing
      values form a group of their own;
    * ``tied_bins``: True where the quick binning of a numeric column has
      fewer bins than requested, because tied quantiles were merged or no
      values are left to bin apart from missing values and special codes.
      The table records this instead of warning for every column, as
      `kivi.binning.bin_edges` does; such an IV understates the column until
      it is binned with custom edges or special codes;
    * ``gini``: the Gini coefficient of the values of a numeric column as a
      score, without missing values and special codes. It is positive when
      events have higher values, negative when they have lower ones, and NaN
      for categorical columns or when the values leave out a class.

    The quick binning gives a first ranking, not a final one: an IV from
    five quantile bins can miss a pattern that other edges would show, and
    a categorical column with many rare labels gets an inflated IV.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.univariate import profile_table
    >>> frame = pd.DataFrame({
    ...     "age": [23, 31, None, 52, 38, 27, 61, 49],
    ...     "housing": ["yes", "no", "yes", "no", "no", "yes", None, "yes"],
    ... })
    >>> profile = profile_table(frame)
    >>> profile[["kind", "count", "missing", "distinct", "top", "top_share"]]
                     kind  count  missing  distinct  top  top_share
    variable
    age           numeric      7        1         7  23.0   0.142857
    housing   categorical      7        1         2  yes   0.571429
    >>> profile.loc["age", ["min", "q50", "max"]].tolist()
    [23.0, 38.0, 61.0]

    With a target, the event rates and the separation of each column:

    >>> target = pd.Series([1, 0, 1, 0, 1, 1, 0, 0])
    >>> profile = profile_table(frame, target, bins=2)
    >>> profile[["missing_event_rate", "non_missing_event_rate", "iv", "gini"]].round(4)
              missing_event_rate  non_missing_event_rate      iv    gini
    variable
    age                      1.0                  0.4286  1.5607 -0.8333
    housing                  0.0                  0.5714  0.7466     NaN
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")
    names = fitted_columns(frame, columns)
    if not len(frame):
        raise ValueError("frame must have at least one row")
    binning._check_count(bins, "bins")
    labels = None if target is None else _target_labels(target, frame, names)
    mass = as_weights(weights, frame.index.to_series(), of="frame")
    if labels is not None and mass is not None:
        totals = np.bincount(labels, weights=mass, minlength=2)
        if not (totals > 0).all():
            raise ValueError(
                "weights must give both target classes a positive total, got "
                f"{float(totals[0])!r} for 0 and {float(totals[1])!r} for 1"
            )
    kinds = {
        name: "numeric" if is_binned_dtype(frame[name].dtype) else "categorical" for name in names
    }
    codes = _special_codes(special, kinds)
    weight = np.ones(len(frame)) if mass is None else mass
    rows = []
    for name in names:
        if kinds[name] == "numeric":
            if pd.api.types.is_complex_dtype(frame[name].dtype):
                raise ValueError(f"column {name!r} must be real, got dtype {frame[name].dtype}")
            # Rows without weight count for nothing, whatever value they hold.
            used = frame[name] if mass is None else frame[name][mass > 0]
            check_finite_column(used, codes.get(name, ()), name)
        rows.append(
            _profile(frame[name], kinds[name], codes.get(name, ()), labels, weight, mass, bins)
        )
    table = pd.DataFrame(rows, index=pd.Index(names, name="variable"))
    if mass is None:
        table[_COUNTS] = table[_COUNTS].astype(np.int64)
    return table


def _profile(
    values: pd.Series,
    kind: str,
    codes: tuple,
    labels: np.ndarray | None,
    weight: np.ndarray,
    mass: np.ndarray | None,
    bins: int,
) -> dict:
    """Return the row of one column; ``weight`` is ``mass``, or ones without weights."""
    missing = values.isna().to_numpy(dtype=bool)
    total = weight.sum()
    row = {"kind": kind, "count": weight[~missing].sum(), "missing": weight[missing].sum()}
    row["missing_share"] = row["missing"] / total
    row.update(_top(values, ~missing & (weight > 0), weight, row["count"]))
    numeric = kind == "numeric"
    special = binning._special_values(list(codes) or None)
    is_special = (
        binning._special_positions(values, special) >= 0 if numeric else np.zeros(len(values), bool)
    )
    row["special"] = weight[is_special].sum()
    row["special_share"] = row["special"] / total
    regular = ~missing & ~is_special & (weight > 0)
    x = values.to_numpy(dtype=float, na_value=np.nan)[regular] if numeric else np.empty(0)
    row.update(_summary(x, None if mass is None else weight[regular]))
    if labels is None:
        return row
    event_rate = _rate(labels, weight, np.ones(len(labels), dtype=bool))
    missing_rate = _rate(labels, weight, missing)
    present_rate = _rate(labels, weight, ~missing)
    row.update(
        event_rate=event_rate,
        missing_event_rate=missing_rate,
        non_missing_event_rate=present_rate,
        missing_lift=missing_rate / event_rate,
        non_missing_lift=present_rate / event_rate,
    )
    if numeric:
        if regular.any():
            with warnings.catch_warnings():
                # The table reports tied quantiles in tied_bins instead.
                warnings.filterwarnings("ignore", message="tied edges", category=UserWarning)
                edges = binning.bin_edges(values, bins, special=list(codes) or None, weights=mass)
        else:
            edges = np.array([-np.inf, np.inf])
        groups = binning.cut(values, edges, special=list(codes) or None)
        tied = len(edges) - 1 < bins
    else:
        groups, tied = values, False
    row["iv"] = float(woe.woe_table(groups, labels, weights=mass)["iv"].sum())
    row["tied_bins"] = tied
    row["gini"] = np.nan
    if numeric and len(np.unique(labels[regular])) == 2:
        row["gini"] = metrics.gini(
            x, labels[regular], weights=None if mass is None else weight[regular]
        )
    return row


def _top(values: pd.Series, present: np.ndarray, weight: np.ndarray, count: float) -> dict:
    """Count the distinct values and find the most frequent one with its share."""
    ordered = isinstance(values.dtype, pd.CategoricalDtype)
    codes, levels = pd.factorize(values[present], sort=ordered)
    if not len(levels):
        return {"distinct": 0, "top": np.nan, "top_share": np.nan}
    sums = np.bincount(codes, weights=weight[present], minlength=len(levels))
    top = levels[int(sums.argmax())]
    return {
        "distinct": len(levels),
        "top": top.item() if isinstance(top, np.generic) else top,
        "top_share": sums.max() / count,
    }


def _summary(x: np.ndarray, weight: np.ndarray | None) -> dict:
    """Mean, standard deviation, range and quantiles of the regular values of a column."""
    names = ["mean", "std", "min", *_QUANTILES, "max"]
    if not len(x):
        return dict.fromkeys(names, np.nan)
    probabilities = np.array(list(_QUANTILES.values()))
    if weight is None:
        mean = x.mean()
        dof = len(x) - 1
        squares = np.sum((x - mean) ** 2)
        quantiles = np.quantile(x, probabilities)
    else:
        mean = np.average(x, weights=weight)
        dof = weight.sum() - 1
        squares = np.sum(weight * (x - mean) ** 2)
        quantiles = binning._weighted_quantile(x, weight, probabilities)
    std = np.sqrt(squares / dof) if dof > 0 else np.nan
    return dict(
        zip(names, [mean, std, x.min(), *quantiles, x.max()], strict=True),
    )


def _rate(labels: np.ndarray, weight: np.ndarray, rows: np.ndarray) -> float:
    """Weighted event rate of ``rows``, or NaN when they have no weight."""
    total = weight[rows].sum()
    return float(np.dot(weight[rows], labels[rows]) / total) if total > 0 else np.nan


def _target_labels(target: npt.ArrayLike, frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    if isinstance(target, pd.Series):
        if target.name is not None and target.name in names:
            raise ValueError(
                f"columns include the target {target.name!r}; leave it out of the profiled columns"
            )
        if not target.index.equals(frame.index):
            raise ValueError("frame and target must have identical indices")
    series = as_series(target, "target")
    if len(series) != len(frame):
        raise ValueError(
            f"frame and target must have the same length, got {len(frame)} and {len(series)}"
        )
    if not series.isin([0, 1]).all() or series.nunique() != 2:
        raise ValueError("target must contain both 0 and 1 and no missing labels")
    return series.to_numpy(dtype=np.intp)


def _special_codes(special: Mapping | None, kinds: dict[str, str]) -> dict[str, tuple]:
    """Check the special codes by column and return them as tuples of numbers."""
    if special is None:
        return {}
    if not isinstance(special, Mapping):
        raise ValueError(f"special must be a dict by column, got {type(special).__name__}")
    unknown = [name for name in special if name not in kinds]
    if unknown:
        raise ValueError(f"special names columns that are not profiled: {shown_list(unknown)}")
    categorical = [name for name in special if kinds[name] != "numeric"]
    if categorical:
        raise ValueError(
            f"special names categorical columns, which keep their labels: {shown_list(categorical)}"
        )
    return {
        name: checked_codes(value, f"special of column {name!r}") for name, value in special.items()
    }
