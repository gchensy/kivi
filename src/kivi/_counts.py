"""Counting and smoothing kernels shared by the metric functions.

The public functions validate their arguments and then call these kernels,
which work on NumPy arrays with one row per group and two count columns.
"""

import numpy as np
import pandas as pd

from ._validation import check_smoothing


def group_counts(
    groups: pd.Series, side: np.ndarray, weights: np.ndarray | None = None
) -> pd.DataFrame:
    """Count observations per group for the two sides coded 0 and 1 in ``side``.

    Categorical groups follow category order with missing values last; other
    groups follow first appearance. With ``weights`` the counts are sums of
    weights, and groups whose weights sum to zero are left out.
    """
    ordered = isinstance(groups.dtype, pd.CategoricalDtype)
    codes, levels = pd.factorize(groups, sort=ordered, use_na_sentinel=False)
    counts = np.bincount(2 * codes + side, weights=weights, minlength=2 * len(levels))
    counts = counts.reshape(-1, 2)
    if weights is not None:
        # Filter the arrays, not a DataFrame: indexing a frame by interval categories hashes
        # them, which warns on NumPy < 2.2.
        keep = counts.sum(axis=1) > 0
        counts, levels = counts[keep], levels[keep]
    return pd.DataFrame(counts, index=pd.Index(levels, name="bin"))


def group_rows(groups: pd.Series, levels: pd.Index) -> np.ndarray:
    """Return the row of `levels`, from `group_counts`, of each observation, or -1 if absent."""
    if isinstance(levels, pd.CategoricalIndex):
        # Map category codes to rows instead of looking up the labels: that hashes
        # interval categories, which warns on NumPy < 2.2. Code -1 is a missing value.
        row_of_code = np.full(len(levels.categories) + 1, -1, dtype=np.intp)
        row_of_code[levels.codes] = np.arange(len(levels))
        return row_of_code[groups.cat.codes.to_numpy()]
    rows = levels.get_indexer(groups)
    missing = groups.isna().to_numpy(dtype=bool)
    missing_row = np.flatnonzero(pd.isna(levels))
    if missing.any():  # get_indexer does not match missing values on pandas 2.0
        rows[missing] = missing_row[0] if len(missing_row) else -1
    return rows


def shares(counts: np.ndarray, smoothing: float) -> np.ndarray:
    """Replace zero counts with ``smoothing``, then normalize each column."""
    check_smoothing(smoothing)
    counts = np.asarray(counts, dtype=float)
    adjusted = np.where(counts == 0, smoothing, counts)
    return adjusted / adjusted.sum(axis=0)


def divergence(counts: np.ndarray, smoothing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shares, log ratios and divergence terms of two count columns.

    For shares ``a`` (first column) and ``b`` (second column) of each row, the
    log ratio is ``ln(b / a)`` and the term is ``(b - a) * ln(b / a)``. With
    (non-event, event) counts these are the WOE and IV of each group; with
    (reference, current) counts the terms are the PSI contributions.
    """
    share = shares(counts, smoothing)
    log_ratio = np.log(share[:, 1] / share[:, 0])
    return share, log_ratio, (share[:, 1] - share[:, 0]) * log_ratio
