"""Counting and smoothing kernels shared by the metric functions.

The public functions validate their arguments and then call these kernels,
which work on NumPy arrays with one row per group and two count columns.
"""

import numpy as np
import pandas as pd

from ._validation import check_smoothing


def group_counts(groups: pd.Series, side: np.ndarray) -> pd.DataFrame:
    """Count observations per group for the two sides coded 0 and 1 in ``side``.

    Categorical groups follow category order with missing values last; other
    groups follow first appearance.
    """
    ordered = isinstance(groups.dtype, pd.CategoricalDtype)
    codes, levels = pd.factorize(groups, sort=ordered, use_na_sentinel=False)
    counts = np.bincount(2 * codes + side, minlength=2 * len(levels)).reshape(-1, 2)
    return pd.DataFrame(counts, index=pd.Index(levels, name="bin"))


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
