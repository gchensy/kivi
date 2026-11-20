"""Common count and smoothing rules for grouped metrics."""

import numpy as np
import pandas as pd

from ._validation import check_smoothing


def group_counts(groups: pd.Series, side: np.ndarray) -> pd.DataFrame:
    """Count observations per group for the two sides coded 0 and 1 in ``side``."""
    codes, levels = pd.factorize(groups, sort=False, use_na_sentinel=False)
    counts = np.bincount(2 * codes + side, minlength=2 * len(levels)).reshape(-1, 2)
    return pd.DataFrame(counts, index=pd.Index(levels, name="bin"))


def shares(counts: pd.DataFrame, smoothing: float) -> pd.DataFrame:
    """Replace zero counts with ``smoothing``, then normalize each column."""
    check_smoothing(smoothing)
    adjusted = counts.astype(float).mask(counts == 0, smoothing)
    return adjusted / adjusted.sum()
