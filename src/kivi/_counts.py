"""Common count and smoothing rules for grouped metrics."""

import numpy as np
import pandas as pd


def group_counts(groups: pd.Series, side: np.ndarray) -> pd.DataFrame:
    codes, levels = pd.factorize(groups, sort=False, use_na_sentinel=False)
    counts = np.zeros((len(levels), 2), dtype=np.int64)
    np.add.at(counts, (codes, side), 1)
    return pd.DataFrame(counts, index=pd.Index(levels, name="bin"))


def shares(counts: pd.DataFrame, smoothing: float) -> pd.DataFrame:
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("smoothing must be finite and positive")
    adjusted = counts.astype(float).mask(counts == 0, smoothing)
    return adjusted / adjusted.sum()
