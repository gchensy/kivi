"""Weight of evidence and information value."""

import numpy as np
import pandas as pd

from ._counts import group_counts, shares


def woe_table(groups: pd.Series, target: pd.Series, *, smoothing: float = 0.5) -> pd.DataFrame:
    """Compute weight of evidence and information value for observed groups.

    Parameters
    ----------
    groups : pandas.Series
        Pre-binned values or category labels. Missing values form a group.
    target : pandas.Series
        Binary labels on the same index, with 1 indicating the event.
    smoothing : float, default 0.5
        Positive replacement for a zero count when calculating shares.

    Returns
    -------
    pandas.DataFrame
        Event and non-event counts, their within-class shares, WOE and IV
        per group. Sum the iv column for the variable's information value.
    """
    if not groups.index.equals(target.index):
        raise ValueError("groups and target must have identical indices")
    if not target.isin([0, 1]).all() or target.nunique() != 2:
        raise ValueError("target must contain both 0 and 1 and no missing labels")
    counts = group_counts(groups, target.to_numpy(dtype=np.intp))
    counts.columns = ["non_event", "event"]
    proportions = shares(counts, smoothing)
    result = counts.join(proportions.add_suffix("_share"))
    result["woe"] = np.log(result.event_share / result.non_event_share)
    result["iv"] = (result.event_share - result.non_event_share) * result.woe
    return result
