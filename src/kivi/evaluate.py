"""Population stability measures."""

import numpy as np
import pandas as pd

from ._counts import group_counts, shares


def psi_table(reference: pd.Series, current: pd.Series, *, smoothing: float = 0.5) -> pd.DataFrame:
    """Compare two samples using their population stability index.

    Parameters
    ----------
    reference, current : pandas.Series
        Labels or values binned using the same edges. Missing values and
        categories observed in only one sample are included.
    smoothing : float, default 0.5
        Positive replacement for a zero count when calculating shares.

    Returns
    -------
    pandas.DataFrame
        Counts, normalized shares and the PSI contribution of each group.
        Sum the psi column for the population stability index.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.evaluate import psi_table
    >>> reference = pd.Series(["x", "x", "y", "y"])
    >>> current = pd.Series(["x", "y", "y", "y"])
    >>> psi_table(reference, current)[["reference", "current", "psi"]].round(4)
         reference  current     psi
    bin
    x            2        1  0.1733
    y            2        3  0.1014
    """
    if reference.empty or current.empty:
        raise ValueError("reference and current must both contain observations")
    groups = pd.concat([reference, current], ignore_index=True)
    side = np.repeat([0, 1], [len(reference), len(current)])
    counts = group_counts(groups, side)
    counts.columns = ["reference", "current"]
    proportions = shares(counts, smoothing)
    result = counts.join(proportions.add_suffix("_share"))
    result["psi"] = (result.current_share - result.reference_share) * np.log(
        result.current_share / result.reference_share
    )
    return result
