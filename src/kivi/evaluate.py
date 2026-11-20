"""Population stability measures."""

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._counts import group_counts, shares
from ._validation import as_series

__all__ = ["psi_table"]


def psi_table(
    reference: npt.ArrayLike, current: npt.ArrayLike, *, smoothing: float = 0.5
) -> pd.DataFrame:
    r"""Compare two samples using their population stability index.

    Each group's share of the current sample is compared with its share of
    the reference sample. Sum the ``psi`` column for the index.

    Parameters
    ----------
    reference, current : array_like
        Labels or values binned using the same edges. Missing values and
        categories observed in only one sample are included.
    smoothing : float, default 0.5
        Positive replacement for a zero count when calculating shares.

    Returns
    -------
    pandas.DataFrame
        Counts, normalized shares and the PSI contribution of each group.
        Sum the psi column for the population stability index.

    Raises
    ------
    ValueError
        If either sample is empty or not one-dimensional, or `smoothing` is
        not finite and positive.

    See Also
    --------
    kivi.binning.bin_edges : Fit edges on the reference sample.
    kivi.woe.woe_table : Relate the groups to a binary target.

    Notes
    -----
    For group :math:`i`, let :math:`q_i` be its share of the reference sample
    and :math:`p_i` its share of the current sample. Then

    .. math::

       \mathrm{PSI}_i = (p_i - q_i) \ln \frac{p_i}{q_i}, \qquad
       \mathrm{PSI} = \sum_i \mathrm{PSI}_i.

    Each zero count is replaced with `smoothing` before the shares are
    normalized, so a group observed in only one sample has a finite
    contribution. The ``reference`` and ``current`` columns always report
    the observed counts.

    Rows follow the order in which the groups first appear, in the reference
    sample and then in the current sample.

    References
    ----------
    .. [1] N. Siddiqi, "Credit Risk Scorecards: Developing and Implementing
       Intelligent Credit Scoring", Wiley, 2006.

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
    reference, current = as_series(reference, "reference"), as_series(current, "current")
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
