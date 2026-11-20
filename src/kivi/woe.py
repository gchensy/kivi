"""Weight of evidence and information value."""

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._counts import group_counts, shares
from ._validation import as_series

__all__ = ["woe_table"]


def woe_table(
    groups: npt.ArrayLike, target: npt.ArrayLike, *, smoothing: float = 0.5
) -> pd.DataFrame:
    r"""Compute weight of evidence and information value for observed groups.

    Each group's share of events is compared with its share of non-events.
    Sum the ``iv`` column for the information value of the variable.

    Parameters
    ----------
    groups : array_like
        Pre-binned values or category labels. Missing values form a group.
    target : array_like
        Binary labels, with 1 indicating the event. If `groups` and `target`
        are both pandas Series, they must have identical indices; otherwise
        they are matched by position.
    smoothing : float, default 0.5
        Positive replacement for a zero count when calculating shares.

    Returns
    -------
    pandas.DataFrame
        Event and non-event counts, their within-class shares, WOE and IV
        per group. Sum the iv column for the variable's information value.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, their indices or lengths do
        not match, `target` does not contain both 0 and 1 or has missing or
        other labels, or `smoothing` is not finite and positive.

    See Also
    --------
    kivi.binning.bin_edges : Fit edges for numeric variables.
    kivi.evaluate.psi_table : Compare the distribution of two samples.

    Notes
    -----
    For group :math:`i`, let :math:`p_i` and :math:`q_i` be its shares of all
    events and of all non-events. Then

    .. math::

       \mathrm{WOE}_i = \ln \frac{p_i}{q_i}, \qquad
       \mathrm{IV}_i = (p_i - q_i)\,\mathrm{WOE}_i, \qquad
       \mathrm{IV} = \sum_i \mathrm{IV}_i.

    Each zero count is replaced with `smoothing` before the shares are
    normalized, so every WOE is finite. The ``non_event`` and ``event``
    columns always report the observed counts.

    Rows follow the order in which the groups first appear. Categories that
    are never observed are omitted.

    References
    ----------
    .. [1] N. Siddiqi, "Credit Risk Scorecards: Developing and Implementing
       Intelligent Credit Scoring", Wiley, 2006.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.woe import woe_table
    >>> groups = pd.Series(["a", "a", "a", "b", "b", "b"])
    >>> target = pd.Series([1, 1, 0, 1, 0, 0])
    >>> woe_table(groups, target)[["event", "non_event", "woe", "iv"]].round(4)
         event  non_event     woe     iv
    bin
    a        2          1  0.6931  0.231
    b        1          2 -0.6931  0.231
    """
    if isinstance(groups, pd.Series) and isinstance(target, pd.Series):
        if not groups.index.equals(target.index):
            raise ValueError("groups and target must have identical indices")
    groups, target = as_series(groups, "groups"), as_series(target, "target")
    if len(groups) != len(target):
        raise ValueError(
            f"groups and target must have the same length, got {len(groups)} and {len(target)}"
        )
    if not target.isin([0, 1]).all() or target.nunique() != 2:
        raise ValueError("target must contain both 0 and 1 and no missing labels")
    counts = group_counts(groups, target.to_numpy(dtype=np.intp))
    counts.columns = ["non_event", "event"]
    proportions = shares(counts, smoothing)
    result = counts.join(proportions.add_suffix("_share"))
    result["woe"] = np.log(result.event_share / result.non_event_share)
    result["iv"] = (result.event_share - result.non_event_share) * result.woe
    return result
