"""Weight of evidence and information value.

``woe_table`` measures groups against a binary target and ``woe_transform``
encodes groups with the fitted WOE. ``woe`` and ``iv`` compute the same
measures from counts that are already aggregated.
"""

from numbers import Real
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._counts import divergence, group_counts
from ._validation import as_count_columns, as_series, as_weights

__all__ = ["iv", "woe", "woe_table", "woe_transform"]


def woe_table(
    groups: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    smoothing: float = 0.5,
    weights: npt.ArrayLike | None = None,
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
    weights : array_like, optional
        Non-negative weight of each observation, for example the inverse of
        its sampling rate, matched like `target`. The counts are then sums of
        weights, and groups whose weights sum to zero are left out, so
        `woe_transform` treats them as unseen. `smoothing` is then in units of
        weight: scale the weights to sum to the number of observations, or
        scale `smoothing` with them.

    Returns
    -------
    pandas.DataFrame
        One row per group with the columns ``non_event``, ``event`` and
        ``total`` (counts), ``non_event_share`` and ``event_share`` (smoothed
        shares within each class), ``total_share`` (share of all
        observations), ``event_rate``, ``woe`` and ``iv``. Sum the iv column
        for the variable's information value.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, their indices or lengths do
        not match, `target` does not contain both 0 and 1 or has missing or
        other labels, or `smoothing` is not finite and positive. Also if
        `weights` is not numeric, does not give each observation one finite,
        non-negative weight, has an index other than that of `groups`, or
        leaves a class with a zero total.

    See Also
    --------
    woe_transform : Replace each group with its WOE.
    woe, iv : The same measures from counts per group.
    kivi.binning.bin_edges : Fit edges for numeric variables.
    kivi.stability.psi_table : Compare the distribution of two samples.

    Notes
    -----
    For group :math:`i`, let :math:`p_i` and :math:`q_i` be its shares of all
    events and of all non-events. Then

    .. math::

       \mathrm{WOE}_i = \ln \frac{p_i}{q_i}, \qquad
       \mathrm{IV}_i = (p_i - q_i)\,\mathrm{WOE}_i, \qquad
       \mathrm{IV} = \sum_i \mathrm{IV}_i.

    Each zero count is replaced with `smoothing` before the shares are
    normalized, so every WOE is finite. The count columns, ``total_share``
    and ``event_rate`` always use the observed counts.

    Categorical groups, such as the result of `kivi.binning.cut` or
    ``pandas.cut``, follow the order of their categories with missing values
    last. Other groups follow the order in which they first appear.
    Categories that are never observed are omitted.

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
    raw_groups = groups
    groups, target = as_series(groups, "groups"), as_series(target, "target")
    if len(groups) != len(target):
        raise ValueError(
            f"groups and target must have the same length, got {len(groups)} and {len(target)}"
        )
    if not target.isin([0, 1]).all() or target.nunique() != 2:
        raise ValueError("target must contain both 0 and 1 and no missing labels")
    mass = as_weights(weights, raw_groups, of="groups")
    counts = group_counts(groups, target.to_numpy(dtype=np.intp), mass)
    totals = counts.to_numpy().sum(axis=0)
    if not (totals > 0).all():
        raise ValueError(
            "weights must give both target classes a positive total, got "
            f"{float(totals[0])!r} for 0 and {float(totals[1])!r} for 1"
        )
    counts.columns = ["non_event", "event"]
    # Assign by position: aligning interval categories hashes them, which warns on NumPy < 2.2.
    share, log_ratio, contribution = divergence(counts.to_numpy(), smoothing)
    total = counts["non_event"] + counts["event"]
    return counts.assign(
        total=total,
        non_event_share=share[:, 0],
        event_share=share[:, 1],
        total_share=total / total.sum(),
        event_rate=counts["event"] / total,
        woe=log_ratio,
        iv=contribution,
    )


def woe(
    event_counts: npt.ArrayLike, non_event_counts: npt.ArrayLike, *, smoothing: float = 0.5
) -> npt.NDArray[np.float64]:
    r"""Compute the weight of evidence of each group from its counts.

    Use this when the counts are already aggregated, for example from a
    database query or a report, instead of one label per observation.

    Parameters
    ----------
    event_counts, non_event_counts : array_like
        One-dimensional, non-negative counts of events and non-events in each
        group, in the same group order. Weighted counts need not be integers.
    smoothing : float, default 0.5
        Positive replacement for a zero count when calculating shares.

    Returns
    -------
    numpy.ndarray
        The WOE of each group.

    Raises
    ------
    ValueError
        If the counts are not one-dimensional, numeric, finite and
        non-negative, have different lengths or a zero total, or `smoothing`
        is not finite and positive.

    See Also
    --------
    iv : The information value of the same counts.
    woe_table : Counts, shares, WOE and IV from group labels and a target.

    Notes
    -----
    With :math:`p_i` and :math:`q_i` the smoothed shares of group :math:`i`
    among all events and all non-events, :math:`\mathrm{WOE}_i = \ln(p_i / q_i)`,
    as in the ``woe`` column of `woe_table`.

    Examples
    --------
    >>> from kivi.woe import woe
    >>> woe([2, 1], [1, 2]).round(4)
    array([ 0.6931, -0.6931])
    """
    counts = as_count_columns(non_event_counts, event_counts, ("non_event_counts", "event_counts"))
    return divergence(counts, smoothing)[1]


def iv(
    event_counts: npt.ArrayLike, non_event_counts: npt.ArrayLike, *, smoothing: float = 0.5
) -> float:
    r"""Compute the information value from counts per group.

    The information value summarizes how well the groups separate events
    from non-events, from counts that are already aggregated.

    Parameters
    ----------
    event_counts, non_event_counts : array_like
        One-dimensional, non-negative counts of events and non-events in each
        group, in the same group order. Weighted counts need not be integers.
    smoothing : float, default 0.5
        Positive replacement for a zero count when calculating shares.

    Returns
    -------
    float
        The information value of the variable.

    Raises
    ------
    ValueError
        If the counts are not one-dimensional, numeric, finite and
        non-negative, have different lengths or a zero total, or `smoothing`
        is not finite and positive.

    See Also
    --------
    woe : The weight of evidence of each group.
    woe_table : The IV contribution of each group from labels and a target.
    kivi.stability.psi : The same divergence between two samples.

    Notes
    -----
    .. math::

       \mathrm{IV} = \sum_i (p_i - q_i) \ln \frac{p_i}{q_i},

    which equals the sum of the ``iv`` column of `woe_table` for the same
    groups.

    Examples
    --------
    >>> from kivi.woe import iv
    >>> round(iv([2, 1], [1, 2]), 4)
    0.4621
    """
    counts = as_count_columns(non_event_counts, event_counts, ("non_event_counts", "event_counts"))
    return float(divergence(counts, smoothing)[2].sum())


def woe_transform(
    groups: npt.ArrayLike,
    table: pd.DataFrame | pd.Series,
    *,
    unseen: float | Literal["raise"] = "raise",
) -> pd.Series | npt.NDArray[np.float64]:
    """Replace each group with its weight of evidence.

    The WOE values come from a table fitted on the development sample, so a
    scorecard model sees the same encoding in development, validation and
    production.

    Parameters
    ----------
    groups : array_like
        One-dimensional group labels formed in the same way as those passed
        to `woe_table`, for example with the same edges and ``pandas.cut``.
    table : pandas.DataFrame or pandas.Series
        The result of `woe_table`, or a Series of WOE values indexed by group.
        A missing label in the index gives the WOE of missing values.
    unseen : float or "raise", default "raise"
        WOE for groups that are not in `table`, such as a new category, an
        interval without development observations, or missing values when
        `table` has no missing group. By default such groups raise an error.

    Returns
    -------
    pandas.Series or numpy.ndarray
        Float WOE values in the order of `groups`. A Series input gives a
        Series with the same index and name.

    Raises
    ------
    ValueError
        If `groups` is not one-dimensional, `table` has no ``woe`` column or
        repeats a group, `unseen` is neither ``"raise"`` nor a number, or
        `unseen` is ``"raise"`` and some groups are not in `table`.

    See Also
    --------
    woe_table : Fit the WOE of each group.
    kivi.binning.bin_edges : Fit the edges used to form numeric groups.

    Notes
    -----
    Labels are matched by equality, so an interval from ``pandas.cut`` matches
    the interval with the same edges in `table`. Use the edges fitted on the
    development sample for every sample that is transformed.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.woe import woe_table, woe_transform
    >>> groups = pd.Series(["a", "a", "a", "b", "b", "b"])
    >>> target = pd.Series([1, 1, 0, 1, 0, 0])
    >>> table = woe_table(groups, target)
    >>> woe_transform(pd.Series(["b", "a", None]), table, unseen=0.0).round(4).tolist()
    [-0.6931, 0.6931, 0.0]
    """
    woe = _woe_values(table)
    raise_unseen = isinstance(unseen, str) and unseen == "raise"
    if not raise_unseen and (isinstance(unseen, (bool, str)) or not isinstance(unseen, Real)):
        raise ValueError(f"unseen must be 'raise' or a number, got {unseen!r}")
    series = as_series(groups, "groups")
    codes, levels = pd.factorize(series, sort=False, use_na_sentinel=False)
    levels = np.asarray(levels, dtype=object)
    is_missing = pd.isna(levels)
    labelled = ~pd.isna(woe.index)
    position = pd.Index(np.asarray(woe.index[labelled], dtype=object)).get_indexer(levels)
    position[is_missing] = -1
    values = woe.to_numpy(dtype=float)
    level_woe = np.full(len(levels), np.nan)
    found = position >= 0
    level_woe[found] = values[labelled][position[found]]
    if not labelled.all():
        level_woe[is_missing] = values[~labelled][0]
        found |= is_missing
    if not found.all():
        if raise_unseen:
            missing = levels[~found]
            shown = ", ".join(repr(label) for label in missing[:5])
            raise ValueError(
                f"{len(missing)} group(s) are not in table, such as {shown}; "
                "pass unseen to give them a WOE"
            )
        level_woe[~found] = unseen
    result = level_woe[codes]
    if isinstance(groups, pd.Series):
        return pd.Series(result, index=series.index, name=series.name)
    return result


def _woe_values(table: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(table, pd.DataFrame):
        if "woe" not in table.columns:
            raise ValueError("table must have a woe column, as returned by woe_table")
        table = table["woe"]
    elif not isinstance(table, pd.Series):
        raise ValueError(
            f"table must be a woe_table result or a Series, got {type(table).__name__}"
        )
    if table.index.has_duplicates or pd.isna(table.index).sum() > 1:
        raise ValueError("table must have one row per group")
    return table
