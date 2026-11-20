"""Population and characteristic stability measures.

``psi_table`` compares two samples of group labels; ``psi`` computes the same
index from counts that are already aggregated. ``csi_table`` weights the
change of each group's share with the group's scorecard points.
"""

import math
from collections.abc import Mapping
from numbers import Real

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._counts import divergence, group_counts
from ._validation import as_count_columns, as_series, as_weights

__all__ = ["csi_table", "psi", "psi_table"]


def psi_table(
    reference: npt.ArrayLike,
    current: npt.ArrayLike,
    *,
    smoothing: float = 0.5,
    reference_weights: npt.ArrayLike | None = None,
    current_weights: npt.ArrayLike | None = None,
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
    reference_weights, current_weights : array_like, optional
        Non-negative weight of each observation in the reference and the
        current sample, matched like the samples. The counts are then sums of
        weights, and groups whose weights sum to zero in both samples are left
        out. `smoothing` is then in units of weight.

    Returns
    -------
    pandas.DataFrame
        Counts, normalized shares and the PSI contribution of each group.
        Sum the psi column for the population stability index.

    Raises
    ------
    ValueError
        If either sample is empty or not one-dimensional, or `smoothing` is
        not finite and positive. Also if a weight array is not numeric, does
        not give each observation of its sample one finite, non-negative weight
        with a positive total, or has an index other than that of its sample.

    See Also
    --------
    psi : The same index from counts per group.
    csi_table : Weight the change of each group's share with its points.
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

    If both samples are categorical with the same categories, rows follow
    the order of the categories with missing values last. Otherwise they
    follow the order in which the groups first appear, in the reference
    sample and then in the current sample.

    References
    ----------
    .. [1] N. Siddiqi, "Credit Risk Scorecards: Developing and Implementing
       Intelligent Credit Scoring", Wiley, 2006.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.stability import psi_table
    >>> reference = pd.Series(["x", "x", "y", "y"])
    >>> current = pd.Series(["x", "y", "y", "y"])
    >>> psi_table(reference, current)[["reference", "current", "psi"]].round(4)
         reference  current     psi
    bin
    x            2        1  0.1733
    y            2        3  0.1014
    """
    counts = _sample_counts(reference, current, reference_weights, current_weights)
    share, _, contribution = divergence(counts.to_numpy(), smoothing)
    return counts.assign(reference_share=share[:, 0], current_share=share[:, 1], psi=contribution)


def psi(
    reference_counts: npt.ArrayLike, current_counts: npt.ArrayLike, *, smoothing: float = 0.5
) -> float:
    r"""Compute the population stability index from counts per group.

    Use this when the samples are already aggregated, for example monthly
    counts per score band from a monitoring report.

    Parameters
    ----------
    reference_counts, current_counts : array_like
        One-dimensional, non-negative counts of each group in the reference
        and the current sample, in the same group order. Weighted counts
        need not be integers.
    smoothing : float, default 0.5
        Positive replacement for a zero count when calculating shares.

    Returns
    -------
    float
        The population stability index.

    Raises
    ------
    ValueError
        If the counts are not one-dimensional, numeric, finite and
        non-negative, have different lengths or a zero total, or `smoothing`
        is not finite and positive.

    See Also
    --------
    psi_table : The index and its contributions from group labels.
    kivi.woe.iv : The same divergence between events and non-events.

    Notes
    -----
    With :math:`q_i` and :math:`p_i` the smoothed shares of group :math:`i`
    in the reference and the current sample,

    .. math::

       \mathrm{PSI} = \sum_i (p_i - q_i) \ln \frac{p_i}{q_i},

    which equals the sum of the ``psi`` column of `psi_table` for the same
    groups.

    Examples
    --------
    >>> from kivi.stability import psi
    >>> round(psi([2, 2], [1, 3]), 4)
    0.2747
    """
    counts = as_count_columns(
        reference_counts, current_counts, ("reference_counts", "current_counts")
    )
    return float(divergence(counts, smoothing)[2].sum())


def csi_table(
    reference: npt.ArrayLike,
    current: npt.ArrayLike,
    points: pd.DataFrame | pd.Series | Mapping[object, float],
    *,
    unseen_points: float | None = None,
    reference_weights: npt.ArrayLike | None = None,
    current_weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Measure how much a variable's shift moves the average score.

    The characteristic stability index (CSI) weights the change of each
    group's share, from the reference to the current sample, with the
    scorecard points of the group. Sum the ``csi`` column for the change of
    the variable's average points.

    Parameters
    ----------
    reference, current : array_like
        Labels or values binned using the same edges, as for `psi_table`.
        Missing values and groups observed in only one sample are included.
    points : pandas.DataFrame, pandas.Series or mapping
        Points of each group. A DataFrame holds the groups of one variable
        in a ``bin`` column, or in its index without one, and their points
        in a ``points`` column, such as ``table.loc[[variable]]`` of a
        `kivi.scorecard.points_table` result. A Series or mapping gives the
        points by group. A missing label gives the points of missing values.
    unseen_points : float, optional
        Points of observed groups that are missing from `points`, such as a
        new category that `kivi.woe.woe_transform` encoded with ``unseen``.
        Its applicants receive the `kivi.scorecard.woe_to_points` result for
        that WOE, not the WOE itself; see the Notes. By default such groups
        raise an error.
    reference_weights, current_weights : array_like, optional
        Non-negative weight of each observation in the reference and the
        current sample, matched like the samples. The counts and shares are
        then sums of weights, and groups whose weights sum to zero in both
        samples are left out.

    Returns
    -------
    pandas.DataFrame
        One row per observed group, indexed by ``bin``, with the counts
        ``reference`` and ``current``, their shares ``reference_share`` and
        ``current_share``, the group's ``points`` and its contribution
        ``csi``. Sum the csi column for the index.

    Raises
    ------
    ValueError
        If either sample is empty or not one-dimensional, `points` is not a
        DataFrame, Series or mapping, holds more than one variable, repeats
        a group or has values that are not finite numbers, `unseen_points`
        is not a finite number, or an observed group has no points and
        `unseen_points` is not given. Also if a weight array is invalid, as
        in `psi_table`.

    See Also
    --------
    psi_table : The population stability index of the same groups.
    kivi.scorecard.points_table : The points of every group of a scorecard.
    kivi.binning.bin_edges : Fit edges on the reference sample.

    Notes
    -----
    For group :math:`i` with points :math:`s_i`, let :math:`q_i` be its
    share of the reference sample and :math:`p_i` its share of the current
    sample. Then

    .. math::

       \mathrm{CSI}_i = (p_i - q_i)\, s_i, \qquad
       \mathrm{CSI} = \sum_i \mathrm{CSI}_i
       = \sum_i p_i s_i - \sum_i q_i s_i,

    the average points of the variable in the current sample minus those in
    the reference sample [1]_. The shares sum to 1 in both samples, so adding
    the same number to the points of every group leaves the CSI unchanged:
    only the differences between the groups count, not the share of the
    intercept and offset that `kivi.scorecard.points_table` gives each
    variable. Over all variables of a scorecard, the CSI values add up to
    the change of the average score.

    Unlike in `psi_table`, the shares are not smoothed. A contribution has
    no logarithm, so an empty group contributes a finite amount without
    smoothing, and a smoothed share would credit an empty group with
    applicants it does not have, so the sum would no longer be the change of
    the average points.

    `unseen_points` is in points, like `points`. For applicants that
    ``woe_transform(..., unseen=woe)`` encoded, it is
    ``woe_to_points`` of a one-row frame that holds that WOE; a WOE of 0
    gives the variable's share of the intercept and offset, the same for
    every variable. Passing the WOE itself would give a wrong CSI without
    an error.

    Some texts also call the PSI of a single variable its CSI; that is
    `psi_table` of the variable's groups. Rows are ordered as in
    `psi_table`, and groups of `points` observed in neither sample are left
    out, since they contribute nothing.

    References
    ----------
    .. [1] N. Siddiqi, "Credit Risk Scorecards: Developing and Implementing
       Intelligent Credit Scoring", Wiley, 2006.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.stability import csi_table
    >>> reference = pd.Series(["x", "x", "y", "y"])
    >>> current = pd.Series(["x", "y", "y", "y"])
    >>> csi_table(reference, current, {"x": 40.0, "y": 25.0})
         reference  current  reference_share  current_share  points    csi
    bin
    x            2        1              0.5           0.25    40.0 -10.00
    y            2        3              0.5           0.75    25.0   6.25

    The average points fall from 32.5 to 28.75, by the sum of the ``csi``
    column, -3.75.
    """
    labels, values = _points_by_group(points)
    if unseen_points is not None and (
        isinstance(unseen_points, bool)
        or not isinstance(unseen_points, Real)
        or not math.isfinite(unseen_points)
    ):
        raise ValueError(f"unseen_points must be a finite number, got {unseen_points!r}")
    counts = _sample_counts(reference, current, reference_weights, current_weights)
    unseen = None if unseen_points is None else float(unseen_points)
    group_points = _match(counts.index, labels, values, unseen)
    observed = counts.to_numpy()
    share = observed / observed.sum(axis=0)
    return counts.assign(
        reference_share=share[:, 0],
        current_share=share[:, 1],
        points=group_points,
        csi=(share[:, 1] - share[:, 0]) * group_points,
    )


def _sample_counts(
    reference: npt.ArrayLike,
    current: npt.ArrayLike,
    reference_weights: npt.ArrayLike | None,
    current_weights: npt.ArrayLike | None,
) -> pd.DataFrame:
    """Count each group in the reference and the current sample, as `psi_table` reports them."""
    raw_reference, raw_current = reference, current
    reference, current = as_series(reference, "reference"), as_series(current, "current")
    if reference.empty or current.empty:
        raise ValueError("reference and current must both contain observations")
    reference_mass = as_weights(reference_weights, raw_reference, "reference_weights", "reference")
    current_mass = as_weights(current_weights, raw_current, "current_weights", "current")
    # pd.concat and join hash interval categories, which warns on NumPy < 2.2, so samples
    # binned with the same edges are joined by their codes and shares assigned by position.
    if isinstance(reference.dtype, pd.CategoricalDtype) and reference.dtype == current.dtype:
        codes = np.concatenate([reference.cat.codes, current.cat.codes])
        groups = pd.Series(pd.Categorical.from_codes(codes, dtype=reference.dtype))
    else:
        groups = pd.concat([reference, current], ignore_index=True)
    side = np.repeat([0, 1], [len(reference), len(current)])
    mass = None
    if reference_mass is not None or current_mass is not None:
        mass = np.concatenate(
            [
                np.ones(len(reference)) if reference_mass is None else reference_mass,
                np.ones(len(current)) if current_mass is None else current_mass,
            ]
        )
    counts = group_counts(groups, side, mass)
    counts.columns = ["reference", "current"]
    return counts


def _points_by_group(points) -> tuple[np.ndarray, np.ndarray]:
    """Return the group labels, as an object array, and the float points of `points`."""
    if isinstance(points, pd.DataFrame):
        if "points" not in points.columns:
            raise ValueError(
                f"points must have a points column, got columns {points.columns.tolist()}"
            )
        if points.index.name == "variable" and points.index.nunique() > 1:
            shown = ", ".join(repr(variable) for variable in points.index.unique()[:5])
            raise ValueError(
                f"points must hold the groups of one variable, got {points.index.nunique()} "
                f"variables: {shown}; select one with table.loc[[variable]]"
            )
        labels = points["bin"] if "bin" in points.columns else points.index
        values = points["points"]
    elif isinstance(points, pd.Series):
        labels, values = points.index, points
    elif isinstance(points, Mapping):
        labels, values = list(points.keys()), list(points.values())
    else:
        raise ValueError(
            "points must be a points_table result, a Series or a mapping of group to points, "
            f"got {type(points).__name__}"
        )
    # fromiter keeps tuple labels whole, where np.asarray would add a dimension.
    labels = np.fromiter(labels, dtype=object, count=len(labels))
    values = pd.Series(values, dtype=object).tolist()
    for label, value in zip(labels, values, strict=True):
        if pd.isna(value) or (isinstance(value, Real) and not math.isfinite(value)):
            raise ValueError(f"points must be finite, got {value!r} for group {label!r}")
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"points must be numbers, got {value!r} for group {label!r}")
    missing = pd.isna(labels)
    index = pd.Index(labels[~missing], dtype=object)
    if index.has_duplicates or missing.sum() > 1:
        repeated = labels[missing][0] if missing.sum() > 1 else index[index.duplicated()][0]
        raise ValueError(f"points must give each group once, got {repeated!r} twice")
    return labels, np.asarray(values, dtype=float)


def _match(
    groups: pd.Index, labels: np.ndarray, values: np.ndarray, unseen: float | None
) -> np.ndarray:
    """Points of each group, looked up by equality; `unseen` for groups without points."""
    levels = np.fromiter(groups, dtype=object, count=len(groups))
    # Object indices compare labels with ==, so an interval matches the interval with the
    # same edges; an IntervalIndex would hash them, which warns on NumPy < 2.2.
    missing = pd.isna(labels)
    position = pd.Index(labels[~missing], dtype=object).get_indexer(levels)
    is_missing = pd.isna(levels)
    position[is_missing] = -1
    result = np.full(len(levels), np.nan)
    found = position >= 0
    result[found] = values[~missing][position[found]]
    if missing.any():
        result[is_missing] = values[missing][0]
        found |= is_missing
    if not found.all():
        if unseen is None:
            absent = levels[~found]
            shown = ", ".join(repr(label) for label in absent[:5])
            raise ValueError(
                f"{len(absent)} group(s) have no points, such as {shown}; "
                "pass unseen_points to give them points"
            )
        result[~found] = unseen
    return result
