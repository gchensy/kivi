"""Population stability measures.

``psi_table`` compares two samples of group labels; ``psi`` computes the same
index from counts that are already aggregated.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._counts import divergence, group_counts
from ._validation import as_count_columns, as_series

__all__ = ["psi", "psi_table"]


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
    psi : The same index from counts per group.
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
    reference, current = as_series(reference, "reference"), as_series(current, "current")
    if reference.empty or current.empty:
        raise ValueError("reference and current must both contain observations")
    # pd.concat and join hash interval categories, which warns on NumPy < 2.2, so samples
    # binned with the same edges are joined by their codes and shares assigned by position.
    if isinstance(reference.dtype, pd.CategoricalDtype) and reference.dtype == current.dtype:
        codes = np.concatenate([reference.cat.codes, current.cat.codes])
        groups = pd.Series(pd.Categorical.from_codes(codes, dtype=reference.dtype))
    else:
        groups = pd.concat([reference, current], ignore_index=True)
    side = np.repeat([0, 1], [len(reference), len(current)])
    counts = group_counts(groups, side)
    counts.columns = ["reference", "current"]
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
