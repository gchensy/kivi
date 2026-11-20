"""Performance of a score for a binary target.

The functions take a binary ``target`` and a score that is higher for a higher
event probability, such as a predicted probability of default. Scorecard
points run the other way, so negate them before passing them in.
"""

from numbers import Integral

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._validation import as_series

__all__ = ["roc_auc", "gini", "ks", "gains_table", "brier_score"]


def roc_auc(target: npt.ArrayLike, score: npt.ArrayLike) -> float:
    r"""Compute the area under the ROC curve of a score.

    The AUC is the probability that a randomly chosen event has a higher
    score than a randomly chosen non-event, with ties counted as one half.

    Parameters
    ----------
    target : array_like
        Binary labels, with 1 indicating the event. Both classes must be
        present. If `target` and `score` are both pandas Series, they must
        have identical indices; otherwise they are matched by position.
    score : array_like
        Finite numeric scores that are higher for a higher event probability.
        Negate scorecard points, which are higher for a lower risk.

    Returns
    -------
    float
        The AUC, from 0 to 1. A score unrelated to the target gives about
        0.5, and a value below 0.5 suggests that the score runs the other way.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, their indices or lengths do
        not match, `target` has a label other than 0 and 1 or lacks one of
        them, or `score` is not numeric or has a missing or infinite value.

    See Also
    --------
    gini : The same ranking measure on a scale from -1 to 1.
    ks : Largest gap between the score distributions of the two classes.
    gains_table : Event rates and captured events by score band.

    Notes
    -----
    With :math:`n_1` events and :math:`n_0` non-events, let :math:`R_1` be the
    sum of the ranks of the events among all scores, where tied scores share
    their average rank. The Mann-Whitney statistic gives

    .. math::

       \mathrm{AUC} = \frac{R_1 - n_1 (n_1 + 1) / 2}{n_1 n_0}.

    References
    ----------
    .. [1] J. A. Hanley and B. J. McNeil, "The meaning and use of the area
       under a receiver operating characteristic (ROC) curve", Radiology,
       143(1), pp. 29-36, 1982.

    Examples
    --------
    >>> from kivi.metrics import roc_auc
    >>> target = [0, 0, 1, 0, 1, 1]
    >>> score = [0.1, 0.4, 0.4, 0.8, 0.7, 0.9]
    >>> round(roc_auc(target, score), 4)
    0.7222

    Scorecard points are higher for a lower risk, so negate them:

    >>> import numpy as np
    >>> points = np.array([720, 650, 650, 560, 590, 510])
    >>> round(roc_auc(target, -points), 4)
    0.7222
    """
    labels, values = _labels_and_scores(target, score, "score")
    non_event, event = _distinct_counts(labels, values).T
    # U counts each (event, non-event) pair in which the event scores higher, a tie as one half;
    # it equals R_1 - n_1 (n_1 + 1) / 2 without subtracting large rank sums.
    below = np.cumsum(non_event) - non_event
    u = np.sum(event * (below + 0.5 * non_event))
    return float(u / (float(event.sum()) * float(non_event.sum())))


def gini(target: npt.ArrayLike, score: npt.ArrayLike) -> float:
    r"""Compute the Gini coefficient of a score.

    The Gini coefficient rescales the AUC to run from -1 to 1, so that a score
    unrelated to the target gives about 0.

    Parameters
    ----------
    target : array_like
        Binary labels, with 1 indicating the event. Both classes must be
        present. If `target` and `score` are both pandas Series, they must
        have identical indices; otherwise they are matched by position.
    score : array_like
        Finite numeric scores that are higher for a higher event probability.
        Negate scorecard points, which are higher for a lower risk.

    Returns
    -------
    float
        The Gini coefficient, from -1 to 1. A negative value suggests that
        the score runs the other way.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, their indices or lengths do
        not match, `target` has a label other than 0 and 1 or lacks one of
        them, or `score` is not numeric or has a missing or infinite value.

    See Also
    --------
    roc_auc : The area under the ROC curve.
    ks : Largest gap between the score distributions of the two classes.

    Notes
    -----
    The Gini coefficient, also called the accuracy ratio, is

    .. math::

       \mathrm{Gini} = 2\,\mathrm{AUC} - 1.

    Examples
    --------
    >>> from kivi.metrics import gini
    >>> target = [0, 0, 1, 0, 1, 1]
    >>> score = [0.1, 0.4, 0.4, 0.8, 0.7, 0.9]
    >>> round(gini(target, score), 4)
    0.4444
    """
    return 2 * roc_auc(target, score) - 1


def ks(target: npt.ArrayLike, score: npt.ArrayLike) -> float:
    r"""Compute the Kolmogorov-Smirnov statistic of a score.

    The KS statistic is the largest gap between the cumulative distribution
    of the scores of events and that of non-events.

    Parameters
    ----------
    target : array_like
        Binary labels, with 1 indicating the event. Both classes must be
        present. If `target` and `score` are both pandas Series, they must
        have identical indices; otherwise they are matched by position.
    score : array_like
        Finite numeric scores, such as predicted event probabilities.

    Returns
    -------
    float
        The largest gap, from 0 to 1.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, their indices or lengths do
        not match, `target` has a label other than 0 and 1 or lacks one of
        them, or `score` is not numeric or has a missing or infinite value.

    See Also
    --------
    gains_table : The gap at the edge of each score band.
    roc_auc : The probability that an event scores higher than a non-event.

    Notes
    -----
    Let :math:`F_1` and :math:`F_0` be the empirical distribution functions of
    the scores of events and of non-events. Then

    .. math::

       \mathrm{KS} = \max_t \, \lvert F_1(t) - F_0(t) \rvert,

    where :math:`t` runs over the observed scores. The gap does not depend on
    the direction of the score, so negated scores give the same KS; use the
    sign of `gini` to check the direction.

    References
    ----------
    .. [1] L. C. Thomas, D. B. Edelman and J. N. Crook, "Credit Scoring and
       Its Applications", SIAM, 2002.

    Examples
    --------
    >>> from kivi.metrics import ks
    >>> target = [0, 0, 1, 0, 1, 1]
    >>> score = [0.1, 0.4, 0.4, 0.8, 0.7, 0.9]
    >>> round(ks(target, score), 4)
    0.3333
    """
    labels, values = _labels_and_scores(target, score, "score")
    non_event, event = _distinct_counts(labels, values).T
    gap = np.cumsum(event) / event.sum() - np.cumsum(non_event) / non_event.sum()
    return float(np.max(np.abs(gap)))


def gains_table(
    target: npt.ArrayLike, score: npt.ArrayLike, *, bins: int | npt.ArrayLike = 10
) -> pd.DataFrame:
    """Summarize events by score band, from the highest scores to the lowest.

    Each row shows how many events a band holds and what share of all events
    is captured by reviewing that band and every band above it.

    Parameters
    ----------
    target : array_like
        Binary labels, with 1 indicating the event. Both classes must be
        present. If `target` and `score` are both pandas Series, they must
        have identical indices; otherwise they are matched by position.
    score : array_like
        Finite numeric scores that are higher for a higher event probability.
        Negate scorecard points, which are higher for a lower risk.
    bins : int or array_like, default 10
        Number of quantile bands, or increasing band edges such as those from
        `kivi.binning.bin_edges`. Tied scores can leave fewer quantile bands
        than requested. Pass the edges fitted on a development sample to
        compare later samples on the same bands.

    Returns
    -------
    pandas.DataFrame
        One row per score band that has observations, from the highest band
        to the lowest, indexed by the interval of the band. The columns are
        described in the Notes.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, their indices or lengths do
        not match, `target` has a label other than 0 and 1 or lacks one of
        them, `score` is not numeric or has a missing or infinite value,
        `bins` is neither a positive integer nor increasing edges, or a score
        lies outside the edges.

    See Also
    --------
    ks : The largest gap over all scores rather than band edges.
    roc_auc : Ranking quality over all pairs of events and non-events.
    kivi.binning.bin_edges : Fit band edges on a development sample.

    Notes
    -----
    Bands are intervals closed on the right, as in ``pandas.cut``, so tied
    scores always share a band. An integer `bins` places the interior edges
    at the sample quantiles of `score`, computed with the default linear
    interpolation of ``numpy.quantile``, and repeated edges are merged. The
    outer edges are infinite. Bands without observations are omitted.

    The columns are:

    - ``count`` and ``event``: observations and events in the band;
    - ``event_rate``: ``event`` divided by ``count``;
    - ``lift``: ``event_rate`` divided by the event rate of the sample;
    - ``cum_count_share``, ``cum_event_share`` and ``cum_non_event_share``:
      shares of all observations, events and non-events in the band and
      the bands above it;
    - ``ks``: the absolute difference between ``cum_event_share`` and
      ``cum_non_event_share``;
    - ``cum_lift``: ``cum_event_share`` divided by ``cum_count_share``.

    The largest ``ks`` is at most the value of `ks`, and the two are equal
    when each distinct score forms its own band.

    Examples
    --------
    >>> from kivi.metrics import gains_table
    >>> target = [0, 0, 1, 0, 1, 1, 0, 0]
    >>> score = [1, 2, 3, 3, 4, 5, 1, 2]
    >>> table = gains_table(target, score, bins=3)
    >>> table[["count", "event", "event_rate", "cum_event_share", "ks", "lift"]].round(3)
                count  event  event_rate  cum_event_share     ks  lift
    bin
    (3.0, inf]      2      2         1.0            0.667  0.667  2.667
    (2.0, 3.0]      2      1         0.5            1.000  0.800  1.333
    (-inf, 2.0]     4      0         0.0            1.000  0.000  0.000
    """
    labels, values = _labels_and_scores(target, score, "score")
    edges = _band_edges(values, bins)
    band = np.searchsorted(edges, values, side="left") - 1
    outside = (band < 0) | (band >= len(edges) - 1)
    if outside.any():
        raise ValueError(
            f"score has {outside.sum()} value(s) outside the edges in bins, such as "
            f"{float(values[outside][0])!r}; use infinite outer edges"
        )
    counts = np.bincount(2 * band + labels, minlength=2 * (len(edges) - 1)).reshape(-1, 2)
    observed = counts.sum(axis=1) > 0
    index = pd.IntervalIndex.from_breaks(edges, closed="right", name="bin")[observed][::-1]
    non_event, event = counts[observed][::-1].T
    count = non_event + event
    event_rate = event / count
    cum_event_share = np.cumsum(event) / event.sum()
    cum_non_event_share = np.cumsum(non_event) / non_event.sum()
    cum_count_share = np.cumsum(count) / count.sum()
    return pd.DataFrame(
        {
            "count": count,
            "event": event,
            "event_rate": event_rate,
            "lift": event_rate * (count.sum() / event.sum()),
            "cum_count_share": cum_count_share,
            "cum_event_share": cum_event_share,
            "cum_non_event_share": cum_non_event_share,
            "ks": np.abs(cum_event_share - cum_non_event_share),
            "cum_lift": cum_event_share / cum_count_share,
        },
        index=index,
    )


def brier_score(target: npt.ArrayLike, probability: npt.ArrayLike) -> float:
    r"""Compute the Brier score of predicted event probabilities.

    The Brier score is the mean squared difference between the predicted
    probability and the observed outcome, so a lower value is better.

    Parameters
    ----------
    target : array_like
        Binary labels, with 1 indicating the event. If `target` and
        `probability` are both pandas Series, they must have identical
        indices; otherwise they are matched by position.
    probability : array_like
        Predicted event probabilities from 0 to 1.

    Returns
    -------
    float
        The Brier score, from 0 for perfect forecasts to 1.

    Raises
    ------
    ValueError
        If the inputs are empty or not one-dimensional, their indices or
        lengths do not match, `target` has a label other than 0 and 1, or
        `probability` is not numeric or has a missing value or a value
        outside 0 to 1.

    See Also
    --------
    roc_auc : Ranking quality, which ignores the probability scale.

    Notes
    -----
    For outcomes :math:`y_i` and probabilities :math:`p_i`,

    .. math::

       \mathrm{BS} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2.

    Unlike the ranking measures, the Brier score depends on the probability
    values themselves, so it rewards calibration, and it is defined even when
    the sample has only one class. Predicting the event rate :math:`\bar y`
    for everyone scores :math:`\bar y (1 - \bar y)`, a useful baseline.

    References
    ----------
    .. [1] G. W. Brier, "Verification of forecasts expressed in terms of
       probability", Monthly Weather Review, 78(1), pp. 1-3, 1950.

    Examples
    --------
    >>> from kivi.metrics import brier_score
    >>> target = [0, 0, 1, 1]
    >>> round(brier_score(target, [0.1, 0.3, 0.6, 0.8]), 4)
    0.075
    """
    labels, values = _labels_and_scores(target, probability, "probability", both_classes=False)
    outside = (values < 0) | (values > 1)
    if outside.any():
        raise ValueError(f"probability must be between 0 and 1, got {float(values[outside][0])!r}")
    return float(np.mean((values - labels) ** 2))


def _labels_and_scores(
    target: npt.ArrayLike, score: npt.ArrayLike, name: str, *, both_classes: bool = True
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    """0/1 labels and finite float scores, matched by index or position like ``woe_table``."""
    if isinstance(target, pd.Series) and isinstance(score, pd.Series):
        if not target.index.equals(score.index):
            raise ValueError(f"target and {name} must have identical indices")
    target, score = as_series(target, "target"), as_series(score, name)
    if len(target) != len(score):
        raise ValueError(
            f"target and {name} must have the same length, got {len(target)} and {len(score)}"
        )
    if not len(target):
        raise ValueError(f"target and {name} must not be empty")
    # For nullable dtypes pandas 2.0 returns an object array, which ~ would negate as integers.
    invalid = ~target.isin([0, 1]).to_numpy(dtype=bool)
    if invalid.any():
        raise ValueError(f"target must contain only 0 and 1, got {_scalar(target[invalid])!r}")
    labels = target.to_numpy(dtype=np.intp)
    if both_classes and labels.sum() in (0, len(labels)):
        raise ValueError(f"target must contain both 0 and 1, got only {labels[0]}")
    if not pd.api.types.is_numeric_dtype(score.dtype):
        raise ValueError(f"{name} must be numeric, got dtype {score.dtype}")
    values = score.to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError(
            f"{name} must be finite, got {(~finite).sum()} missing or infinite value(s) "
            f"such as {float(values[~finite][0])!r}"
        )
    return labels, values


def _scalar(values: pd.Series):
    """First value of ``values`` as a Python scalar, for error messages."""
    value = values.iloc[0]
    return value.item() if isinstance(value, np.generic) else value


def _distinct_counts(labels: np.ndarray, values: np.ndarray) -> npt.NDArray[np.int64]:
    """(non_event, event) counts at each distinct score, from the lowest score to the highest."""
    distinct, position = np.unique(values, return_inverse=True)
    counts = np.bincount(2 * position.ravel() + labels, minlength=2 * len(distinct))
    return counts.reshape(-1, 2)


def _band_edges(values: np.ndarray, bins) -> npt.NDArray[np.float64]:
    """Increasing band edges: quantile edges for an integer ``bins``, else ``bins`` itself."""
    message = "bins must be a positive integer or at least two increasing edges, got {!r}"
    if isinstance(bins, Integral) and not isinstance(bins, bool):
        if bins < 1:
            raise ValueError(message.format(bins))
        # Keep an interior quantile equal to the minimum: (-inf, min] then holds the tied lowest
        # scores. One equal to the maximum leaves (max, inf] empty, which gains_table omits.
        interior = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])
        return np.r_[-np.inf, np.unique(interior), np.inf]
    try:
        edges = np.asarray(bins, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(message.format(bins)) from None
    if edges.ndim != 1 or len(edges) < 2 or not (np.diff(edges) > 0).all():
        raise ValueError(message.format(bins))
    return edges
