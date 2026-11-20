"""Performance of a score for a binary target.

Every function takes the score first and the binary ``target`` second, as
``kivi.woe.woe_table`` takes the groups first; scikit-learn's metrics take the
target first. The score is higher for a higher event probability, such as a
predicted probability of default. Scorecard points run the other way, so
negate them before passing them in.
"""

from numbers import Integral

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._validation import as_series, as_weights
from .binning import bin_edges

__all__ = ["roc_auc", "gini", "ks", "gains_table", "brier_score"]


def roc_auc(score: npt.ArrayLike, target: npt.ArrayLike) -> float:
    r"""Compute the area under the ROC curve of a score.

    The AUC is the probability that a randomly chosen event has a higher
    score than a randomly chosen non-event, with ties counted as one half.

    The score comes first and the target second, as in `kivi.woe.woe_table`;
    scikit-learn's metrics take the target first.

    Parameters
    ----------
    score : array_like
        Finite numeric scores that are higher for a higher event probability.
        Negate scorecard points, which are higher for a lower risk.
    target : array_like
        Binary labels, with 1 indicating the event. Both classes must be
        present. If `score` and `target` are both pandas Series, they must
        have identical indices; otherwise they are matched by position.

    Returns
    -------
    float
        The AUC, from 0 to 1. A score unrelated to the target gives about
        0.5, and a value below 0.5 suggests that the score runs the other way.

    Raises
    ------
    ValueError
        If the inputs are empty or not one-dimensional, their indices or
        lengths do not match, `target` has a label other than 0 and 1 or
        lacks one of them, or `score` is not real and numeric or has a
        missing or infinite value.

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
    >>> score = [0.1, 0.4, 0.4, 0.8, 0.7, 0.9]
    >>> target = [0, 0, 1, 0, 1, 1]
    >>> round(roc_auc(score, target), 4)
    0.7222

    Scorecard points are higher for a lower risk, so negate them:

    >>> import numpy as np
    >>> points = np.array([720, 650, 650, 560, 590, 510])
    >>> round(roc_auc(-points, target), 4)
    0.7222
    """
    values, labels, _ = _scores_and_labels(score, target, "score")
    non_event, event = _distinct_counts(labels, values).T
    # U counts each (event, non-event) pair in which the event scores higher, a tie as one half;
    # it equals R_1 - n_1 (n_1 + 1) / 2 without subtracting large rank sums.
    below = np.cumsum(non_event) - non_event
    u = np.sum(event * (below + 0.5 * non_event))
    return float(u / (float(event.sum()) * float(non_event.sum())))


def gini(score: npt.ArrayLike, target: npt.ArrayLike) -> float:
    r"""Compute the Gini coefficient of a score.

    The Gini coefficient rescales the AUC to run from -1 to 1, so that a score
    unrelated to the target gives about 0.

    The score comes first and the target second, as in `kivi.woe.woe_table`;
    scikit-learn's metrics take the target first.

    Parameters
    ----------
    score : array_like
        Finite numeric scores that are higher for a higher event probability.
        Negate scorecard points, which are higher for a lower risk.
    target : array_like
        Binary labels, with 1 indicating the event. Both classes must be
        present. If `score` and `target` are both pandas Series, they must
        have identical indices; otherwise they are matched by position.

    Returns
    -------
    float
        The Gini coefficient, from -1 to 1. A negative value suggests that
        the score runs the other way.

    Raises
    ------
    ValueError
        If the inputs are empty or not one-dimensional, their indices or
        lengths do not match, `target` has a label other than 0 and 1 or
        lacks one of them, or `score` is not real and numeric or has a
        missing or infinite value.

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
    >>> score = [0.1, 0.4, 0.4, 0.8, 0.7, 0.9]
    >>> target = [0, 0, 1, 0, 1, 1]
    >>> round(gini(score, target), 4)
    0.4444
    """
    return 2 * roc_auc(score, target) - 1


def ks(score: npt.ArrayLike, target: npt.ArrayLike) -> float:
    r"""Compute the Kolmogorov-Smirnov statistic of a score.

    The KS statistic is the largest gap between the cumulative distribution
    of the scores of events and that of non-events.

    The score comes first and the target second, as in `kivi.woe.woe_table`;
    scikit-learn's metrics take the target first.

    Parameters
    ----------
    score : array_like
        Finite numeric scores, such as predicted event probabilities.
    target : array_like
        Binary labels, with 1 indicating the event. Both classes must be
        present. If `score` and `target` are both pandas Series, they must
        have identical indices; otherwise they are matched by position.

    Returns
    -------
    float
        The largest gap, from 0 to 1.

    Raises
    ------
    ValueError
        If the inputs are empty or not one-dimensional, their indices or
        lengths do not match, `target` has a label other than 0 and 1 or
        lacks one of them, or `score` is not real and numeric or has a
        missing or infinite value.

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

    where :math:`t` runs over the observed scores. As in the two-sample
    Kolmogorov-Smirnov test, the gap is taken in absolute value, so negated
    scores give the same KS; use the sign of `gini` to check the direction.

    The largest gap can come from a region where the score runs the wrong
    way, for example when events gather among the lowest scores as well as
    the highest. The KS then measures how far apart the two distributions
    are rather than how well the score ranks events above non-events. The
    cumulative shares in `gains_table` show where the gap occurs.

    References
    ----------
    .. [1] L. C. Thomas, D. B. Edelman and J. N. Crook, "Credit Scoring and
       Its Applications", SIAM, 2002.

    Examples
    --------
    >>> from kivi.metrics import ks
    >>> score = [0.1, 0.4, 0.4, 0.8, 0.7, 0.9]
    >>> target = [0, 0, 1, 0, 1, 1]
    >>> round(ks(score, target), 4)
    0.3333
    """
    values, labels, _ = _scores_and_labels(score, target, "score")
    non_event, event = _distinct_counts(labels, values).T
    gap = np.cumsum(event) / event.sum() - np.cumsum(non_event) / non_event.sum()
    return float(np.max(np.abs(gap)))


def gains_table(
    score: npt.ArrayLike, target: npt.ArrayLike, *, bins: int | npt.ArrayLike = 10
) -> pd.DataFrame:
    """Summarize events by score band, from the highest scores to the lowest.

    Each row shows how many events a band holds and what share of all events
    is captured by reviewing that band and every band above it.

    The score comes first and the target second, as in `kivi.woe.woe_table`;
    scikit-learn's metrics take the target first.

    Parameters
    ----------
    score : array_like
        Finite numeric scores that are higher for a higher event probability.
        Negate scorecard points, which are higher for a lower risk.
    target : array_like
        Binary labels, with 1 indicating the event. Both classes must be
        present. If `score` and `target` are both pandas Series, they must
        have identical indices; otherwise they are matched by position.
    bins : int or array_like, default 10
        Number of quantile bands, or increasing band edges. An integer is the
        same as passing ``kivi.binning.bin_edges(score, bins=bins)``, so tied
        scores can leave fewer bands than requested. Pass the edges fitted on
        a development sample to compare later samples on the same bands.

    Returns
    -------
    pandas.DataFrame
        One row per score band, from the highest band to the lowest, indexed
        by the interval of the band. The columns are described in the Notes.

    Raises
    ------
    ValueError
        If the inputs are empty or not one-dimensional, their indices or
        lengths do not match, `target` has a label other than 0 and 1 or
        lacks one of them, `score` is not real and numeric or has a missing
        or infinite value, `bins` is neither a positive integer nor
        increasing edges, or a score lies outside the edges.

    Warns
    -----
    UserWarning
        If `bins` is an integer and tied scores leave fewer bands than
        requested, as in `kivi.binning.bin_edges`.

    See Also
    --------
    ks : The largest gap over all scores rather than band edges.
    roc_auc : Ranking quality over all pairs of events and non-events.
    kivi.binning.bin_edges : Fit band edges on a development sample.

    Notes
    -----
    Bands are intervals closed on the right, as in ``pandas.cut``, so tied
    scores always share a band. Every band is reported, including bands
    without observations, so samples cut on the same edges give tables with
    the same rows. An empty band has a NaN ``event_rate`` and ``lift``, and
    ``cum_lift`` is NaN until the first band with observations.

    The columns are:

    - ``non_event``, ``event`` and ``total``: non-events, events and all
      observations in the band;
    - ``event_rate``: ``event`` divided by ``total``;
    - ``lift``: ``event_rate`` divided by the event rate of the sample;
    - ``cum_non_event_share``, ``cum_event_share`` and ``cum_total_share``:
      shares of all non-events, events and observations in the band and the
      bands above it;
    - ``ks``: the absolute difference between ``cum_event_share`` and
      ``cum_non_event_share``;
    - ``cum_lift``: ``cum_event_share`` divided by ``cum_total_share``.

    The largest ``ks`` is at most the value of `ks`, and the two are equal
    when each distinct score forms its own band.

    Examples
    --------
    >>> from kivi.metrics import gains_table
    >>> score = [1, 2, 3, 3, 4, 5, 1, 2]
    >>> target = [0, 0, 1, 0, 1, 1, 0, 0]
    >>> table = gains_table(score, target, bins=3)
    >>> table[["total", "event", "event_rate", "cum_event_share", "ks", "lift"]].round(3)
                total  event  event_rate  cum_event_share     ks  lift
    bin
    (3.0, inf]      2      2         1.0            0.667  0.667  2.667
    (2.0, 3.0]      2      1         0.5            1.000  0.800  1.333
    (-inf, 2.0]     4      0         0.0            1.000  0.000  0.000
    """
    values, labels, _ = _scores_and_labels(score, target, "score")
    edges = _band_edges(values, bins)
    band = np.searchsorted(edges, values, side="left") - 1
    outside = (band < 0) | (band >= len(edges) - 1)
    if outside.any():
        raise ValueError(
            f"score has {outside.sum()} value(s) outside the edges in bins, such as "
            f"{float(values[outside][0])!r}; use infinite outer edges"
        )
    counts = np.bincount(2 * band + labels, minlength=2 * (len(edges) - 1)).reshape(-1, 2)
    # Highest band first; empty bands stay so that tables cut on the same edges line up.
    index = pd.IntervalIndex.from_breaks(edges, closed="right", name="bin")[::-1]
    non_event, event = counts[::-1].T
    total = non_event + event
    inverse_rate = total.sum() / event.sum()
    event_rate = _ratio(event, total)
    cum_event_share = np.cumsum(event) / event.sum()
    cum_non_event_share = np.cumsum(non_event) / non_event.sum()
    return pd.DataFrame(
        {
            "non_event": non_event,
            "event": event,
            "total": total,
            "event_rate": event_rate,
            "lift": event_rate * inverse_rate,
            "cum_non_event_share": cum_non_event_share,
            "cum_event_share": cum_event_share,
            "cum_total_share": np.cumsum(total) / total.sum(),
            "ks": np.abs(cum_event_share - cum_non_event_share),
            "cum_lift": _ratio(np.cumsum(event), np.cumsum(total)) * inverse_rate,
        },
        index=index,
    )


def brier_score(probability: npt.ArrayLike, target: npt.ArrayLike) -> float:
    r"""Compute the Brier score of predicted event probabilities.

    The Brier score is the mean squared difference between the predicted
    probability and the observed outcome, so a lower value is better.

    The probabilities come first and the target second, as in
    `kivi.woe.woe_table`; scikit-learn's metrics take the target first.

    Parameters
    ----------
    probability : array_like
        Predicted event probabilities from 0 to 1.
    target : array_like
        Binary labels, with 1 indicating the event. If `probability` and
        `target` are both pandas Series, they must have identical indices;
        otherwise they are matched by position.

    Returns
    -------
    float
        The Brier score, from 0 for perfect forecasts to 1.

    Raises
    ------
    ValueError
        If the inputs are empty or not one-dimensional, their indices or
        lengths do not match, `target` has a label other than 0 and 1, or
        `probability` is not real and numeric or has a missing value or a
        value outside 0 to 1.

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
    >>> round(brier_score([0.1, 0.3, 0.6, 0.8], target), 4)
    0.075
    """
    values, labels, _ = _scores_and_labels(probability, target, "probability", both_classes=False)
    outside = (values < 0) | (values > 1)
    if outside.any():
        raise ValueError(f"probability must be between 0 and 1, got {float(values[outside][0])!r}")
    return float(np.mean((values - labels) ** 2))


def _scores_and_labels(
    score: npt.ArrayLike,
    target: npt.ArrayLike,
    name: str,
    *,
    both_classes: bool = True,
    weights: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp], npt.NDArray[np.float64] | None]:
    """Finite float scores, 0/1 labels and weights, matched by index or position like ``woe_table``.

    The weights are None without `weights`. With `both_classes`, both classes
    must be present and, with `weights`, have a positive total weight.
    """
    if isinstance(score, pd.Series) and isinstance(target, pd.Series):
        if not score.index.equals(target.index):
            raise ValueError(f"{name} and target must have identical indices")
    raw_score = score
    score, target = as_series(score, name), as_series(target, "target")
    if len(score) != len(target):
        raise ValueError(
            f"{name} and target must have the same length, got {len(score)} and {len(target)}"
        )
    if not len(score):
        raise ValueError(f"{name} and target must not be empty")
    if not pd.api.types.is_numeric_dtype(score.dtype):
        raise ValueError(f"{name} must be numeric, got dtype {score.dtype}")
    if pd.api.types.is_complex_dtype(score.dtype):
        raise ValueError(f"{name} must be real, got dtype {score.dtype}")
    values = score.to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError(
            f"{name} must be finite, got {(~finite).sum()} missing or infinite value(s) "
            f"such as {float(values[~finite][0])!r}"
        )
    # For nullable dtypes pandas 2.0 returns an object array, which ~ would negate as integers.
    invalid = ~target.isin([0, 1]).to_numpy(dtype=bool)
    if invalid.any():
        raise ValueError(f"target must contain only 0 and 1, got {_scalar(target[invalid])!r}")
    labels = target.to_numpy(dtype=np.intp)
    if both_classes and labels.sum() in (0, len(labels)):
        raise ValueError(f"target must contain both 0 and 1, got only {labels[0]}")
    mass = as_weights(weights, raw_score, of=name)
    if both_classes and mass is not None:
        if not (np.bincount(labels, weights=mass, minlength=2) > 0).all():
            raise ValueError("weights must give both target classes a positive total")
    return values, labels, mass


def _scalar(values: pd.Series):
    """First value of ``values`` as a Python scalar, for error messages."""
    value = values.iloc[0]
    return value.item() if isinstance(value, np.generic) else value


def _distinct_counts(labels: np.ndarray, values: np.ndarray) -> npt.NDArray[np.int64]:
    """(non_event, event) counts at each distinct score, from the lowest score to the highest."""
    distinct, position = np.unique(values, return_inverse=True)
    counts = np.bincount(2 * position.ravel() + labels, minlength=2 * len(distinct))
    return counts.reshape(-1, 2)


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> npt.NDArray[np.float64]:
    """Elementwise ratio that is NaN, without a warning, where the denominator is 0."""
    result = np.full(len(numerator), np.nan)
    return np.divide(numerator, denominator, out=result, where=denominator > 0)


def _band_edges(values: np.ndarray, bins) -> npt.NDArray[np.float64]:
    """Increasing band edges: quantile edges for an integer ``bins``, else ``bins`` itself."""
    message = "bins must be a positive integer or at least two increasing edges, got {!r}"
    if isinstance(bins, Integral) and not isinstance(bins, bool):
        if bins < 1:
            raise ValueError(message.format(bins))
        # The same edges, and the same warning for tied quantiles, as bin_edges.
        return bin_edges(values, bins=int(bins))
    try:
        edges = np.asarray(bins, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(message.format(bins)) from None
    if edges.ndim != 1 or len(edges) < 2 or not (np.diff(edges) > 0).all():
        raise ValueError(message.format(bins))
    return edges
