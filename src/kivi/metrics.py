"""Performance of a score for a binary target.

Every function takes the score first and the binary ``target`` second, as
``kivi.woe.woe_table`` takes the groups first; scikit-learn's metrics take the
target first. The score is higher for a higher event probability, such as a
predicted probability of default. Scorecard points run the other way, so
negate them before passing them in.
"""

import math
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._stats import normal_quantile
from ._validation import scores_and_labels
from .binning import _fit_edges

__all__ = [
    "roc_auc",
    "gini",
    "ks",
    "roc_table",
    "roc_auc_interval",
    "delong_test",
    "gains_table",
    "brier_score",
]


def roc_auc(
    score: npt.ArrayLike, target: npt.ArrayLike, *, weights: npt.ArrayLike | None = None
) -> float:
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
    weights : array_like, optional
        Non-negative weight of each observation, such as the inverse of its
        sampling rate, matched like `target`. Every count becomes a sum of
        weights, so integer weights give the same result as repeating rows.

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
        missing or infinite value. Also if `weights` is not numeric, does
        not give each observation one finite, non-negative weight, has an
        index other than that of `score`, or leaves a class without weight.

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

    With weights, each pair of an event :math:`i` and a non-event :math:`j`
    counts with weight :math:`w_i w_j`, and :math:`n_1` and :math:`n_0` are
    the total weights of the two classes.

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
    values, labels, mass = scores_and_labels(score, target, "score", weights=weights)
    return _auc(values, labels, mass)


def gini(
    score: npt.ArrayLike, target: npt.ArrayLike, *, weights: npt.ArrayLike | None = None
) -> float:
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
    weights : array_like, optional
        Non-negative weight of each observation, such as the inverse of its
        sampling rate, matched like `target`. Every count becomes a sum of
        weights, so integer weights give the same result as repeating rows.

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
        missing or infinite value. Also if `weights` is not numeric, does
        not give each observation one finite, non-negative weight, has an
        index other than that of `score`, or leaves a class without weight.

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
    return 2 * roc_auc(score, target, weights=weights) - 1


def ks(
    score: npt.ArrayLike, target: npt.ArrayLike, *, weights: npt.ArrayLike | None = None
) -> float:
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
    weights : array_like, optional
        Non-negative weight of each observation, such as the inverse of its
        sampling rate, matched like `target`. Every count becomes a sum of
        weights, so integer weights give the same result as repeating rows.

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
        missing or infinite value. Also if `weights` is not numeric, does
        not give each observation one finite, non-negative weight, has an
        index other than that of `score`, or leaves a class without weight.

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

    where :math:`t` runs over the observed scores. With weights, the
    distribution functions are weighted. As in the two-sample
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
    values, labels, mass = scores_and_labels(score, target, "score", weights=weights)
    non_event, event = _distinct_counts(labels, values, mass).T
    gap = np.cumsum(event) / event.sum() - np.cumsum(non_event) / non_event.sum()
    return float(np.max(np.abs(gap)))


def roc_table(
    score: npt.ArrayLike, target: npt.ArrayLike, *, weights: npt.ArrayLike | None = None
) -> pd.DataFrame:
    r"""Tabulate the ROC curve of a score, with one row per distinct score.

    Each row flags the observations that score at or above its threshold and
    counts the events and non-events among them. Plotting
    ``true_positive_rate`` against ``false_positive_rate`` gives the ROC
    curve, from the highest threshold to the lowest.

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
    weights : array_like, optional
        Non-negative weight of each observation, such as the inverse of its
        sampling rate, matched like `target`. The counts become sums of
        weights, and a score whose observations all have zero weight has no
        row, so integer weights give the same table as repeating rows.

    Returns
    -------
    pandas.DataFrame
        One row per threshold, in an index named ``threshold``: first
        infinity, which flags no observation, then every distinct score from
        the highest to the lowest, which flags them all. The columns are
        described in the Notes.

    Raises
    ------
    ValueError
        If the inputs are empty or not one-dimensional, their indices or
        lengths do not match, `target` has a label other than 0 and 1 or
        lacks one of them, or `score` is not real and numeric or has a
        missing or infinite value. Also if `weights` is not numeric, does
        not give each observation one finite, non-negative weight, has an
        index other than that of `score`, or leaves a class without weight.

    See Also
    --------
    roc_auc : The area under the curve.
    ks : The largest gap between the two rates.
    gains_table : The same cumulative shares by score band.

    Notes
    -----
    The columns are:

    - ``cum_non_event`` and ``cum_event``: non-events and events that score
      at or above the threshold, or the sums of their weights;
    - ``false_positive_rate`` and ``true_positive_rate``: these counts as
      shares of all non-events and of all events;
    - ``precision``: ``cum_event`` divided by the sum of both counts, the
      event rate among the flagged observations, which is NaN in the first
      row.

    Tied scores share a row, so the curve crosses a tie diagonally. The
    trapezoids under the curve therefore add up to `roc_auc`, which counts a
    tied pair as one half,

    .. math::

       \mathrm{AUC} = \sum_{k \ge 1} (\mathrm{FPR}_k - \mathrm{FPR}_{k-1})
       \, \frac{\mathrm{TPR}_k + \mathrm{TPR}_{k-1}}{2},

    and the largest gap between the two rates, :math:`\max_k \lvert
    \mathrm{TPR}_k - \mathrm{FPR}_k \rvert`, is `ks`. The rows are those of
    ``sklearn.metrics.roc_curve`` with ``drop_intermediate=False``.

    Examples
    --------
    >>> from kivi.metrics import roc_table
    >>> score = [0.1, 0.4, 0.4, 0.8, 0.7, 0.9]
    >>> target = [0, 0, 1, 0, 1, 1]
    >>> table = roc_table(score, target)
    >>> table[["cum_non_event", "cum_event", "false_positive_rate", "true_positive_rate"]].round(3)
               cum_non_event  cum_event  false_positive_rate  true_positive_rate
    threshold
    inf                    0          0                0.000               0.000
    0.9                    0          1                0.000               0.333
    0.8                    1          1                0.333               0.333
    0.7                    1          2                0.333               0.667
    0.4                    2          3                0.667               1.000
    0.1                    3          3                1.000               1.000
    >>> table["precision"].round(3).tolist()
    [nan, 1.0, 0.5, 0.667, 0.6, 0.5]
    """
    values, labels, mass = scores_and_labels(score, target, "score", weights=weights)
    if mass is not None:
        # Repeated rows would leave out an observation of zero weight, and with it its score.
        kept = mass > 0
        values, labels, mass = values[kept], labels[kept], mass[kept]
    distinct, counts, _ = _score_counts(values, labels, mass)
    # Highest score first, after a row that flags no observation.
    counts = np.concatenate([np.zeros((1, 2), dtype=counts.dtype), counts[::-1]])
    cum_non_event, cum_event = np.cumsum(counts, axis=0).T
    return pd.DataFrame(
        {
            "cum_non_event": cum_non_event,
            "cum_event": cum_event,
            "false_positive_rate": cum_non_event / cum_non_event[-1],
            "true_positive_rate": cum_event / cum_event[-1],
            "precision": _ratio(cum_event, cum_non_event + cum_event),
        },
        index=pd.Index(np.concatenate([[np.inf], distinct[::-1]]), name="threshold"),
    )


def roc_auc_interval(
    score: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    level: float = 0.95,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Estimate the standard error and a confidence interval of the AUC.

    The standard error is that of DeLong, DeLong and Clarke-Pearson [1]_,
    which makes no assumption about the distribution of the scores. The
    interval of the Gini coefficient follows from that of the AUC.

    The score comes first and the target second, as in `kivi.woe.woe_table`;
    scikit-learn's metrics take the target first.

    Parameters
    ----------
    score : array_like
        Finite numeric scores that are higher for a higher event probability.
        Negate scorecard points, which are higher for a lower risk.
    target : array_like
        Binary labels, with 1 indicating the event. Without `weights` there
        must be at least two events and two non-events. If `score` and
        `target` are both pandas Series, they must have identical indices;
        otherwise they are matched by position.
    level : float, default 0.95
        Confidence level of the interval, between 0 and 1.
    weights : array_like, optional
        Non-negative frequency weight of each observation, matched like
        `target`. An observation of weight :math:`w` counts as :math:`w`
        observations, so integer weights give the same result as repeating
        rows, standard error included. The weights of each class must add up
        to more than 1.

    Returns
    -------
    pandas.DataFrame
        Two rows, ``auc`` and ``gini``, and the columns ``estimate``, ``se``
        for the standard error, and ``lower`` and ``upper`` for the bounds of
        the interval.

    Raises
    ------
    ValueError
        If the inputs are empty or not one-dimensional, their indices or
        lengths do not match, `target` has a label other than 0 and 1 or,
        without `weights`, fewer than two of either, or `score` is not real
        and numeric or has a missing or infinite value. Also if `level` is
        not a number between 0 and 1, or if `weights` is not numeric, does
        not give each observation one finite, non-negative weight, has an
        index other than that of `score`, or gives a class a total weight of
        1 or less.

    See Also
    --------
    roc_auc : The AUC alone.
    delong_test : Compare the AUCs of two scores on the same observations.

    Notes
    -----
    Let :math:`m` be the number of events and :math:`n` that of non-events.
    The placement value :math:`V_{10,i}` of event :math:`i` is the share of
    non-events that score below it, and the placement value :math:`V_{01,j}`
    of non-event :math:`j` is the share of events that score above it, with
    ties counted as one half. Both average to the AUC :math:`\hat\theta`, and
    its variance is estimated by

    .. math::

       \widehat{\mathrm{Var}}(\hat\theta) = \frac{S_{10}}{m} + \frac{S_{01}}{n},
       \qquad
       S_{10} = \frac{1}{m - 1} \sum_{i=1}^{m} (V_{10,i} - \hat\theta)^2,

    with :math:`S_{01}` defined in the same way over the non-events. The
    placement values come from midranks, as in the algorithm of Sun and
    Xu [2]_: the rank of an event among all scores minus its rank among the
    events is the number of non-events below it, a tie counting as one
    half. The computation thus takes :math:`O(N \log N)` time for :math:`N`
    observations instead of comparing every pair.

    The interval is :math:`\hat\theta \pm z \, \mathrm{SE}`, cut off at 0
    and 1, where :math:`z` is the quantile of the standard normal
    distribution at :math:`(1 + \mathrm{level}) / 2`. For the Gini
    coefficient, :math:`2\,\mathrm{AUC} - 1`, the estimate and the bounds
    are rescaled in the same way and the standard error doubles.

    The normal approximation is poor when the AUC is close to 0 or 1 or a
    class has few observations. A score that separates the two classes
    perfectly has a standard error of 0.

    With weights, :math:`m` and :math:`n` are the total weights of the two
    classes, and each squared deviation counts with the weight of its
    observation. Weights that stand for a population larger than the sample,
    such as inverse sampling rates, therefore give too small a standard
    error.

    References
    ----------
    .. [1] E. R. DeLong, D. M. DeLong and D. L. Clarke-Pearson, "Comparing
       the areas under two or more correlated receiver operating
       characteristic curves: a nonparametric approach", Biometrics, 44(3),
       pp. 837-845, 1988.
    .. [2] X. Sun and W. Xu, "Fast implementation of DeLong's algorithm for
       comparing the areas under correlated receiver operating
       characteristic curves", IEEE Signal Processing Letters, 21(11),
       pp. 1389-1393, 2014.

    Examples
    --------
    Six observations leave a wide interval, whose upper bound is cut off at
    1:

    >>> from kivi.metrics import roc_auc_interval
    >>> score = [0.1, 0.4, 0.4, 0.8, 0.7, 0.9]
    >>> target = [0, 0, 1, 0, 1, 1]
    >>> roc_auc_interval(score, target).round(4)
          estimate      se   lower  upper
    auc     0.7222  0.2485  0.2353    1.0
    gini    0.4444  0.4969 -0.5295    1.0
    """
    values, labels, mass = scores_and_labels(score, target, "score", weights=weights)
    _check_class_totals(labels, mass)
    z = _critical_value(level)
    auc = _auc(values, labels, mass)
    se = math.sqrt(_delong_variance(_placements(values, labels, mass), labels, mass))
    # A weighted AUC can round to just past 0 or 1, so clip both bounds.
    lower, upper = np.clip([auc - z * se, auc + z * se], 0.0, 1.0).tolist()
    return pd.DataFrame(
        {
            "estimate": [auc, 2 * auc - 1],
            "se": [se, 2 * se],
            "lower": [lower, 2 * lower - 1],
            "upper": [upper, 2 * upper - 1],
        },
        index=["auc", "gini"],
    )


def delong_test(
    current: npt.ArrayLike,
    new: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    level: float = 0.95,
    weights: npt.ArrayLike | None = None,
) -> pd.Series:
    r"""Test whether two scores of the same observations have the same AUC.

    DeLong's test [1]_ compares two scores of the same sample, such as a new
    model and the model it would replace. Both scores rank the same pairs of
    events and non-events, so their AUCs are correlated, and the standard
    error of the difference takes that into account.

    The scores come first and the target last, as in
    `kivi.strategy.swap_set`; scikit-learn's metrics take the target first.

    Parameters
    ----------
    current : array_like
        Finite numeric scores of one model, such as the model in use, that
        are higher for a higher event probability.
    new : array_like
        Finite numeric scores of the other model for the same observations,
        in the same direction.
    target : array_like
        Binary labels, with 1 indicating the event. Without `weights` there
        must be at least two events and two non-events. Any two of
        `current`, `new` and `target` that are pandas Series must have
        identical indices; otherwise they are matched by position.
    level : float, default 0.95
        Confidence level of the interval of the difference, between 0 and 1.
    weights : array_like, optional
        Non-negative frequency weight of each observation, matched like
        `target`. An observation of weight :math:`w` counts as :math:`w`
        observations, so integer weights give the same result as repeating
        rows. The weights of each class must add up to more than 1.

    Returns
    -------
    pandas.Series
        ``current_auc`` and ``new_auc``, the AUCs of the two scores;
        ``difference``, which is ``new_auc`` minus ``current_auc``; ``se``,
        its standard error; ``lower`` and ``upper``, the bounds of its
        interval at `level`, cut off at -1 and 1; ``z``, the difference
        divided by its standard error; and ``p_value``, the two-sided
        p-value of ``z``.

    Raises
    ------
    ValueError
        If the inputs are empty or not one-dimensional, their indices or
        lengths do not match, `target` has a label other than 0 and 1 or,
        without `weights`, fewer than two of either, or `current` or `new`
        is not real and numeric or has a missing or infinite value. Also if
        `level` is not a number between 0 and 1, or if `weights` is not
        numeric, does not give each observation one finite, non-negative
        weight, has an index other than that of a Series among the scores,
        or gives a class a total weight of 1 or less.

    See Also
    --------
    roc_auc_interval : The standard error and interval of one AUC.
    kivi.strategy.swap_set : The applicants that two scores decide
        differently at their cut-offs.

    Notes
    -----
    With the placement values of `roc_auc_interval` computed for each score,
    let :math:`D_{10,i}` be the placement value of event :math:`i` under the
    new score minus that under the current one, and :math:`D_{01,j}` the
    same difference for non-event :math:`j`. They average to the difference
    of the AUCs, and its variance is estimated by

    .. math::

       \widehat{\mathrm{Var}}(\hat\theta_{\mathrm{new}} -
       \hat\theta_{\mathrm{current}}) = \frac{S^D_{10}}{m} +
       \frac{S^D_{01}}{n},

    where :math:`S^D_{10}` and :math:`S^D_{01}` are the sample variances of
    :math:`D_{10,i}` over the :math:`m` events and of :math:`D_{01,j}` over
    the :math:`n` non-events. This equals the variance of the contrast in
    DeLong et al., which adds the two variances and subtracts twice the
    covariance. Under the hypothesis of equal AUCs, :math:`z` is
    approximately standard normal.

    For a one-sided test that the new score ranks better, halve
    ``p_value`` when ``z`` is positive. When the two scores order every pair
    of an event and a non-event in the same way, the difference and its
    standard error are both 0, and ``z`` and ``p_value`` are NaN; with
    fractional weights, a difference and a standard error that are both
    within rounding error of 0 count as 0. A difference with a standard
    error of 0, such as that between a constant score and one that
    separates the classes, gives an infinite ``z`` and a ``p_value`` of 0.

    Weights count as in `roc_auc_interval`, so inverse sampling rates give
    too small a standard error.

    References
    ----------
    .. [1] E. R. DeLong, D. M. DeLong and D. L. Clarke-Pearson, "Comparing
       the areas under two or more correlated receiver operating
       characteristic curves: a nonparametric approach", Biometrics, 44(3),
       pp. 837-845, 1988.

    Examples
    --------
    The new score separates the six observations perfectly:

    >>> from kivi.metrics import delong_test
    >>> current = [0.1, 0.4, 0.4, 0.8, 0.7, 0.9]
    >>> new = [0.2, 0.3, 0.6, 0.5, 0.7, 0.9]
    >>> target = [0, 0, 1, 0, 1, 1]
    >>> delong_test(current, new, target).round(4)
    current_auc    0.7222
    new_auc        1.0000
    difference     0.2778
    se             0.2485
    lower         -0.2092
    upper          0.7647
    z              1.1180
    p_value        0.2636
    dtype: float64
    """
    if isinstance(current, pd.Series) and isinstance(new, pd.Series):
        if not current.index.equals(new.index):
            raise ValueError("current and new must have identical indices")
    current_values, labels, mass = scores_and_labels(current, target, "current", weights=weights)
    new_values, _, _ = scores_and_labels(new, target, "new", weights=weights)
    _check_class_totals(labels, mass)
    z_level = _critical_value(level)
    current_auc = _auc(current_values, labels, mass)
    new_auc = _auc(new_values, labels, mass)
    placement = _placements(new_values, labels, mass) - _placements(current_values, labels, mass)
    se = math.sqrt(_delong_variance(placement, labels, mass))
    difference = new_auc - current_auc
    if mass is not None:
        # Counts are exact, but sums of fractional weights are rounded, and in another order for
        # each score: scores that order every pair alike can differ by rounding alone.
        rounding = 8 * np.finfo(float).eps * len(labels)
        if abs(difference) <= rounding and se <= rounding:
            difference = se = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = float(np.float64(difference) / se)
    p_value = math.nan if math.isnan(z) else math.erfc(abs(z) / math.sqrt(2))
    lower, upper = np.clip([difference - z_level * se, difference + z_level * se], -1.0, 1.0)
    return pd.Series(
        {
            "current_auc": current_auc,
            "new_auc": new_auc,
            "difference": difference,
            "se": se,
            "lower": float(lower),
            "upper": float(upper),
            "z": z,
            "p_value": p_value,
        }
    )


def gains_table(
    score: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    bins: int | npt.ArrayLike = 10,
    weights: npt.ArrayLike | None = None,
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
        With `weights`, an integer gives weighted quantile edges, as
        ``bin_edges(score, bins=bins, weights=weights)`` does.
    weights : array_like, optional
        Non-negative weight of each observation, such as the inverse of its
        sampling rate, matched like `target`. The counts become sums of
        weights. On the same edges, integer weights give the same table as
        repeating rows; an integer `bins` can give other edges, because
        weighted quantiles use the inverted CDF.

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
        increasing edges, or a score lies outside the edges. Also if
        `weights` is not numeric, does not give each observation one finite,
        non-negative weight, has an index other than that of `score`, or
        leaves a class without weight.

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
      observations in the band, or the sums of their weights;
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
    values, labels, mass = scores_and_labels(score, target, "score", weights=weights)
    edges = _band_edges(values, bins, mass)
    band = np.searchsorted(edges, values, side="left") - 1
    outside = (band < 0) | (band >= len(edges) - 1)
    if outside.any():
        raise ValueError(
            f"score has {outside.sum()} value(s) outside the edges in bins, such as "
            f"{float(values[outside][0])!r}; use infinite outer edges"
        )
    counts = np.bincount(2 * band + labels, weights=mass, minlength=2 * (len(edges) - 1))
    counts = counts.reshape(-1, 2)
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


def brier_score(
    probability: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
) -> float:
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
    weights : array_like, optional
        Non-negative weight of each observation, matched like `target`. The
        score is then the weighted mean of the squared differences.

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
        value outside 0 to 1. Also if `weights` is not numeric, does not
        give each observation one finite, non-negative weight with a positive
        total, or has an index other than that of `probability`.

    See Also
    --------
    roc_auc : Ranking quality, which ignores the probability scale.

    Notes
    -----
    For outcomes :math:`y_i` and probabilities :math:`p_i`,

    .. math::

       \mathrm{BS} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2,

    or :math:`\sum_i w_i (p_i - y_i)^2 / \sum_i w_i` with weights.

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
    values, labels, mass = scores_and_labels(
        probability, target, "probability", both_classes=False, weights=weights
    )
    outside = (values < 0) | (values > 1)
    if outside.any():
        raise ValueError(f"probability must be between 0 and 1, got {float(values[outside][0])!r}")
    return float(np.average((values - labels) ** 2, weights=mass))


def _auc(values: np.ndarray, labels: np.ndarray, weights: np.ndarray | None) -> float:
    """AUC from validated scores, labels and weights, with ties counted as one half."""
    if weights is not None:
        # Scaling by a power of two is exact and keeps the product of the class totals finite.
        weights = np.ldexp(weights, -int(np.frexp(weights.max())[1]))
    non_event, event = _distinct_counts(labels, values, weights).T
    # U counts each (event, non-event) pair in which the event scores higher, a tie as one half;
    # it equals R_1 - n_1 (n_1 + 1) / 2 without subtracting large rank sums.
    below = np.cumsum(non_event) - non_event
    u = np.sum(event * (below + 0.5 * non_event))
    return float(u / (float(event.sum()) * float(non_event.sum())))


def _placements(
    values: np.ndarray, labels: np.ndarray, weights: np.ndarray | None
) -> npt.NDArray[np.float64]:
    """Placement value of each observation, the structural components of DeLong et al.

    For an event, the share of non-events that score below it; for a non-event, the share of
    events that score above it; a tie counts as one half. With weights, the shares are shares of
    weight, as they would be among repeated rows.
    """
    _, counts, position = _score_counts(values, labels, weights)
    cum_non_event, cum_event = np.cumsum(counts, axis=0).T
    non_event, event = counts.T
    # Rank among all scores minus rank among the own class, both midranks (Sun and Xu, 2014).
    event_placement = (cum_non_event - 0.5 * non_event) / cum_non_event[-1]
    non_event_placement = (cum_event[-1] - cum_event + 0.5 * event) / cum_event[-1]
    return np.where(labels == 1, event_placement[position], non_event_placement[position])


def _delong_variance(
    placement: np.ndarray, labels: np.ndarray, weights: np.ndarray | None
) -> float:
    """S10 / m + S01 / n from placement values, reading weights as frequency weights."""
    variance = 0.0
    for side in (0, 1):
        in_class = labels == side
        values = placement[in_class]
        mass = np.ones(len(values)) if weights is None else weights[in_class]
        total = mass.sum()
        deviation = values - np.sum(mass * values) / total
        # Dividing twice keeps a huge total weight from overflowing the product.
        variance += np.sum(mass * deviation**2) / (total - 1) / total
    return float(variance)


def _check_class_totals(labels: np.ndarray, weights: np.ndarray | None) -> None:
    """Require the sample sizes that a standard error needs: more than 1 in each class."""
    totals = np.bincount(labels, weights=weights, minlength=2)
    if weights is None:
        if (totals < 2).any():
            raise ValueError(
                "target must contain at least two events and two non-events for a standard "
                f"error, got {totals[1]} event(s) and {totals[0]} non-event(s)"
            )
    elif not (totals > 1).all():
        raise ValueError(
            "weights must give each target class a total above 1 for a standard error, got "
            f"{float(totals[0])!r} for 0 and {float(totals[1])!r} for 1"
        )


def _critical_value(level) -> float:
    """Return the normal quantile at (1 + level) / 2, for an interval at confidence `level`."""
    # The check is on the float that is used, so a level that rounds to 1 is rejected too.
    if isinstance(level, bool) or not isinstance(level, Real) or not 0 < float(level) < 1:
        raise ValueError(f"level must be a number between 0 and 1, got {level!r}")
    # (1 - level) / 2 keeps its precision when level is close to 1; 0.5 + level / 2 would not.
    return -normal_quantile((1 - float(level)) / 2)


def _score_counts(
    values: np.ndarray, labels: np.ndarray, weights: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the distinct scores, their (non_event, event) counts and each observation's position.

    The distinct scores run from the lowest to the highest.
    """
    distinct, position = np.unique(values, return_inverse=True)
    position = position.ravel()
    counts = np.bincount(2 * position + labels, weights=weights, minlength=2 * len(distinct))
    return distinct, counts.reshape(-1, 2), position


def _distinct_counts(
    labels: np.ndarray, values: np.ndarray, weights: np.ndarray | None = None
) -> npt.NDArray[np.int64] | npt.NDArray[np.float64]:
    """(non_event, event) counts at each distinct score, from the lowest score to the highest."""
    return _score_counts(values, labels, weights)[1]


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> npt.NDArray[np.float64]:
    """Elementwise ratio that is NaN, without a warning, where the denominator is 0."""
    result = np.full(len(numerator), np.nan)
    return np.divide(numerator, denominator, out=result, where=denominator > 0)


def _band_edges(values: np.ndarray, bins, weights: np.ndarray | None) -> npt.NDArray[np.float64]:
    """Increasing band edges: quantile edges for an integer ``bins``, else ``bins`` itself."""
    message = "bins must be a positive integer or at least two increasing edges, got {!r}"
    if isinstance(bins, Integral) and not isinstance(bins, bool):
        if bins < 1:
            raise ValueError(message.format(bins))
        # The edges of bin_edges; the warning for tied quantiles points at the caller of
        # gains_table and gives advice that applies to it.
        return _fit_edges(
            values,
            int(bins),
            weights=weights,
            advice="pass fewer bins or explicit edges",
            stacklevel=4,
        )
    try:
        edges = np.asarray(bins, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(message.format(bins)) from None
    if edges.ndim != 1 or len(edges) < 2 or not (np.diff(edges) > 0).all():
        raise ValueError(message.format(bins))
    return edges
