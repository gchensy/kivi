"""Calibration of event probabilities and rating grades."""

import math
import warnings
from numbers import Real
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import _logit
from ._counts import group_counts, group_rows
from ._stats import betainc, chi2_sf
from ._validation import (
    as_series,
    as_weights,
    check_smoothing,
    is_integral,
    is_real,
    like_input,
    scores_and_labels,
)

__all__ = [
    "assign_grades",
    "binomial_test",
    "calibration_table",
    "corrected_intercept",
    "fit_grade_pd",
    "herfindahl_index",
    "hosmer_lemeshow",
    "isotonic_fit",
    "isotonic_predict",
    "jeffreys_test",
    "master_scale",
    "platt_fit",
    "platt_predict",
    "prior_correction",
]


def prior_correction(
    probability: npt.ArrayLike, *, sample_rate: float, population_rate: float
) -> float | pd.Series | npt.NDArray[np.float64]:
    r"""Correct event probabilities for a development sample with another event rate.

    A model developed on a sample that over- or undersamples events
    predicts probabilities for that sample. The correction rescales the odds
    so the probabilities refer to the population.

    Parameters
    ----------
    probability : array_like
        Event probabilities from the model, between 0 and 1. Missing values
        stay missing.
    sample_rate : float
        Event rate of the development sample, strictly between 0 and 1.
    population_rate : float
        Event rate of the population, strictly between 0 and 1.

    Returns
    -------
    float, pandas.Series or numpy.ndarray
        Corrected probabilities with the shape of `probability`. A Series input
        gives a Series with the same index and name, and a scalar gives a
        float.

    Raises
    ------
    ValueError
        If `probability` has values outside [0, 1], or a rate is not strictly
        between 0 and 1.

    See Also
    --------
    corrected_intercept : Apply the same correction to a logistic intercept.
    calibration_table : Compare predicted and observed event rates.

    Notes
    -----
    With sample event rate :math:`\bar y` and population event rate
    :math:`\tau`, the odds of every probability are multiplied by

    .. math::

       \omega = \frac{\tau / (1 - \tau)}{\bar y / (1 - \bar y)}, \qquad
       \tilde p = \frac{\omega p}{\omega p + 1 - p}.

    This is the prior correction of King and Zeng [1]_. It changes the
    level of the probabilities but not their order. For a logistic
    regression it equals adding :math:`\ln \omega` to the intercept.

    References
    ----------
    .. [1] G. King and L. Zeng, "Logistic Regression in Rare Events Data",
       Political Analysis, vol. 9, no. 2, pp. 137-163, 2001.

    Examples
    --------
    >>> from kivi.calibration import prior_correction
    >>> prior_correction([0.5, 0.2, 0.05], sample_rate=0.5, population_rate=0.02).round(4)
    array([0.02  , 0.0051, 0.0011])
    """
    omega = _odds_ratio(sample_rate, population_rate)
    values = np.asarray(probability, dtype=float)
    _check_unit_interval(values, "probability")
    corrected = omega * values / (omega * values + 1 - values)
    if isinstance(probability, pd.Series):
        return pd.Series(corrected, index=probability.index, name=probability.name)
    return float(corrected) if corrected.ndim == 0 else corrected


def corrected_intercept(intercept: float, *, sample_rate: float, population_rate: float) -> float:
    r"""Correct a logistic intercept for a development sample with another event rate.

    The corrected model gives the probabilities of `prior_correction`: only
    the intercept changes, so the coefficients and the order of applicants
    stay as they are.

    Parameters
    ----------
    intercept : float
        Intercept of a logistic regression of the event.
    sample_rate : float
        Event rate of the development sample, strictly between 0 and 1.
    population_rate : float
        Event rate of the population, strictly between 0 and 1.

    Returns
    -------
    float
        The intercept for the population, ``intercept + ln(omega)``.

    Raises
    ------
    ValueError
        If `intercept` is not a finite number, or a rate is not strictly
        between 0 and 1.

    See Also
    --------
    prior_correction : Apply the correction to probabilities.

    Notes
    -----
    King and Zeng [1]_ subtract
    :math:`\ln\left[\frac{1 - \tau}{\tau} \cdot \frac{\bar y}{1 - \bar y}\right]
    = -\ln \omega` from the intercept; see `prior_correction`.

    References
    ----------
    .. [1] G. King and L. Zeng, "Logistic Regression in Rare Events Data",
       Political Analysis, vol. 9, no. 2, pp. 137-163, 2001.

    Examples
    --------
    >>> from kivi.calibration import corrected_intercept
    >>> print(f"{corrected_intercept(0.0, sample_rate=0.5, population_rate=0.02):.4f}")
    -3.8918
    """
    if isinstance(intercept, bool) or not is_real(intercept) or not math.isfinite(intercept):
        raise ValueError(f"intercept must be a finite number, got {intercept!r}")
    return intercept + math.log(_odds_ratio(sample_rate, population_rate))


def platt_fit(
    score: npt.ArrayLike, target: npt.ArrayLike, *, weights: npt.ArrayLike | None = None
) -> tuple[float, float]:
    r"""Fit Platt scaling, a logistic curve from a score to event probabilities.

    The slope and intercept are fitted once, on a development or calibration
    sample, and `platt_predict` applies them to the scores of any sample.

    Parameters
    ----------
    score : array_like
        Finite numeric scores, such as the log odds of a model, a PD or
        scorecard points. The direction does not matter: a score that falls
        as the risk rises gets a negative slope.
    target : array_like
        Binary labels, with 1 indicating the event. Both classes must be
        present. If `score` and `target` are both pandas Series, they must
        have identical indices; otherwise they are matched by position.
    weights : array_like, optional
        Non-negative frequency weight of each observation, matched like
        `target`: a weight of 2 counts an observation twice.

    Returns
    -------
    slope : float
        Coefficient of the score in the log odds of the event.
    intercept : float
        Log odds of the event at a score of 0.

    Raises
    ------
    kivi.selection.SeparationError
        If the score separates the classes, so that the slope that
        maximizes the likelihood is infinite. It is a ValueError.
    ValueError
        If the inputs are empty or not one-dimensional, their indices or
        lengths do not match, `target` has a label other than 0 and 1 or
        lacks one of them, `score` is not real and numeric, has a missing or
        infinite value or does not vary, or `weights` is invalid or leaves a
        class without weight.

    Warns
    -----
    RuntimeWarning
        If the fit does not converge.

    See Also
    --------
    platt_predict : Apply the slope and intercept to scores.
    isotonic_fit : A calibration that only assumes that the rate rises.
    kivi.selection.logistic_regression : The same kind of fit with standard errors.

    Notes
    -----
    The event probability at score :math:`s` is modelled as

    .. math::

       p(s) = \frac{1}{1 + \exp(-(a s + b))},

    and the slope :math:`a` and the intercept :math:`b` maximize the
    log-likelihood, fitted by iteratively reweighted least squares as in
    `kivi.selection.logistic_regression`. At the maximum, the mean fitted
    probability equals the event rate of the sample, weighted with
    `weights`.

    When the score is the log odds of a model, this is the logistic
    recalibration of Cox [2]_: a slope of 1 and an intercept of 0 mean that
    the model is calibrated on the sample, and a slope below 1 that its
    probabilities are too extreme, as after overfitting.

    Platt [1]_, and scikit-learn's ``CalibratedClassifierCV`` after him,
    replace the labels 1 and 0 with :math:`(n_1 + 1) / (n_1 + 2)` and
    :math:`1 / (n_0 + 2)` for :math:`n_1` events and :math:`n_0`
    non-events. This matters only in small samples, where it also keeps the
    slope finite when the score separates the classes. Here the labels are
    used as they are, so a separating score raises an error instead;
    `isotonic_fit` handles such a score.

    References
    ----------
    .. [1] J. C. Platt, "Probabilistic outputs for support vector machines
       and comparisons to regularized likelihood methods", in Advances in
       Large Margin Classifiers, MIT Press, pp. 61-74, 1999.
    .. [2] D. R. Cox, "Two further applications of a model for binary
       regression", Biometrika, 45(3-4), pp. 562-565, 1958.

    Examples
    --------
    >>> from kivi.calibration import platt_fit, platt_predict
    >>> score = [-2.0, -1.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0]
    >>> target = [0, 0, 1, 0, 1, 0, 1, 1]
    >>> slope, intercept = platt_fit(score, target)
    >>> print(f"slope = {slope:.4f}, intercept = {intercept:.4f}")
    slope = 0.8824, intercept = -0.1250
    >>> platt_predict([-1.0, 0.0, 1.0], slope, intercept).round(4)
    array([0.2675, 0.4688, 0.6808])
    """
    values, labels, mass = scores_and_labels(score, target, "score", weights=weights)
    if mass is not None:
        # Rows without weight must not enter the centring and scaling of the fit.
        used = mass > 0
        values, labels, mass = values[used], labels[used], mass[used]
    if values.min() == values.max():
        raise ValueError(f"score must vary to fit a slope, got only {float(values[0])!r}")
    # Scale by a power of two, which is exact, so that squaring the scores cannot overflow.
    exponent = int(np.frexp(np.abs(values).max())[1])
    design = np.column_stack([np.ones(len(values)), np.ldexp(values, -exponent)])
    try:
        fitted = _logit.fit(design, labels.astype(float), weights=mass)
    except _logit.SeparationError:
        raise _logit.SeparationError(
            "score separates the classes of target, so the slope that maximizes the "
            "likelihood is infinite; isotonic_fit handles a separating score"
        ) from None
    except _logit.SingularError:
        raise ValueError(
            "score varies too little relative to its size to fit a slope, got scores from "
            f"{float(values.min())!r} to {float(values.max())!r}"
        ) from None
    if not fitted.converged:
        warnings.warn(
            f"Platt scaling did not converge after {fitted.n_iter} iterations",
            RuntimeWarning,
            stacklevel=2,
        )
    intercept, slope = fitted.coef
    return float(np.ldexp(slope, -exponent)), float(intercept)


def platt_predict(
    score: npt.ArrayLike, slope: float, intercept: float
) -> float | pd.Series | npt.NDArray[np.float64]:
    r"""Turn scores into event probabilities with a fitted Platt scaling.

    The probabilities follow the logistic curve with the slope and
    intercept from `platt_fit`, so the same calibration applies to the
    development sample, a validation sample and new applicants.

    Parameters
    ----------
    score : array_like
        Numeric scores on the scale of the fit. Missing values give missing
        probabilities.
    slope : float
        Coefficient of the score, as returned by `platt_fit`.
    intercept : float
        Intercept, as returned by `platt_fit`.

    Returns
    -------
    float, pandas.Series or numpy.ndarray
        Event probabilities with the shape of `score`. A Series input gives
        a Series with the same index and name, and a scalar gives a float.

    Raises
    ------
    ValueError
        If `score` is not real and numeric, or `slope` or `intercept` is not
        a finite number.

    See Also
    --------
    platt_fit : Fit the slope and intercept.
    isotonic_predict : Apply an isotonic calibration.

    Notes
    -----
    The probability is :math:`1 / (1 + \exp(-(a s + b)))` for slope
    :math:`a`, intercept :math:`b` and score :math:`s`, computed without
    overflow for extreme scores.

    Examples
    --------
    >>> from kivi.calibration import platt_predict
    >>> platt_predict([-2.0, 0.0, 2.0], 1.5, -0.5).round(4)
    array([0.0293, 0.3775, 0.9241])
    """
    for name, value in (("slope", slope), ("intercept", intercept)):
        if isinstance(value, bool) or not is_real(value) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    values = _float_scores(score, "score")
    # A huge score overflows to an infinite log odds, and logaddexp warns on missing values.
    with np.errstate(over="ignore", invalid="ignore"):
        probability = np.exp(-np.logaddexp(0, -(slope * values + intercept)))
    return like_input(score, probability)


def isotonic_fit(
    score: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
    increasing: bool = True,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Fit an isotonic calibration, a step function from a score to event probabilities.

    The score range is split into bands whose event rates rise with the
    score, so the calibration assumes only that a higher score means a
    higher risk. Fit the bands once and apply them to the scores of any
    sample with `isotonic_predict`.

    Parameters
    ----------
    score : array_like
        Finite numeric scores, such as a PD or the log odds of a model.
    target : array_like
        Binary labels, with 1 indicating the event. Both classes must be
        present. If `score` and `target` are both pandas Series, they must
        have identical indices; otherwise they are matched by position.
    weights : array_like, optional
        Non-negative frequency weight of each observation, matched like
        `target`: a weight of 2 counts an observation twice. Scores whose
        weights sum to zero are ignored.
    increasing : bool, default True
        Whether the probability rises with the score, as for a PD. Pass
        False for scorecard points, which are higher for a lower risk.

    Returns
    -------
    edges : numpy.ndarray
        Increasing band edges from ``-inf`` to ``inf``, like those of
        `kivi.binning.bin_edges`. The bands are closed on the right, as in
        ``pandas.cut``, and each interior edge is the highest score of the
        band below it.
    probabilities : numpy.ndarray
        Event rate of each band, one fewer than `edges`. The rates are
        strictly increasing, or strictly decreasing with
        ``increasing=False``.

    Raises
    ------
    ValueError
        If the inputs are empty or not one-dimensional, their indices or
        lengths do not match, `target` has a label other than 0 and 1 or
        lacks one of them, `score` is not real and numeric or has a missing
        or infinite value, `weights` is invalid or leaves a class without
        weight, or `increasing` is not a boolean.

    See Also
    --------
    isotonic_predict : Apply the bands to scores.
    platt_fit : A calibration by a logistic curve.
    calibration_table : Compare the bands with the outcomes of another sample.

    Notes
    -----
    Among all probabilities :math:`p_i` that rise with the score, isotonic
    regression minimizes :math:`\sum_i w_i (y_i - p_i)^2`, and the same
    solution maximizes the binomial likelihood [1]_. The
    pool-adjacent-violators algorithm finds it: each distinct score starts
    as a band with its event rate, and adjacent bands are merged while the
    lower one has a rate at least as high as the one above it. The
    probability of a band is then its pooled event rate. Tied scores always
    share a band, and since bands with equal rates are merged too, no two
    adjacent bands have the same probability.

    The outer edges are infinite, so a score below or above the fitted
    range gets the probability of the lowest or highest band. A score
    between the observed scores of two bands falls into the upper one,
    whose interval it belongs to; with ``increasing=False``, that is the
    band with the lower risk. scikit-learn's ``IsotonicRegression``
    interpolates linearly between the bands instead; both give the same
    probability at every score of the fitted sample [2]_.

    A band without events has probability 0 and a band with only events
    has probability 1; a score that separates the classes gives just these
    two bands. Floor the probabilities where 0 is not allowed, for example
    before `kivi.scorecard.pd_to_score`. The bands can also serve as the
    grades of `calibration_table`, by cutting the scores at `edges`.

    References
    ----------
    .. [1] M. Ayer, H. D. Brunk, G. M. Ewing, W. T. Reid and E. Silverman,
       "An empirical distribution function for sampling with incomplete
       information", The Annals of Mathematical Statistics, 26(4),
       pp. 641-647, 1955.
    .. [2] B. Zadrozny and C. Elkan, "Transforming classifier scores into
       accurate multiclass probability estimates", Proceedings of the 8th
       ACM SIGKDD Conference, pp. 694-699, 2002.

    Examples
    --------
    >>> from kivi.calibration import isotonic_fit, isotonic_predict
    >>> score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    >>> target = [0, 1, 0, 0, 1, 0, 1, 1]
    >>> edges, probabilities = isotonic_fit(score, target)
    >>> edges
    array([-inf,  0.1,  0.4,  0.6,  inf])
    >>> probabilities
    array([0.        , 0.33333333, 0.5       , 1.        ])
    >>> isotonic_predict([0.05, 0.25, 0.45, 0.9], edges, probabilities).round(4)
    array([0.    , 0.3333, 0.5   , 1.    ])
    """
    if not isinstance(increasing, (bool, np.bool_)):
        raise ValueError(f"increasing must be True or False, got {increasing!r}")
    values, labels, mass = scores_and_labels(score, target, "score", weights=weights)
    distinct, position = np.unique(values, return_inverse=True)
    position = position.ravel()  # NumPy 2.0 keeps the input's shape here
    total = np.bincount(position, weights=mass, minlength=len(distinct))
    event_mass = labels if mass is None else mass * labels
    event = np.bincount(position, weights=event_mass, minlength=len(distinct))
    used = total > 0
    last, pooled_total, pooled_event = _pool_adjacent_violators(
        total[used], event[used], increasing=bool(increasing)
    )
    edges = np.concatenate([[-np.inf], distinct[used][last[:-1]], [np.inf]])
    return edges, pooled_event / pooled_total


def isotonic_predict(
    score: npt.ArrayLike, edges: npt.ArrayLike, probabilities: npt.ArrayLike
) -> float | pd.Series | npt.NDArray[np.float64]:
    """Give each score the probability of its band from an isotonic calibration.

    The bands come from `isotonic_fit` on a development or calibration
    sample, so the same step function applies to every later sample.

    Parameters
    ----------
    score : array_like
        Numeric scores on the scale of the fit. Missing values give missing
        probabilities.
    edges : array_like
        Increasing band edges from ``-inf`` to ``inf``, as returned by
        `isotonic_fit`.
    probabilities : array_like
        Probability of each band, between 0 and 1, one fewer than `edges`.

    Returns
    -------
    float, pandas.Series or numpy.ndarray
        Event probabilities with the shape of `score`. A Series input gives
        a Series with the same index and name, and a scalar gives a float.

    Raises
    ------
    ValueError
        If `score` is not real and numeric, `edges` is not increasing from
        ``-inf`` to ``inf``, or `probabilities` does not give one value
        between 0 and 1 for each band.

    See Also
    --------
    isotonic_fit : Fit the bands and their probabilities.
    platt_predict : Apply a Platt scaling.

    Notes
    -----
    The bands are closed on the right, as in ``pandas.cut``: a score equal
    to an interior edge gets the probability of the band below the edge.
    Since the outer edges are infinite, every score that is not missing
    falls into a band.

    Examples
    --------
    >>> from kivi.calibration import isotonic_predict
    >>> edges = [-float("inf"), 0.2, 0.5, float("inf")]
    >>> isotonic_predict([0.1, 0.2, 0.3, 0.9], edges, [0.05, 0.2, 0.6])
    array([0.05, 0.05, 0.2 , 0.6 ])
    """
    edges, probabilities = _isotonic_bands(edges, probabilities)
    values = _float_scores(score, "score")
    band = np.searchsorted(edges[1:-1], values, side="left")
    result = probabilities[np.minimum(band, len(probabilities) - 1)]
    result = np.where(np.isnan(values), np.nan, result)
    return like_input(score, result)


def calibration_table(
    groups: npt.ArrayLike,
    target: npt.ArrayLike,
    probability: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    """Compare the predicted and observed event rates of each group.

    The groups are typically the grades of a master scale, formed by cutting
    the probabilities at fixed edges, or score bands. Each row has the
    inputs of the calibration tests.

    Parameters
    ----------
    groups : array_like
        Grade or group labels. Missing values form a group.
    target : array_like
        Binary labels, with 1 indicating the event. A group, or the whole
        sample, may have no events.
    probability : array_like
        Predicted event probabilities between 0 and 1. Series inputs must
        have identical indices; otherwise the inputs are matched by
        position.
    weights : array_like, optional
        Non-negative weight of each observation, such as the inverse of its
        sampling rate, matched like `target`. ``total`` and ``event`` are
        then sums of weights, the rate and the mean probability are
        weighted, and groups whose weights sum to zero are left out.

    Returns
    -------
    pandas.DataFrame
        One row per group with the number of observations ``total``, the
        number of events ``event``, the observed ``event_rate`` and the
        mean predicted probability ``mean_probability``.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, are empty, or their indices
        or lengths do not match, `target` has labels other than 0 and 1, or
        `probability` has missing values or values outside [0, 1]. Also if
        `weights` is not numeric, does not give each observation one finite,
        non-negative weight with a positive total, or has an index other
        than that of `groups`.

    See Also
    --------
    binomial_test : Test whether a grade has more events than predicted.
    jeffreys_test : The Bayesian counterpart of the binomial test.
    hosmer_lemeshow : Test the calibration of all groups at once.
    herfindahl_index : Measure how concentrated the grades are.

    Notes
    -----
    Groups are counted as in `kivi.woe.woe_table`. Rows follow the order of
    the categories when `groups` is categorical, such as the result of
    ``pandas.cut``, with missing values last. Otherwise they follow the order
    in which the groups first appear. Categories that are never observed are
    omitted.

    The binomial, Jeffreys and Hosmer-Lemeshow tests need whole numbers of
    observations and events, so run them on an unweighted table.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.calibration import calibration_table
    >>> grades = pd.Series(["A", "A", "B", "B", "B"])
    >>> target = pd.Series([0, 0, 0, 1, 1])
    >>> probability = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> calibration_table(grades, target, probability)
         total  event  event_rate  mean_probability
    bin
    A        2      0    0.000000       0.15
    B        3      2    0.666667       0.40
    """
    inputs = {"groups": groups, "target": target, "probability": probability}
    indexed = [values.index for values in inputs.values() if isinstance(values, pd.Series)]
    if any(not index.equals(indexed[0]) for index in indexed[1:]):
        raise ValueError(
            "groups, target and probability must have identical indices when they are Series"
        )
    raw_groups = groups
    groups, target, probability = (as_series(values, name) for name, values in inputs.items())
    if len({len(groups), len(target), len(probability)}) > 1:
        raise ValueError(
            "groups, target and probability must have the same length, "
            f"got {len(groups)}, {len(target)} and {len(probability)}"
        )
    if groups.empty:
        raise ValueError("groups, target and probability must contain observations")
    # For nullable dtypes pandas 2.0 returns an object array, which ~ would negate as integers.
    invalid = ~target.isin([0, 1]).to_numpy(dtype=bool)
    if invalid.any():
        raise ValueError(f"target must contain only 0 and 1, got {target[invalid].tolist()[0]!r}")
    probabilities = probability.to_numpy(dtype=float)
    if np.isnan(probabilities).any():
        raise ValueError("probability must not have missing values")
    _check_unit_interval(probabilities, "probability")
    others = {name: inputs[name] for name in ("target", "probability")}
    mass = as_weights(weights, raw_groups, of="groups", others=others)
    counts = group_counts(groups, target.to_numpy(dtype=np.intp), mass)
    rows = group_rows(groups, counts.index)
    total = counts.to_numpy().sum(axis=1)
    event = counts.to_numpy()[:, 1]
    # Only observations of zero weight lie in a group that group_counts left out.
    weighted = probabilities if mass is None else mass * probabilities
    kept = rows >= 0
    prob_sum = np.bincount(rows[kept], weights=weighted[kept], minlength=len(counts))
    return pd.DataFrame(
        {
            "total": total,
            "event": event,
            "event_rate": event / total,
            "mean_probability": prob_sum / total,
        },
        index=counts.index,
    )


def binomial_test(
    event: npt.ArrayLike, total: npt.ArrayLike, probability: npt.ArrayLike
) -> float | pd.Series | npt.NDArray[np.float64]:
    r"""Test whether each grade has more events than its probability predicts.

    The one-sided exact binomial test assumes independent events. A small
    p-value means that the probability underestimates the event rate of
    the grade.

    Parameters
    ----------
    event : array_like
        Number of events in each grade.
    total : array_like
        Number of observations in each grade, a positive whole number.
    probability : array_like
        Predicted event probability of each grade, between 0 and 1, such as
        the ``mean_probability`` column of `calibration_table`. The three inputs
        are broadcast together; Series inputs must have identical indices.

    Returns
    -------
    float, pandas.Series or numpy.ndarray
        One-sided p-values: a Series with the index of the Series inputs, a
        float for scalar inputs and an array otherwise.

    Raises
    ------
    ValueError
        If the inputs cannot be broadcast or their Series indices differ,
        `total` is not a positive whole number, `event` is not a whole number
        between 0 and `total`, or `probability` is outside [0, 1].

    See Also
    --------
    jeffreys_test : The Bayesian counterpart of this test.
    calibration_table : The counts and predicted probabilities of each grade.

    Notes
    -----
    Under the null hypothesis the number of events :math:`D` among :math:`N`
    observations follows a binomial distribution with probability
    :math:`p`. The p-value is

    .. math::

       P(D \ge d) = \sum_{k=d}^{N} \binom{N}{k} p^k (1 - p)^{N - k}
       = I_p(d, N - d + 1),

    with the regularized incomplete beta function :math:`I`, and 1 when
    :math:`d = 0`. Correlated defaults, as in an economic downturn, make
    the test reject more often than its nominal level [1]_.

    References
    ----------
    .. [1] Basel Committee on Banking Supervision, "Studies on the
       Validation of Internal Rating Systems", Working Paper No. 14, 2005.

    Examples
    --------
    >>> from kivi.calibration import binomial_test
    >>> binomial_test([0, 3, 8], [200, 200, 200], 0.02).round(4)
    array([1.    , 0.7649, 0.0493])
    """
    event, total, probability, index = _grade_inputs(event, total, probability)
    pvalue = np.ones(event.shape)
    some = event > 0
    pvalue[some] = betainc(event[some], total[some] - event[some] + 1, probability[some])
    return _pvalues(pvalue, index)


def jeffreys_test(
    event: npt.ArrayLike, total: npt.ArrayLike, probability: npt.ArrayLike
) -> float | pd.Series | npt.NDArray[np.float64]:
    r"""Test each grade's probability against its event rate with the Jeffreys prior.

    The p-value is the posterior probability that the true event rate is at
    most the predicted probability. A small p-value means that the
    probability underestimates the event rate of the grade.

    Parameters
    ----------
    event : array_like
        Number of events in each grade.
    total : array_like
        Number of observations in each grade, a positive whole number.
    probability : array_like
        Predicted event probability of each grade, between 0 and 1, such as
        the ``mean_probability`` column of `calibration_table`. The three inputs
        are broadcast together; Series inputs must have identical indices.

    Returns
    -------
    float, pandas.Series or numpy.ndarray
        One-sided p-values: a Series with the index of the Series inputs, a
        float for scalar inputs and an array otherwise.

    Raises
    ------
    ValueError
        If the inputs cannot be broadcast or their Series indices differ,
        `total` is not a positive whole number, `event` is not a whole number
        between 0 and `total`, or `probability` is outside [0, 1].

    See Also
    --------
    binomial_test : The frequentist counterpart of this test.
    calibration_table : The counts and predicted probabilities of each grade.

    Notes
    -----
    With the Jeffreys prior Beta(1/2, 1/2), the posterior of the event rate
    after :math:`d` events among :math:`N` observations is
    :math:`\mathrm{Beta}(d + 1/2, N - d + 1/2)`, and the p-value is its
    distribution function at the predicted probability :math:`p`:

    .. math::

       I_p\left(d + \tfrac12, N - d + \tfrac12\right).

    This is the test that the European Central Bank asks for in the
    validation of PD models [1]_. Unlike the binomial test, a grade without
    events still has a p-value below 1.

    References
    ----------
    .. [1] European Central Bank, "Instructions for reporting the
       validation results of internal models: IRB Pillar I models for
       credit risk", 2019.

    Examples
    --------
    >>> from kivi.calibration import jeffreys_test
    >>> jeffreys_test([0, 3, 8], [200, 200, 200], 0.02).round(4)
    array([0.9956, 0.6701, 0.032 ])
    """
    event, total, probability, index = _grade_inputs(event, total, probability)
    return _pvalues(betainc(event + 0.5, total - event + 0.5, probability), index)


def hosmer_lemeshow(
    event: npt.ArrayLike, total: npt.ArrayLike, probability: npt.ArrayLike, *, ddof: int = 2
) -> tuple[float, float]:
    r"""Test the calibration of all groups at once with the Hosmer-Lemeshow statistic.

    The statistic adds up the squared differences between observed and
    predicted events of each group. A small p-value means that the
    probabilities do not match the observed event rates.

    Parameters
    ----------
    event : array_like
        Number of events in each group.
    total : array_like
        Number of observations in each group, a positive whole number.
    probability : array_like
        Mean predicted event probability of each group, strictly between 0
        and 1, such as the ``mean_probability`` column of `calibration_table`.
        Series inputs must have identical indices.
    ddof : int, default 2
        The chi-square distribution has as many degrees of freedom as there
        are groups, minus `ddof`.

    Returns
    -------
    statistic : float
        The Hosmer-Lemeshow statistic.
    pvalue : float
        Its upper tail probability under the chi-square distribution.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, have different lengths or
        Series indices, `total` is not a positive whole number, `event` is
        not a whole number between 0 and `total`, `probability` is not strictly
        between 0 and 1, or `ddof` is negative or leaves no degree of
        freedom.

    See Also
    --------
    calibration_table : The counts and predicted probabilities of each group.
    binomial_test : Test each grade on its own.

    Notes
    -----
    For :math:`g` groups with :math:`d_k` events among :math:`n_k`
    observations and mean predicted probability :math:`\bar p_k`,

    .. math::

       H = \sum_{k=1}^{g} \frac{(d_k - n_k \bar p_k)^2}{n_k \bar p_k (1 - \bar p_k)},

    compared with a chi-square distribution with :math:`g - \mathrm{ddof}`
    degrees of freedom. Hosmer and Lemeshow [1]_ form about ten groups of
    similar size from the sorted probabilities of the sample the model was
    fitted on, which gives :math:`g - 2`. For probabilities that were not
    fitted on the tested sample, such as a validation sample, use
    ``ddof=0`` [2]_.

    References
    ----------
    .. [1] D. W. Hosmer and S. Lemeshow, "Goodness of fit tests for the
       multiple logistic regression model", Communications in Statistics -
       Theory and Methods, vol. 9, no. 10, pp. 1043-1069, 1980.
    .. [2] D. W. Hosmer, S. Lemeshow and R. X. Sturdivant, "Applied
       Logistic Regression", 3rd ed., Wiley, 2013.

    Examples
    --------
    >>> from kivi.calibration import hosmer_lemeshow
    >>> statistic, pvalue = hosmer_lemeshow([2, 6, 15], [100, 100, 100], [0.03, 0.06, 0.12])
    >>> print(f"statistic = {statistic:.3f}, pvalue = {pvalue:.3f}")
    statistic = 1.196, pvalue = 0.274
    """
    if isinstance(ddof, bool) or not is_integral(ddof) or ddof < 0:
        raise ValueError(f"ddof must be a non-negative integer, got {ddof!r}")
    event, total, probability, _ = _grade_inputs(event, total, probability)
    if event.ndim != 1:
        raise ValueError(
            f"event, total and probability must be one-dimensional, got {event.ndim} dimensions"
        )
    invalid = (probability <= 0) | (probability >= 1)
    if invalid.any():
        raise ValueError(
            f"probability must be strictly between 0 and 1, got {_first(probability, invalid)}"
        )
    dof = len(event) - ddof
    if dof < 1:
        raise ValueError(f"{len(event)} group(s) with ddof={ddof} leave {dof} degrees of freedom")
    expected = total * probability
    with np.errstate(over="ignore"):  # a vanishing expected count gives an infinite statistic
        statistic = float(np.sum((event - expected) ** 2 / (expected * (1 - probability))))
    return statistic, chi2_sf(statistic, dof)


def herfindahl_index(counts: npt.ArrayLike, *, normalize: bool = False) -> float:
    r"""Measure how concentrated a portfolio is across grades.

    The Herfindahl-Hirschman index is the sum of the squared shares of the
    grades. It is 1 when every observation is in one grade and :math:`1/k`
    when :math:`k` grades are equally full.

    Parameters
    ----------
    counts : array_like
        Number of observations, or exposure, in each grade, such as the
        ``total`` column of `calibration_table`.
    normalize : bool, default False
        Rescale the index to run from 0 for equally full grades to 1 for a
        single full grade.

    Returns
    -------
    float
        The index.

    Raises
    ------
    ValueError
        If `counts` is not one-dimensional, has negative or non-finite
        values or sums to zero, or `normalize` is not a boolean or is set
        with fewer than two grades.

    See Also
    --------
    calibration_table : The number of observations in each grade.

    Notes
    -----
    For grade shares :math:`s_i = c_i / \sum_j c_j` of :math:`k` grades,

    .. math::

       \mathrm{HHI} = \sum_{i=1}^{k} s_i^2, \qquad
       \mathrm{HHI}^* = \frac{\mathrm{HHI} - 1/k}{1 - 1/k}.

    A rating scale whose observations crowd into a few grades
    differentiates risk poorly. Empty grades count towards :math:`k`.

    Examples
    --------
    >>> from kivi.calibration import herfindahl_index
    >>> herfindahl_index([50, 30, 20])
    0.38
    >>> print(f"{herfindahl_index([50, 30, 20], normalize=True):.2f}")
    0.07
    """
    if not isinstance(normalize, (bool, np.bool_)):
        raise ValueError(f"normalize must be True or False, got {normalize!r}")
    if np.ndim(counts) != 1:
        raise ValueError(f"counts must be one-dimensional, got {np.ndim(counts)} dimensions")
    values = np.asarray(counts, dtype=float)
    invalid = ~np.isfinite(values) | (values < 0)
    if invalid.any():
        raise ValueError(f"counts must be finite and non-negative, got {_first(values, invalid)}")
    if not values.sum() > 0:
        raise ValueError(f"counts must have a positive sum, got {float(values.sum())!r}")
    index = float(np.sum((values / values.sum()) ** 2))
    if not normalize:
        return index
    if len(values) < 2:
        raise ValueError(f"normalize requires at least two grades, got {len(values)}")
    return (index - 1 / len(values)) / (1 - 1 / len(values))


def master_scale(
    probability: npt.ArrayLike,
    *,
    boundary: Literal["geometric", "arithmetic"] = "geometric",
    labels: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Build a master scale from the PD of each grade.

    A master scale divides the PD range from 0 to 1 into grades, each with
    one PD. The bounds between adjacent grades lie between their PDs, so
    that `assign_grades` can give every predicted PD its grade.

    Parameters
    ----------
    probability : array_like
        PD of each grade, strictly increasing from the grade with the lowest
        risk and strictly between 0 and 1. ``numpy.geomspace(first, last,
        grades)`` gives a geometric grid, and `fit_grade_pd` fits the PDs to
        observed default rates.
    boundary : {"geometric", "arithmetic"}, default "geometric"
        Bound between two adjacent grades: the geometric mean of their PDs,
        the midpoint on a logarithmic scale, or their arithmetic mean.
    labels : array_like, optional
        Unique label of each grade, such as ``list("ABCDEFG")``. By default
        the grades are numbered from 1.

    Returns
    -------
    pandas.DataFrame
        One row per grade, indexed by its label in an index named
        ``grade``, with the columns ``lower``, ``probability`` and
        ``upper``. The first grade starts at 0, the last one ends at 1, and
        each grade ends where the next one starts.

    Raises
    ------
    ValueError
        If `probability` is empty, not one-dimensional or not real and
        numeric, has a missing value, is not strictly increasing or is not
        strictly between 0 and 1, `boundary` is not recognized, or `labels`
        does not give one unique, non-missing label per grade.

    See Also
    --------
    fit_grade_pd : Fit the PD of each grade to observed default rates.
    assign_grades : Give each predicted PD the grade of the scale.
    calibration_table : Compare the grades with their observed default rates.

    Notes
    -----
    For the PDs :math:`p_1 < \dots < p_K` of the grades, the bound between
    grades :math:`k` and :math:`k + 1` is

    .. math::

       b_k = \sqrt{p_k \, p_{k+1}} \quad \text{(geometric)}, \qquad
       b_k = \frac{p_k + p_{k+1}}{2} \quad \text{(arithmetic)},

    and grade :math:`k` holds the PDs in :math:`(b_{k-1}, b_k]`, with
    :math:`b_0 = 0` and :math:`b_K = 1`. PDs of a master scale usually grow
    by a constant factor from grade to grade, so they are evenly spaced on a
    logarithmic scale. The geometric bound is the midpoint on that scale: a
    PD goes to the grade whose PD is nearer by ratio. The arithmetic mean,
    which the 0.2 versions of kivi used, always lies above the geometric
    one, so each grade reaches further towards the grade above it than
    towards the grade below. With PDs that double from grade to grade, a
    grade with PD :math:`p` then holds the PDs up to :math:`1.5 p` instead
    of :math:`\sqrt 2 p \approx 1.41 p`, and its observed default rate tends
    to exceed its PD.

    Examples
    --------
    >>> import numpy as np
    >>> from kivi.calibration import master_scale
    >>> master_scale(np.geomspace(0.01, 0.16, 5), labels=list("ABCDE")).round(4)
            lower  probability   upper
    grade
    A      0.0000         0.01  0.0141
    B      0.0141         0.02  0.0283
    C      0.0283         0.04  0.0566
    D      0.0566         0.08  0.1131
    E      0.1131         0.16  1.0000

    The arithmetic mean of adjacent PDs gives higher bounds:

    >>> master_scale([0.01, 0.02, 0.04], boundary="arithmetic")
           lower  probability  upper
    grade
    1      0.000         0.01  0.015
    2      0.015         0.02  0.030
    3      0.030         0.04  1.000
    """
    if boundary not in ("geometric", "arithmetic"):
        raise ValueError(f"boundary must be 'geometric' or 'arithmetic', got {boundary!r}")
    values = _numeric_values(probability, "probability")
    if not len(values):
        raise ValueError("probability must give the PD of at least one grade")
    if np.isnan(values).any():
        raise ValueError("probability must not have missing values")
    invalid = (values <= 0) | (values >= 1)
    if invalid.any():
        raise ValueError(
            f"probability must be strictly between 0 and 1, got {_first(values, invalid)}"
        )
    falling = np.flatnonzero(np.diff(values) <= 0)
    if len(falling):
        i = falling[0]
        raise ValueError(
            "probability must increase strictly from grade to grade, got "
            f"{float(values[i])!r} followed by {float(values[i + 1])!r}"
        )
    index = _grade_labels(labels, len(values))
    if boundary == "geometric":
        product = values[:-1] * values[1:]
        # Split the square root only where the product of two tiny PDs would underflow.
        bounds = np.where(
            product >= np.finfo(float).tiny,
            np.sqrt(product),
            np.sqrt(values[:-1]) * np.sqrt(values[1:]),
        )
    else:
        bounds = (values[:-1] + values[1:]) / 2
    return pd.DataFrame(
        {"lower": np.r_[0.0, bounds], "probability": values, "upper": np.r_[bounds, 1.0]},
        index=index,
    )


def fit_grade_pd(
    event: npt.ArrayLike, total: npt.ArrayLike, *, smoothing: float = 0.5
) -> pd.Series | npt.NDArray[np.float64]:
    r"""Fit the PD of each grade to observed default rates, log-linear in the grade.

    The logarithm of the default rate is fitted as a straight line in the
    grade number, so the fitted PDs grow by a constant factor from grade to
    grade. Pass the result to `master_scale` for the bounds of the grades.

    Parameters
    ----------
    event : array_like
        Number of defaults in each grade, ordered from the grade with the
        lowest risk to the grade with the highest. Weighted counts, such as
        those of a weighted `calibration_table`, are accepted.
    total : array_like
        Number of observations in each grade. A grade without observations
        is left out of the fit but still gets a fitted PD, so pass a 0 for
        every grade of the scale that the data do not cover. Series inputs
        must have identical indices.
    smoothing : float, default 0.5
        Number of defaults that replaces a count of zero, whose logarithm
        is undefined, as `kivi.woe.woe_table` replaces an empty cell.

    Returns
    -------
    pandas.Series or numpy.ndarray
        Fitted PD of each grade, strictly increasing and below 1: a Series
        with the index of the Series inputs, an array otherwise.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, have different lengths or
        Series indices, are not finite and non-negative, or `event` exceeds
        `total`; if fewer than two grades have observations or `smoothing`
        is not finite and positive. Also if the fitted PDs do not rise with
        the grade or reach 1.

    See Also
    --------
    master_scale : Build the scale from the fitted PDs.
    calibration_table : Count the observations and defaults of each grade.

    Notes
    -----
    With :math:`d_k` defaults among :math:`n_k` observations in grade
    :math:`k = 1, \dots, K`, the intercept :math:`a` and the slope
    :math:`b` minimize the weighted sum of squares

    .. math::

       \sum_{k : n_k > 0} n_k \left(\ln \frac{d_k}{n_k} - a - b k\right)^2,

    where :math:`d_k` is replaced by `smoothing` when it is 0. The fitted
    PD of grade :math:`k` is :math:`e^{a + b k}`, and consecutive PDs have
    the constant ratio :math:`e^b`. The weights give each observation the
    same influence, so integer counts weigh like repeated observations;
    the 0.2 versions of kivi fitted the line by ordinary least squares,
    giving each grade the same influence instead.

    The fit extrapolates beyond the observed grades, so grades at the
    edges of the scale can get PDs no observation supports. A default rate
    of zero says only that the grade is safe, not how safe: the smoothed
    rate :math:`0.5 / n_k` stands in for it.

    Examples
    --------
    >>> from kivi.calibration import fit_grade_pd
    >>> event = [0, 3, 5, 18, 30]
    >>> total = [400, 500, 300, 400, 200]
    >>> fit_grade_pd(event, total).round(4)
    array([0.0015, 0.0049, 0.0155, 0.0494, 0.1572])
    """
    indices = [values.index for values in (event, total) if isinstance(values, pd.Series)]
    if len(indices) == 2 and not indices[0].equals(indices[1]):
        raise ValueError("event and total must have identical indices when they are Series")
    check_smoothing(smoothing)
    events = _numeric_values(event, "event")
    totals = _numeric_values(total, "total")
    if len(events) != len(totals):
        raise ValueError(
            f"event and total must have the same length, got {len(events)} and {len(totals)}"
        )
    for name, values in (("event", events), ("total", totals)):
        invalid = ~np.isfinite(values) | (values < 0)
        if invalid.any():
            raise ValueError(
                f"{name} must be finite and non-negative, got {_first(values, invalid)}"
            )
    invalid = events > totals
    if invalid.any():
        raise ValueError(
            f"event must not exceed total, got {_first(events, invalid)} events "
            f"among {_first(totals, invalid)}"
        )
    observed = totals > 0
    if observed.sum() < 2:
        raise ValueError(
            f"total must have observations in at least two grades, got {int(observed.sum())}"
        )
    grade = np.arange(1, len(totals) + 1, dtype=float)
    x, weight = grade[observed], totals[observed]
    y = np.log(np.where(events > 0, events, smoothing)[observed] / weight)
    # Centre on the weighted means, so that the slope does not lose digits to the intercept.
    x_mean, y_mean = np.average(x, weights=weight), np.average(y, weights=weight)
    slope = np.sum(weight * (x - x_mean) * (y - y_mean)) / np.sum(weight * (x - x_mean) ** 2)
    if not slope > 0:
        raise ValueError(
            "the fitted PDs must rise with the grade; order event and total from the lowest "
            f"risk to the highest, got a slope of {float(slope)!r} in ln(PD) per grade"
        )
    probability = np.exp(y_mean + slope * (grade - x_mean))
    if probability[-1] >= 1:
        raise ValueError(
            f"the fitted PD of the last grade must be below 1, got {float(probability[-1])!r}"
        )
    if indices:
        return pd.Series(probability, index=indices[0])
    return probability


def assign_grades(probability: npt.ArrayLike, scale: pd.DataFrame) -> pd.Series | pd.Categorical:
    """Give each predicted PD the grade of a master scale.

    The grades are an ordered categorical of the labels of the scale, so
    `calibration_table` counts them directly, in the order of the scale,
    for the tests of each grade.

    Parameters
    ----------
    probability : array_like
        One-dimensional predicted PDs between 0 and 1. Missing values stay
        missing.
    scale : pandas.DataFrame
        Master scale, as returned by `master_scale`: one row per grade,
        indexed by unique labels, with the columns ``lower`` and ``upper``.
        The first grade must start at 0, each grade where the one before it
        ends, and the last one must end at 1.

    Returns
    -------
    pandas.Series or pandas.Categorical
        Grade of each PD, an ordered categorical whose categories are the
        labels of the scale from the lowest risk to the highest. A Series
        input gives a Series with the same index and name.

    Raises
    ------
    ValueError
        If `probability` is not one-dimensional, is not real and numeric,
        or has values outside [0, 1], or `scale` is not a DataFrame with
        unique, non-missing labels and ``lower`` and ``upper`` columns that
        cover [0, 1] with increasing, adjacent grades.

    See Also
    --------
    master_scale : Build the scale.
    calibration_table : Compare the observed and predicted default rates of each grade.
    pandas.cut : Assign values to intervals given by their edges.

    Notes
    -----
    Grade :math:`k` holds the PDs above its ``lower`` bound up to and
    including its ``upper`` bound, as the intervals of ``pandas.cut`` are
    closed on the right, and a PD of 0 belongs to the first grade, as with
    ``include_lowest=True``. The result equals
    ``pandas.cut(probability, edges, labels=scale.index, include_lowest=True)``
    for the edges ``[0, *scale["upper"]]``; the ``upper`` column, not the
    PDs, decides the grade. The 0.2 versions of kivi gave a PD on a bound
    the grade above it instead.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.calibration import assign_grades, master_scale
    >>> scale = master_scale([0.01, 0.04, 0.16])
    >>> scale
           lower  probability  upper
    grade
    1       0.00         0.01   0.02
    2       0.02         0.04   0.08
    3       0.08         0.16   1.00
    >>> assign_grades(pd.Series([0.0, 0.02, 0.05, None, 0.9]), scale)
    0      1
    1      1
    2      2
    3    NaN
    4      3
    dtype: category
    Categories (3, int64): [1 < 2 < 3]
    """
    upper = _scale_bounds(scale)
    series = as_series(probability, "probability")
    values = _numeric_values(series, "probability")
    _check_unit_interval(values, "probability")
    codes = np.searchsorted(upper[:-1], values, side="left")
    codes[np.isnan(values)] = -1
    grades = pd.Categorical.from_codes(codes, categories=scale.index, ordered=True)
    if isinstance(probability, pd.Series):
        return pd.Series(grades, index=series.index, name=series.name)
    return grades


def _pool_adjacent_violators(
    total: np.ndarray, event: npt.NDArray[np.float64], *, increasing: bool
) -> tuple[npt.NDArray[np.intp], np.ndarray, npt.NDArray[np.float64]]:
    """Pool adjacent blocks of ordered scores until their event rates are strictly monotone.

    Returns the position of the last score of each block and the block's total
    and event weight.
    """
    sign = 1.0 if increasing else -1.0
    last: list[int]
    totals: list[float]
    events: list[float]
    last, totals, events = [], [], []
    for position, (block_total, block_event) in enumerate(
        zip(total.tolist(), event.tolist(), strict=True)
    ):
        # Compare the rates as they are returned, so that the returned rates are strictly
        # monotone even where rounding separates rates that are equal in exact arithmetic.
        while totals and sign * events[-1] / totals[-1] >= sign * block_event / block_total:
            block_total += totals.pop()
            block_event += events.pop()
            last.pop()
        last.append(position)
        totals.append(block_total)
        events.append(block_event)
    return np.array(last, dtype=np.intp), np.array(totals), np.array(events)


def _isotonic_bands(
    edges: npt.ArrayLike, probabilities: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Check and return the band edges and probabilities of an isotonic calibration."""
    edges = _numeric_values(edges, "edges")
    probabilities = _numeric_values(probabilities, "probabilities")
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not (np.diff(edges) > 0).all()
        or edges[0] != -np.inf
        or edges[-1] != np.inf
    ):
        raise ValueError(f"edges must increase from -inf to inf, got {edges.tolist()!r}")
    if probabilities.shape != (len(edges) - 1,):
        raise ValueError(
            f"probabilities must give one value for each of the {len(edges) - 1} band(s), "
            f"got shape {probabilities.shape}"
        )
    if np.isnan(probabilities).any():
        raise ValueError("probabilities must not have missing values")
    _check_unit_interval(probabilities, "probabilities")
    return edges, probabilities


def _grade_labels(labels: npt.ArrayLike | None, grades: int) -> pd.Index:
    """Return the labels of a master scale's grades as an index named ``grade``."""
    if labels is None:
        return pd.Index(np.arange(1, grades + 1), name="grade")
    if isinstance(labels, str):
        raise ValueError(f"labels must be one label per grade, got the string {labels!r}")
    series = as_series(labels, "labels")
    if len(series) != grades:
        raise ValueError(
            f"labels must give one label for each of the {grades} grade(s), got {len(series)}"
        )
    if series.isna().any():
        raise ValueError("labels must not have missing values")
    repeated = series[series.duplicated()]
    if len(repeated):
        raise ValueError(f"labels must be unique, got {repeated.iloc[0]!r} more than once")
    return pd.Index(series.to_numpy(), name="grade")


def _scale_bounds(scale: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Check a master scale and return the upper bounds of its grades."""
    if not isinstance(scale, pd.DataFrame):
        raise ValueError(f"scale must be a DataFrame from master_scale, got {type(scale).__name__}")
    absent = [name for name in ("lower", "upper") if name not in scale.columns]
    if absent:
        raise ValueError(f"scale must have the columns 'lower' and 'upper', missing {absent[0]!r}")
    if scale.empty:
        raise ValueError("scale must have at least one grade")
    if isinstance(scale.index, pd.MultiIndex) or scale.index.hasnans or not scale.index.is_unique:
        raise ValueError("scale must be indexed by unique, non-missing grade labels")
    lower = _numeric_values(scale["lower"], "lower of scale")
    upper = _numeric_values(scale["upper"], "upper of scale")
    if lower[0] != 0 or upper[-1] != 1:
        raise ValueError(
            f"scale must cover [0, 1], got grades from {float(lower[0])!r} to {float(upper[-1])!r}"
        )
    if not (np.diff(np.r_[0.0, upper]) > 0).all() or not np.array_equal(lower[1:], upper[:-1]):
        raise ValueError(
            "scale must have increasing grades, each starting where the one before it ends, "
            f"got upper bounds {upper.tolist()!r}"
        )
    return upper


def _float_scores(values: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Return a scalar or one-dimensional scores as floats, with missing values as NaN.

    A scalar gives a zero-dimensional array, so `like_input` returns a float.
    """
    if np.ndim(values) == 0 and not isinstance(values, pd.Series):
        return _numeric_values([values], name).reshape(())  # type: ignore[arg-type]  # [scalar] is one-dimensional
    if np.ndim(values) > 1:
        raise ValueError(
            f"{name} must be a scalar or one-dimensional, got {np.ndim(values)} dimensions"
        )
    return _numeric_values(values, name)


def _numeric_values(values: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Return one-dimensional real numeric values as floats, with missing values as NaN.

    Lists and object arrays are judged by their elements, as a Series built
    from them, so numbers held as strings are rejected however they arrive.
    """
    series = as_series(values, name)
    if pd.api.types.is_object_dtype(series.dtype):
        series = series.infer_objects()
    dtype = series.dtype
    if series.empty and pd.api.types.is_object_dtype(dtype):
        return np.empty(0)
    if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_complex_dtype(dtype):
        shown = next(
            (
                value
                for value in series
                if not isinstance(value, Real)
                and not (pd.api.types.is_scalar(value) and pd.isna(value))
            ),
            series.iloc[0],
        )
        raise ValueError(f"{name} must be real and numeric, got {shown!r} of dtype {dtype}")
    return series.to_numpy(dtype=float, na_value=np.nan)


def _odds_ratio(sample_rate: float, population_rate: float) -> float:
    for name, rate in (("sample_rate", sample_rate), ("population_rate", population_rate)):
        if isinstance(rate, bool) or not is_real(rate) or not 0 < rate < 1:
            raise ValueError(f"{name} must be strictly between 0 and 1, got {rate!r}")
    return (population_rate / (1 - population_rate)) / (sample_rate / (1 - sample_rate))


def _check_unit_interval(values: npt.NDArray[np.float64], name: str) -> None:
    outside = (values < 0) | (values > 1)
    if outside.any():
        raise ValueError(f"{name} must be between 0 and 1, got {_first(values, outside)}")


def _grade_inputs(
    event: npt.ArrayLike, total: npt.ArrayLike, probability: npt.ArrayLike
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], pd.Index | None
]:
    """Broadcast and check the per-grade inputs of the calibration tests.

    Also returns the index shared by the Series inputs, or None without Series.
    """
    indices = [
        values.index for values in (event, total, probability) if isinstance(values, pd.Series)
    ]
    index = indices[0] if indices else None
    if any(not other.equals(index) for other in indices[1:]):
        raise ValueError(
            "event, total and probability must have identical indices when they are Series"
        )
    try:
        arrays = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (event, total, probability))
        )
    except ValueError as error:
        raise ValueError(
            f"event, total and probability cannot be broadcast together: {error}"
        ) from None
    event, total, probability = arrays
    if index is not None and event.shape != (len(index),):
        raise ValueError(
            "event, total and probability must broadcast to the length "
            f"{len(index)} of their Series index, got shape {event.shape}"
        )
    invalid = ~_whole(total) | (total < 1)
    if invalid.any():
        raise ValueError(f"total must be a positive whole number, got {_first(total, invalid)}")
    invalid = ~_whole(event) | (event < 0)
    if invalid.any():
        raise ValueError(f"event must be a non-negative whole number, got {_first(event, invalid)}")
    invalid = event > total
    if invalid.any():
        raise ValueError(
            f"event must not exceed total, got {_first(event, invalid)} events "
            f"among {_first(total, invalid)}"
        )
    if np.isnan(probability).any():
        raise ValueError("probability must not have missing values")
    _check_unit_interval(probability, "probability")
    return event, total, probability, index


def _pvalues(
    pvalue: npt.NDArray[np.float64], index: pd.Index | None
) -> float | pd.Series | npt.NDArray[np.float64]:
    if index is not None:
        return pd.Series(pvalue, index=index)
    return float(pvalue) if pvalue.ndim == 0 else pvalue


def _whole(values: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    finite = np.isfinite(values)
    return finite & (np.where(finite, values, 0) % 1 == 0)


def _first(values: npt.NDArray[np.float64], invalid: npt.NDArray[np.bool_]) -> str:
    return repr(float(values[invalid].flat[0]))
