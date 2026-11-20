"""Calibration of event probabilities and rating grades."""

import math
import warnings
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import _logit
from ._counts import group_counts
from ._stats import betainc, chi2_sf
from ._validation import as_series, as_weights
from .metrics import _scores_and_labels

__all__ = [
    "binomial_test",
    "calibration_table",
    "corrected_intercept",
    "herfindahl_index",
    "hosmer_lemeshow",
    "isotonic_fit",
    "isotonic_predict",
    "jeffreys_test",
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
    if (
        isinstance(intercept, bool)
        or not isinstance(intercept, Real)
        or not math.isfinite(intercept)
    ):
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
        If the fit has not converged after 100 iterations.

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
    values, labels, mass = _scores_and_labels(score, target, "score", weights=weights)
    used = values if mass is None else values[mass > 0]
    if used.min() == used.max():
        raise ValueError(f"score must vary to fit a slope, got only {float(used[0])!r}")
    design = np.column_stack([np.ones(len(values)), values])
    try:
        fitted = _logit.fit(design, labels.astype(float), weights=mass)
    except _logit.SeparationError:
        raise _logit.SeparationError(
            "score separates the classes of target, so the slope that maximizes the "
            "likelihood is infinite; isotonic_fit handles a separating score"
        ) from None
    except _logit.SingularError:
        raise ValueError(
            "score varies too little relative to its size to fit a slope; centre or rescale it"
        ) from None
    if not fitted.converged:
        warnings.warn(
            f"Platt scaling did not converge after {fitted.n_iter} iterations",
            RuntimeWarning,
            stacklevel=2,
        )
    intercept, slope = fitted.coef
    return float(slope), float(intercept)


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
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    values = _float_scores(score, "score")
    # A huge score overflows to an infinite log odds, and logaddexp warns on missing values.
    with np.errstate(over="ignore", invalid="ignore"):
        probability = np.exp(-np.logaddexp(0, -(slope * values + intercept)))
    return _shaped(score, probability)


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
    whose interval it belongs to. scikit-learn's ``IsotonicRegression``
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
    values, labels, mass = _scores_and_labels(score, target, "score", weights=weights)
    distinct, position = np.unique(values, return_inverse=True)
    position = position.ravel()  # NumPy 2.0 keeps the input's shape here
    total = np.bincount(position, weights=mass, minlength=len(distinct))
    event = np.bincount(position, weights=labels if mass is None else mass * labels)
    event = np.pad(event, (0, len(distinct) - len(event)))
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
    return _shaped(score, result)


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

    The binomial, Jeffreys and Hosmer-Lemeshow tests need whole counts.
    With weights that only correct for sampling, pass the counts of the
    unweighted table to the tests and compare the weighted rates.

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
    mass = as_weights(weights, raw_groups, of="groups")
    counts = group_counts(groups, target.to_numpy(dtype=np.intp), mass)
    rows = _group_rows(groups, counts.index)
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
    if isinstance(ddof, bool) or not isinstance(ddof, Integral) or ddof < 0:
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


def _pool_adjacent_violators(
    total: np.ndarray, event: np.ndarray, *, increasing: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool adjacent blocks of ordered scores until their event rates are strictly monotone.

    Returns the position of the last score of each block and the block's total
    and event weight.
    """
    sign = 1.0 if increasing else -1.0
    last, totals, events = [], [], []
    for position, (block_total, block_event) in enumerate(
        zip(total.tolist(), event.tolist(), strict=True)
    ):
        # The previous rate is at least the current one: e' / t' >= e / t, cross-multiplied.
        while totals and sign * events[-1] * block_total >= sign * block_event * totals[-1]:
            block_total += totals.pop()
            block_event += events.pop()
            last.pop()
        last.append(position)
        totals.append(block_total)
        events.append(block_event)
    return np.array(last, dtype=np.intp), np.array(totals), np.array(events)


def _isotonic_bands(edges, probabilities) -> tuple[np.ndarray, np.ndarray]:
    """Check and return the band edges and probabilities of an isotonic calibration."""
    try:
        edges = np.asarray(edges, dtype=float)
        probabilities = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("edges and probabilities must be numeric") from None
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


def _float_scores(values: npt.ArrayLike, name: str) -> np.ndarray:
    """Scores as a float array, with missing values as NaN."""
    if isinstance(values, pd.Series):
        dtype = values.dtype
        if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_complex_dtype(dtype):
            raise ValueError(f"{name} must be real and numeric, got dtype {dtype}")
        return values.to_numpy(dtype=float, na_value=np.nan)
    array = np.asarray(values)
    if array.dtype.kind in "cUSMm":
        raise ValueError(f"{name} must be real and numeric, got dtype {array.dtype}")
    try:
        return array.astype(float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be real and numeric, got dtype {array.dtype}") from None


def _shaped(template: npt.ArrayLike, values: np.ndarray) -> float | pd.Series | np.ndarray:
    """Return values shaped like the input: a Series, a float or an array."""
    if isinstance(template, pd.Series):
        return pd.Series(values, index=template.index, name=template.name)
    return float(values) if values.ndim == 0 else values


def _odds_ratio(sample_rate: float, population_rate: float) -> float:
    for name, rate in (("sample_rate", sample_rate), ("population_rate", population_rate)):
        if isinstance(rate, bool) or not isinstance(rate, Real) or not 0 < rate < 1:
            raise ValueError(f"{name} must be strictly between 0 and 1, got {rate!r}")
    return (population_rate / (1 - population_rate)) / (sample_rate / (1 - sample_rate))


def _check_unit_interval(values: np.ndarray, name: str) -> None:
    outside = (values < 0) | (values > 1)
    if outside.any():
        raise ValueError(f"{name} must be between 0 and 1, got {_first(values, outside)}")


def _group_rows(groups: pd.Series, levels: pd.Index) -> np.ndarray:
    """Return the row of `levels`, from `group_counts`, of each observation, or -1 if absent."""
    if isinstance(levels, pd.CategoricalIndex):
        # Map category codes to rows instead of looking up the labels: that hashes
        # interval categories, which warns on NumPy < 2.2. Code -1 is a missing value.
        row_of_code = np.full(len(levels.categories) + 1, -1, dtype=np.intp)
        row_of_code[levels.codes] = np.arange(len(levels))
        return row_of_code[groups.cat.codes.to_numpy()]
    rows = levels.get_indexer(groups)
    missing = groups.isna().to_numpy(dtype=bool)
    missing_row = np.flatnonzero(pd.isna(levels))
    if missing.any():  # get_indexer does not match missing values on pandas 2.0
        rows[missing] = missing_row[0] if len(missing_row) else -1
    return rows


def _grade_inputs(
    event, total, probability
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index | None]:
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


def _pvalues(pvalue: np.ndarray, index: pd.Index | None) -> float | pd.Series | np.ndarray:
    if index is not None:
        return pd.Series(pvalue, index=index)
    return float(pvalue) if pvalue.ndim == 0 else pvalue


def _whole(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    return finite & (np.where(finite, values, 0) % 1 == 0)


def _first(values: np.ndarray, invalid: np.ndarray) -> str:
    return repr(float(values[invalid].flat[0]))
