"""Calibration of event probabilities and rating grades."""

import math
import typing
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import _counts, _logit, _stats, _validation

__all__ = [
    "assign_grades",
    "backtest_table",
    "binomial_test",
    "brier_decomposition",
    "calibration_table",
    "concentration_test",
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
    "spiegelhalter_test",
]


@typing.overload
def prior_correction(
    probability: pd.Series, *, sample_rate: float, population_rate: float
) -> pd.Series: ...
@typing.overload
def prior_correction(
    probability: float, *, sample_rate: float, population_rate: float
) -> float: ...
@typing.overload
def prior_correction(
    probability: _validation.ListLike, *, sample_rate: float, population_rate: float
) -> npt.NDArray[np.float64]: ...
@typing.overload
def prior_correction(
    probability: npt.ArrayLike, *, sample_rate: float, population_rate: float
) -> float | pd.Series | npt.NDArray[np.float64]: ...
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
    values = _validation.as_real(probability, "probability", booleans=True)
    _check_unit_interval(values, "probability")
    return _validation.like_input(probability, omega * values / (omega * values + 1 - values))


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
        or not _validation.is_real(intercept)
        or not math.isfinite(intercept)
    ):
        raise ValueError(
            f"intercept must be a finite number, got {_validation.shown_argument(intercept)}"
        )
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
    values, labels, mass = _validation.scores_and_labels(score, target, "score", weights=weights)
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


@typing.overload
def platt_predict(score: pd.Series, slope: float, intercept: float) -> pd.Series: ...
@typing.overload
def platt_predict(score: float, slope: float, intercept: float) -> float: ...
@typing.overload
def platt_predict(
    score: _validation.ListLike, slope: float, intercept: float
) -> npt.NDArray[np.float64]: ...
@typing.overload
def platt_predict(
    score: npt.ArrayLike, slope: float, intercept: float
) -> float | pd.Series | npt.NDArray[np.float64]: ...
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
        if isinstance(value, bool) or not _validation.is_real(value) or not math.isfinite(value):
            raise ValueError(
                f"{name} must be a finite number, got {_validation.shown_argument(value)}"
            )
    values = _float_scores(score, "score")
    # A huge score overflows to an infinite log odds, and logaddexp warns on missing values.
    with np.errstate(over="ignore", invalid="ignore"):
        probability = np.exp(-np.logaddexp(0, -(slope * values + intercept)))
    return _validation.like_input(score, probability)


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
    probability : numpy.ndarray
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
    >>> edges, probability = isotonic_fit(score, target)
    >>> edges
    array([-inf,  0.1,  0.4,  0.6,  inf])
    >>> probability
    array([0.        , 0.33333333, 0.5       , 1.        ])
    >>> isotonic_predict([0.05, 0.25, 0.45, 0.9], edges, probability).round(4)
    array([0.    , 0.3333, 0.5   , 1.    ])
    """
    if not isinstance(increasing, (bool, np.bool_)):
        raise ValueError(
            f"increasing must be True or False, got {_validation.shown_argument(increasing)}"
        )
    values, labels, mass = _validation.scores_and_labels(score, target, "score", weights=weights)
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


@typing.overload
def isotonic_predict(
    score: pd.Series, edges: npt.ArrayLike, probability: npt.ArrayLike
) -> pd.Series: ...
@typing.overload
def isotonic_predict(score: float, edges: npt.ArrayLike, probability: npt.ArrayLike) -> float: ...
@typing.overload
def isotonic_predict(
    score: _validation.ListLike, edges: npt.ArrayLike, probability: npt.ArrayLike
) -> npt.NDArray[np.float64]: ...
@typing.overload
def isotonic_predict(
    score: npt.ArrayLike, edges: npt.ArrayLike, probability: npt.ArrayLike
) -> float | pd.Series | npt.NDArray[np.float64]: ...
def isotonic_predict(
    score: npt.ArrayLike, edges: npt.ArrayLike, probability: npt.ArrayLike
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
    probability : array_like
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
        ``-inf`` to ``inf``, or `probability` does not give one value
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
    edges, bands = _isotonic_bands(edges, probability)
    values = _float_scores(score, "score")
    band = np.searchsorted(edges[1:-1], values, side="left")
    result = bands[np.minimum(band, len(bands) - 1)]
    result = np.where(np.isnan(values), np.nan, result)
    return _validation.like_input(score, result)


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
    _, probabilities, mass, counts, rows = _grouped_probabilities(
        groups, target, probability, weights
    )
    total = counts.to_numpy().sum(axis=1)
    event = counts.to_numpy()[:, 1]
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


@typing.overload
def binomial_test(
    event: pd.Series, total: npt.ArrayLike, probability: npt.ArrayLike
) -> pd.Series: ...
@typing.overload
def binomial_test(
    event: npt.ArrayLike, total: pd.Series, probability: npt.ArrayLike
) -> pd.Series: ...
@typing.overload
def binomial_test(
    event: npt.ArrayLike, total: npt.ArrayLike, probability: pd.Series
) -> pd.Series: ...
@typing.overload
def binomial_test(event: float, total: float, probability: float) -> float: ...
@typing.overload
def binomial_test(
    event: _validation.ListLike,
    total: float | _validation.ListLike,
    probability: float | _validation.ListLike,
) -> npt.NDArray[np.float64]: ...
@typing.overload
def binomial_test(
    event: npt.ArrayLike, total: npt.ArrayLike, probability: npt.ArrayLike
) -> float | pd.Series | npt.NDArray[np.float64]: ...
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
    pvalue[some] = _stats.betainc(event[some], total[some] - event[some] + 1, probability[some])
    return _pvalues(pvalue, index)


@typing.overload
def jeffreys_test(
    event: pd.Series, total: npt.ArrayLike, probability: npt.ArrayLike
) -> pd.Series: ...
@typing.overload
def jeffreys_test(
    event: npt.ArrayLike, total: pd.Series, probability: npt.ArrayLike
) -> pd.Series: ...
@typing.overload
def jeffreys_test(
    event: npt.ArrayLike, total: npt.ArrayLike, probability: pd.Series
) -> pd.Series: ...
@typing.overload
def jeffreys_test(event: float, total: float, probability: float) -> float: ...
@typing.overload
def jeffreys_test(
    event: _validation.ListLike,
    total: float | _validation.ListLike,
    probability: float | _validation.ListLike,
) -> npt.NDArray[np.float64]: ...
@typing.overload
def jeffreys_test(
    event: npt.ArrayLike, total: npt.ArrayLike, probability: npt.ArrayLike
) -> float | pd.Series | npt.NDArray[np.float64]: ...
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
    return _pvalues(_stats.betainc(event + 0.5, total - event + 0.5, probability), index)


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
    if isinstance(ddof, bool) or not _validation.is_integral(ddof) or ddof < 0:
        raise ValueError(
            f"ddof must be a non-negative integer, got {_validation.shown_argument(ddof)}"
        )
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
    return statistic, _stats.chi2_sf(statistic, dof)


def spiegelhalter_test(
    probability: npt.ArrayLike, target: npt.ArrayLike, *, weights: npt.ArrayLike | None = None
) -> tuple[float, float]:
    r"""Test the calibration of individual probabilities with Spiegelhalter's z statistic.

    The test compares the Brier score of the probabilities with the score
    they would have if they were the true event probabilities. Unlike the
    tests of grouped counts, it needs no groups. A small p-value means that
    the probabilities do not match the outcomes, in either direction.

    Parameters
    ----------
    probability : array_like
        Predicted event probability of each observation, between 0 and 1.
    target : array_like
        Binary labels, with 1 indicating the event. The sample may have no
        events. If `probability` and `target` are both pandas Series, they
        must have identical indices; otherwise they are matched by position.
    weights : array_like, optional
        Non-negative frequency weight of each observation, matched like
        `target`: integer weights give the same result as repeating rows.

    Returns
    -------
    statistic : float
        The z statistic, approximately standard normal when the
        probabilities are the true ones. It is positive when the Brier
        score of the probabilities exceeds its expectation under them, and
        negative when it falls short of it. When every probability is
        below 1/2, as PDs usually are, a positive statistic means more
        events than the probabilities predict and a negative one fewer;
        both reject the probabilities.
    pvalue : float
        Its two-sided p-value.

    Raises
    ------
    ValueError
        If the inputs are empty or not one-dimensional, their indices or
        lengths do not match, `target` has a label other than 0 and 1, or
        `probability` is not real and numeric, has a missing value or a
        value outside [0, 1]. Also if `weights` is not numeric, does not
        give each observation one finite, non-negative weight with a
        positive total, or has an index other than that of `probability`.

    See Also
    --------
    hosmer_lemeshow : Test grouped counts against their mean probabilities.
    brier_decomposition : Split the Brier score into calibration and
        resolution.
    kivi.metrics.brier_score : The Brier score itself.

    Notes
    -----
    For probabilities :math:`p_i` and outcomes :math:`y_i` of :math:`N`
    observations, the Brier score :math:`B = \sum_i (y_i - p_i)^2 / N` has
    the expectation :math:`\sum_i p_i (1 - p_i) / N` and the variance
    :math:`\sum_i (1 - 2 p_i)^2 p_i (1 - p_i) / N^2` when each :math:`y_i`
    is an independent Bernoulli draw with probability :math:`p_i`.
    Spiegelhalter [1]_ standardizes the score with them:

    .. math::

       z = \frac{\sum_i (y_i - p_i)(1 - 2 p_i)}
       {\sqrt{\sum_i (1 - 2 p_i)^2 \, p_i (1 - p_i)}},

    and the p-value is :math:`2 (1 - \Phi(|z|))`. With weights, every sum
    weighs its terms with the weights of the observations.

    The denominator is 0 when every probability is 0, 1/2 or 1. ``z`` is
    then infinite, with a p-value of 0, when an event has a probability of
    0 or a non-event one of 1, and NaN otherwise.

    The test weighs observations with probabilities near 0 or 1 most, so a
    few confident mistakes can decide it. Like the other calibration
    tests, it assumes independent outcomes.

    References
    ----------
    .. [1] D. J. Spiegelhalter, "Probabilistic prediction in patient
       management and clinical trials", Statistics in Medicine, 5(5),
       pp. 421-433, 1986.

    Examples
    --------
    Two events among the three lowest probabilities make the probabilities
    look too confident:

    >>> from kivi.calibration import spiegelhalter_test
    >>> probability = [0.1, 0.2, 0.2, 0.4, 0.7, 0.9]
    >>> statistic, pvalue = spiegelhalter_test(probability, [0, 1, 1, 0, 0, 1])
    >>> print(f"z = {statistic:.3f}, p = {pvalue:.3f}")
    z = 1.912, p = 0.056

    Outcomes that follow the probabilities more closely than chance give a
    Brier score below its expectation and a negative ``z``:

    >>> statistic, pvalue = spiegelhalter_test(probability, [0, 0, 0, 1, 1, 1])
    >>> print(f"z = {statistic:.3f}, p = {pvalue:.3f}")
    z = -0.765, p = 0.444
    """
    values, labels, mass = _validation.scores_and_labels(
        probability, target, "probability", both_classes=False, weights=weights
    )
    _check_unit_interval(values, "probability")
    if mass is None:
        mass = np.ones(len(values))
    slope = 1 - 2 * values
    numerator = float(np.sum(mass * (labels - values) * slope))
    variance = float(np.sum(mass * slope**2 * values * (1 - values)))
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = float(np.float64(numerator) / math.sqrt(variance))
    pvalue = math.nan if math.isnan(statistic) else math.erfc(abs(statistic) / math.sqrt(2))
    return statistic, pvalue


def brier_decomposition(
    groups: npt.ArrayLike,
    target: npt.ArrayLike,
    probability: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
) -> pd.Series:
    r"""Split the Brier score into reliability, resolution, uncertainty and two within-group terms.

    The decomposition shows why a model scores as it does. Reliability
    measures how far the mean probability of each group lies from its event
    rate, resolution how far the event rates of the groups lie from the
    overall event rate, and uncertainty how hard the outcomes are to
    predict at all. The two within-group terms account for the
    probabilities that vary within a group.

    Parameters
    ----------
    groups : array_like
        Grade or group labels, such as the grades of a master scale.
        Missing values form a group.
    target : array_like
        Binary labels, with 1 indicating the event. A group, or the whole
        sample, may have no events.
    probability : array_like
        Predicted event probabilities between 0 and 1. Series inputs must
        have identical indices; otherwise the inputs are matched by
        position.
    weights : array_like, optional
        Non-negative frequency weight of each observation, matched like
        `target`: integer weights give the same result as repeating rows.

    Returns
    -------
    pandas.Series
        ``brier``, the Brier score, and its five terms ``reliability``,
        ``resolution``, ``uncertainty``, ``within_variance`` and
        ``within_covariance``, in an index named ``statistic``.

    Raises
    ------
    ValueError
        In the same cases as `calibration_table`.

    See Also
    --------
    calibration_table : The event rate and mean probability of each group.
    kivi.metrics.brier_score : The Brier score alone.
    spiegelhalter_test : Test whether the Brier score is as low as the
        probabilities predict.

    Notes
    -----
    Let group :math:`k` hold :math:`n_k` of the :math:`N` observations,
    with mean probability :math:`\bar f_k` and event rate :math:`\bar o_k`,
    and let :math:`\bar o` be the overall event rate. The Brier score of
    probabilities :math:`f_i` and outcomes :math:`o_i` is

    .. math::

       \mathrm{BS} = \frac{1}{N} \sum_i (f_i - o_i)^2
       = \mathrm{REL} - \mathrm{RES} + \mathrm{UNC} + \mathrm{WBV} - \mathrm{WBC},

    with

    .. math::

       \mathrm{REL} = \frac{1}{N} \sum_k n_k (\bar f_k - \bar o_k)^2, \qquad
       \mathrm{RES} = \frac{1}{N} \sum_k n_k (\bar o_k - \bar o)^2, \qquad
       \mathrm{UNC} = \bar o (1 - \bar o),

       \mathrm{WBV} = \frac{1}{N} \sum_k \sum_{i \in k} (f_i - \bar f_k)^2, \qquad
       \mathrm{WBC} = \frac{2}{N} \sum_k \sum_{i \in k}
       (f_i - \bar f_k)(o_i - \bar o_k).

    The identity holds exactly for any grouping, up to rounding. When all
    observations of a group have the same probability, as on a master
    scale whose grades each have one PD, the two within-group terms are 0
    and the decomposition is the three-term partition of Murphy [1]_; the
    within-group terms are those of Stephenson, Coelho and Jolliffe [2]_.

    A lower reliability is better and a higher resolution is better.
    Uncertainty depends on the outcomes alone. With weights, :math:`n_k`
    and :math:`N` are sums of weights and every sum weighs its terms.

    References
    ----------
    .. [1] A. H. Murphy, "A new vector partition of the probability score",
       Journal of Applied Meteorology, 12(4), pp. 595-600, 1973.
    .. [2] D. B. Stephenson, C. A. S. Coelho and I. T. Jolliffe, "Two extra
       components in the Brier score decomposition", Weather and
       Forecasting, 23(4), pp. 752-757, 2008.

    Examples
    --------
    >>> from kivi.calibration import brier_decomposition
    >>> grades = ["A", "A", "A", "B", "B", "B"]
    >>> target = [0, 0, 1, 0, 1, 1]
    >>> brier_decomposition(grades, target, [0.2, 0.2, 0.2, 0.6, 0.6, 0.6]).round(4)
    statistic
    brier                0.2333
    reliability          0.0111
    resolution           0.0278
    uncertainty          0.2500
    within_variance      0.0000
    within_covariance    0.0000
    dtype: float64

    Each grade has one probability, so the within-group terms are 0 and
    0.0111 - 0.0278 + 0.25 is the Brier score.
    """
    labels, probabilities, mass, counts, rows = _grouped_probabilities(
        groups, target, probability, weights
    )
    kept = rows >= 0
    rows, forecast, outcome = rows[kept], probabilities[kept], labels[kept].astype(float)
    mass = np.ones(len(rows)) if mass is None else mass[kept]
    total = counts.to_numpy().sum(axis=1).astype(float)
    size = float(total.sum())
    event_rate = counts.to_numpy()[:, 1] / total
    overall = float(counts.to_numpy()[:, 1].sum()) / size
    mean_forecast, deviation = _group_means(forecast, rows, mass, total)
    terms = {
        "brier": np.sum(mass * (forecast - outcome) ** 2) / size,
        "reliability": np.sum(total * (mean_forecast - event_rate) ** 2) / size,
        "resolution": np.sum(total * (event_rate - overall) ** 2) / size,
        "uncertainty": overall * (1 - overall),
        "within_variance": np.sum(mass * deviation**2) / size,
        "within_covariance": 2 * np.sum(mass * deviation * (outcome - event_rate[rows])) / size,
    }
    return pd.Series(
        [float(value) for value in terms.values()],
        index=pd.Index(list(terms), name="statistic"),
    )


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
        raise ValueError(
            f"normalize must be True or False, got {_validation.shown_argument(normalize)}"
        )
    shares = _grade_shares(counts, "counts")
    index = float(np.sum(shares**2))
    if not normalize:
        return index
    if len(shares) < 2:
        raise ValueError(f"normalize requires at least two grades, got {len(shares)}")
    return (index - 1 / len(shares)) / (1 - 1 / len(shares))


def concentration_test(
    counts: npt.ArrayLike,
    *,
    initial_counts: npt.ArrayLike | None = None,
    initial_cv: float | None = None,
) -> pd.Series:
    r"""Test whether the grades of a rating scale are more concentrated than at development.

    The European Central Bank asks for this test to check that a rating
    scale still spreads its customers over the grades [1]_. It compares the
    coefficient of variation of the grade shares with its value at the
    initial validation of the model. A small p-value means that the grades
    are now more concentrated.

    Parameters
    ----------
    counts : array_like
        Number of customers, or exposure, in each of the grades of the
        scale, empty grades included, such as
        ``grades.value_counts(sort=False)``. At least two grades.
    initial_counts : array_like, optional
        Number of customers in each grade at the initial validation, for
        the same grades in the same order. If `counts` and `initial_counts`
        are both Series, they must have identical indices.
    initial_cv : float, optional
        The coefficient of variation at the initial validation, when only
        that is known. Pass `initial_counts` or `initial_cv`, or neither to
        compute only the HI, as the European Central Bank asks for the
        exposure-weighted shares.

    Returns
    -------
    pandas.Series
        ``hi`` and ``cv``, the HI and the coefficient of variation of
        `counts`; ``initial_hi`` and ``initial_cv``, those at the initial
        validation; ``statistic``, the test statistic; and ``p_value``, its
        one-sided p-value. The last four are NaN without `initial_counts`
        and `initial_cv`. The index is named ``statistic``.

    Raises
    ------
    ValueError
        If `counts` or `initial_counts` is not one-dimensional, has
        negative or non-finite values or sums to zero, if `counts` has
        fewer than two grades or `initial_counts` another number of grades
        or another index, if `initial_cv` is not a finite, non-negative
        number or exceeds :math:`\sqrt{K - 1}`, or if both `initial_counts`
        and `initial_cv` are given.

    See Also
    --------
    herfindahl_index : The Herfindahl-Hirschman index of the shares.
    calibration_table : The number of observations in each grade.

    Notes
    -----
    For the shares :math:`R_i` of the :math:`K` grades, the coefficient of
    variation and the HI are

    .. math::

       \mathrm{CV} = \sqrt{K \sum_{i=1}^{K} \left(R_i - \frac{1}{K}\right)^2},
       \qquad
       \mathrm{HI} = 1 + \frac{\ln\left((\mathrm{CV}^2 + 1) / K\right)}{\ln K}
       = 1 + \frac{\ln \mathrm{HHI}}{\ln K},

    where :math:`\mathrm{HHI} = \sum_i R_i^2` is the result of
    `herfindahl_index`. The HI is 0 when every grade has the same share
    and 1 when one grade holds every customer, and the CV runs from 0 to
    :math:`\sqrt{K - 1}`. The test statistic

    .. math::

       S = \frac{\sqrt{K - 1}\,(\mathrm{CV} - \mathrm{CV}_\mathrm{init})}
       {\sqrt{\mathrm{CV}^2 (0.5 + \mathrm{CV}^2)}}

    is approximately standard normal under the null hypothesis that the
    current concentration is not above the initial one, which is taken as
    known, and the p-value is :math:`1 - \Phi(S)`. The variance in the
    denominator is the asymptotic variance of a coefficient of variation in
    Miller [2]_. A CV of 0, for equally full grades, gives a statistic of
    minus infinity and a p-value of 1, or NaN for an initial CV of 0.

    References
    ----------
    .. [1] European Central Bank, "Instructions for reporting the
       validation results of internal models: IRB Pillar I models for
       credit risk", February 2019, section 2.5.5.3.
    .. [2] G. E. Miller, "Asymptotic test statistics for coefficients of
       variation", Communications in Statistics - Theory and Methods,
       20(10), pp. 3351-3363, 1991.

    Examples
    --------
    >>> from kivi.calibration import concentration_test
    >>> concentration_test([400, 250, 200, 150], initial_counts=[300, 300, 250, 150]).round(4)
    statistic
    hi            0.0945
    cv            0.3742
    initial_hi    0.0420
    initial_cv    0.2449
    statistic     0.7477
    p_value       0.2273
    dtype: float64

    The scale is more concentrated than at development, but not
    significantly so.
    """
    shares = _grade_shares(counts, "counts")
    grades = len(shares)
    if grades < 2:
        raise ValueError(f"counts must have at least two grades, got {grades}")
    if initial_counts is not None and initial_cv is not None:
        raise ValueError("pass initial_counts or initial_cv, not both")
    _validation.shared_index({"counts": counts, "initial_counts": initial_counts})
    hi, cv = _concentration(shares)
    initial_hi = start_cv = math.nan
    if initial_counts is not None:
        initial = _grade_shares(initial_counts, "initial_counts")
        if len(initial) != grades:
            raise ValueError(
                f"initial_counts must give one count for each of the {grades} grades of counts, "
                f"got {len(initial)}"
            )
        initial_hi, start_cv = _concentration(initial)
    elif initial_cv is not None:
        if (
            isinstance(initial_cv, (bool, np.bool_))
            or not _validation.is_real(initial_cv)
            or not (math.isfinite(initial_cv) and initial_cv >= 0)
        ):
            raise ValueError(
                "initial_cv must be a finite, non-negative number, got "
                f"{_validation.shown_argument(initial_cv)}"
            )
        start_cv = float(initial_cv)
        # A share of 1 in one grade gives the largest CV, sqrt(K - 1).
        if start_cv > math.sqrt(grades - 1) * (1 + 1e-12):
            raise ValueError(
                f"initial_cv must be at most {math.sqrt(grades - 1):.6g}, the coefficient of "
                f"variation of one full grade among {grades}, got {start_cv!r}"
            )
        initial_hi = 1 + math.log((start_cv**2 + 1) / grades) / math.log(grades)
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = float(
            np.float64(math.sqrt(grades - 1) * (cv - start_cv)) / (cv * math.sqrt(0.5 + cv**2))
        )
    values = {
        "hi": hi,
        "cv": cv,
        "initial_hi": initial_hi,
        "initial_cv": start_cv,
        "statistic": statistic,
        "p_value": float(_stats.normal_cdf(-statistic)),
    }
    return pd.Series(list(values.values()), index=pd.Index(list(values), name="statistic"))


def backtest_table(
    groups: npt.ArrayLike,
    realized: npt.ArrayLike,
    estimated: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Back-test continuous estimates, such as LGDs or CCFs, against realized values.

    For each group, the one-sample t-test for paired observations of the
    European Central Bank [1]_ compares the realized values with their
    estimates. It tests the hypothesis that the estimates are not too
    low, so a small p-value means that the group's estimates understate
    its realized values.

    Parameters
    ----------
    groups : array_like
        Grade, pool or segment labels, such as the facility grades of an
        LGD model or the ECB's LGD segments cut from the estimates. Missing
        values form a group. For the whole portfolio, pass one label for
        every observation.
    realized : array_like
        Realized value of each observation, such as the realized LGD of a
        closed default. Finite real numbers, which may lie outside [0, 1].
    estimated : array_like
        Estimate of each observation, such as its estimated LGD, ELBE or
        CCF. Finite real numbers. Series inputs must have identical
        indices; otherwise the inputs are matched by position.
    weights : array_like, optional
        Non-negative frequency weight of each observation, matched like
        `realized`: integer weights give the same result as repeating rows.
        Groups whose weights sum to zero are left out.

    Returns
    -------
    pandas.DataFrame
        One row per group, in an index named ``bin``, with the number of
        observations ``total``; the means ``estimated`` and ``realized``;
        ``difference``, the mean of realized minus estimated; ``variance``,
        the sample variance of that difference; ``statistic``, the t
        statistic; and ``p_value``, its one-sided p-value.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, are empty, or their indices
        or lengths do not match, or `realized` or `estimated` is not real
        and numeric or has a missing or infinite value. Also if `weights`
        is not numeric, does not give each observation one finite,
        non-negative weight with a positive total, or has an index other
        than that of `groups`.

    See Also
    --------
    calibration_table : The same comparison for event probabilities.
    jeffreys_test : The back-test of PDs.
    kivi.metrics.generalized_auc : The ranking of continuous estimates.

    Notes
    -----
    For the :math:`N` observations of a group with differences
    :math:`d_i = r_i - e_i` between realized and estimated values,

    .. math::

       T = \frac{\sqrt{N}\,\bar d}{s}, \qquad
       s^2 = \frac{1}{N - 1} \sum_{i=1}^{N} (d_i - \bar d)^2,

    which follows Student's t distribution with :math:`N - 1` degrees of
    freedom under the null hypothesis, and the p-value is
    :math:`1 - S_{N-1}(T)`. This is the test of sections 2.6.2.1, 2.8.2.1,
    2.9.3.1 and 2.9.3.2 of [1]_ for LGDs, LGDs in default, CCFs and the
    EAD estimates of facilities with an EAD approach, whose realized
    value is the amount drawn at default. Section 2.7.2.1 tests the ELBE
    two-sided; its p-value is ``2 * min(p_value, 1 - p_value)``.

    A group of one observation has no variance, and its ``variance``,
    ``statistic`` and ``p_value`` are NaN. A group whose differences are
    all equal has a variance of 0 and an infinite statistic, or a NaN one
    when the differences are all 0. With weights, :math:`N` is the sum of
    the weights, the means are weighted, and so is the variance, with the
    denominator :math:`N - 1`; a group with a total weight of 1 or less has
    NaN in the last three columns.

    Groups are counted as in `calibration_table`: categorical groups, such
    as the result of ``pandas.cut``, follow the order of their categories,
    with missing values last, and other groups the order in which they
    first appear.

    References
    ----------
    .. [1] European Central Bank, "Instructions for reporting the
       validation results of internal models: IRB Pillar I models for
       credit risk", February 2019, sections 2.6.2.1, 2.7.2.1, 2.8.2.1,
       2.9.3.1 and 2.9.3.2.

    Examples
    --------
    >>> from kivi.calibration import backtest_table
    >>> grades = ["F1", "F1", "F1", "F2", "F2", "F2", "F2"]
    >>> realized = [0.05, 0.35, 0.20, 0.55, 0.20, 0.90, 0.75]
    >>> estimated = [0.15, 0.15, 0.15, 0.45, 0.45, 0.45, 0.45]
    >>> backtest_table(grades, realized, estimated).round(4)
         total  estimated  realized  difference  variance  statistic  p_value
    bin
    F1       3       0.15       0.2        0.05    0.0225     0.5774   0.3110
    F2       4       0.45       0.6        0.15    0.0917     0.9909   0.1974

    In grade F1, the differences -0.1, 0.2 and 0.05 have the mean 0.05 and
    the variance 0.0225, so :math:`T = \sqrt{3} \cdot 0.05 / 0.15`.
    """
    inputs = {"groups": groups, "realized": realized, "estimated": estimated}
    _validation.shared_index(inputs)
    raw_groups = groups
    groups, realized, estimated = (
        _validation.as_series(values, name) for name, values in inputs.items()
    )
    if len({len(groups), len(realized), len(estimated)}) > 1:
        raise ValueError(
            "groups, realized and estimated must have the same length, "
            f"got {len(groups)}, {len(realized)} and {len(estimated)}"
        )
    if groups.empty:
        raise ValueError("groups, realized and estimated must contain observations")
    realized_values = _finite_values(realized, "realized")
    estimated_values = _finite_values(estimated, "estimated")
    others = {name: inputs[name] for name in ("realized", "estimated")}
    mass = _validation.as_weights(weights, raw_groups, of="groups", others=others)
    counts = _counts.group_counts(groups, np.zeros(len(groups), dtype=np.intp), mass)
    rows = _counts.group_rows(groups, counts.index)
    kept = rows >= 0
    rows = rows[kept]
    mass = np.ones(len(rows)) if mass is None else mass[kept]
    total = counts.to_numpy()[:, 0]
    means = {
        name: np.bincount(rows, weights=mass * values[kept], minlength=len(total)) / total
        for name, values in (("estimated", estimated_values), ("realized", realized_values))
    }
    difference, deviation = _group_means(
        realized_values[kept] - estimated_values[kept], rows, mass, total.astype(float)
    )
    squares = np.bincount(rows, weights=mass * deviation**2, minlength=len(total))
    enough = total > 1
    variance = np.divide(squares, total - 1, out=np.full(len(total), np.nan), where=enough)
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = difference / np.sqrt(variance / total)
    p_value = np.full(len(total), np.nan)
    known = ~np.isnan(statistic)
    p_value[known] = _stats.student_t_sf(statistic[known], total[known] - 1)
    return pd.DataFrame(
        {
            "total": total,
            "estimated": means["estimated"],
            "realized": means["realized"],
            "difference": difference,
            "variance": variance,
            "statistic": statistic,
            "p_value": p_value,
        },
        index=counts.index,
    )


def master_scale(
    probability: npt.ArrayLike,
    *,
    boundary: typing.Literal["geometric", "arithmetic"] = "geometric",
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
        the index of `probability` if it is a Series, such as the result of
        `fit_grade_pd` for a table of grades; otherwise the grades are
        numbered from 1.

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
        (or the index of a Series `probability` without `labels`) does not
        give one unique, non-missing label per grade.

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

    A Series gives its index as the labels:

    >>> import pandas as pd
    >>> master_scale(pd.Series([0.01, 0.04], index=["low", "high"])).round(4)
           lower  probability  upper
    grade
    low     0.00         0.01   0.02
    high    0.02         0.04   1.00
    """
    if not isinstance(boundary, str) or boundary not in ("geometric", "arithmetic"):  # type: ignore[redundant-expr]  # rejects non-str values
        raise ValueError(
            "boundary must be 'geometric' or 'arithmetic', got "
            f"{_validation.shown_argument(boundary)}"
        )
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
    if labels is None and isinstance(probability, pd.Series):
        index = _grade_labels(probability.index, len(values), "the index of probability")
    else:
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


@typing.overload
def fit_grade_pd(
    event: pd.Series, total: npt.ArrayLike, *, smoothing: float = ...
) -> pd.Series: ...
@typing.overload
def fit_grade_pd(
    event: npt.ArrayLike, total: pd.Series, *, smoothing: float = ...
) -> pd.Series: ...
@typing.overload
def fit_grade_pd(
    event: _validation.ListLike, total: _validation.ListLike, *, smoothing: float = ...
) -> npt.NDArray[np.float64]: ...
@typing.overload
def fit_grade_pd(
    event: npt.ArrayLike, total: npt.ArrayLike, *, smoothing: float = ...
) -> pd.Series | npt.NDArray[np.float64]: ...
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
        those of a weighted `calibration_table`, must be on the scale of
        observation counts, because `smoothing` is a number of defaults: on
        a smaller scale, such as with weights that sum to 1, the smoothed
        rate of a grade without defaults is too high.
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
        `total`; if fewer than two grades have observations, or `smoothing`
        is not finite and positive or not smaller than the total of a grade
        without defaults. Also if the fitted PDs do not rise with the grade
        or reach 1.

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
    smoothing = _validation.check_smoothing(smoothing)
    events = _numeric_values(event, "event", booleans=False)
    totals = _numeric_values(total, "total", booleans=False)
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
    # A smoothed rate of 1 or more would leave the logarithm without meaning.
    unsmoothable = observed & (events == 0) & (totals <= smoothing)
    if unsmoothable.any():
        position = int(np.argmax(unsmoothable))
        label = indices[0][position] if indices else position + 1
        raise ValueError(
            "smoothing must be smaller than the total of each grade without defaults, got "
            f"{_validation.shown_argument(smoothing)} for grade {_validation.shown(label)} with a "
            f"total of {_first(totals, unsmoothable)}; pass counts on the scale of observations "
            "or a smaller smoothing"
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


@typing.overload
def assign_grades(probability: pd.Series, scale: pd.DataFrame) -> pd.Series: ...
@typing.overload
def assign_grades(probability: _validation.ListLike, scale: pd.DataFrame) -> pd.Categorical: ...
@typing.overload
def assign_grades(
    probability: npt.ArrayLike, scale: pd.DataFrame
) -> pd.Series | pd.Categorical: ...
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
    series = _validation.as_series(probability, "probability")
    values = _numeric_values(series, "probability")
    _check_unit_interval(values, "probability")
    codes = np.searchsorted(upper[:-1], values, side="left")
    codes[np.isnan(values)] = -1
    grades = pd.Categorical.from_codes(codes, categories=scale.index, ordered=True)
    if isinstance(probability, pd.Series):
        return pd.Series(grades, index=series.index, name=series.name)
    return grades


def _grouped_probabilities(
    groups: npt.ArrayLike,
    target: npt.ArrayLike,
    probability: npt.ArrayLike,
    weights: npt.ArrayLike | None,
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64] | None,
    pd.DataFrame,
    npt.NDArray[np.intp],
]:
    """Check the groups, labels, probabilities and weights of one sample, and count the groups.

    Returns the 0/1 labels, the probabilities and the weights of the observations, the
    (non_event, event) counts of `kivi._counts.group_counts` and the row of each observation in
    them. Only an observation of zero weight lies in a group that the counts leave out, and its
    row is -1.
    """
    inputs = {"groups": groups, "target": target, "probability": probability}
    indexed = [values.index for values in inputs.values() if isinstance(values, pd.Series)]
    if any(not index.equals(indexed[0]) for index in indexed[1:]):
        raise ValueError(
            "groups, target and probability must have identical indices when they are Series"
        )
    raw_groups = groups
    groups, target, probability = (
        _validation.as_series(values, name) for name, values in inputs.items()
    )
    if len({len(groups), len(target), len(probability)}) > 1:
        raise ValueError(
            "groups, target and probability must have the same length, "
            f"got {len(groups)}, {len(target)} and {len(probability)}"
        )
    if groups.empty:
        raise ValueError("groups, target and probability must contain observations")
    labels = _validation.binary_labels(target, both_classes=False)
    probabilities = _validation.as_real(probability, "probability", booleans=True)
    if np.isnan(probabilities).any():
        raise ValueError("probability must not have missing values")
    _check_unit_interval(probabilities, "probability")
    others = {name: inputs[name] for name in ("target", "probability")}
    mass = _validation.as_weights(weights, raw_groups, of="groups", others=others)
    counts = _counts.group_counts(groups, labels, mass)
    return labels, probabilities, mass, counts, _counts.group_rows(groups, counts.index)


def _group_means(
    values: npt.NDArray[np.float64],
    rows: npt.NDArray[np.intp],
    mass: npt.NDArray[np.float64],
    total: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Weighted mean of each group, and the deviation of each observation from its group's mean.

    ``rows`` gives the group of each observation, and every group has an observation of positive
    weight. The values are shifted by the first value of positive weight in their group before
    they are averaged, so that a group whose weighted values are equal has that value as its mean
    and deviations of exactly 0 where the weights are positive.
    """
    weighted = np.flatnonzero(mass > 0)
    _, first = np.unique(rows[weighted], return_index=True)
    shift = values[weighted[first]]
    shifted = values - shift[rows]
    mean_shift = np.bincount(rows, weights=mass * shifted, minlength=len(total)) / total
    return shift + mean_shift, shifted - mean_shift[rows]


def _grade_shares(counts: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Check the counts or exposures of the grades and return the share of each grade."""
    if np.ndim(counts) != 1:
        raise ValueError(f"{name} must be one-dimensional, got {np.ndim(counts)} dimensions")
    values = _validation.as_real(counts, name)
    invalid = ~np.isfinite(values) | (values < 0)
    if invalid.any():
        raise ValueError(f"{name} must be finite and non-negative, got {_first(values, invalid)}")
    if not values.sum() > 0:
        raise ValueError(f"{name} must have a positive sum, got {float(values.sum())!r}")
    return values / values.sum()


def _concentration(shares: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Return the HI and the coefficient of variation of the shares of the grades."""
    grades = len(shares)
    cv = math.sqrt(grades * float(np.sum((shares - 1 / grades) ** 2)))
    return 1 + math.log(float(np.sum(shares**2))) / math.log(grades), cv


def _finite_values(values: pd.Series, name: str) -> npt.NDArray[np.float64]:
    """Finite real values as floats; booleans, missing values and infinities raise."""
    array = _validation.as_real(values, name)
    if np.isnan(array).any():
        raise ValueError(f"{name} must not have missing values")
    infinite = np.isinf(array)
    if infinite.any():
        raise ValueError(f"{name} must be finite, got {_first(array, infinite)}")
    return array


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
    edges: npt.ArrayLike, probability: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Check and return the band edges and probabilities of an isotonic calibration."""
    edges = _numeric_values(edges, "edges", booleans=False)
    probabilities = _numeric_values(probability, "probability")
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not (np.diff(edges) > 0).all()
        or edges[0] != -np.inf
        or edges[-1] != np.inf
    ):
        raise ValueError(
            f"edges must increase from -inf to inf, got {_validation.shown(edges.tolist())}"
        )
    if probabilities.shape != (len(edges) - 1,):
        raise ValueError(
            f"probability must give one value for each of the {len(edges) - 1} band(s), "
            f"got shape {probabilities.shape}"
        )
    if np.isnan(probabilities).any():
        raise ValueError("probability must not have missing values")
    _check_unit_interval(probabilities, "probability")
    return edges, probabilities


def _grade_labels(labels: npt.ArrayLike | None, grades: int, name: str = "labels") -> pd.Index:
    """Return the labels of a master scale's grades as an index named ``grade``.

    ``name`` is the argument the labels came from, for the error messages.
    """
    if labels is None:
        return pd.Index(np.arange(1, grades + 1), name="grade")
    if isinstance(labels, str):
        raise ValueError(
            "labels must be one label per grade, got the string "
            f"{_validation.shown_argument(labels)}"
        )
    series = _validation.as_series(labels, "labels")
    if len(series) != grades:
        raise ValueError(
            f"labels must give one label for each of the {grades} grade(s), got {len(series)}"
        )
    if series.isna().any():
        raise ValueError(f"{name} must not have missing values")
    repeated = series[series.duplicated()]
    if len(repeated):
        raise ValueError(
            f"{name} must be unique, got {_validation.shown(repeated.iloc[0])} more than once"
        )
    return pd.Index(series.to_numpy(), name="grade")


def _scale_bounds(scale: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Check a master scale and return the upper bounds of its grades."""
    if not isinstance(scale, pd.DataFrame):
        raise ValueError(f"scale must be a DataFrame from master_scale, got {type(scale).__name__}")
    absent = [name for name in ("lower", "upper") if name not in scale.columns]
    if absent:
        raise ValueError(
            "scale must have the columns 'lower' and 'upper', missing "
            f"{_validation.shown(absent[0])}"
        )
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
            f"got upper bounds {_validation.shown(upper.tolist())}"
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


def _numeric_values(
    values: npt.ArrayLike, name: str, *, booleans: bool = True
) -> npt.NDArray[np.float64]:
    """Return one-dimensional real numeric values as floats, with missing values as NaN.

    Booleans count as 1 and 0, as in probabilities and scores, unless ``booleans`` is
    False, as for counts and edges.
    """
    return _validation.as_real(_validation.as_series(values, name), name, booleans=booleans)


def _odds_ratio(sample_rate: float, population_rate: float) -> float:
    for name, rate in (("sample_rate", sample_rate), ("population_rate", population_rate)):
        if isinstance(rate, bool) or not _validation.is_real(rate) or not 0 < rate < 1:
            raise ValueError(
                f"{name} must be strictly between 0 and 1, got {_validation.shown_argument(rate)}"
            )
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
    values = (
        _validation.as_real(event, "event"),
        _validation.as_real(total, "total"),
        _validation.as_real(probability, "probability", booleans=True),
    )
    try:
        arrays = np.broadcast_arrays(*values)
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
