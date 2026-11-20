"""Calibration of event probabilities and rating grades."""

import math
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._counts import group_counts
from ._stats import betainc, chi2_sf
from ._validation import as_series

__all__ = [
    "binomial_test",
    "calibration_table",
    "corrected_intercept",
    "herfindahl_index",
    "hosmer_lemeshow",
    "jeffreys_test",
    "prior_correction",
]


def prior_correction(
    prob: npt.ArrayLike, *, sample_rate: float, population_rate: float
) -> float | pd.Series | npt.NDArray[np.float64]:
    r"""Correct event probabilities for a development sample with another event rate.

    A model developed on a sample that over- or undersamples events
    predicts probabilities for that sample. The correction rescales the odds
    so the probabilities refer to the population.

    Parameters
    ----------
    prob : array_like
        Event probabilities from the model, between 0 and 1. Missing values
        stay missing.
    sample_rate : float
        Event rate of the development sample, strictly between 0 and 1.
    population_rate : float
        Event rate of the population, strictly between 0 and 1.

    Returns
    -------
    float, pandas.Series or numpy.ndarray
        Corrected probabilities with the shape of `prob`. A Series input
        gives a Series with the same index and name, and a scalar gives a
        float.

    Raises
    ------
    ValueError
        If `prob` has values outside [0, 1], or a rate is not strictly
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
    values = np.asarray(prob, dtype=float)
    _check_unit_interval(values, "prob")
    corrected = omega * values / (omega * values + 1 - values)
    if isinstance(prob, pd.Series):
        return pd.Series(corrected, index=prob.index, name=prob.name)
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


def calibration_table(
    groups: npt.ArrayLike, target: npt.ArrayLike, prob: npt.ArrayLike
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
    prob : array_like
        Predicted event probabilities between 0 and 1. Series inputs must
        have identical indices; otherwise the inputs are matched by
        position.

    Returns
    -------
    pandas.DataFrame
        One row per group with the number of observations ``total``, the
        number of events ``event``, the observed ``event_rate`` and the
        mean predicted probability ``mean_prob``.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, are empty, or their indices
        or lengths do not match, `target` has labels other than 0 and 1, or
        `prob` has missing values or values outside [0, 1].

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

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.calibration import calibration_table
    >>> grades = pd.Series(["A", "A", "B", "B", "B"])
    >>> target = pd.Series([0, 0, 0, 1, 1])
    >>> prob = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> calibration_table(grades, target, prob)
         total  event  event_rate  mean_prob
    bin
    A        2      0    0.000000       0.15
    B        3      2    0.666667       0.40
    """
    inputs = {"groups": groups, "target": target, "prob": prob}
    indexed = [values.index for values in inputs.values() if isinstance(values, pd.Series)]
    if any(not index.equals(indexed[0]) for index in indexed[1:]):
        raise ValueError("groups, target and prob must have identical indices when they are Series")
    groups, target, prob = (as_series(values, name) for name, values in inputs.items())
    if len({len(groups), len(target), len(prob)}) > 1:
        raise ValueError(
            "groups, target and prob must have the same length, "
            f"got {len(groups)}, {len(target)} and {len(prob)}"
        )
    if groups.empty:
        raise ValueError("groups, target and prob must contain observations")
    invalid = ~target.isin([0, 1]).to_numpy()
    if invalid.any():
        raise ValueError(f"target must contain only 0 and 1, got {target[invalid].tolist()[0]!r}")
    probability = prob.to_numpy(dtype=float)
    if np.isnan(probability).any():
        raise ValueError("prob must not have missing values")
    _check_unit_interval(probability, "prob")
    counts = group_counts(groups, target.to_numpy(dtype=np.intp))
    rows = _group_rows(groups, counts.index)
    total = counts.to_numpy().sum(axis=1)
    event = counts.to_numpy()[:, 1]
    prob_sum = np.bincount(rows, weights=probability, minlength=len(counts))
    return pd.DataFrame(
        {
            "total": total,
            "event": event,
            "event_rate": event / total,
            "mean_prob": prob_sum / total,
        },
        index=counts.index,
    )


def binomial_test(
    event: npt.ArrayLike, total: npt.ArrayLike, prob: npt.ArrayLike
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
    prob : array_like
        Predicted event probability of each grade, between 0 and 1, such as
        the ``mean_prob`` column of `calibration_table`. The three inputs
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
        between 0 and `total`, or `prob` is outside [0, 1].

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
    event, total, prob, index = _grade_inputs(event, total, prob)
    pvalue = np.ones(event.shape)
    some = event > 0
    pvalue[some] = betainc(event[some], total[some] - event[some] + 1, prob[some])
    return _pvalues(pvalue, index)


def jeffreys_test(
    event: npt.ArrayLike, total: npt.ArrayLike, prob: npt.ArrayLike
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
    prob : array_like
        Predicted event probability of each grade, between 0 and 1, such as
        the ``mean_prob`` column of `calibration_table`. The three inputs
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
        between 0 and `total`, or `prob` is outside [0, 1].

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
    event, total, prob, index = _grade_inputs(event, total, prob)
    return _pvalues(betainc(event + 0.5, total - event + 0.5, prob), index)


def hosmer_lemeshow(
    event: npt.ArrayLike, total: npt.ArrayLike, prob: npt.ArrayLike, *, ddof: int = 2
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
    prob : array_like
        Mean predicted event probability of each group, strictly between 0
        and 1, such as the ``mean_prob`` column of `calibration_table`.
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
        not a whole number between 0 and `total`, `prob` is not strictly
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
    event, total, prob, _ = _grade_inputs(event, total, prob)
    if event.ndim != 1:
        raise ValueError(
            f"event, total and prob must be one-dimensional, got {event.ndim} dimensions"
        )
    invalid = (prob <= 0) | (prob >= 1)
    if invalid.any():
        raise ValueError(f"prob must be strictly between 0 and 1, got {_first(prob, invalid)}")
    dof = len(event) - ddof
    if dof < 1:
        raise ValueError(f"{len(event)} group(s) with ddof={ddof} leave {dof} degrees of freedom")
    expected = total * prob
    with np.errstate(over="ignore"):  # a vanishing expected count gives an infinite statistic
        statistic = float(np.sum((event - expected) ** 2 / (expected * (1 - prob))))
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
    """Return the row of `levels`, from `group_counts`, of each observation."""
    if isinstance(levels, pd.CategoricalIndex):
        # Map category codes to rows instead of looking up the labels: that hashes
        # interval categories, which warns on NumPy < 2.2. Code -1 is a missing value.
        row_of_code = np.zeros(len(levels.categories) + 1, dtype=np.intp)
        row_of_code[levels.codes] = np.arange(len(levels))
        return row_of_code[groups.cat.codes.to_numpy()]
    rows = levels.get_indexer(groups)
    missing = groups.isna().to_numpy()
    if missing.any():  # get_indexer does not match missing values on pandas 2.0
        rows[missing] = np.flatnonzero(pd.isna(levels))[0]
    return rows


def _grade_inputs(event, total, prob) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index | None]:
    """Broadcast and check the per-grade inputs of the calibration tests.

    Also returns the index shared by the Series inputs, or None without Series.
    """
    indices = [values.index for values in (event, total, prob) if isinstance(values, pd.Series)]
    index = indices[0] if indices else None
    if any(not other.equals(index) for other in indices[1:]):
        raise ValueError("event, total and prob must have identical indices when they are Series")
    try:
        arrays = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (event, total, prob)))
    except ValueError as error:
        raise ValueError(f"event, total and prob cannot be broadcast together: {error}") from None
    event, total, prob = arrays
    if index is not None and event.shape != (len(index),):
        raise ValueError(
            f"event, total and prob must broadcast to the length {len(index)} of their Series "
            f"index, got shape {event.shape}"
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
    if np.isnan(prob).any():
        raise ValueError("prob must not have missing values")
    _check_unit_interval(prob, "prob")
    return event, total, prob, index


def _pvalues(pvalue: np.ndarray, index: pd.Index | None) -> float | pd.Series | np.ndarray:
    if index is not None:
        return pd.Series(pvalue, index=index)
    return float(pvalue) if pvalue.ndim == 0 else pvalue


def _whole(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    return finite & (np.where(finite, values, 0) % 1 == 0)


def _first(values: np.ndarray, invalid: np.ndarray) -> str:
    return repr(float(values[invalid].flat[0]))
