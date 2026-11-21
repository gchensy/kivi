"""Logistic regression fitted by iteratively reweighted least squares.

`fit` takes a design matrix that already holds the intercept column and
returns the maximum likelihood estimates with their Wald statistics. The
public wrappers in `kivi.selection` validate the inputs and label the result.
"""

import math
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

# A fitted probability within 1e-10 of 0 or 1 means |eta| > 23.
_EXTREME_ETA = -math.log(1e-10)
# A step moves an observation against its class if its change of eta is below
# -_AGAINST times the largest change; smaller changes are rounding error.
_AGAINST = 1e-6
_MAX_HALVINGS = 30


class LogitFit(NamedTuple):
    """Estimates of a logistic regression, one entry per design column."""

    coef: npt.NDArray[np.float64]
    se: npt.NDArray[np.float64]
    z: npt.NDArray[np.float64]
    p_value: npt.NDArray[np.float64]
    cov: npt.NDArray[np.float64]
    log_likelihood: float
    n_iter: int
    converged: bool


class SeparationError(ValueError):
    """The classes are separated, so the logistic regression has no estimates.

    A combination of the variables predicts the target without error, for
    every observation (complete separation) or for every observation off a
    boundary (quasi-complete separation). The likelihood then keeps growing
    as the coefficients grow, so the maximum likelihood estimates do not
    exist.

    See Also
    --------
    logistic_regression : Raises this error.
    stepwise : Leaves out candidates that separate the classes.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.selection import SeparationError, logistic_regression
    >>> X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    >>> try:
    ...     logistic_regression(X, [0, 0, 1, 1])
    ... except SeparationError:
    ...     print("x separates the classes")
    x separates the classes
    """


# Defined here for `fit`; documented and exported by the public module.
SeparationError.__module__ = "kivi.selection"


class SingularError(ValueError):
    """The Fisher information is numerically singular at `column` of the design."""

    def __init__(self, column: int) -> None:
        super().__init__(f"design column {column} is nearly a linear combination of the others")
        self.column = column


def fit(
    design: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    *,
    weights: npt.NDArray[np.float64] | None = None,
    start: npt.NDArray[np.float64] | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> LogitFit:
    r"""Maximize the logistic log-likelihood by Newton-Raphson.

    For the logit link, Newton-Raphson is iteratively reweighted least
    squares: each step solves :math:`(X^\top W X)\,\delta = X^\top (y - p)`
    with :math:`W = \mathrm{diag}(w_i p_i (1 - p_i))`. The iterations run on
    centred and scaled columns, so an offset or a unit of measurement does
    not square into the conditioning of :math:`X^\top W X`; the estimates
    and their covariance are transformed back. A step that lowers the
    log-likelihood by more than rounding error is halved until it does not.
    The fit has converged when every scaled coefficient changes by at most
    `tol` times one plus its absolute value.

    Parameters
    ----------
    design : numpy.ndarray
        Two-dimensional float design matrix of full column rank, including
        the intercept column. See `dependent_column`.
    target : numpy.ndarray
        Labels 0 and 1, one per row of `design`.
    weights : numpy.ndarray, optional
        Non-negative frequency weights: a row with weight 2 counts as two
        identical observations. Default 1 for every row.
    start : numpy.ndarray, optional
        Starting coefficients. Default 0.
    max_iter : int, default 100
        Maximum number of Newton steps.
    tol : float, default 1e-8
        Largest change of a scaled coefficient at convergence, relative to
        one plus its absolute value.

    Returns
    -------
    LogitFit
        Coefficients, standard errors from the inverse Fisher information,
        Wald z statistics, two-sided p-values, the covariance matrix, the
        log-likelihood, the number of steps and whether the fit converged.
        A fit that stops after `max_iter` steps is returned with
        ``converged=False``; the caller decides whether to warn.

    Raises
    ------
    SeparationError
        If some fitted probabilities reach 0 or 1 while a Newton step moves
        every observation towards its own class, which proves that a
        combination of the columns separates the classes.
    SingularError
        If the Fisher information is numerically singular.
    """
    weights = np.ones(len(target)) if weights is None else weights
    used = weights > 0
    sign = np.where(target == 1, 1.0, -1.0)
    scaled, transform = _standardize(design)
    coef = (
        np.zeros(design.shape[1])
        if start is None
        else np.linalg.solve(transform, np.asarray(start, dtype=float))
    )
    eta = scaled @ coef
    event, non_event, log_likelihood = _evaluate(eta, target, weights)
    # The pairwise sum of n terms of the log-likelihood is exact to about log2(n) roundings.
    rounding = (math.log2(len(target)) + 1) * np.finfo(float).eps
    converged = False
    n_iter = 0
    while n_iter < max_iter and not converged:
        n_iter += 1
        gradient = scaled.T @ (weights * np.where(target == 1, non_event, -event))
        information = _information(scaled, weights, event, non_event)
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            raise SingularError(_weakest_column(scaled, weights, event, non_event)) from None
        if not np.isfinite(step).all():
            raise SingularError(_weakest_column(scaled, weights, event, non_event))
        converged = bool((np.abs(step) <= tol * (1 + np.abs(coef))).all())
        for _ in range(_MAX_HALVINGS):
            trial = coef + step
            trial_eta = scaled @ trial
            evaluated = _evaluate(trial_eta, target, weights)
            if converged or evaluated[2] >= log_likelihood - rounding * abs(log_likelihood):
                break
            step = step / 2
        else:
            break  # no ascent at working precision
        change = trial_eta - eta
        coef, eta = trial, trial_eta
        event, non_event, log_likelihood = evaluated
        if not converged and np.abs(eta[used]).max() > _EXTREME_ETA:
            towards = (sign * change)[used]
            if towards.max() > 0 and towards.min() >= -_AGAINST * np.abs(towards).max():
                raise SeparationError(
                    "the classes of target are separated by X: some fitted probabilities tend "
                    "to 0 or 1, so the maximum likelihood estimates do not exist; merge the "
                    "bins or drop the variables that predict target without error"
                )
    try:
        cov = np.linalg.inv(_information(scaled, weights, event, non_event))
    except np.linalg.LinAlgError:
        raise SingularError(_weakest_column(scaled, weights, event, non_event)) from None
    variance = np.diagonal(cov)
    if not (np.isfinite(variance).all() and (variance > 0).all()):
        raise SingularError(_weakest_column(scaled, weights, event, non_event))
    coef = transform @ coef
    cov = transform @ cov @ transform.T
    se = np.sqrt(np.diagonal(cov))
    z = coef / se
    p_value = np.array([math.erfc(abs(value) / math.sqrt(2)) for value in z])
    return LogitFit(coef, se, z, p_value, cov, log_likelihood, n_iter, converged)


def dependent_column(design: npt.NDArray[np.float64]) -> int | None:
    """Return the first column that is a linear combination of the columns before it.

    Parameters
    ----------
    design : numpy.ndarray
        Two-dimensional float matrix.

    Returns
    -------
    int or None
        Position of the first dependent column, or None if `design` has full
        column rank.
    """
    rows, columns = design.shape
    if not columns:
        return None
    diagonal = np.abs(np.diagonal(np.linalg.qr(design, mode="r")))
    norms = np.linalg.norm(design[:, : len(diagonal)], axis=0)
    # The diagonal of R is the part of each column orthogonal to the columns before it.
    dependent = np.flatnonzero(diagonal <= max(rows, columns) * np.finfo(float).eps * norms)
    if len(dependent):
        return int(dependent[0])
    return rows if rows < columns else None


def _standardize(
    design: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return centred, unit-variance columns and the matrix that maps their coefficients back.

    With ``scaled, transform = _standardize(design)``, ``scaled @ b`` equals
    ``design @ (transform @ b)``. Columns are centred only if `design` has a
    non-zero constant column, the intercept, which absorbs the means.
    """
    columns = design.shape[1]
    constant = design.min(axis=0) == design.max(axis=0)
    intercepts = np.flatnonzero(constant & (design[0] != 0))
    transform = np.eye(columns)
    scaled = np.empty_like(design)
    varying = np.flatnonzero(~constant)
    if len(intercepts):
        intercept = intercepts[0]
        value = design[0, intercept]
        mean = design[:, varying].mean(axis=0)
        centred = design[:, varying] - mean
        scale = np.sqrt((centred**2).mean(axis=0))
        scaled[:, varying] = centred / scale
        transform[intercept, varying] = -mean / (scale * value)
    else:
        intercept = None
        centred = design[:, varying]
        scale = np.sqrt((centred**2).mean(axis=0))
        scaled[:, varying] = centred / scale
    transform[varying, varying] = 1 / scale
    for column in np.flatnonzero(constant):
        value = design[0, column]
        divisor = value if value != 0 else 1.0
        scaled[:, column] = design[:, column] / divisor
        transform[column, column] = 1 / divisor
    return scaled, transform


def _weakest_column(
    scaled: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    event: npt.NDArray[np.float64],
    non_event: npt.NDArray[np.float64],
) -> int:
    """Return the column of the weighted design that the others explain best."""
    weighted = scaled * np.sqrt(weights * event * non_event)[:, None]
    diagonal = np.abs(np.diagonal(np.linalg.qr(weighted, mode="r")))
    norms = np.linalg.norm(weighted[:, : len(diagonal)], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        independent = np.where(norms > 0, diagonal / norms, 0.0)
    return int(np.argmin(independent))


def _evaluate(
    eta: npt.NDArray[np.float64], target: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Return P(y = 1), P(y = 0) and the log-likelihood at the linear predictor `eta`."""
    # exp(-|eta|) never overflows, and computing both probabilities from it keeps the
    # smaller one accurate instead of rounding it to 0 through 1 - p.
    tail = np.exp(-np.abs(eta))
    likelier = 1 / (1 + tail)
    rarer = tail * likelier
    above = eta >= 0
    event = np.where(above, likelier, rarer)
    non_event = np.where(above, rarer, likelier)
    # ln(1 + exp(eta)) = max(eta, 0) + ln(1 + exp(-|eta|)); np.sum adds pairwise.
    log_likelihood = float(np.sum(weights * (target * eta - np.maximum(eta, 0) - np.log1p(tail))))
    return event, non_event, log_likelihood


def _information(
    design: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    event: npt.NDArray[np.float64],
    non_event: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Fisher information, the negative Hessian of the log-likelihood."""
    return (design * (weights * event * non_event)[:, None]).T @ design
