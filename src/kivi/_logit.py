"""Logistic regression fitted by iteratively reweighted least squares.

`fit` takes a design matrix that already holds the intercept column and
returns the maximum likelihood estimates with their Wald statistics. The
public wrappers in `kivi.selection` validate the inputs and label the result.
"""

import math
import warnings
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

# A fit that does not converge and leaves a fitted probability this close to 0 or 1
# (|eta| > 23) is diverging because the classes are separated: each Newton step then
# pushes the separated observations further towards certainty.
_SEPARATION = 1e-10
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
    """The classes are separated, so the maximum likelihood estimates do not exist."""


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
    with :math:`W = \mathrm{diag}(w_i p_i (1 - p_i))`. A step that lowers the
    log-likelihood by more than rounding error is halved until it does not,
    so the iterations never move away from the maximum. The fit has
    converged when every coefficient changes by at most `tol` times one
    plus its absolute value.

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
        Largest coefficient change at convergence, relative to one plus the
        absolute value of the coefficient.

    Returns
    -------
    LogitFit
        Coefficients, standard errors from the inverse Fisher information,
        Wald z statistics, two-sided p-values, the covariance matrix, the
        log-likelihood, the number of steps and whether the fit converged.

    Raises
    ------
    SeparationError
        If the fit does not converge because some fitted probabilities tend
        to 0 or 1, which happens when a combination of the columns separates
        the classes completely or quasi-completely.

    Warns
    -----
    RuntimeWarning
        If the fit does not converge in `max_iter` steps for another reason.
    """
    weights = np.ones(len(target)) if weights is None else weights
    is_event = target == 1
    coef = np.zeros(design.shape[1]) if start is None else np.array(start, dtype=float)
    eta = design @ coef
    event, non_event, log_likelihood = _evaluate(eta, target, weights)
    converged = False
    n_iter = 0
    while n_iter < max_iter and not converged:
        n_iter += 1
        gradient = design.T @ (weights * np.where(is_event, non_event, -event))
        try:
            step = np.linalg.solve(_information(design, weights, event, non_event), gradient)
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(step).all():
            break
        converged = bool((np.abs(step) <= tol * (1 + np.abs(coef))).all())
        # Near the maximum, the change in the log-likelihood is lost in its rounding error.
        rounding = len(target) * np.finfo(float).eps * abs(log_likelihood)
        for _ in range(_MAX_HALVINGS):
            trial = coef + step
            trial_eta = design @ trial
            evaluated = _evaluate(trial_eta, target, weights)
            if converged or evaluated[2] >= log_likelihood - rounding:
                break
            step = step / 2
        else:
            break  # no ascent at working precision
        coef, eta = trial, trial_eta
        event, non_event, log_likelihood = evaluated
    if not converged:
        if np.abs(eta[weights > 0]).max() > -math.log(_SEPARATION):
            raise SeparationError(
                "the classes of target are separated by X: some fitted probabilities tend "
                "to 0 or 1, so the maximum likelihood estimates do not exist; merge the "
                "bins or drop the variables that predict target without error"
            )
        warnings.warn(
            f"logistic regression did not converge after {n_iter} iterations",
            RuntimeWarning,
            stacklevel=2,
        )
    cov = np.linalg.inv(_information(design, weights, event, non_event))
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
    # ln(1 + exp(eta)) = max(eta, 0) + ln(1 + exp(-|eta|))
    log_likelihood = float(weights @ (target * eta - np.maximum(eta, 0) - np.log1p(tail)))
    return event, non_event, log_likelihood


def _information(
    design: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    event: npt.NDArray[np.float64],
    non_event: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Fisher information, the negative Hessian of the log-likelihood."""
    return (design * (weights * event * non_event)[:, None]).T @ design
