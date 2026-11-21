"""Variable selection for scorecards built on WOE-encoded variables."""

import warnings
from collections.abc import Hashable
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import _logit
from ._logit import SeparationError
from ._validation import as_series, as_weights, is_real

__all__ = ["SeparationError", "correlation_filter", "logistic_regression", "stepwise", "vif"]


def correlation_filter(
    X: pd.DataFrame | npt.ArrayLike,
    *,
    threshold: float = 0.7,
    priority: npt.ArrayLike | None = None,
) -> pd.Index:
    """Drop one variable of each strongly correlated pair.

    Variables are visited from the highest `priority` to the lowest. A
    variable is kept unless its absolute Pearson correlation with a variable
    that is already kept exceeds `threshold`.

    Parameters
    ----------
    X : pandas.DataFrame or array_like
        Two-dimensional numeric values with one column per variable, such as
        WOE values from `kivi.woe.woe_transform`.
    threshold : float, default 0.7
        Largest absolute correlation allowed between two kept variables,
        between 0 and 1.
    priority : array_like, optional
        One number per column, such as its information value; of two
        correlated variables, the one with the higher priority is kept. A
        pandas Series is matched to the columns by label, other inputs by
        position. Ties, and every column when `priority` is omitted, follow
        the column order, so earlier columns are kept.

    Returns
    -------
    pandas.Index
        Labels of the kept columns, in the order of `X`. Column positions
        when `X` is not a DataFrame.

    Raises
    ------
    ValueError
        If `X` is not two-dimensional, has fewer than 2 rows or no columns,
        repeats a column label, or has a column that is not numeric, contains
        missing or infinite values or is constant; if `threshold` is not between 0 and
        1; or if `priority` does not give one number for every column.

    See Also
    --------
    vif : Measure how well the other variables explain each variable.
    stepwise : Select variables by their significance in a logistic regression.

    Notes
    -----
    A variable is dropped only because of a variable that is kept. When
    ``a`` is correlated with ``b`` and ``b`` with ``c``, but ``a`` not with
    ``c``, and the priorities decrease from ``a`` to ``c``, then ``b`` is
    dropped and ``c`` is kept.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.selection import correlation_filter
    >>> X = pd.DataFrame({
    ...     "a": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     "b": [1.0, 2.0, 3.0, 5.0, 4.0],
    ...     "c": [2.0, 1.0, 4.0, 3.0, 1.0],
    ... })
    >>> X.corr().round(2)
         a     b     c
    a  1.0  0.90  0.00
    b  0.9  1.00  0.24
    c  0.0  0.24  1.00
    >>> correlation_filter(X).tolist()
    ['a', 'c']
    >>> iv = pd.Series({"a": 0.12, "b": 0.31, "c": 0.05})
    >>> correlation_filter(X, priority=iv).tolist()
    ['b', 'c']
    """
    if isinstance(threshold, bool) or not is_real(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold!r}")
    frame, values = _as_frame(X)
    _check_not_constant(frame, values, "its correlation is undefined")
    order = np.argsort(-_priority(priority, frame.columns), kind="stable")
    correlation = np.abs(np.corrcoef(values, rowvar=False).reshape(len(order), len(order)))
    kept: list[int] = []
    for column in order:
        if not (correlation[column, kept] > threshold).any():
            kept.append(column)
    return frame.columns[np.sort(kept)]


def vif(X: pd.DataFrame | npt.ArrayLike) -> pd.Series:
    r"""Compute the variance inflation factor of each variable.

    The VIF of a variable measures how well the other variables explain it
    in a linear regression. It is 1 for a variable that is uncorrelated with
    the others and grows without bound as the variable approaches a linear
    combination of them.

    Parameters
    ----------
    X : pandas.DataFrame or array_like
        Two-dimensional numeric values with one column per variable, such as
        WOE values from `kivi.woe.woe_transform`.

    Returns
    -------
    pandas.Series
        VIF of each column, indexed by the column labels of `X` (positions
        when `X` is not a DataFrame) and named ``vif``. It is infinite for a
        column that is a linear combination of the others.

    Raises
    ------
    ValueError
        If `X` is not two-dimensional, has fewer than 2 rows or no columns,
        repeats a column label, or has a column that is not numeric, contains
        missing or infinite values or is constant.

    See Also
    --------
    correlation_filter : Drop one variable of each strongly correlated pair.
    logistic_regression : Fit the scorecard model.

    Notes
    -----
    For column :math:`j`, let :math:`R_j^2` be the coefficient of
    determination of the least-squares regression of column :math:`j` on an
    intercept and the other columns. Then

    .. math::

       \mathrm{VIF}_j = \frac{1}{1 - R_j^2}.

    The square root of the VIF is the factor by which collinearity inflates
    the standard error of the variable's coefficient. With two columns of
    correlation :math:`r`, both have a VIF of :math:`1 / (1 - r^2)`.

    A column counts as a linear combination of the others, with an
    infinite VIF, when it is one up to rounding: when the singular value
    decomposition of the standardized columns gives it a component in
    their null space, or when its VIF reaches
    :math:`1 / \epsilon \approx 4.5 \cdot 10^{15}`, so that :math:`1 - R_j^2`
    is at most the machine epsilon :math:`\epsilon`, the spacing of
    double-precision numbers at 1. Two rows, for example, make any two
    columns collinear.

    Examples
    --------
    Two columns with a correlation of 0.9 both have a VIF of
    :math:`1 / (1 - 0.9^2) = 5.263`:

    >>> import pandas as pd
    >>> from kivi.selection import vif
    >>> X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [1.0, 2.0, 3.0, 5.0, 4.0]})
    >>> print(f"{X['a'].corr(X['b']):.2f}")
    0.90
    >>> vif(X).round(3)
    a    5.263
    b    5.263
    Name: vif, dtype: float64

    Adding their sum makes every column a linear combination of the others:

    >>> vif(X.assign(total=X["a"] + X["b"]))
    a        inf
    b        inf
    total    inf
    Name: vif, dtype: float64
    """
    frame, values = _as_frame(X)
    _check_not_constant(frame, values, "its VIF is undefined")
    centered = values - values.mean(axis=0)
    # With unit-length centred columns Z, Z'Z is the correlation matrix and its inverse,
    # whose diagonal holds the VIFs, is V diag(1 / s^2) V' from the SVD Z = U diag(s) V'.
    standardized = centered / np.linalg.norm(centered, axis=0)
    rows, columns = standardized.shape
    # With fewer rows than columns, only the full V has a row for every direction.
    singular, vt = np.linalg.svd(standardized, full_matrices=rows < columns)[1:]
    eps = np.finfo(float).eps
    rank = int((singular > max(rows, columns) * eps * singular[0]).sum())
    result = ((vt[:rank] / singular[:rank, None]) ** 2).sum(axis=0)
    # A column with a component in the null space of Z, the rows of V beyond the rank, is a
    # combination of the others, and so is a column whose 1 - R^2 = 1 / VIF is at most eps.
    null = (vt[rank:] ** 2).sum(axis=0)
    result[(null > columns * eps) | (result >= 1 / eps)] = np.inf
    return pd.Series(result, index=frame.columns, name="vif")


def logistic_regression(
    X: pd.DataFrame | npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Fit a logistic regression and test each coefficient.

    The model has an intercept and one coefficient per column of `X`. The
    coefficients are maximum likelihood estimates, and each is tested
    against zero with a Wald test.

    Parameters
    ----------
    X : pandas.DataFrame or array_like
        Two-dimensional numeric values with one column per variable, such as
        WOE values from `kivi.woe.woe_transform`.
    target : array_like
        Binary labels, with 1 indicating the event. If `X` is a DataFrame and
        `target` a Series, they must have identical indices; otherwise they
        are matched by position.
    weights : array_like, optional
        Non-negative frequency weight of each row, matched like `target`: a
        row with weight 2 counts as two identical observations, so integer
        weights give the same estimates, standard errors and log-likelihood
        as repeating rows. Rows with zero weight are left out of the fit.
        See the Notes on weights that do not count observations.

    Returns
    -------
    pandas.DataFrame
        One row for the intercept, labelled ``intercept``, and one for each
        column of `X`, with the coefficient, its standard error, the Wald z
        statistic and its two-sided p-value. The index has the name of the
        columns of `X`. ``attrs["log_likelihood"]`` holds the maximized
        log-likelihood and ``attrs["converged"]`` whether the fit converged.

    Raises
    ------
    SeparationError
        If the columns separate the classes, so that the estimates do not
        exist.
    ValueError
        If `X` is not two-dimensional, has fewer than 2 rows or no columns,
        repeats a column label, has a column labelled ``intercept``, or has a
        column that is not numeric, contains missing or infinite values, is
        constant or is a linear combination of the intercept and the columns
        before it, or so close to one that its coefficient cannot be
        estimated; if `target` is not one-dimensional, its index or length
        does not match `X`, or it does not contain both 0 and 1 or has
        missing or other labels; or if `weights` is not numeric, does not
        give each row one finite, non-negative weight, or leaves a class
        without weight. With `weights`, only the checks for constant columns
        and linear combinations are limited to the rows with positive
        weight.

    Warns
    -----
    RuntimeWarning
        If the fit does not converge for a reason other than separation.

    See Also
    --------
    stepwise : Select the columns of the model by their p-values.
    vif : Check the columns for collinearity.
    kivi.woe.woe_transform : Encode groups with their WOE.

    Notes
    -----
    The model is

    .. math::

       \ln \frac{P(y = 1)}{P(y = 0)} = \beta_0 + \sum_j \beta_j x_j.

    The estimates are found by Newton-Raphson, also known as iteratively
    reweighted least squares, on centred and scaled columns, so that an
    offset or the unit of a column does not affect their accuracy. The
    standard errors are the square roots of the diagonal of the inverse
    Fisher information, :math:`z = \hat\beta / \mathrm{se}`, and the p-value
    is :math:`P(|Z| > |z|)` for a standard normal :math:`Z`.

    With WOE-encoded variables, the coefficients are expected to be
    positive: see `stepwise`.

    With `weights` :math:`w_i`, the estimates maximize the weighted
    log-likelihood
    :math:`\sum_i w_i [y_i \ln p_i + (1 - y_i) \ln(1 - p_i)]`, and the
    standard errors come from the weighted Fisher information
    :math:`\sum_i w_i p_i (1 - p_i) x_i x_i^\top`. These are frequency
    weights, as in statsmodels' ``GLM(freq_weights=...)``: the total weight
    counts as the number of independent observations, so multiplying every
    weight by :math:`c` divides the variances by :math:`c`. Integer weights
    that count repeated rows therefore give the standard errors of the
    repeated rows. Weights that do not count observations do not: those of
    `kivi.reject_inference.fuzzy_augmentation`, which split each rejected
    applicant into a bad and a good row, and those of
    `kivi.reject_inference.reweighting`, which let each accepted applicant
    stand for several, make the standard errors too small and the p-values
    too low. The inferred rows and the extra weight count as observations,
    although they add no observed outcomes.

    References
    ----------
    .. [1] D. W. Hosmer, S. Lemeshow and R. X. Sturdivant, "Applied Logistic
       Regression", 3rd ed., Wiley, 2013.

    Examples
    --------
    With a single binary variable, the coefficient is the log odds ratio:
    the odds of the event are 1:3 at ``x = 0`` and 3:1 at ``x = 1``, so the
    coefficient is :math:`\ln 9` and the intercept :math:`\ln(1/3)`.

    >>> import pandas as pd
    >>> from kivi.selection import logistic_regression
    >>> X = pd.DataFrame({"x": [0, 0, 0, 0, 1, 1, 1, 1]})
    >>> target = pd.Series([0, 0, 0, 1, 0, 1, 1, 1])
    >>> logistic_regression(X, target).round(4)
                 coef      se       z  p_value
    intercept -1.0986  1.1547 -0.9514   0.3414
    x          2.1972  1.6330  1.3455   0.1785

    Integer weights count repeated rows, so the four distinct rows with
    their counts give the same fit:

    >>> distinct = pd.DataFrame({"x": [0, 0, 1, 1]})
    >>> logistic_regression(distinct, [0, 1, 0, 1], weights=[3, 1, 1, 3]).round(4)
                 coef      se       z  p_value
    intercept -1.0986  1.1547 -0.9514   0.3414
    x          2.1972  1.6330  1.3455   0.1785
    """
    frame, values, labels, mass = _model_inputs(X, target, weights)
    _check_independent(frame, values, weighted=mass is not None)
    result = _fit(_with_intercept(values), labels, frame.columns, mass)
    if not result.converged:
        _warn_not_converged(result, stacklevel=2)
    table = pd.DataFrame(
        {"coef": result.coef, "se": result.se, "z": result.z, "p_value": result.p_value},
        index=pd.Index(["intercept"], name=frame.columns.name).append(frame.columns),
    )
    table.attrs = {"log_likelihood": result.log_likelihood, "converged": result.converged}
    return table


def stepwise(
    X: pd.DataFrame | npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    p_enter: float = 0.05,
    p_remove: float = 0.10,
    direction: Literal["forward", "backward", "both"] = "both",
    positive: bool = True,
    weights: npt.ArrayLike | None = None,
) -> pd.Index:
    """Select variables for a logistic regression by their p-values.

    Forward selection starts from the intercept and adds, one at a time, the
    variable with the smallest p-value while that p-value is below
    `p_enter`. Backward elimination starts from all variables and removes,
    one at a time, the variable with the largest p-value while that p-value
    is above `p_remove`. Selection in both directions adds variables like
    forward selection, but after each addition removes the variables that
    no longer meet `p_remove`.

    Parameters
    ----------
    X : pandas.DataFrame or array_like
        Two-dimensional numeric values with one column per candidate
        variable, such as WOE values from `kivi.woe.woe_transform`.
    target : array_like
        Binary labels, with 1 indicating the event. If `X` is a DataFrame and
        `target` a Series, they must have identical indices; otherwise they
        are matched by position.
    p_enter : float, default 0.05
        A variable enters only if its p-value in the extended model is below
        this level, which is greater than 0 and at most 1.
    p_remove : float, default 0.10
        A variable is removed if its p-value is above this level, which is
        greater than 0 and at most 1 and, in both directions, not below
        `p_enter`.
    direction : {"both", "forward", "backward"}, default "both"
        Whether variables are added and removed, only added, or only removed.
    positive : bool, default True
        Keep every coefficient of the selected model positive. A variable
        enters only if all coefficients of the extended model are positive,
        and a variable whose coefficient is not positive is removed before
        any other, starting with the smallest z statistic.
    weights : array_like, optional
        Non-negative frequency weight of each row, as in
        `logistic_regression`. Integer weights select the same variables as
        repeating rows. The p-values depend on the scale of the weights as
        they do on the sample size. The weights of reject inference make
        them too low, so that more variables enter than the observed
        outcomes support; select the variables on the accepted applicants
        instead.

    Returns
    -------
    pandas.Index
        Labels of the selected columns, in the order of `X`. Column positions
        when `X` is not a DataFrame.

    Raises
    ------
    SeparationError
        If `direction` is ``"backward"`` and the model with every column
        separates the classes.
    ValueError
        If `X`, `target` or `weights` is invalid as described for
        `logistic_regression`, in every direction; if `p_enter` or
        `p_remove` is not greater than 0 and at most 1, or `p_enter` exceeds
        `p_remove` when `direction` is ``"both"``; or if `direction` is not
        one of the three options or `positive` is not a boolean.

    Warns
    -----
    RuntimeWarning
        If the model of the selected variables does not converge for a
        reason other than separation.

    See Also
    --------
    logistic_regression : Fit the selected model and read its coefficients.
    correlation_filter : Drop strongly correlated variables first.

    Notes
    -----
    The p-values are those of the Wald tests reported by
    `logistic_regression`. Each candidate is tested in the model made of
    the selected variables and the candidate, so every step refits one
    model per candidate. A candidate does not enter if that model would
    separate the classes, does not converge or is numerically singular. In
    both directions, a variable cannot enter if that would recreate a set of
    variables that was selected before, which guarantees that the selection
    stops.

    kivi defines the WOE of a group as the log of its share of events over
    its share of non-events, so a higher WOE means a higher event rate. In a
    logistic regression of `target` on a single WOE-encoded variable, the
    coefficient is therefore positive; it is exactly 1 when no cell needed
    smoothing. A negative coefficient in a model with several variables
    means that the variable's effect is reversed by the others, usually
    because of collinearity, and makes the scorecard points of its bins
    contradict their event rates. ``positive=True`` rules out such models.

    References
    ----------
    .. [1] D. W. Hosmer, S. Lemeshow and R. X. Sturdivant, "Applied Logistic
       Regression", 3rd ed., Wiley, 2013.
    .. [2] N. Siddiqi, "Credit Risk Scorecards: Developing and Implementing
       Intelligent Credit Scoring", Wiley, 2006.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.datasets import Dataset
    >>> from kivi.selection import stepwise
    >>> from kivi.woe import woe_table, woe_transform
    >>> data = Dataset.bank_data()
    >>> columns = ["poutcome", "contact", "housing", "loan", "marital", "default"]
    >>> X = pd.DataFrame({
    ...     column: woe_transform(data[column], woe_table(data[column], data["target"]))
    ...     for column in columns
    ... })
    >>> stepwise(X, data["target"]).tolist()
    ['poutcome', 'contact', 'housing', 'loan', 'marital']
    """
    _check_level(p_enter, "p_enter")
    _check_level(p_remove, "p_remove")
    if direction not in ("forward", "backward", "both"):
        raise ValueError(f"direction must be 'forward', 'backward' or 'both', got {direction!r}")
    if direction == "both" and p_enter > p_remove:
        raise ValueError(
            f"p_enter must not exceed p_remove, got p_enter={p_enter!r} and p_remove={p_remove!r}"
        )
    if not isinstance(positive, (bool, np.bool_)):
        raise ValueError(f"positive must be True or False, got {positive!r}")
    frame, values, labels, mass = _model_inputs(X, target, weights)
    # Every subset of independent columns is independent, so candidate models have full rank.
    _check_independent(frame, values, weighted=mass is not None)
    design = _with_intercept(values)
    selected = list(range(values.shape[1])) if direction == "backward" else []
    visited = {frozenset(selected)}
    fitted = None
    while True:
        if direction != "forward":
            fitted = _remove(
                design, labels, mass, frame.columns, selected, visited, p_remove, positive, fitted
            )
        if direction == "backward":
            break
        entering = _enter(design, labels, mass, selected, visited, p_enter, positive, fitted)
        if entering is None:
            break
        fitted = entering[1]
        selected.append(entering[0])
        visited.add(frozenset(selected))
    return frame.columns[np.sort(np.array(selected, dtype=np.intp))]


def _remove(
    design: npt.NDArray[np.float64],
    labels: npt.NDArray[np.float64],
    mass: npt.NDArray[np.float64] | None,
    columns: pd.Index,
    selected: list[int],
    visited: set[frozenset[int]],
    p_remove: float,
    positive: bool,
    fitted: _logit.LogitFit | None,
) -> _logit.LogitFit:
    """Remove failing variables from `selected` in place and return the fit of the rest.

    `fitted` is the converged fit of `selected` when the last step added a variable.
    """
    start = None
    while True:
        if fitted is None:
            fitted = _fit(
                design[:, _model_columns(selected)], labels, columns[selected], mass, start
            )
            if not fitted.converged:
                _warn_not_converged(fitted, stacklevel=3)
        coef, z, p_value = fitted.coef[1:], fitted.z[1:], fitted.p_value[1:]
        failing = p_value > p_remove
        if positive:
            failing |= coef <= 0
        if not failing.any():
            return fitted
        # The signed z puts negative coefficients first; |z| orders by p-value without underflow.
        worst = int(np.argmin(np.where(failing, z if positive else np.abs(z), np.inf)))
        # Start the smaller model from the same mean linear predictor.
        start = np.delete(fitted.coef, worst + 1)
        start[0] += fitted.coef[worst + 1] * np.average(
            design[:, selected[worst] + 1], weights=mass
        )
        del selected[worst]
        visited.add(frozenset(selected))
        fitted = None


def _enter(
    design: npt.NDArray[np.float64],
    labels: npt.NDArray[np.float64],
    mass: npt.NDArray[np.float64] | None,
    selected: list[int],
    visited: set[frozenset[int]],
    p_enter: float,
    positive: bool,
    fitted: _logit.LogitFit | None,
) -> tuple[int, _logit.LogitFit] | None:
    """Return the best candidate to add and its fit, or None."""
    start = None if fitted is None else np.r_[fitted.coef, 0.0]
    best: tuple[int, _logit.LogitFit] | None
    best, best_z = None, -np.inf
    for candidate in range(design.shape[1] - 1):
        if candidate in selected or frozenset([*selected, candidate]) in visited:
            continue
        try:
            trial = _warm_fit(
                design[:, _model_columns([*selected, candidate])], labels, mass, start
            )
        except (_logit.SeparationError, _logit.SingularError):
            continue
        if not trial.converged or trial.p_value[-1] >= p_enter:
            continue
        if positive and (trial.coef[1:] <= 0).any():
            continue
        if abs(trial.z[-1]) > best_z:
            best, best_z = (candidate, trial), abs(trial.z[-1])
    return best


def _warm_fit(
    design: npt.NDArray[np.float64],
    labels: npt.NDArray[np.float64],
    mass: npt.NDArray[np.float64] | None,
    start: npt.NDArray[np.float64] | None,
) -> _logit.LogitFit:
    """Fit from `start`, and from zero if the information becomes singular on the way."""
    try:
        return _logit.fit(design, labels, weights=mass, start=start)
    except _logit.SingularError:
        if start is None:
            raise
        return _logit.fit(design, labels, weights=mass)


def _fit(
    design: npt.NDArray[np.float64],
    labels: npt.NDArray[np.float64],
    columns: pd.Index,
    mass: npt.NDArray[np.float64] | None,
    start: npt.NDArray[np.float64] | None = None,
) -> _logit.LogitFit:
    """Fit a model on the intercept and `columns`, naming the column of a singular fit."""
    try:
        return _warm_fit(design, labels, mass, start)
    except _logit.SingularError as error:
        label = "intercept" if error.column == 0 else columns[error.column - 1]
        raise ValueError(
            f"X column {label!r} is so close to a linear combination of the intercept and "
            "the other columns that its coefficient cannot be estimated"
        ) from None


def _warn_not_converged(fitted: _logit.LogitFit, stacklevel: int) -> None:
    warnings.warn(
        f"logistic regression did not converge after {fitted.n_iter} iterations",
        RuntimeWarning,
        stacklevel=stacklevel + 1,
    )


def _model_columns(selected: list[int]) -> list[int]:
    return [0, *(column + 1 for column in selected)]


def _with_intercept(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.column_stack([np.ones(len(values)), values])


def _check_level(level: float, name: str) -> None:
    if isinstance(level, bool) or not is_real(level) or not 0 < level <= 1:
        raise ValueError(f"{name} must be greater than 0 and at most 1, got {level!r}")


def _as_frame(X: pd.DataFrame | npt.ArrayLike) -> tuple[pd.DataFrame, npt.NDArray[np.float64]]:
    """Return `X` as a DataFrame, without copying one, and its float values."""
    if isinstance(X, pd.DataFrame):
        frame = X
    else:
        if np.ndim(X) != 2:
            raise ValueError(f"X must be two-dimensional, got {np.ndim(X)} dimensions")
        frame = pd.DataFrame(X)  # type: ignore[arg-type]  # a two-dimensional array_like
    if len(frame) < 2:
        raise ValueError(f"X must have at least 2 rows, got {len(frame)}")
    if not frame.shape[1]:
        raise ValueError("X must have at least one column")
    if frame.columns.has_duplicates:
        label: Hashable = frame.columns[frame.columns.duplicated()][0]
        raise ValueError(f"X must have unique column labels, got {label!r} more than once")
    for label, dtype in frame.dtypes.items():
        if not pd.api.types.is_numeric_dtype(dtype):
            raise ValueError(
                f"X column {label!r} must be numeric, such as WOE values from woe_transform, "
                f"got dtype {dtype}"
            )
    values = frame.to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(values).all(axis=0)
    if not finite.all():
        label = frame.columns[np.argmin(finite)]  # type: ignore[call-overload]  # a NumPy integer selects one label
        raise ValueError(f"X column {label!r} must not contain missing or infinite values")
    return frame, values


def _model_inputs(
    X: pd.DataFrame | npt.ArrayLike,
    target: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
) -> tuple[
    pd.DataFrame, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64] | None
]:
    """Validate a model's inputs like `woe_table`.

    Return the frame, and the values, labels and weights of the rows with
    positive weight; the weights are None without `weights`.
    """
    for name, values in (("target", target), ("weights", weights)):
        if isinstance(X, pd.DataFrame) and isinstance(values, pd.Series):
            if not X.index.equals(values.index):
                raise ValueError(f"X and {name} must have identical indices")
    frame, values = _as_frame(X)
    if "intercept" in frame.columns:
        raise ValueError("X must not have a column labelled 'intercept'")
    raw_target = target
    target = as_series(target, "target")
    if len(target) != len(frame):
        raise ValueError(
            f"X and target must have the same number of rows, got {len(frame)} and {len(target)}"
        )
    if not target.isin([0, 1]).all() or target.nunique() != 2:
        raise ValueError("target must contain both 0 and 1 and no missing labels")
    labels = target.to_numpy(dtype=float)
    if weights is None:
        return frame, values, labels, None
    if np.ndim(weights) == 1 and len(weights) != len(frame):  # type: ignore[arg-type]  # one-dimensional
        raise ValueError(
            f"weights must have one weight per row of X, got {len(weights)} for {len(frame)}"  # type: ignore[arg-type]  # one-dimensional
        )
    mass = cast(
        npt.NDArray[np.float64], as_weights(weights, frame, of="X", others={"target": raw_target})
    )
    totals = np.bincount(labels.astype(np.intp), weights=mass, minlength=2)
    if not (totals > 0).all():
        raise ValueError(
            "weights must give both target classes a positive total, got "
            f"{float(totals[0])!r} for 0 and {float(totals[1])!r} for 1"
        )
    # Rows without weight must not enter the centring and scaling of the fit.
    used = mass > 0
    return frame, values[used], labels[used], mass[used]


def _priority(priority: npt.ArrayLike | None, columns: pd.Index) -> npt.NDArray[np.float64]:
    if priority is None:
        return np.zeros(len(columns))
    if isinstance(priority, pd.Series):
        if priority.index.has_duplicates:
            raise ValueError("priority must have one value per column label")
        absent = columns[~columns.isin(priority.index)]
        if len(absent):
            raise ValueError(f"priority has no value for X column {absent[0]!r}")
        priority = priority.reindex(columns)
    priority = as_series(priority, "priority")
    if len(priority) != len(columns):
        raise ValueError(
            f"priority must have one value per column of X, got {len(priority)} values "
            f"for {len(columns)} columns"
        )
    if not pd.api.types.is_numeric_dtype(priority) or priority.isna().any():
        raise ValueError("priority must contain numbers only, without missing values")
    return priority.to_numpy(dtype=float)


def _check_not_constant(
    frame: pd.DataFrame, values: npt.NDArray[np.float64], reason: str, where: str = ""
) -> None:
    constant = np.ptp(values, axis=0) == 0
    if constant.any():
        raise ValueError(
            f"X column {frame.columns[np.argmax(constant)]!r} is constant{where}; {reason}"  # type: ignore[call-overload]  # a NumPy integer selects one label
        )


def _check_independent(
    frame: pd.DataFrame, values: npt.NDArray[np.float64], *, weighted: bool
) -> None:
    where = " on the rows with positive weight" if weighted else ""
    _check_not_constant(frame, values, "it cannot be told apart from the intercept", where)
    dependent = _logit.dependent_column(_with_intercept(values))
    if dependent is not None:
        raise ValueError(
            f"X column {frame.columns[dependent - 1]!r} is a linear combination of the "
            f"intercept and the columns before it{where}"
        )
