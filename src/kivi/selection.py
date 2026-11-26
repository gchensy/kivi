"""Variable selection for scorecards built on WOE-encoded variables."""

import typing
from collections import abc

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import _logit, _validation, metrics
from ._logit import SeparationError

__all__ = [
    "SeparationError",
    "correlation_filter",
    "logistic_regression",
    "selection_path",
    "stepwise",
    "vif",
    "vif_filter",
]

# VIFs within this relative difference of the largest tie, as other statistics in kivi do.
_TIE = 1e-10
# The values that the message for a column that is not numeric gives as an example.
_WOE = "WOE values from woe_transform"


def correlation_filter(
    X: pd.DataFrame | npt.ArrayLike,
    *,
    method: typing.Literal["pearson", "spearman"] = "pearson",
    threshold: float = 0.7,
    priority: npt.ArrayLike | None = None,
) -> pd.Index:
    """Drop one variable of each strongly correlated pair.

    Variables are visited from the highest `priority` to the lowest. A
    variable is kept unless its absolute correlation with a variable that is
    already kept exceeds `threshold`.

    Parameters
    ----------
    X : pandas.DataFrame or array_like
        Two-dimensional numeric values with one column per variable, such as
        WOE values from `kivi.woe.woe_transform`.
    method : {"pearson", "spearman"}, default "pearson"
        The correlation: Pearson's linear correlation, or Spearman's rank
        correlation, the Pearson correlation of the ranks of the values with
        ties given their average rank, as ``pandas.DataFrame.corr``
        computes it.
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
        missing or infinite values or is constant; if `method` is not
        ``"pearson"`` or ``"spearman"``; if `threshold` is not between 0 and
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

    Pearson's correlation measures how close two variables are to a straight
    line. It suits WOE values, which the logistic regression combines
    linearly. Spearman's correlation measures how close they are to a
    monotone relation and is not moved by a few extreme values, which suits
    raw variables before binning, such as amounts with a long right tail.

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

    A relation that is strictly monotone but not linear is perfect by rank:

    >>> growth = pd.DataFrame({
    ...     "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ...     "y": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
    ... })
    >>> growth.corr().round(2)
          x     y
    x  1.00  0.85
    y  0.85  1.00
    >>> growth.corr(method="spearman").round(2)
         x    y
    x  1.0  1.0
    y  1.0  1.0
    >>> correlation_filter(growth, threshold=0.9).tolist()
    ['x', 'y']
    >>> correlation_filter(growth, method="spearman", threshold=0.9).tolist()
    ['x']
    """
    if not isinstance(method, str) or method not in ("pearson", "spearman"):  # type: ignore[redundant-expr]  # rejects non-str values
        raise ValueError(
            f"method must be 'pearson' or 'spearman', got {_validation.shown_argument(method)}"
        )
    if isinstance(threshold, bool) or not _validation.is_real(threshold) or not 0 <= threshold <= 1:
        raise ValueError(
            f"threshold must be between 0 and 1, got {_validation.shown_argument(threshold)}"
        )
    frame, values = _logit.as_frame(X, example=_WOE)
    _logit.check_not_constant(frame.columns, values, "its correlation is undefined")
    order = np.argsort(-_priority(priority, frame.columns), kind="stable")
    if method == "pearson":
        matrix = np.corrcoef(values, rowvar=False)
    else:
        matrix = pd.DataFrame(values).corr(method="spearman").to_numpy()
    correlation = np.abs(matrix.reshape(len(order), len(order)))
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
        when `X` is not a DataFrame) in an index named ``variable``, and
        named ``vif``. It is infinite for a column that is a linear
        combination of the others.

    Raises
    ------
    ValueError
        If `X` is not two-dimensional, has fewer than 2 rows or no columns,
        repeats a column label, or has a column that is not numeric, contains
        missing or infinite values or is constant.

    See Also
    --------
    vif_filter : Drop the variable with the largest VIF until no VIF exceeds a threshold.
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
    variable
    a    5.263
    b    5.263
    Name: vif, dtype: float64

    Adding their sum makes every column a linear combination of the others:

    >>> vif(X.assign(total=X["a"] + X["b"]))
    variable
    a        inf
    b        inf
    total    inf
    Name: vif, dtype: float64
    """
    frame, values = _logit.as_frame(X, example=_WOE)
    _logit.check_not_constant(frame.columns, values, "its VIF is undefined")
    return pd.Series(_vif(values), index=frame.columns.rename("variable"), name="vif")


def vif_filter(
    X: pd.DataFrame | npt.ArrayLike,
    *,
    threshold: float = 5.0,
    priority: npt.ArrayLike | None = None,
) -> pd.Index:
    """Drop the variable with the largest VIF until no VIF exceeds a threshold.

    Each round computes the `vif` of the variables that are still kept. If
    the largest is at most `threshold`, they are all kept and the
    elimination stops; otherwise the variable with the largest VIF is
    dropped and the next round starts. A single variable, whose VIF is 1, is
    always kept.

    Parameters
    ----------
    X : pandas.DataFrame or array_like
        Two-dimensional numeric values with one column per variable, such as
        WOE values from `kivi.woe.woe_transform`.
    threshold : float, default 5.0
        Largest VIF allowed for a kept variable, a finite number greater
        than 1.
    priority : array_like, optional
        One number per column, such as its information value, matched to the
        columns as in `correlation_filter`: a pandas Series by label, other
        inputs by position. Of several variables whose VIFs tie for the
        largest, such as variables with infinite VIFs, the one with the
        lowest priority is dropped. Ties of priority, and every tie when
        `priority` is omitted, drop the last of the tied variables in the
        column order, so that earlier columns are kept, as in
        `correlation_filter`.

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
        missing or infinite values or is constant; if `threshold` is not a
        finite number greater than 1; or if `priority` does not give one
        number for every column.

    See Also
    --------
    vif : Compute the VIFs of one set of variables.
    correlation_filter : Drop one variable of each strongly correlated pair.

    Notes
    -----
    Dropping a variable lowers the VIFs of the variables that it helped to
    explain, so a variable whose VIF exceeds `threshold` in the first round
    may be kept in the end. Each round therefore drops a single variable.

    A variable that is a linear combination of the others has an infinite
    VIF, and so does every other variable of that combination (see `vif`).
    Such variables tie, and dropping one of them, the one with the lowest
    priority or else the last in the column order, usually leaves the
    others with finite VIFs. VIFs within a
    relative :math:`10^{-10}` of the largest count as tied, so that rounding
    does not decide between variables whose VIFs are equal, such as the two
    variables of a pair.

    Examples
    --------
    ``c`` is nearly the sum of ``a`` and ``b``, so all three have large
    VIFs, and ``d`` is unrelated to them:

    >>> import pandas as pd
    >>> from kivi.selection import vif, vif_filter
    >>> X = pd.DataFrame({
    ...     "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ...     "b": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
    ...     "c": [3.2, 2.9, 7.1, 6.8, 11.1, 11.0],
    ...     "d": [1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
    ... })
    >>> vif(X).round(1)
    variable
    a     577.8
    b     952.8
    c    2747.5
    d       1.3
    Name: vif, dtype: float64
    >>> kept = vif_filter(X)
    >>> kept.tolist()
    ['a', 'b', 'd']
    >>> vif(X[kept]).round(2)
    variable
    a    3.55
    b    3.55
    d    1.12
    Name: vif, dtype: float64

    Dropping ``c``, the variable with the largest VIF, leaves ``a`` and
    ``b`` with VIFs below 5, although they were above 500 in the first
    round.

    The three variables of an exact linear combination all have an
    infinite VIF. Without `priority`, the last of them in the column order
    is dropped; with the information values as `priority`, the least
    predictive one is:

    >>> Z = pd.DataFrame({"a": X["a"], "d": X["d"], "e": X["a"] + X["d"]})
    >>> vif(Z)
    variable
    a    inf
    d    inf
    e    inf
    Name: vif, dtype: float64
    >>> vif_filter(Z).tolist()
    ['a', 'd']
    >>> vif_filter(Z, priority=pd.Series({"a": 0.3, "d": 0.1, "e": 0.2})).tolist()
    ['a', 'e']
    """
    if (
        isinstance(threshold, bool)
        or not _validation.is_real(threshold)
        or not 1 < threshold < np.inf
    ):
        raise ValueError(
            "threshold must be a finite number greater than 1, got "
            f"{_validation.shown_argument(threshold)}"
        )
    frame, values = _logit.as_frame(X, example=_WOE)
    _logit.check_not_constant(frame.columns, values, "its VIF is undefined")
    rank = _priority(priority, frame.columns)
    standardized = _standardized(values)
    rows, columns = standardized.shape
    # Z = QR leaves the singular values and right singular vectors of every set of columns
    # unchanged, so each round decomposes columns of the small R instead of the rows of Z.
    reduced = np.linalg.qr(standardized, mode="r") if rows > columns else standardized
    kept = np.arange(columns)
    while len(kept) > 1:
        factors = _standardized_vif(reduced[:, kept], rows)
        largest = factors.max()
        if largest <= threshold:
            break
        # An infinite largest VIF ties only with the other infinite ones.
        tied = np.flatnonzero(factors >= largest * (1 - _TIE))
        lowest = rank[kept[tied]] == rank[kept[tied]].min()
        # Of the tied columns with the lowest priority, the last goes, so earlier ones are kept.
        kept = np.delete(kept, tied[lowest][-1])
    return frame.columns[kept]


def _vif(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the VIF of every column of non-constant, finite `values`."""
    standardized = _standardized(values)
    return _standardized_vif(standardized, standardized.shape[0])


def _standardized(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the columns of `values` centred and scaled to unit length."""
    centered = values - values.mean(axis=0)
    return centered / np.linalg.norm(centered, axis=0)


def _standardized_vif(matrix: npt.NDArray[np.float64], rows: int) -> npt.NDArray[np.float64]:
    """Return the VIFs of standardized columns from a matrix with their singular values.

    `matrix` is the standardized columns Z or the matching columns of R from Z = QR, which
    has the same singular values and right singular vectors, and `rows` is the number of
    observations, which sets the rank tolerance in both cases.
    """
    # With unit-length centred columns Z, Z'Z is the correlation matrix and its inverse,
    # whose diagonal holds the VIFs, is V diag(1 / s^2) V' from the SVD Z = U diag(s) V'.
    columns = matrix.shape[1]
    # With fewer rows than columns, only the full V has a row for every direction.
    singular, vt = np.linalg.svd(matrix, full_matrices=matrix.shape[0] < columns)[1:]
    eps = np.finfo(float).eps
    rank = int((singular > max(rows, columns) * eps * singular[0]).sum())
    result = ((vt[:rank] / singular[:rank, None]) ** 2).sum(axis=0)
    # A column with a component in the null space of Z, the rows of V beyond the rank, is a
    # combination of the others, and so is a column whose 1 - R^2 = 1 / VIF is at most eps.
    null = (vt[rank:] ** 2).sum(axis=0)
    result[(null > columns * eps) | (result >= 1 / eps)] = np.inf
    return result


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
        statistic and its two-sided p-value, in an index named
        ``variable``. ``attrs["log_likelihood"]`` holds the maximized
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
    variable
    intercept -1.0986  1.1547 -0.9514   0.3414
    x          2.1972  1.6330  1.3455   0.1785

    Integer weights count repeated rows, so the four distinct rows with
    their counts give the same fit:

    >>> distinct = pd.DataFrame({"x": [0, 0, 1, 1]})
    >>> logistic_regression(distinct, [0, 1, 0, 1], weights=[3, 1, 1, 3]).round(4)
                 coef      se       z  p_value
    variable
    intercept -1.0986  1.1547 -0.9514   0.3414
    x          2.1972  1.6330  1.3455   0.1785
    """
    frame, values, labels, mass = _model_inputs(X, target, weights)
    _logit.check_independent(frame.columns, values, weighted=mass is not None)
    result = _fit(_logit.with_intercept(values), labels, frame.columns, mass)
    if not result.converged:
        _logit.warn_not_converged(result, stacklevel=2)
    table = pd.DataFrame(
        {"coef": result.coef, "se": result.se, "z": result.z, "p_value": result.p_value},
        index=pd.Index(["intercept"]).append(frame.columns).rename("variable"),
    )
    table.attrs = {"log_likelihood": result.log_likelihood, "converged": result.converged}
    return table


def stepwise(
    X: pd.DataFrame | npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    p_enter: float = 0.05,
    p_remove: float = 0.10,
    direction: typing.Literal["forward", "backward", "both"] = "both",
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
    selection_path : Fit the models of a fixed order of variables.
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
    >>> from kivi.datasets import load_bank
    >>> from kivi.selection import stepwise
    >>> from kivi.woe import woe_table, woe_transform
    >>> data = load_bank()
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
    if not isinstance(direction, str) or direction not in ("forward", "backward", "both"):  # type: ignore[redundant-expr]  # rejects non-str values
        raise ValueError(
            "direction must be 'forward', 'backward' or 'both', got "
            f"{_validation.shown_argument(direction)}"
        )
    if direction == "both" and p_enter > p_remove:
        raise ValueError(
            f"p_enter must not exceed p_remove, got p_enter={_validation.shown_argument(p_enter)} "
            f"and p_remove={_validation.shown_argument(p_remove)}"
        )
    if not isinstance(positive, (bool, np.bool_)):
        raise ValueError(
            f"positive must be True or False, got {_validation.shown_argument(positive)}"
        )
    frame, values, labels, mass = _model_inputs(X, target, weights)
    # Every subset of independent columns is independent, so candidate models have full rank.
    _logit.check_independent(frame.columns, values, weighted=mass is not None)
    design = _logit.with_intercept(values)
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


def selection_path(
    X: pd.DataFrame | npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    order: abc.Sequence[abc.Hashable] | npt.ArrayLike | None = None,
    validation: tuple[pd.DataFrame | npt.ArrayLike, npt.ArrayLike] | None = None,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Fit a logistic regression on each leading part of an order of variables.

    Step :math:`k` is the logistic regression of `target` on the first
    :math:`k` variables of `order`, fitted as `logistic_regression` fits it.
    Each step reports how well its model ranks the development sample and,
    optionally, a validation sample, and the largest p-value and the
    smallest coefficient of its variables, so that the table shows where
    adding variables stops paying.

    Parameters
    ----------
    X : pandas.DataFrame or array_like
        Two-dimensional numeric values with one column per candidate
        variable, such as WOE values from `kivi.woe.woe_transform`.
    target : array_like
        Binary labels, with 1 indicating the event. If `X` is a DataFrame and
        `target` a Series, they must have identical indices; otherwise they
        are matched by position.
    order : sequence, optional
        The variables in the order in which they enter, such as the columns
        sorted by information value: labels of the columns of `X`, or column
        positions when `X` is not a DataFrame. It may leave columns out and
        gives one step per entry. Default every column, in the order of `X`.
    validation : tuple, optional
        A validation sample ``(X_valid, target_valid)``, scored with the
        models fitted on `X`. ``X_valid`` holds the variables of `order`: a
        DataFrame with their labels when `X` is a DataFrame, otherwise
        values with as many columns as `X`, matched by position.
        ``target_valid`` holds both classes and is matched to ``X_valid`` as
        `target` is to `X`.
    weights : array_like, optional
        Non-negative frequency weight of each row of `X`, matched like
        `target`, as in `logistic_regression`. The fits and the metrics on
        `X` use them, so integer weights give the same table as repeating
        rows. Each row of the validation sample counts once.

    Returns
    -------
    pandas.DataFrame
        One row per step, in an index named ``step`` that counts the steps
        from 1, with the columns:

        - ``variable``: the variable that the step adds;
        - ``auc``, ``gini`` and ``ks``: the area under the ROC curve, the
          Gini coefficient and the KS statistic of the fitted probability
          on `X`, as `kivi.metrics.roc_auc`, `kivi.metrics.gini` and
          `kivi.metrics.ks` compute them;
        - ``max_p_value``: the largest Wald p-value of the step's variables,
          without the intercept;
        - ``min_coefficient``: the smallest coefficient of the step's
          variables, without the intercept;
        - ``validation_auc``, ``validation_gini`` and ``validation_ks``,
          only with `validation`: the same metrics on the validation sample.

    Raises
    ------
    SeparationError
        If the model of a step separates the classes; the message names the
        step and the variable that it adds.
    ValueError
        If `X`, `target` or `weights` is invalid as described for
        `logistic_regression`, where the columns before a variable are those
        before it in `order`; if `order` is not a sequence of distinct
        columns of `X`; or if `validation` is not a pair whose values have
        the variables of `order` and are numeric and finite, and whose
        labels are 0 and 1 with both classes present.

    Warns
    -----
    RuntimeWarning
        If the model of a step does not converge for a reason other than
        separation.

    See Also
    --------
    stepwise : Select the variables of the model by their p-values.
    logistic_regression : Fit the model of one step and read its
        coefficients.
    kivi.metrics.roc_auc : Compute the AUC of a score.

    Notes
    -----
    The steps add the variables in a fixed order; nothing is selected or
    removed on the way. The order comes from outside the model, usually
    from the univariate strength of each variable, as in the forward
    procedure of many scorecard tools. `stepwise` instead chooses each
    variable by its p-value in the model.

    The AUC on the development sample almost always rises with every
    variable, because each model is fitted to that sample. On a validation
    sample it stops rising, or falls, once the variables add more noise
    than information, which is where the order is usually cut. A
    ``min_coefficient`` at or below 0 marks a model in which a WOE-encoded
    variable works against its own event rates, which ``stepwise`` with
    ``positive=True`` rules out.

    The metrics are computed from the linear predictor
    :math:`\hat\beta_0 + \sum_j \hat\beta_j x_j`. It ranks the rows as the
    fitted probability does, without the ties that would appear where
    probabilities close to 0 or 1 round to the same number.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.datasets import load_bank
    >>> from kivi.selection import selection_path
    >>> from kivi.woe import woe_table, woe_transform
    >>> data = load_bank()
    >>> development, validation = data.iloc[:3000], data.iloc[3000:]
    >>> columns = ["poutcome", "contact", "housing", "loan", "marital", "default"]
    >>> tables = {
    ...     column: woe_table(development[column], development["target"]) for column in columns
    ... }
    >>> def encode(sample):
    ...     return pd.DataFrame(
    ...         {column: woe_transform(sample[column], tables[column]) for column in columns}
    ...     )
    >>> path = selection_path(
    ...     encode(development),
    ...     development["target"],
    ...     validation=(encode(validation), validation["target"]),
    ... )
    >>> path[["variable", "auc", "validation_auc", "max_p_value", "min_coefficient"]].round(4)
          variable     auc  validation_auc  max_p_value  min_coefficient
    step
    1     poutcome  0.6065          0.6162       0.0000           1.0000
    2      contact  0.6548          0.6758       0.0000           0.7697
    3      housing  0.6781          0.6956       0.0000           0.6702
    4         loan  0.6918          0.7083       0.0017           0.6783
    5      marital  0.7070          0.7070       0.0021           0.6602
    6      default  0.7078          0.7076       0.7327          -1.3508

    The AUC on the development sample rises with every step, while on the
    validation sample it peaks at step 4. ``default`` adds nothing: its
    p-value is 0.73 and its coefficient negative.
    """
    frame, values, labels, mass = _model_inputs(X, target, weights)
    positions = _order(order, frame.columns)
    columns = frame.columns[positions]
    values = values[:, positions]
    _logit.check_independent(columns, values, weighted=mass is not None)
    if validation is not None:
        validation_values, validation_labels = _validation_sample(
            validation, isinstance(X, pd.DataFrame), frame.columns, positions
        )
    steps = len(positions)
    results = np.empty((steps, 8))
    for step in range(1, steps + 1):
        # The design of logistic_regression on the same columns, so that the fits agree exactly.
        design = _logit.with_intercept(values[:, :step])
        try:
            fitted = _fit(design, labels, columns[:step], mass, step=step)
        except SeparationError:
            raise SeparationError(
                f"the model of step {step}, which adds X column "
                f"{_validation.shown(columns[step - 1])}, separates the classes of target, so its "
                "maximum likelihood estimates do not exist; end order before that column or merge "
                "its bins"
            ) from None
        if not fitted.converged:
            _logit.warn_not_converged(fitted, stacklevel=2, step=step)
        results[step - 1, :5] = (
            *_ranking(_linear_predictor(values, fitted.coef), labels, mass),
            fitted.p_value[1:].max(),
            fitted.coef[1:].min(),
        )
        if validation is not None:
            results[step - 1, 5:] = _ranking(
                _linear_predictor(validation_values, fitted.coef), validation_labels, None
            )
    names = ["auc", "gini", "ks", "max_p_value", "min_coefficient"]
    if validation is not None:
        names += ["validation_auc", "validation_gini", "validation_ks"]
    table = pd.DataFrame(
        results[:, : len(names)], index=pd.RangeIndex(1, steps + 1, name="step"), columns=names
    )
    table.insert(0, "variable", columns)
    return table


def _linear_predictor(
    values: npt.NDArray[np.float64], coef: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return the intercept ``coef[0]`` plus the first columns of `values` times ``coef[1:]``."""
    # Element-wise products and sums, unlike a matrix product, give identical rows identical
    # scores, so that their ties count as ties.
    score = np.full(len(values), coef[0])
    for column, slope in enumerate(coef[1:]):
        score += slope * values[:, column]
    return score


def _ranking(
    score: npt.NDArray[np.float64],
    labels: npt.NDArray[np.float64],
    mass: npt.NDArray[np.float64] | None,
) -> tuple[float, float, float]:
    """Return the AUC, Gini and KS of a score, as `kivi.metrics` computes them."""
    auc = metrics.roc_auc(score, labels, weights=mass)
    # kivi.metrics.gini is 2 * roc_auc - 1.
    return auc, 2 * auc - 1, metrics.ks(score, labels, weights=mass)


def _order(
    order: abc.Sequence[abc.Hashable] | npt.ArrayLike | None, columns: pd.Index
) -> npt.NDArray[np.intp]:
    """Return the positions in `columns` of the labels of `order`, every column by default."""
    if order is None:
        return np.arange(len(columns))
    if (
        isinstance(order, (str, bytes))
        or not isinstance(order, (abc.Sequence, pd.Index, pd.Series, np.ndarray))
        or (isinstance(order, np.ndarray) and order.ndim != 1)
    ):
        raise ValueError(f"order must be a sequence of X column labels, got {type(order).__name__}")
    labels = list(order)
    if not labels:
        raise ValueError("order must name at least one column of X, got []")
    positions = []
    for label in labels:
        try:
            position = columns.get_loc(label)
        except (KeyError, TypeError, pd.errors.InvalidIndexError):
            raise ValueError(
                f"order must name columns of X, got {_validation.shown(label)}"
            ) from None
        if position in positions:
            raise ValueError(f"order must not repeat a column, got {_validation.shown(label)}")
        positions.append(position)
    return np.array(positions, dtype=np.intp)


def _validation_sample(
    validation: tuple[pd.DataFrame | npt.ArrayLike, npt.ArrayLike],
    by_label: bool,
    columns: pd.Index,
    positions: npt.NDArray[np.intp],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the values of the variables of `positions` and the labels of `validation`."""
    if not isinstance(validation, (tuple, list)) or len(validation) != 2:  # type: ignore[redundant-expr]  # rejects other values
        raise ValueError(f"validation must be a pair (X, target), got {type(validation).__name__}")
    X, target = validation
    # The checks of the development sample apply, with messages that name the validation sample.
    try:
        if by_label:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(f"X must be a DataFrame, as X is, got {type(X).__name__}")
            names = columns[positions]
            absent = names[~names.isin(X.columns)]
            if len(absent):
                raise ValueError(f"X has no column {_validation.shown(absent[0])}")
            frame, values = _logit.as_frame(X[names], example=_WOE)
        else:
            frame, values = _logit.as_frame(X, example=_WOE)
            if frame.shape[1] != len(columns):
                raise ValueError(
                    f"X must have {len(columns)} columns, as X has, got {frame.shape[1]}"
                )
            values = values[:, positions]
        if isinstance(X, pd.DataFrame) and isinstance(target, pd.Series):
            if not X.index.equals(target.index):
                raise ValueError("X and target must have identical indices")
        series = _validation.as_series(target, "target")
        if len(series) != len(frame):
            raise ValueError(
                "X and target must have the same number of rows, got "
                f"{len(frame)} and {len(series)}"
            )
        labels = _validation.binary_labels(series).astype(float)
    except ValueError as error:
        raise ValueError(f"validation {error}") from None
    return values, labels


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
                _logit.warn_not_converged(fitted, stacklevel=3)
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
    *,
    step: int | None = None,
) -> _logit.LogitFit:
    """Fit a model on the intercept and `columns`, naming the column of a singular fit.

    `step` names the step of `selection_path` in the message.
    """
    try:
        return _warm_fit(design, labels, mass, start)
    except _logit.SingularError as error:
        model = "" if step is None else f" in the model of step {step}"
        raise _logit.singular_column(error, columns, model) from None


def _model_columns(selected: list[int]) -> list[int]:
    return [0, *(column + 1 for column in selected)]


def _check_level(level: float, name: str) -> None:
    if isinstance(level, bool) or not _validation.is_real(level) or not 0 < level <= 1:
        raise ValueError(
            f"{name} must be greater than 0 and at most 1, got {_validation.shown_argument(level)}"
        )


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
    frame, values = _logit.as_frame(X, example=_WOE)
    if "intercept" in frame.columns:
        raise ValueError("X must not have a column labelled 'intercept'")
    raw_target = target
    target = _validation.as_series(target, "target")
    if len(target) != len(frame):
        raise ValueError(
            f"X and target must have the same number of rows, got {len(frame)} and {len(target)}"
        )
    labels = _validation.binary_labels(target).astype(float)
    if weights is None:
        return frame, values, labels, None
    if np.ndim(weights) == 1 and len(weights) != len(frame):  # type: ignore[arg-type]  # one-dimensional
        raise ValueError(
            f"weights must have one weight per row of X, got {len(weights)} for {len(frame)}"  # type: ignore[arg-type]  # one-dimensional
        )
    mass = typing.cast(
        npt.NDArray[np.float64],
        _validation.as_weights(weights, frame, of="X", others={"target": raw_target}),
    )
    _validation.check_class_weights(labels.astype(np.intp), mass)
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
            raise ValueError(f"priority has no value for X column {_validation.shown(absent[0])}")
        priority = priority.reindex(columns)
    priority = _validation.as_series(priority, "priority")
    if len(priority) != len(columns):
        raise ValueError(
            f"priority must have one value per column of X, got {len(priority)} values "
            f"for {len(columns)} columns"
        )
    if not pd.api.types.is_numeric_dtype(priority):
        raise ValueError(
            "priority must contain numbers only, without missing values, got dtype "
            f"{priority.dtype}"
        )
    if priority.isna().any():
        raise ValueError(
            "priority must contain numbers only, without missing values, got "
            f"{_validation.shown(priority[priority.isna()])}"
        )
    return priority.to_numpy(dtype=float)
