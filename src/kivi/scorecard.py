"""Scorecard scaling, points and reason codes."""

import math
from collections.abc import Container, Mapping
from numbers import Real

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._validation import is_integral, is_real, like_input, woe_values

__all__ = [
    "pd_to_score",
    "points_table",
    "reason_codes",
    "score_scaling",
    "score_to_pd",
    "woe_to_points",
]


def score_scaling(*, pdo: float, base_score: float, base_odds: float) -> tuple[float, float]:
    r"""Return the factor and offset that turn log odds into a score.

    A scorecard score is linear in the log of the good:bad odds. It equals
    `base_score` at odds of `base_odds` and rises by `pdo` points each time
    the odds double.

    Parameters
    ----------
    pdo : float
        Points to double the odds. Must be positive.
    base_score : float
        Score at the base odds.
    base_odds : float
        Good:bad odds at the base score, such as 50 for 50 non-events per
        event. Must be positive.

    Returns
    -------
    factor : float
        Points per unit of log odds, ``pdo / ln 2``.
    offset : float
        Score at even odds, ``base_score - factor * ln(base_odds)``.

    Raises
    ------
    ValueError
        If an argument is not a finite number, or `pdo` or `base_odds` is
        not positive.

    See Also
    --------
    pd_to_score : Convert event probabilities to scores.
    score_to_pd : Convert scores to event probabilities.

    Notes
    -----
    With odds :math:`o = (1 - p) / p` for an event probability :math:`p`,

    .. math::

       \mathrm{score} = \mathrm{offset} + \mathrm{factor} \cdot \ln o,

       \mathrm{factor} = \frac{\mathrm{pdo}}{\ln 2}, \qquad
       \mathrm{offset} = \mathrm{base\_score} - \mathrm{factor} \cdot \ln \mathrm{base\_odds}.

    Examples
    --------
    >>> from kivi.scorecard import score_scaling
    >>> factor, offset = score_scaling(pdo=20, base_score=600, base_odds=50)
    >>> print(f"factor = {factor:.4f}, offset = {offset:.4f}")
    factor = 28.8539, offset = 487.1229
    """
    for name, value in (("pdo", pdo), ("base_score", base_score), ("base_odds", base_odds)):
        if isinstance(value, bool) or not is_real(value) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    if pdo <= 0:
        raise ValueError(f"pdo must be positive, got {pdo!r}")
    if base_odds <= 0:
        raise ValueError(f"base_odds must be positive, got {base_odds!r}")
    factor = pdo / math.log(2)
    return factor, base_score - factor * math.log(base_odds)


def pd_to_score(
    probability: npt.ArrayLike, *, pdo: float, base_score: float, base_odds: float
) -> float | pd.Series | npt.NDArray[np.float64]:
    """Convert event probabilities to scorecard scores.

    A lower probability of the event, such as a default, gives a higher
    score. The scaling is set by `pdo`, `base_score` and `base_odds`, as in
    `score_scaling`.

    Parameters
    ----------
    probability : array_like
        Event probabilities strictly between 0 and 1. Missing values give
        missing scores.
    pdo : float
        Points to double the odds. Must be positive.
    base_score : float
        Score at the base odds.
    base_odds : float
        Good:bad odds at the base score. Must be positive.

    Returns
    -------
    float, pandas.Series or numpy.ndarray
        Scores with the shape of `probability`. A Series input gives a Series with
        the same index and name, and a scalar gives a float.

    Raises
    ------
    ValueError
        If `probability` has values outside the open interval (0, 1), or the
        scaling arguments are invalid.

    See Also
    --------
    score_to_pd : The inverse conversion.
    score_scaling : The factor and offset of the scaling.

    Notes
    -----
    The score is ``offset + factor * ln((1 - probability) / probability)``. A probability
    of exactly 0 or 1 has no finite score and raises an error; see
    `score_to_pd` for the scores whose probability rounds to 1.

    Examples
    --------
    >>> from kivi.scorecard import pd_to_score
    >>> scale = dict(pdo=20, base_score=600, base_odds=50)
    >>> pd_to_score([1 / 51, 1 / 101, 0.1], **scale).round(2)
    array([600.  , 620.  , 550.52])
    """
    factor, offset = score_scaling(pdo=pdo, base_score=base_score, base_odds=base_odds)
    values = np.asarray(probability, dtype=float)
    if np.any((values <= 0) | (values >= 1)):
        shown = float(values[(values <= 0) | (values >= 1)].flat[0])
        raise ValueError(f"probability must be strictly between 0 and 1, got {shown!r}")
    return like_input(probability, offset + factor * (np.log1p(-values) - np.log(values)))


def score_to_pd(
    score: npt.ArrayLike, *, pdo: float, base_score: float, base_odds: float
) -> float | pd.Series | npt.NDArray[np.float64]:
    """Convert scorecard scores to event probabilities.

    This is the inverse of `pd_to_score` for the same scaling, so it gives
    the probability of the event implied by each score.

    Parameters
    ----------
    score : array_like
        Scores. Missing values give missing probabilities.
    pdo : float
        Points to double the odds. Must be positive.
    base_score : float
        Score at the base odds.
    base_odds : float
        Good:bad odds at the base score. Must be positive.

    Returns
    -------
    float, pandas.Series or numpy.ndarray
        Event probabilities with the shape of `score`. A Series input gives
        a Series with the same index and name, and a scalar gives a float.

    Raises
    ------
    ValueError
        If the scaling arguments are invalid.

    See Also
    --------
    pd_to_score : The inverse conversion.
    score_scaling : The factor and offset of the scaling.

    Notes
    -----
    The probability is ``1 / (1 + exp((score - offset) / factor))``,
    computed without overflow for extreme scores. In double precision it
    rounds to exactly 1 for scores more than about ``36.7 * factor`` below
    the offset, -572 for the scale in the example, and `pd_to_score` rejects
    such probabilities. Scores that low do not occur on a practical scale.

    Examples
    --------
    >>> from kivi.scorecard import score_to_pd
    >>> scale = dict(pdo=20, base_score=600, base_odds=50)
    >>> score_to_pd([580, 600, 620], **scale).round(5)
    array([0.03846, 0.01961, 0.0099 ])
    """
    factor, offset = score_scaling(pdo=pdo, base_score=base_score, base_odds=base_odds)
    log_odds = (np.asarray(score, dtype=float) - offset) / factor
    with np.errstate(invalid="ignore"):  # logaddexp warns on missing values
        probability = np.exp(-np.logaddexp(0, log_odds))
    return like_input(score, probability)


def points_table(
    tables: Mapping[object, pd.DataFrame | pd.Series],
    coefficients: Mapping[object, float] | pd.Series,
    intercept: float,
    *,
    pdo: float,
    base_score: float,
    base_odds: float,
) -> pd.DataFrame:
    r"""Turn a logistic regression on WOE values into points for each group.

    The points of an applicant's groups, one per variable, add up to the
    score that `pd_to_score` gives for the model's probability. The
    intercept is spread evenly across the variables.

    Parameters
    ----------
    tables : mapping
        For each variable, its `kivi.woe.woe_table` result or a Series of
        WOE values indexed by group. Tables of variables without a
        coefficient are ignored.
    coefficients : mapping or pandas.Series
        Logistic regression coefficient of each variable's WOE, for a model
        of the event probability. The rows follow this order.
    intercept : float
        Intercept of the same model.
    pdo : float
        Points to double the odds. Must be positive.
    base_score : float
        Score at the base odds.
    base_odds : float
        Good:bad odds at the base score. Must be positive.

    Returns
    -------
    pandas.DataFrame
        One row per variable and group, indexed by ``variable``, with the
        group label ``bin`` and the group's ``woe`` and ``points``. Select a
        variable's rows with ``table.loc[[variable]]``.

    Raises
    ------
    ValueError
        If `tables` is not a mapping or lacks a variable of `coefficients`,
        a table has no ``woe`` column, repeats a group or has non-finite WOE
        values, the coefficients are empty, repeated or not finite numbers,
        or the scaling arguments are invalid.

    See Also
    --------
    woe_to_points : The points of each applicant.
    reason_codes : The variables that cost an applicant the most points.
    kivi.woe.woe_table : Fit the WOE of each group.

    Notes
    -----
    With kivi's WOE, :math:`\ln(p_i / q_i)` for event and non-event shares
    :math:`p_i` and :math:`q_i`, the model is
    :math:`\ln \frac{\mathrm{PD}}{1 - \mathrm{PD}} = \beta_0 + \sum_j \beta_j \mathrm{WOE}_j`.
    A higher WOE means more risk and, for a positive coefficient, fewer
    points. Group :math:`i` of variable :math:`j`, one of :math:`m`
    variables, receives

    .. math::

       \mathrm{points}_{ij} = \frac{\mathrm{offset} - \mathrm{factor} \cdot \beta_0}{m}
       - \mathrm{factor} \cdot \beta_j \, \mathrm{WOE}_{ij},

    with the factor and offset of `score_scaling`. Each variable carries the
    same share of the intercept and offset, so the points of a variable's
    groups differ only by their WOE term.

    The group labels stay in a column rather than an index level, so labels
    that compare equal across variables, such as ``True`` and ``1``, keep
    their own values.

    References
    ----------
    .. [1] N. Siddiqi, "Credit Risk Scorecards: Developing and Implementing
       Intelligent Credit Scoring", Wiley, 2006.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.scorecard import points_table
    >>> tables = {
    ...     "housing": pd.Series({"no": -0.4, "yes": 0.3}),
    ...     "contact": pd.Series({"cellular": -0.2, "unknown": 0.6}),
    ... }
    >>> coefficients = {"housing": 0.8, "contact": 0.5}
    >>> points_table(tables, coefficients, -2.0, pdo=20, base_score=600, base_odds=50).round(2)
                   bin  woe  points
    variable
    housing         no -0.4  281.65
    housing        yes  0.3  265.49
    contact   cellular -0.2  275.30
    contact    unknown  0.6  263.76
    """
    slope, base = _model_points(coefficients, intercept, pdo, base_score, base_odds)
    if not isinstance(tables, Mapping):
        raise ValueError(
            f"tables must be a mapping of variable to table, got {type(tables).__name__}"
        )
    _check_present(slope.index, tables, "tables")
    woe = [_woe_column(tables[variable], variable) for variable in slope.index]
    variable_codes = np.repeat(np.arange(len(woe)), [len(column) for column in woe])
    values = np.concatenate([column.to_numpy(dtype=float) for column in woe])
    # The same operations as woe_to_points, so both give identical points.
    points = base - slope.to_numpy()[variable_codes] * values
    table = pd.DataFrame(
        {"woe": values, "points": points}, index=slope.index.take(variable_codes).rename("variable")
    )
    # A column keeps every label as given: an index level would merge labels that compare
    # equal, such as True and 1, and would hash interval labels, which warns on NumPy < 2.2.
    labels = np.concatenate([np.asarray(column.index, dtype=object) for column in woe])
    table.insert(0, "bin", pd.array(labels, dtype=object))  # type: ignore[arg-type]  # pandas-stubs omit dtype=object
    return table.astype({"bin": object})


def woe_to_points(
    woe: pd.DataFrame | Mapping[object, npt.ArrayLike],
    coefficients: Mapping[object, float] | pd.Series,
    intercept: float,
    *,
    pdo: float,
    base_score: float,
    base_odds: float,
) -> pd.DataFrame:
    r"""Give each applicant the points of each variable from its WOE values.

    The points are those of `points_table` for the same model and scaling.
    Their row sums are the scores, equal to `pd_to_score` of the model's
    probabilities.

    Parameters
    ----------
    woe : pandas.DataFrame or mapping
        WOE values with one column per variable, such as the result of
        `kivi.woe.woe_transform` for each variable. Columns without a
        coefficient are ignored.
    coefficients : mapping or pandas.Series
        Logistic regression coefficient of each variable's WOE, for a model
        of the event probability. The columns follow this order.
    intercept : float
        Intercept of the same model.
    pdo : float
        Points to double the odds. Must be positive.
    base_score : float
        Score at the base odds.
    base_odds : float
        Good:bad odds at the base score. Must be positive.

    Returns
    -------
    pandas.DataFrame
        Points with the index of `woe` and one column per variable.

    Raises
    ------
    ValueError
        If `woe` is not a DataFrame or mapping, lacks or repeats a variable
        of `coefficients` or has non-finite values, the coefficients are empty,
        repeated or not finite, or the scaling arguments are invalid.

    See Also
    --------
    points_table : The points of every group.
    reason_codes : The variables that cost an applicant the most points.

    Notes
    -----
    The points of variable :math:`j` are
    :math:`(\mathrm{offset} - \mathrm{factor} \cdot \beta_0) / m
    - \mathrm{factor} \cdot \beta_j \, \mathrm{WOE}_j` for :math:`m`
    variables, as in `points_table`. They are computed in the same way, so
    rounding both gives the same points.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.scorecard import woe_to_points
    >>> woe = pd.DataFrame({"housing": [-0.4, 0.3], "contact": [0.6, -0.2]})
    >>> points = woe_to_points(
    ...     woe, {"housing": 0.8, "contact": 0.5}, -2.0, pdo=20, base_score=600, base_odds=50
    ... )
    >>> points.round(2)
       housing  contact
    0   281.65   263.76
    1   265.49   275.30
    >>> points.sum(axis=1).round(2).tolist()
    [545.41, 540.79]
    """
    slope, base = _model_points(coefficients, intercept, pdo, base_score, base_odds)
    if isinstance(woe, Mapping):
        woe = pd.DataFrame(woe)  # type: ignore[arg-type]  # a mapping of columns
    elif not isinstance(woe, pd.DataFrame):
        raise ValueError(f"woe must be a DataFrame or a mapping, got {type(woe).__name__}")
    _check_present(slope.index, woe.columns, "woe")
    repeated = woe.columns[woe.columns.duplicated() & woe.columns.isin(slope.index)]
    if len(repeated):
        raise ValueError(f"woe must have one column per variable, got {repeated[0]!r} twice")
    values = woe[list(slope.index)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        column = slope.index[~np.isfinite(values).all(axis=0)][0]
        raise ValueError(f"woe must be finite, got missing or infinite values in {column!r}")
    return pd.DataFrame(base - slope.to_numpy() * values, index=woe.index, columns=slope.index)


def reason_codes(
    points: pd.DataFrame,
    table: pd.DataFrame,
    *,
    top: int = 3,
    atol: float = 1e-4,
    unseen_points: float | Mapping[object, float] | pd.Series | None = None,
) -> pd.DataFrame:
    """Name the variables that cost each applicant the most points.

    A variable costs an applicant the difference between the points of the
    variable's best group and the points of the applicant's group. Variables
    are listed from the largest cost; those that cost nothing are not
    listed.

    Parameters
    ----------
    points : pandas.DataFrame
        Points of each applicant, as returned by `woe_to_points`.
    table : pandas.DataFrame
        The `points_table` result for the same model and scaling.
    top : int, default 3
        Number of reasons per applicant.
    atol : float, default 1e-4
        Largest difference, in points, between an applicant's points and the
        points of a group for the applicant to be in that group. It absorbs
        rounding, for example of WOE values stored with six decimals.
    unseen_points : float, mapping or pandas.Series, optional
        Points of applicants in groups that `kivi.woe.woe_transform` encoded
        with ``unseen``, one value for every variable or values by variable.
        Points that match no group but lie within `atol` of their variable's
        `unseen_points` cost nothing, so the variable is not a reason for
        those applicants. They are the `woe_to_points` result for the WOE
        passed as ``unseen``; see the Examples. By default every point must
        match a group.

    Returns
    -------
    pandas.DataFrame
        Variable names in columns ``reason_1`` to ``reason_<top>``, with the
        index of `points`. An applicant with fewer costly variables has
        None in the remaining columns.

    Raises
    ------
    ValueError
        If `points` is not a DataFrame or has non-finite values, `table` is
        not a `points_table` result or lacks a variable of `points`, some
        points are farther than `atol` from every group of their variable
        and from its `unseen_points`, `top` is not a positive integer, `atol`
        is not a finite positive number, or `unseen_points` is not a finite
        number or a mapping or Series of finite numbers.

    See Also
    --------
    points_table : The points of every group.
    woe_to_points : The points of each applicant.

    Notes
    -----
    Each applicant's points are matched to the nearest group of the variable
    in `table`, and the costs are differences between the points of
    `table`'s groups. Costs that differ by less than about `atol` count as
    equal, so such variables keep the order of the columns of `points`, and
    a cost below about `atol` is no cost.

    Points that match no group, such as those from a different intercept or
    scaling, or WOE values rounded more coarsely than `atol` allows, raise
    an error rather than giving reasons against the wrong groups. So do, by
    default, the points of applicants in a group that had no development
    observations, such as a new category, whose WOE came from ``unseen`` in
    `kivi.woe.woe_transform`. That WOE is a value chosen for the group, not
    evidence about the applicant, so with `unseen_points` such points give
    no reason instead of the distance to the best group. Only points near
    `unseen_points` are exempt; any other mismatch still raises. Points that
    lie within `atol` of a group are matched to that group, even when their
    WOE came from ``unseen``.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.scorecard import points_table, reason_codes, woe_to_points
    >>> tables = {
    ...     "housing": pd.Series({"no": -0.4, "yes": 0.3}),
    ...     "contact": pd.Series({"cellular": -0.2, "unknown": 0.6}),
    ... }
    >>> model = ({"housing": 0.8, "contact": 0.5}, -2.0)
    >>> scale = dict(pdo=20, base_score=600, base_odds=50)
    >>> table = points_table(tables, *model, **scale)
    >>> woe = pd.DataFrame({"housing": [-0.4, 0.3, 0.3], "contact": [-0.2, 0.6, -0.2]})
    >>> reason_codes(woe_to_points(woe, *model, **scale), table, top=2)
      reason_1 reason_2
    0     None     None
    1  housing  contact
    2  housing     None

    An applicant with a housing status that the development sample did not
    have was encoded with ``woe_transform(..., unseen=0.0)``. The points of
    a WOE of 0 are the `woe_to_points` result for a row of zeros, the same
    base points for every variable. With these points as `unseen_points`,
    housing is not one of the applicant's reasons:

    >>> zero = woe_to_points(pd.DataFrame({"housing": [0.0], "contact": [0.0]}), *model, **scale)
    >>> zero.round(2)
       housing  contact
    0   272.42   272.42
    >>> new = pd.DataFrame({"housing": [0.0], "contact": [0.6]})
    >>> reason_codes(
    ...     woe_to_points(new, *model, **scale), table, top=2, unseen_points=zero.iloc[0]
    ... )
      reason_1 reason_2
    0  contact     None
    """
    if not isinstance(points, pd.DataFrame):
        raise ValueError(f"points must be a DataFrame, got {type(points).__name__}")
    if isinstance(top, bool) or not is_integral(top) or top < 1:
        raise ValueError(f"top must be a positive integer, got {top!r}")
    if isinstance(atol, bool) or not is_real(atol) or not 0 < atol < math.inf:
        raise ValueError(f"atol must be a finite positive number, got {atol!r}")
    if not isinstance(table, pd.DataFrame) or table.index.name != "variable":  # type: ignore[redundant-expr]  # checked at runtime
        raise ValueError(f"table must be a points_table result, got {_describe(table)}")
    if "points" not in table.columns:
        raise ValueError(f"table must have a points column, got columns {table.columns.tolist()}")
    groups = dict(iter(table["points"].groupby(level="variable", sort=False)))
    _check_present(points.columns, groups, "table")
    exempt = _unseen_points(unseen_points, points.columns)
    values = points.to_numpy(dtype=float)
    invalid = ~np.isfinite(values)
    if invalid.any():
        row, column = np.argwhere(invalid)[0]
        value, variable = float(values[row, column]), points.columns[column]
        raise ValueError(f"points must be finite, got {value!r} in {variable!r}")
    cost = np.empty_like(values)
    for column, variable in enumerate(points.columns):
        group_points = np.sort(groups[variable].to_numpy(dtype=float))
        nearest = _nearest(group_points, values[:, column])
        far = np.abs(values[:, column] - nearest) > atol
        unseen = exempt[column]
        if not np.isnan(unseen):
            far &= np.abs(values[:, column] - unseen) > atol
        if far.any():
            row = np.flatnonzero(far)[0]
            also = "" if np.isnan(unseen) else f" or unseen_points {float(unseen)!r}"
            raise ValueError(
                f"points of {variable!r} must match a group of table within atol={atol!r}"
                f"{also}, got {float(values[row, column])!r}, nearest group "
                f"{float(nearest[row])!r}; check atol and that points and table come from the "
                "same model and scaling, and for groups that woe_transform encoded with unseen "
                "pass their points as unseen_points"
            )
        matched = np.abs(values[:, column] - nearest) <= atol
        cost[:, column] = np.where(matched, group_points[-1] - nearest, 0.0)
    # Costs that differ by less than atol tie, so the stable sort keeps the column order.
    steps = np.round(cost / atol)
    order = np.argsort(-steps, axis=1, kind="stable")[:, :top]
    names = np.asarray(points.columns, dtype=object)[order]
    names[np.take_along_axis(steps, order, axis=1) <= 0] = None
    reasons = np.full((len(points), top), None, dtype=object)
    reasons[:, : names.shape[1]] = names
    columns = [f"reason_{rank}" for rank in range(1, top + 1)]
    return pd.DataFrame(reasons, index=points.index, columns=columns, dtype=object)


def _unseen_points(
    unseen_points: float | Mapping[object, float] | pd.Series | None, variables: pd.Index
) -> npt.NDArray[np.float64]:
    """Return the unseen points of each variable, NaN for a variable without them."""
    if unseen_points is None:
        return np.full(len(variables), np.nan)
    if isinstance(unseen_points, (Mapping, pd.Series)):
        by_variable = dict(unseen_points.items())
        if isinstance(unseen_points, pd.Series) and unseen_points.index.has_duplicates:
            repeated = unseen_points.index[unseen_points.index.duplicated()][0]
            raise ValueError(f"unseen_points must name each variable once, got {repeated!r} twice")
    else:
        by_variable = dict.fromkeys(variables, unseen_points)
    result = np.full(len(variables), np.nan)
    for position, variable in enumerate(variables):
        if variable not in by_variable:
            continue
        value = by_variable[variable]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(
                f"unseen_points must be finite numbers, got {value!r} for {variable!r}"
            )
        result[position] = float(value)
    return result


def _model_points(
    coefficients: Mapping[object, float] | pd.Series,
    intercept: float,
    pdo: float,
    base_score: float,
    base_odds: float,
) -> tuple[pd.Series, float]:
    """Return the points per unit of WOE and the base points of each variable."""
    factor, offset = score_scaling(pdo=pdo, base_score=base_score, base_odds=base_odds)
    if isinstance(intercept, bool) or not is_real(intercept) or not math.isfinite(intercept):
        raise ValueError(f"intercept must be a finite number, got {intercept!r}")
    if isinstance(coefficients, Mapping):
        coefficients = pd.Series(dict(coefficients), dtype=object)
    elif not isinstance(coefficients, pd.Series):
        kind = type(coefficients).__name__  # type: ignore[unreachable]  # checked at runtime
        raise ValueError(f"coefficients must be a mapping or Series by variable, got {kind}")
    if coefficients.empty:
        raise ValueError("coefficients must name at least one variable, got none")
    if coefficients.index.has_duplicates:
        repeated = coefficients.index[coefficients.index.duplicated()][0]
        raise ValueError(f"coefficients must name each variable once, got {repeated!r} twice")
    for variable, value in zip(coefficients.index, coefficients.tolist(), strict=True):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"coefficients must be finite numbers, got {value!r} for {variable!r}")
    beta = pd.Series(coefficients.to_numpy(dtype=float), index=coefficients.index)
    return factor * beta, (offset - factor * intercept) / len(beta)


def _woe_column(table: pd.DataFrame | pd.Series, variable: object) -> pd.Series:
    try:
        woe = woe_values(table)
    except ValueError as error:
        raise ValueError(f"invalid table for {variable!r}: {error}") from None
    values = woe.to_numpy(dtype=float)
    invalid = ~np.isfinite(values)
    if invalid.any():
        raise ValueError(
            f"the table of {variable!r} must have finite WOE values, "
            f"got {float(values[invalid][0])!r} for group {woe.index[invalid][0]!r}"
        )
    return woe


def _check_present(variables: pd.Index, available: Container[object], name: str) -> None:
    missing = [variable for variable in variables if variable not in available]
    if missing:
        shown = ", ".join(repr(variable) for variable in missing[:5])
        raise ValueError(f"{name} has no entry for {len(missing)} variable(s): {shown}")


def _nearest(
    sorted_values: npt.NDArray[np.float64], values: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return the element of the sorted array nearest to each value."""
    right = np.clip(np.searchsorted(sorted_values, values), 0, len(sorted_values) - 1)
    left = np.clip(right - 1, 0, len(sorted_values) - 1)
    closer_left = np.abs(values - sorted_values[left]) <= np.abs(values - sorted_values[right])
    return np.where(closer_left, sorted_values[left], sorted_values[right])


def _describe(value: object) -> str:
    if isinstance(value, pd.DataFrame):
        return f"a DataFrame indexed by {value.index.name!r} with columns {value.columns.tolist()}"
    return type(value).__name__
