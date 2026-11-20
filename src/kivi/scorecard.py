"""Scorecard scaling, points and reason codes."""

import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt
import pandas as pd

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
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    if pdo <= 0:
        raise ValueError(f"pdo must be positive, got {pdo!r}")
    if base_odds <= 0:
        raise ValueError(f"base_odds must be positive, got {base_odds!r}")
    factor = pdo / math.log(2)
    return factor, base_score - factor * math.log(base_odds)


def pd_to_score(
    prob: npt.ArrayLike, *, pdo: float, base_score: float, base_odds: float
) -> float | pd.Series | npt.NDArray[np.float64]:
    """Convert event probabilities to scorecard scores.

    A lower probability of the event, such as a default, gives a higher
    score. The scaling is set by `pdo`, `base_score` and `base_odds`, as in
    `score_scaling`.

    Parameters
    ----------
    prob : array_like
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
        Scores with the shape of `prob`. A Series input gives a Series with
        the same index and name, and a scalar gives a float.

    Raises
    ------
    ValueError
        If `prob` has values outside the open interval (0, 1), or the
        scaling arguments are invalid.

    See Also
    --------
    score_to_pd : The inverse conversion.
    score_scaling : The factor and offset of the scaling.

    Notes
    -----
    The score is ``offset + factor * ln((1 - prob) / prob)``.

    Examples
    --------
    >>> from kivi.scorecard import pd_to_score
    >>> scale = dict(pdo=20, base_score=600, base_odds=50)
    >>> pd_to_score([1 / 51, 1 / 101, 0.1], **scale).round(2)
    array([600.  , 620.  , 550.52])
    """
    factor, offset = score_scaling(pdo=pdo, base_score=base_score, base_odds=base_odds)
    values = np.asarray(prob, dtype=float)
    if np.any((values <= 0) | (values >= 1)):
        shown = float(values[(values <= 0) | (values >= 1)].flat[0])
        raise ValueError(f"prob must be strictly between 0 and 1, got {shown!r}")
    return _like(prob, offset + factor * (np.log1p(-values) - np.log(values)))


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
    computed without overflow for extreme scores.

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
        prob = np.exp(-np.logaddexp(0, log_odds))
    return _like(score, prob)


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
        One row per variable and group, indexed by ``variable`` and
        ``bin``, with the group's ``woe`` and ``points``.

    Raises
    ------
    ValueError
        If `tables` is not a mapping or lacks a variable of `coefficients`,
        a table has no ``woe`` column or non-finite WOE values, the
        coefficients are empty, repeated or not finite, or the scaling
        arguments are invalid.

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
                       woe  points
    variable bin
    housing  no       -0.4  281.65
             yes       0.3  265.49
    contact  cellular -0.2  275.30
             unknown   0.6  263.76
    """
    slope, base = _model_points(coefficients, intercept, pdo, base_score, base_odds)
    if not isinstance(tables, Mapping):
        raise ValueError(
            f"tables must be a mapping of variable to table, got {type(tables).__name__}"
        )
    _check_present(slope.index, tables, "tables")
    woe = [_woe_column(tables[variable], variable) for variable in slope.index]
    variable_codes = np.repeat(np.arange(len(woe)), [len(column) for column in woe])
    # Explicit levels keep the labels as objects; letting pandas infer an interval level
    # hashes infinite edges, which warns on NumPy < 2.2.
    labels = np.concatenate([np.asarray(column.index, dtype=object) for column in woe])
    label_codes, label_levels = pd.factorize(labels)
    index = pd.MultiIndex(
        levels=[slope.index, pd.Index(label_levels, dtype=object)],
        codes=[variable_codes, label_codes],
        names=["variable", "bin"],
    )
    values = np.concatenate([column.to_numpy(dtype=float) for column in woe])
    # The same operations as woe_to_points, so both give identical points.
    points = base - slope.to_numpy()[variable_codes] * values
    return pd.DataFrame({"woe": values, "points": points}, index=index)


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
        If `woe` is not a DataFrame or mapping, lacks a variable of
        `coefficients` or has non-finite values, the coefficients are empty,
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
        woe = pd.DataFrame(woe)
    elif not isinstance(woe, pd.DataFrame):
        raise ValueError(f"woe must be a DataFrame or a mapping, got {type(woe).__name__}")
    _check_present(slope.index, woe.columns, "woe")
    values = woe[list(slope.index)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        column = slope.index[~np.isfinite(values).all(axis=0)][0]
        raise ValueError(f"woe must be finite, got missing or infinite values in {column!r}")
    return pd.DataFrame(base - slope.to_numpy() * values, index=woe.index, columns=slope.index)


def reason_codes(points: pd.DataFrame, table: pd.DataFrame, *, top: int = 3) -> pd.DataFrame:
    """Name the variables that cost each applicant the most points.

    A variable costs an applicant the difference between the points of the
    variable's best group and the points the applicant received. Variables
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
        not a `points_table` result or lacks a variable of `points`, or
        `top` is not a positive integer.

    See Also
    --------
    points_table : The points of every group.
    woe_to_points : The points of each applicant.

    Notes
    -----
    Variables that cost the same number of points keep the order of the
    columns of `points`. Compare points that are rounded in the same way:
    the best group's points are read from `table`.

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
    """
    if not isinstance(points, pd.DataFrame):
        raise ValueError(f"points must be a DataFrame, got {type(points).__name__}")
    if isinstance(top, bool) or not isinstance(top, Integral) or top < 1:
        raise ValueError(f"top must be a positive integer, got {top!r}")
    if (
        not isinstance(table, pd.DataFrame)
        or "points" not in table.columns
        or table.index.nlevels != 2
    ):
        raise ValueError("table must be a points_table result")
    best = table["points"].groupby(level=0, sort=False).max()
    _check_present(points.columns, best.index, "table")
    values = points.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("points must be finite")
    cost = best.reindex(points.columns).to_numpy() - values
    order = np.argsort(-cost, axis=1, kind="stable")[:, :top]
    names = np.asarray(points.columns, dtype=object)[order]
    names[np.take_along_axis(cost, order, axis=1) <= 0] = None
    reasons = np.full((len(points), top), None, dtype=object)
    reasons[:, : names.shape[1]] = names
    columns = [f"reason_{rank}" for rank in range(1, top + 1)]
    return pd.DataFrame(reasons, index=points.index, columns=columns, dtype=object)


def _model_points(coefficients, intercept, pdo, base_score, base_odds) -> tuple[pd.Series, float]:
    """Return the points per unit of WOE and the base points of each variable."""
    factor, offset = score_scaling(pdo=pdo, base_score=base_score, base_odds=base_odds)
    if (
        isinstance(intercept, bool)
        or not isinstance(intercept, Real)
        or not math.isfinite(intercept)
    ):
        raise ValueError(f"intercept must be a finite number, got {intercept!r}")
    if isinstance(coefficients, Mapping):
        coefficients = pd.Series(dict(coefficients), dtype=object)
    elif not isinstance(coefficients, pd.Series):
        kind = type(coefficients).__name__
        raise ValueError(f"coefficients must be a mapping or Series by variable, got {kind}")
    if coefficients.empty:
        raise ValueError("coefficients must name at least one variable")
    if coefficients.index.has_duplicates:
        raise ValueError("coefficients must name each variable once")
    beta = pd.to_numeric(coefficients, errors="coerce").astype(float)
    invalid = ~np.isfinite(beta.to_numpy())
    if invalid.any():
        position = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            f"coefficients must be finite numbers, got {coefficients.tolist()[position]!r} "
            f"for {coefficients.index[position]!r}"
        )
    return factor * beta, (offset - factor * intercept) / len(beta)


def _woe_column(table: pd.DataFrame | pd.Series, variable: object) -> pd.Series:
    if isinstance(table, pd.DataFrame):
        if "woe" not in table.columns:
            raise ValueError(f"the table of {variable!r} must have a woe column")
        table = table["woe"]
    elif not isinstance(table, pd.Series):
        raise ValueError(
            f"the table of {variable!r} must be a woe_table result or a Series, "
            f"got {type(table).__name__}"
        )
    if not np.isfinite(table.to_numpy(dtype=float)).all():
        raise ValueError(f"the table of {variable!r} must have finite WOE values")
    return table


def _check_present(variables: pd.Index, available, name: str) -> None:
    missing = [variable for variable in variables if variable not in available]
    if missing:
        shown = ", ".join(repr(variable) for variable in missing[:5])
        raise ValueError(f"{name} has no entry for {len(missing)} variable(s): {shown}")


def _like(template: npt.ArrayLike, values: np.ndarray) -> float | pd.Series | np.ndarray:
    """Return values shaped like the input: a Series, a float or an array."""
    if isinstance(template, pd.Series):
        return pd.Series(values, index=template.index, name=template.name)
    if values.ndim == 0:
        return float(values)
    return values
