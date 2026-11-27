"""Scorecard scaling, points, reason codes and saved scorecards."""

import contextlib
import dataclasses
import json
import math
import numbers
import types
import typing
from collections import abc

import numpy as np
import numpy.typing as npt
import pandas as pd

import kivi

from . import _validation, calibration, spec

__all__ = [
    "ScorecardSpec",
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
        if not _is_finite(value):
            raise ValueError(
                f"{name} must be a finite number, got {_validation.shown_argument(value)}"
            )
    if pdo <= 0:
        raise ValueError(f"pdo must be positive, got {_validation.shown_argument(pdo)}")
    if base_odds <= 0:
        raise ValueError(f"base_odds must be positive, got {_validation.shown_argument(base_odds)}")
    # Python floats, so that a float32 argument does not round the scores to float32.
    factor = float(pdo) / math.log(2)
    return factor, float(base_score) - factor * math.log(float(base_odds))


@typing.overload
def pd_to_score(
    probability: pd.Series, *, pdo: float, base_score: float, base_odds: float
) -> pd.Series: ...
@typing.overload
def pd_to_score(
    probability: float, *, pdo: float, base_score: float, base_odds: float
) -> float: ...
@typing.overload
def pd_to_score(
    probability: _validation.ListLike, *, pdo: float, base_score: float, base_odds: float
) -> npt.NDArray[np.float64]: ...
@typing.overload
def pd_to_score(
    probability: npt.ArrayLike, *, pdo: float, base_score: float, base_odds: float
) -> float | pd.Series | npt.NDArray[np.float64]: ...
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
        If `probability` is not real and numeric or has values outside the
        open interval (0, 1), or the scaling arguments are invalid.

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
    values = _validation.as_real(probability, "probability", booleans=True)
    if np.any((values <= 0) | (values >= 1)):
        outside = float(values[(values <= 0) | (values >= 1)].flat[0])
        raise ValueError(
            f"probability must be strictly between 0 and 1, got {_validation.shown(outside)}"
        )
    return _validation.like_input(
        probability, offset + factor * (np.log1p(-values) - np.log(values))
    )


@typing.overload
def score_to_pd(
    score: pd.Series, *, pdo: float, base_score: float, base_odds: float
) -> pd.Series: ...
@typing.overload
def score_to_pd(score: float, *, pdo: float, base_score: float, base_odds: float) -> float: ...
@typing.overload
def score_to_pd(
    score: _validation.ListLike, *, pdo: float, base_score: float, base_odds: float
) -> npt.NDArray[np.float64]: ...
@typing.overload
def score_to_pd(
    score: npt.ArrayLike, *, pdo: float, base_score: float, base_odds: float
) -> float | pd.Series | npt.NDArray[np.float64]: ...
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
        If `score` is not real and numeric, or the scaling arguments are
        invalid.

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
    log_odds = (_validation.as_real(score, "score", booleans=True) - offset) / factor
    with np.errstate(invalid="ignore"):  # logaddexp warns on missing values
        probability = np.exp(-np.logaddexp(0, log_odds))
    return _validation.like_input(score, probability)


def points_table(
    tables: abc.Mapping[object, pd.DataFrame | pd.Series],
    coefficients: abc.Mapping[object, float] | pd.Series,
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
        values, the coefficients are empty, repeated or not finite numbers
        or include the intercept, or the scaling arguments are invalid.

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
    if not isinstance(tables, abc.Mapping):
        raise ValueError(
            f"tables must be a mapping of variable to table, got {type(tables).__name__}"
        )
    _check_variables(slope.index, tables, "tables")
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
    woe: pd.DataFrame | abc.Mapping[object, npt.ArrayLike],
    coefficients: abc.Mapping[object, float] | pd.Series,
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
        coefficient are ignored. The Series of a mapping must have
        identical indices; other values are matched by position.
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
        of `coefficients`, has non-finite values or is a mapping of Series
        with different indices, the coefficients are empty, repeated, not
        finite or include the intercept, or the scaling arguments are
        invalid.

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
    if isinstance(woe, abc.Mapping):
        # pandas would align the Series by label, which is not how kivi matches observations.
        _validation.shared_index(
            {f"woe[{_validation.shown(variable)}]": values for variable, values in woe.items()}
        )
        woe = pd.DataFrame(woe)  # type: ignore[arg-type]  # a mapping of columns
    elif not isinstance(woe, pd.DataFrame):
        raise ValueError(f"woe must be a DataFrame or a mapping, got {type(woe).__name__}")
    _check_variables(slope.index, woe.columns, "woe")
    repeated = woe.columns[woe.columns.duplicated() & woe.columns.isin(slope.index)]
    if len(repeated):
        raise ValueError(
            f"woe must have one column per variable, got {_validation.shown(repeated[0])} twice"
        )
    values = woe[list(slope.index)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        column = slope.index[~np.isfinite(values).all(axis=0)][0]
        raise ValueError(
            f"woe must be finite, got missing or infinite values in {_validation.shown(column)}"
        )
    return pd.DataFrame(base - slope.to_numpy() * values, index=woe.index, columns=slope.index)


def reason_codes(
    points: pd.DataFrame,
    table: pd.DataFrame,
    *,
    top: int = 3,
    atol: float = 1e-4,
    unseen_points: float | abc.Mapping[object, float] | pd.Series | None = None,
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
        If `points` is not a DataFrame, repeats a column or has non-finite
        values, `table` is not a `points_table` result, lacks a variable of
        `points` or has non-finite points for one, some points are farther
        than `atol` from every group of their variable and from its
        `unseen_points`, `top` is not a positive integer, `atol` is not a
        finite positive number, or `unseen_points` is not a finite number or
        a mapping or Series of finite numbers.

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
    _check_reason_options(top, atol)
    if not isinstance(table, pd.DataFrame) or table.index.name != "variable":  # type: ignore[redundant-expr]  # checked at runtime
        raise ValueError(f"table must be a points_table result, got {_describe(table)}")
    if "points" not in table.columns:
        raise ValueError(f"table must have a points column, got columns {table.columns.tolist()}")
    repeated = points.columns[points.columns.duplicated()]
    if len(repeated):
        raise ValueError(
            f"points must have one column per variable, got {_validation.shown(repeated[0])} twice"
        )
    groups = dict(iter(table["points"].groupby(level="variable", sort=False)))
    _check_present(points.columns, groups, "table")
    exempt = _unseen_points(unseen_points, points.columns)
    values = points.to_numpy(dtype=float)
    invalid = ~np.isfinite(values)
    if invalid.any():
        row, column = np.argwhere(invalid)[0]
        value, variable = float(values[row, column]), points.columns[column]
        raise ValueError(
            f"points must be finite, got {_validation.shown(value)} in "
            f"{_validation.shown(variable)}"
        )
    cost = np.empty_like(values)
    for column, variable in enumerate(points.columns):
        group_points = np.sort(groups[variable].to_numpy(dtype=float))
        if not np.isfinite(group_points).all():
            bad = float(group_points[~np.isfinite(group_points)][0])
            raise ValueError(
                f"table must have finite points, got {_validation.shown(bad)} for "
                f"{_validation.shown(variable)}"
            )
        nearest = _nearest(group_points, values[:, column])
        far = np.abs(values[:, column] - nearest) > atol
        unseen = exempt[column]
        if not np.isnan(unseen):
            far &= np.abs(values[:, column] - unseen) > atol
        if far.any():
            row = np.flatnonzero(far)[0]
            also = "" if np.isnan(unseen) else f" or unseen_points {float(unseen)!r}"
            raise ValueError(
                f"points of {_validation.shown(variable)} must match a group of table within "
                f"atol={_validation.shown_argument(atol)}{also}, got "
                f"{float(values[row, column])!r}, nearest group {float(nearest[row])!r}; check "
                "atol and that points and table come from the same model and scaling, and for "
                "groups that woe_transform encoded with unseen pass their points as unseen_points"
            )
        matched = np.abs(values[:, column] - nearest) <= atol
        cost[:, column] = np.where(matched, group_points[-1] - nearest, 0.0)
    return _ranked_reasons(cost, points.columns, points.index, top, atol)


def _check_reason_options(top: int, atol: float) -> None:
    """Check the number of reasons and the tolerance of `reason_codes`."""
    if isinstance(top, bool) or not _validation.is_integral(top) or top < 1:
        raise ValueError(f"top must be a positive integer, got {_validation.shown_argument(top)}")
    if isinstance(atol, bool) or not _validation.is_real(atol) or not 0 < atol < math.inf:
        raise ValueError(
            f"atol must be a finite positive number, got {_validation.shown_argument(atol)}"
        )


def _ranked_reasons(
    cost: npt.NDArray[np.float64], variables: pd.Index, index: pd.Index, top: int, atol: float
) -> pd.DataFrame:
    """Name the ``top`` costliest variables of each row; costs below about ``atol`` give None."""
    # Costs that differ by less than atol tie, so the stable sort keeps the column order.
    steps = np.round(cost / atol)
    order = np.argsort(-steps, axis=1, kind="stable")[:, :top]
    names = np.asarray(variables, dtype=object)[order]
    names[np.take_along_axis(steps, order, axis=1) <= 0] = None
    reasons = np.full((len(index), top), None, dtype=object)
    reasons[:, : names.shape[1]] = names
    columns = [f"reason_{rank}" for rank in range(1, top + 1)]
    return pd.DataFrame(reasons, index=index, columns=columns, dtype=object)


_FORMAT = "kivi.ScorecardSpec"
_FORMAT_VERSION = 1
_KEYS = (
    "format",
    "format_version",
    "kivi_version",
    "scaling",
    "binning",
    "points",
    "unseen_points",
    "calibration",
    "scale",
)
_SCALING = ("pdo", "base_score", "base_odds")
# The parameters of each kind of calibration, in the order of the JSON format.
_CALIBRATIONS = {
    "shift": ("shift",),
    "platt": ("slope", "intercept"),
    "isotonic": ("edges", "probability"),
}
_SCALE_COLUMNS = ("lower", "probability", "upper")
_Frame = typing.TypeVar("_Frame", pd.DataFrame, pd.DataFrame | None)


class _Copied(typing.Generic[_Frame]):
    """A dataclass field that keeps a DataFrame and returns a new copy on each access."""

    def __init__(self, *, optional: bool = False) -> None:
        self._optional = optional

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = f"_{name}"

    def __get__(self, instance: object, owner: type | None = None) -> _Frame:
        if instance is None:
            # The default of the field: None if it is optional, and otherwise no default.
            if self._optional:
                return typing.cast(_Frame, None)
            raise AttributeError(self._name)
        value = instance.__dict__[self._name]
        return typing.cast(_Frame, None if value is None else value.copy())

    def __set__(self, instance: object, value: _Frame) -> None:
        instance.__dict__[self._name] = value


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class ScorecardSpec:
    r"""A whole scorecard, to score, calibrate and grade any sample the same way.

    A scorecard specification holds what scoring in production needs: the
    `kivi.spec.BinningSpec` that groups and encodes each characteristic,
    the points of every group, the scaling of the score, the points of
    values whose group was not in development, and optionally a
    calibration of the probability and a master scale. `points`, `score`,
    `probability`, `grade` and `reason_codes` apply it to any sample, and
    `to_json` and `from_json` save and load it. Build a specification with
    `from_model` from a logistic regression on the WOE values, or with the
    constructor from a points table whose points were rounded or adjusted.

    A specification is immutable. ``table`` and ``scale`` return a new copy
    on each access, and ``unseen_points`` and ``calibration`` are read-only
    mappings. To change a field, use `dataclasses.replace`, which validates
    the result again.

    Parameters
    ----------
    binning : kivi.spec.BinningSpec
        Binning and WOE of the characteristics of the scorecard, and of no
        other column, such as the result of `kivi.spec.BinningSpec.select`.
    table : pandas.DataFrame
        Points of every group, as returned by `points_table` for the tables
        of `binning`: indexed by ``variable``, with the columns ``bin``,
        ``woe`` and ``points``. The points may be rounded or changed, but
        ``bin`` and ``woe`` must equal the groups and the WOE of `binning`
        row for row, with the variables in the order of its columns.
    pdo : float
        Points to double the odds. Must be positive.
    base_score : float
        Score at the base odds.
    base_odds : float
        Good:bad odds at the base score. Must be positive.
    unseen_points : mapping or pandas.Series, optional
        Points of values whose group is not in their column's table, for
        each column whose unseen policy in `binning` is a WOE, and only for
        those. A column whose policy is ``"raise"`` raises an error for such
        values, and a column whose policy names a group gives them the
        points of that group.
    calibration : mapping, optional
        Calibration of `probability`, one of ``{"kind": "shift", "shift":
        shift}``, ``{"kind": "platt", "slope": slope, "intercept":
        intercept}`` and ``{"kind": "isotonic", "edges": edges,
        "probability": probability}``; see the Notes. By default the
        probability is the one that the scaling gives the score.
    scale : pandas.DataFrame, optional
        Master scale of `grade`, as returned by
        `kivi.calibration.master_scale`.
    kivi_version : str, optional
        Version of kivi that built the specification. By default, or if
        None, the installed version.

    Attributes
    ----------
    FORMAT_VERSION : int
        Version of the JSON format that `to_json` writes and the newest
        that `from_json` reads.

    Raises
    ------
    ValueError
        If `binning` is not a `kivi.spec.BinningSpec`; if `table` is not a
        `points_table` result with one row per group of every column of
        `binning`, in its order, and the groups and WOE of `binning`, or has
        points that are not finite numbers; if the scaling arguments are
        invalid; if `unseen_points` lacks a column whose unseen policy is a
        WOE, has another column or has points that are not finite numbers;
        if `calibration` is not one of the three kinds with finite
        parameters, a slope that is not negative and, for ``"isotonic"``,
        edges from ``-inf`` to ``inf`` and probabilities between 0 and 1
        that do not rise with the score; if `scale` is not a master scale
        with the columns ``lower``, ``probability`` and ``upper`` and
        labels that are strings, integers, finite floats or booleans; or if
        `kivi_version` is not a string.

    See Also
    --------
    from_model : Build the specification from a logistic regression.
    points_table : The points of every group.
    kivi.spec.BinningSpec : The binning, which the specification includes.

    Notes
    -----
    With the factor and offset of `score_scaling`, a score :math:`s` stands
    for the log odds of the event

    .. math::

       \ell = \frac{\mathrm{offset} - s}{\mathrm{factor}},

    which `probability` computes from the score directly, so that it does
    not round a PD near 0 or 1 first. Without calibration, the probability
    is :math:`\sigma(\ell) = 1 / (1 + e^{-\ell})`, the result of
    `score_to_pd`. The calibrations give

    * ``"shift"``: :math:`\sigma(\ell + \Delta)`, for the shift
      :math:`\Delta` of `kivi.calibration.central_tendency_shift` fitted to
      the uncalibrated probabilities;
    * ``"platt"``: :math:`\sigma(a \ell + b)`, for the slope :math:`a` and
      intercept :math:`b` of `kivi.calibration.platt_fit` or
      `kivi.calibration.quasi_moment_fit` fitted to the log odds
      :math:`\ell`, as `kivi.calibration.platt_predict` gives;
    * ``"isotonic"``: the probability of the band of the score, as
      `kivi.calibration.isotonic_predict` gives for the edges and
      probabilities of `kivi.calibration.isotonic_fit` fitted to the scores
      with ``increasing=False``, because a higher score means a lower risk.

    The specification stores points rather than coefficients: the points
    table is the scorecard that production uses, and once its points are
    rounded, the probability follows from the rounded score, as in
    production. Feature definitions such as those of
    `kivi.features.FeatureSpec` are not part of it, because the features are
    computed from other tables before the scorecard scores them; save the
    two specifications side by side.

    Two scorecard specifications are equal when `to_json` gives the same
    text for both, which includes `kivi_version`.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.scorecard import ScorecardSpec
    >>> from kivi.spec import fit_spec
    >>> frame = pd.DataFrame({
    ...     "age": [23, 31, 45, 52, 38, 27, 61, 49],
    ...     "housing": ["yes", "no", "yes", "no", "no", "yes", "no", "yes"],
    ... })
    >>> target = pd.Series([1, 0, 0, 0, 1, 1, 0, 0])
    >>> binning = fit_spec(frame, target, bins=2, unseen={"housing": 0.0})
    >>> card = ScorecardSpec.from_model(
    ...     binning, {"age": 0.8, "housing": 0.5}, -1.1, pdo=20, base_score=600, base_odds=50,
    ...     decimals=0,
    ... )
    >>> card
    ScorecardSpec(columns=['age', 'housing'])
    >>> card.table
                       bin       woe  points
    variable
    age       (-inf, 41.5]  1.455287   226.0
    age        (41.5, inf] -1.722767   299.0
    housing            yes  0.510826   252.0
    housing             no -0.587787   268.0

    A housing status that development did not have gets the WOE 0, whose
    points are kept with the table:

    >>> dict(card.unseen_points)
    {'housing': 259.0}
    >>> card.score(pd.DataFrame({"age": [30, 70], "housing": ["rented", "yes"]}))
    0    485.0
    1    551.0
    Name: score, dtype: float64
    """

    binning: spec.BinningSpec
    table: _Copied[pd.DataFrame] = _Copied()
    _: dataclasses.KW_ONLY
    pdo: float
    base_score: float
    base_odds: float
    unseen_points: abc.Mapping[str, float] | None = None
    calibration: abc.Mapping[str, typing.Any] | None = None
    scale: _Copied[pd.DataFrame | None] = _Copied(optional=True)
    kivi_version: str | None = None
    FORMAT_VERSION: typing.ClassVar[int] = _FORMAT_VERSION

    def __post_init__(self) -> None:
        """Validate the fields and replace them with read-only copies."""
        binning = self.binning
        if not isinstance(binning, spec.BinningSpec):
            raise ValueError(
                f"binning must be a BinningSpec, such as a fit_spec result, got "
                f"{type(binning).__name__}"
            )
        scaling = score_scaling(pdo=self.pdo, base_score=self.base_score, base_odds=self.base_odds)
        with _fitting_floats("table"):
            points = _checked_points(self.__dict__["_table"], binning)
        unseen = _checked_unseen_points(self.unseen_points, binning)
        with _fitting_floats("calibration"):
            fitted = _checked_calibration(self.calibration)
        scale = self.__dict__["_scale"]
        if scale is not None:
            with _fitting_floats("scale"):
                scale = _checked_scale(scale)
        version = kivi.__version__ if self.kivi_version is None else self.kivi_version
        if not isinstance(version, str):
            raise ValueError(
                f"kivi_version must be a string, got {_validation.shown_argument(version)}"
            )
        # Adding 0.0 turns -0.0 into 0.0 throughout, so that specifications that score alike
        # have the same JSON text, and so compare equal and hash alike.
        for name in _SCALING:
            object.__setattr__(self, name, float(getattr(self, name)) + 0.0)
        object.__setattr__(self, "table", _points_frame(binning, points))
        object.__setattr__(
            self, "unseen_points", None if unseen is None else types.MappingProxyType(unseen)
        )
        object.__setattr__(
            self, "calibration", None if fitted is None else types.MappingProxyType(fitted)
        )
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "kivi_version", version)
        object.__setattr__(self, "_scaling", scaling)
        object.__setattr__(self, "_points", points)
        object.__setattr__(self, "_unseen", unseen or {})
        # The binning with the position of each group in place of its WOE, so that its
        # woe_transform finds the group of every value as BinningSpec.woe_transform does.
        object.__setattr__(self, "_positions", _positions(binning))

    def __repr__(self) -> str:
        """Name the columns, which is shorter than listing every field."""
        return f"{type(self).__name__}(columns={list(self.binning.columns)!r})"

    def __eq__(self, other: object) -> bool:
        """Compare the JSON text."""
        if not isinstance(other, ScorecardSpec):
            return NotImplemented
        return self.to_json() == other.to_json()

    def __hash__(self) -> int:
        """Hash the JSON text, consistently with equality."""
        return hash(self.to_json())

    def __reduce__(
        self,
    ) -> tuple[
        abc.Callable[[abc.Mapping[str, typing.Any]], "ScorecardSpec"], tuple[dict[str, typing.Any]]
    ]:
        """Pickle through the dict format, which validates the fields again."""
        return (type(self).from_dict, (self.to_dict(),))

    @classmethod
    def from_model(
        cls,
        binning: spec.BinningSpec,
        coefficients: abc.Mapping[object, float] | pd.Series,
        intercept: float,
        *,
        pdo: float,
        base_score: float,
        base_odds: float,
        decimals: int | None = None,
        calibration: abc.Mapping[str, typing.Any] | None = None,
        scale: pd.DataFrame | None = None,
    ) -> "ScorecardSpec":
        """Build the specification of a logistic regression on the WOE values of a binning.

        The specification keeps the columns of `binning` that have a
        coefficient, in the order of `coefficients`, and the points that
        `points_table` gives their groups. The points of values in groups
        that development did not have, for the columns whose unseen policy
        is a WOE, are those that `woe_to_points` gives that WOE.

        Parameters
        ----------
        binning : kivi.spec.BinningSpec
            Binning and WOE of every column of the model, and possibly of
            other columns.
        coefficients : mapping or pandas.Series
            Logistic regression coefficient of each column's WOE, for a model
            of the event probability, without the intercept.
        intercept : float
            Intercept of the same model.
        pdo : float
            Points to double the odds. Must be positive.
        base_score : float
            Score at the base odds.
        base_odds : float
            Good:bad odds at the base score. Must be positive.
        decimals : int, optional
            Number of decimals to round the points and the unseen points to,
            with pandas ``round``, which rounds halves to even. By default
            the points are not rounded.
        calibration : mapping, optional
            Calibration of the probability, as for the constructor.
        scale : pandas.DataFrame, optional
            Master scale of `grade`, as for the constructor.

        Returns
        -------
        ScorecardSpec
            The specification of the model's columns.

        Raises
        ------
        ValueError
            If `binning` is not a `kivi.spec.BinningSpec` or lacks a column
            of `coefficients`, `decimals` is not an integer, or for the
            reasons that `points_table` and the constructor give.

        See Also
        --------
        points_table : The points of every group.
        woe_to_points : The points of each applicant from its WOE values.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.scorecard import ScorecardSpec
        >>> from kivi.spec import fit_spec
        >>> frame = pd.DataFrame({
        ...     "age": [23, 31, 45, 52, 38, 27, 61, 49],
        ...     "housing": ["yes", "no", "yes", "no", "no", "yes", "no", "yes"],
        ... })
        >>> binning = fit_spec(frame, [1, 0, 0, 0, 1, 1, 0, 0], bins=2)
        >>> card = ScorecardSpec.from_model(
        ...     binning, {"age": 0.8}, -1.1, pdo=20, base_score=600, base_odds=50
        ... )
        >>> card.table.round(2)
                       bin   woe  points
        variable
        age       (-inf, 41.5]  1.46  485.27
        age        (41.5, inf] -1.72  558.63
        """
        if not isinstance(binning, spec.BinningSpec):
            raise ValueError(
                f"binning must be a BinningSpec, such as a fit_spec result, got "
                f"{type(binning).__name__}"
            )
        if decimals is not None and (
            isinstance(decimals, (bool, np.bool_)) or not _validation.is_integral(decimals)
        ):
            raise ValueError(
                f"decimals must be an integer, got {_validation.shown_argument(decimals)}"
            )
        scaling = {"pdo": pdo, "base_score": base_score, "base_odds": base_odds}
        slope, _ = _model_points(coefficients, intercept, pdo, base_score, base_odds)
        _check_variables(slope.index, binning.columns, "binning")
        selected = binning.select(list(slope.index))
        tables: dict[object, pd.DataFrame] = {
            name: selected.tables[name] for name in selected.columns
        }
        table = points_table(tables, coefficients, intercept, **scaling)
        unseen: dict[str, float] | None = None
        woe_columns = _woe_policies(selected)
        if woe_columns:
            # The points of the unseen WOE from woe_to_points itself, so that both agree exactly.
            row = pd.DataFrame({name: [woe_columns.get(name, 0.0)] for name in selected.columns})
            points = woe_to_points(row, coefficients, intercept, **scaling).iloc[0]
            unseen_series = points[list(woe_columns)]
            if decimals is not None:
                unseen_series = unseen_series.round(int(decimals))
            unseen = {str(name): float(value) for name, value in unseen_series.items()}
        if decimals is not None:
            table["points"] = table["points"].round(int(decimals))
        return cls(
            selected,
            table,
            pdo=pdo,
            base_score=base_score,
            base_odds=base_odds,
            unseen_points=unseen,
            calibration=calibration,
            scale=scale,
        )

    def points(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Give each row of a sample the points of its group in every column.

        Each column is grouped as by `kivi.spec.BinningSpec.woe_transform`
        and each group gets its points in `table`. A value whose group is not
        in its column's table gets the points of the column's unseen group
        or its `unseen_points`, or raises an error if the column's unseen
        policy is ``"raise"``, as `kivi.spec.BinningSpec.woe_transform`
        does.

        Parameters
        ----------
        frame : pandas.DataFrame
            Sample that has every column of the scorecard. Other columns are
            ignored.

        Returns
        -------
        pandas.DataFrame
            Float points of each column of the scorecard, in its order, with
            the index of `frame`.

        Raises
        ------
        ValueError
            For the reasons that `kivi.spec.BinningSpec.woe_transform`
            gives, such as a missing column or a value whose group is not in
            the table of a column with the unseen policy ``"raise"``.

        See Also
        --------
        score : The sum of the points.
        woe_to_points : The points of WOE values.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.scorecard import ScorecardSpec
        >>> from kivi.spec import fit_spec
        >>> frame = pd.DataFrame({
        ...     "age": [23, 31, 45, 52, 38, 27, 61, 49],
        ...     "housing": ["yes", "no", "yes", "no", "no", "yes", "no", "yes"],
        ... })
        >>> target = pd.Series([1, 0, 0, 0, 1, 1, 0, 0])
        >>> binning = fit_spec(frame, target, bins=2, unseen={"housing": 0.0})
        >>> card = ScorecardSpec.from_model(
        ...     binning, {"age": 0.8, "housing": 0.5}, -1.1, pdo=20, base_score=600, base_odds=50,
        ...     decimals=0,
        ... )
        >>> card.points(pd.DataFrame({"age": [30, 70], "housing": ["no", "rented"]}))
             age  housing
        0  226.0    268.0
        1  299.0    259.0
        """
        return pd.DataFrame(self._point_columns(frame), index=frame.index)

    def score(self, frame: pd.DataFrame) -> pd.Series:
        """Score each row of a sample.

        The score of a row is the sum of its `points`, added column by
        column in the order of the scorecard rather than by a matrix
        product, so that rows with the same groups get exactly the same
        score on every processor.

        Parameters
        ----------
        frame : pandas.DataFrame
            Sample that has every column of the scorecard. Other columns are
            ignored.

        Returns
        -------
        pandas.Series
            Float scores named ``score``, with the index of `frame`.

        Raises
        ------
        ValueError
            For the reasons that `points` gives.

        See Also
        --------
        points : The points of each column.
        probability : The probability of the event of each row.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.scorecard import ScorecardSpec
        >>> from kivi.spec import fit_spec
        >>> frame = pd.DataFrame({
        ...     "age": [23, 31, 45, 52, 38, 27, 61, 49],
        ...     "housing": ["yes", "no", "yes", "no", "no", "yes", "no", "yes"],
        ... })
        >>> target = pd.Series([1, 0, 0, 0, 1, 1, 0, 0])
        >>> binning = fit_spec(frame, target, bins=2)
        >>> card = ScorecardSpec.from_model(
        ...     binning, {"age": 0.8, "housing": 0.5}, -1.1, pdo=20, base_score=600, base_odds=50,
        ...     decimals=0,
        ... )
        >>> card.score(frame.head(3))
        0    478.0
        1    494.0
        2    551.0
        Name: score, dtype: float64
        """
        columns = self._point_columns(frame)
        total = np.zeros(len(frame))
        for values in columns.values():
            total += values
        return pd.Series(total, index=frame.index, name="score")

    def probability(self, frame: pd.DataFrame) -> pd.Series:
        """Give each row of a sample the probability of the event, such as its PD.

        The probability follows from the `score` of the row through the
        scaling and, if the specification has one, its calibration; see the
        Notes of `ScorecardSpec`.

        Parameters
        ----------
        frame : pandas.DataFrame
            Sample that has every column of the scorecard. Other columns are
            ignored.

        Returns
        -------
        pandas.Series
            Probabilities named ``probability``, with the index of `frame`.

        Raises
        ------
        ValueError
            For the reasons that `points` gives.

        See Also
        --------
        score : The score of each row.
        grade : The grade of each probability on the master scale.
        score_to_pd : The probability of a score without calibration.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.scorecard import ScorecardSpec
        >>> from kivi.spec import fit_spec
        >>> frame = pd.DataFrame({
        ...     "age": [23, 31, 45, 52, 38, 27, 61, 49],
        ...     "housing": ["yes", "no", "yes", "no", "no", "yes", "no", "yes"],
        ... })
        >>> target = pd.Series([1, 0, 0, 0, 1, 1, 0, 0])
        >>> binning = fit_spec(frame, target, bins=2)
        >>> scaling = dict(pdo=20, base_score=600, base_odds=50)
        >>> card = ScorecardSpec.from_model(binning, {"age": 0.8, "housing": 0.5}, -1.1, **scaling)
        >>> card.probability(frame.head(3)).round(4)
        0    0.5792
        1    0.4428
        2    0.0977
        Name: probability, dtype: float64

        With a shift of the log odds, such as one from
        `kivi.calibration.central_tendency_shift`:

        >>> shifted = ScorecardSpec.from_model(
        ...     binning, {"age": 0.8, "housing": 0.5}, -1.1, **scaling,
        ...     calibration={"kind": "shift", "shift": -0.5},
        ... )
        >>> shifted.probability(frame.head(3)).round(4)
        0    0.4550
        1    0.3253
        2    0.0616
        Name: probability, dtype: float64
        """
        score = self.score(frame)
        return pd.Series(self._calibrated(score.to_numpy()), index=frame.index, name="probability")

    def grade(self, frame: pd.DataFrame) -> pd.Series:
        """Give each row of a sample the grade of its probability on the master scale.

        The grades are those that `kivi.calibration.assign_grades` gives the
        `probability` of each row on `scale`.

        Parameters
        ----------
        frame : pandas.DataFrame
            Sample that has every column of the scorecard. Other columns are
            ignored.

        Returns
        -------
        pandas.Series
            Grades named ``grade``, an ordered categorical of the labels of
            the scale from the lowest risk to the highest, with the index of
            `frame`.

        Raises
        ------
        ValueError
            If the specification has no master scale, or for the reasons
            that `points` gives.

        See Also
        --------
        probability : The probability of each row.
        kivi.calibration.master_scale : Build a master scale.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.calibration import master_scale
        >>> from kivi.scorecard import ScorecardSpec
        >>> from kivi.spec import fit_spec
        >>> frame = pd.DataFrame({
        ...     "age": [23, 31, 45, 52, 38, 27, 61, 49],
        ...     "housing": ["yes", "no", "yes", "no", "no", "yes", "no", "yes"],
        ... })
        >>> target = pd.Series([1, 0, 0, 0, 1, 1, 0, 0])
        >>> binning = fit_spec(frame, target, bins=2)
        >>> card = ScorecardSpec.from_model(
        ...     binning, {"age": 0.8, "housing": 0.5}, -1.1, pdo=20, base_score=600, base_odds=50,
        ...     scale=master_scale([0.05, 0.2, 0.5]),
        ... )
        >>> card.grade(frame.head(3))
        0    3
        1    3
        2    1
        Name: grade, dtype: category
        Categories (3, int64): [1 < 2 < 3]
        """
        scale = self.__dict__["_scale"]
        if scale is None:
            raise ValueError(
                "grade needs a master scale; build the specification with scale, such as a "
                "master_scale result"
            )
        grades = calibration.assign_grades(self.probability(frame), scale)
        return grades.rename("grade")

    def reason_codes(
        self, frame: pd.DataFrame, *, top: int = 3, atol: float = 1e-4
    ) -> pd.DataFrame:
        """Name the columns that cost each row of a sample the most points.

        A column costs a row the difference between the points of the
        column's best group and the points of the row's group, and the
        columns are listed from the largest cost, as by the module function
        `reason_codes`. A value whose group was not in development costs
        nothing in a column with `unseen_points`, even when those points
        equal the points of a group, so the column is not its reason.

        Parameters
        ----------
        frame : pandas.DataFrame
            Sample that has every column of the scorecard. Other columns are
            ignored.
        top : int, default 3
            Number of reasons per row.
        atol : float, default 1e-4
            Tolerance in points: costs that differ by less than about `atol`
            count as equal and keep the order of the columns, and a cost
            below about `atol` is no cost.

        Returns
        -------
        pandas.DataFrame
            Column names in columns ``reason_1`` to ``reason_<top>``, with
            the index of `frame`. A row with fewer costly columns has None
            in the remaining columns.

        Raises
        ------
        ValueError
            If `top` is not a positive integer or `atol` is not a finite
            positive number, or for the reasons that `points` gives.

        See Also
        --------
        points : The points of each column.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.scorecard import ScorecardSpec
        >>> from kivi.spec import fit_spec
        >>> frame = pd.DataFrame({
        ...     "age": [23, 31, 45, 52, 38, 27, 61, 49],
        ...     "housing": ["yes", "no", "yes", "no", "no", "yes", "no", "yes"],
        ... })
        >>> target = pd.Series([1, 0, 0, 0, 1, 1, 0, 0])
        >>> binning = fit_spec(frame, target, bins=2, unseen={"housing": 0.0})
        >>> card = ScorecardSpec.from_model(
        ...     binning, {"age": 0.8, "housing": 0.5}, -1.1, pdo=20, base_score=600, base_odds=50,
        ...     decimals=0,
        ... )
        >>> new = pd.DataFrame({"age": [30, 70, 30], "housing": ["no", "yes", "rented"]})
        >>> card.reason_codes(new, top=2)
          reason_1 reason_2
        0      age     None
        1  housing     None
        2      age     None
        """
        _check_reason_options(top, atol)
        positions = self._group_positions(frame)
        names = self.binning.columns
        cost = np.empty((len(frame), len(names)))
        for column, (name, points) in enumerate(zip(names, self.__dict__["_points"], strict=True)):
            position = positions[name]
            # A seen group costs what reason_codes gives it; an unseen value costs nothing.
            own = points[np.maximum(position, 0)]
            cost[:, column] = np.where(position >= 0, points.max() - own, 0.0)
        return _ranked_reasons(cost, names, frame.index, top, atol)

    def to_dict(self) -> dict[str, typing.Any]:
        """Return the scorecard specification as a dict of JSON types.

        The result holds only dicts, lists, strings, numbers, booleans and
        None, so ``json.dumps`` can write it without further conversion. Its
        keys are always in the same order. `to_json` writes this dict.

        Returns
        -------
        dict
            The keys ``format``, ``format_version``, ``kivi_version``,
            ``scaling``, ``binning``, ``points``, ``unseen_points``,
            ``calibration`` and ``scale``, with None for an unset field. See
            the user guide for the format.

        See Also
        --------
        from_dict : Rebuild the scorecard specification.
        to_json : Write the dict as JSON.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.scorecard import ScorecardSpec
        >>> from kivi.spec import fit_spec
        >>> frame = pd.DataFrame({"age": [23, 31, 45, 52, 38, 27, 61, 49]})
        >>> binning = fit_spec(frame, [1, 0, 0, 0, 1, 1, 0, 0], bins=2)
        >>> card = ScorecardSpec.from_model(
        ...     binning, {"age": 0.8}, -1.1, pdo=20, base_score=600, base_odds=50, decimals=0
        ... )
        >>> data = card.to_dict()
        >>> list(data)
        ['format', 'format_version', 'kivi_version', 'scaling', 'binning', 'points',
         'unseen_points', 'calibration', 'scale']
        >>> data["points"]
        {'age': [485.0, 559.0]}
        """
        return {
            "format": _FORMAT,
            "format_version": _FORMAT_VERSION,
            "kivi_version": self.kivi_version,
            "scaling": {name: getattr(self, name) for name in _SCALING},
            "binning": self.binning.to_dict(),
            "points": {
                name: values.tolist()
                for name, values in zip(self.binning.columns, self.__dict__["_points"], strict=True)
            },
            "unseen_points": None if self.unseen_points is None else dict(self.unseen_points),
            "calibration": _calibration_json(self.calibration),
            "scale": _scale_json(self.__dict__["_scale"]),
        }

    @classmethod
    def from_dict(cls, data: abc.Mapping[str, typing.Any]) -> "ScorecardSpec":
        """Rebuild a scorecard specification from the result of `to_dict`.

        The dict is checked before the specification is built, so that a
        damaged or hand-edited file fails with a message that names the
        problem instead of scoring with wrong values.

        Parameters
        ----------
        data : mapping
            A dict in the format of `to_dict`, for example from
            ``json.load``. Every key that `to_dict` writes must be present.

        Returns
        -------
        ScorecardSpec
            The scorecard specification, equal to the one that wrote `data`.

        Raises
        ------
        ValueError
            If `data` is not a kivi scorecard specification, has a
            ``format_version`` newer than this version of kivi reads, lacks a
            key or has an unknown one, or holds an invalid value, including
            an invalid binning for `kivi.spec.BinningSpec.from_dict`.

        See Also
        --------
        to_dict : The dict format.
        from_json : Read the scorecard specification from JSON text.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.scorecard import ScorecardSpec
        >>> from kivi.spec import fit_spec
        >>> binning = fit_spec(pd.DataFrame({"age": [23, 31, 45, 52]}), [1, 0, 0, 1], bins=2)
        >>> card = ScorecardSpec.from_model(
        ...     binning, {"age": 0.8}, -1.1, pdo=20, base_score=600, base_odds=50
        ... )
        >>> ScorecardSpec.from_dict(card.to_dict()) == card
        True
        """
        if not isinstance(data, abc.Mapping):
            raise ValueError(f"data must be a mapping, got {type(data).__name__}")
        if data.get("format") != _FORMAT:
            raise ValueError(
                f"data is not a kivi scorecard specification: format must be {_FORMAT!r}, "
                f"got {_validation.shown(data.get('format'))}"
            )
        version = data.get("format_version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError(
                f"format_version must be a positive integer, got {_validation.shown(version)}"
            )
        if version > _FORMAT_VERSION:
            raise ValueError(
                f"format_version {version} is newer than {_FORMAT_VERSION}, the newest that "
                f"kivi {kivi.__version__} reads; upgrade kivi to load this specification"
            )
        _check_keys(data, set(_KEYS), "the specification")
        if not isinstance(data["kivi_version"], str):
            raise ValueError(
                f"kivi_version must be a string, got {_validation.shown(data['kivi_version'])}"
            )
        scaling = data["scaling"]
        if not isinstance(scaling, abc.Mapping):
            raise ValueError(f"scaling must be a mapping, got {_json_kind(scaling)}")
        _check_keys(scaling, set(_SCALING), "scaling")
        for name in _SCALING:
            _json_number(scaling[name], f"{name} of scaling")
        try:
            binning = spec.BinningSpec.from_dict(data["binning"])
        except ValueError as error:
            raise ValueError(f"binning: {error}") from error
        table = _points_frame(binning, _decoded_points(data["points"], binning))
        return cls(
            binning,
            table,
            **{name: scaling[name] for name in _SCALING},
            unseen_points=_decoded_unseen_points(data["unseen_points"]),
            calibration=_decoded_calibration(data["calibration"]),
            scale=_decoded_scale(data["scale"]),
            kivi_version=data["kivi_version"],
        )

    def to_json(self, *, indent: int | None = None) -> str:
        """Return the scorecard specification as JSON text.

        The text is the dict of `to_dict` written by ``json.dumps``. The same
        scorecard specification always gives the same text, so saved
        specifications can be compared and kept under version control.
        Non-ASCII labels are written as they are, so save the text as UTF-8.

        Parameters
        ----------
        indent : int, optional
            Indentation of nested values, as in ``json.dumps``. By default
            the text is a single line.

        Returns
        -------
        str
            JSON text that `from_json` reads back into an equal scorecard
            specification.

        See Also
        --------
        from_json : Read the scorecard specification back.
        to_dict : The same content as a dict.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.scorecard import ScorecardSpec
        >>> from kivi.spec import fit_spec
        >>> frame = pd.DataFrame({"age": [23, 31, 45, 52, 38, 27, 61, 49]})
        >>> binning = fit_spec(frame, [1, 0, 0, 0, 1, 1, 0, 0], bins=2)
        >>> card = ScorecardSpec.from_model(
        ...     binning, {"age": 0.8}, -1.1, pdo=20, base_score=600, base_odds=50, decimals=0
        ... )
        >>> print(card.to_json(indent=2))
        {
          "format": "kivi.ScorecardSpec",
          "format_version": 1,
          "kivi_version": "...",
          "scaling": {
            "pdo": 20.0,
            "base_score": 600.0,
            "base_odds": 50.0
          },
          "binning": {
            "format": "kivi.BinningSpec",
        ...
          },
          "points": {
            "age": [
              485.0,
              559.0
            ]
          },
          "unseen_points": null,
          "calibration": null,
          "scale": null
        }

        Write the text to a file with
        ``Path("scorecard.json").write_text(text, encoding="utf-8")``.
        """
        if indent is not None:
            return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=indent)
        # The specification is frozen, so the compact text can be kept for equality and hashing.
        text = self.__dict__.get("_compact_json")
        if text is None:
            text = json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)
            object.__setattr__(self, "_compact_json", text)
        return text

    @classmethod
    def from_json(cls, text: str | bytes) -> "ScorecardSpec":
        """Read a scorecard specification from the JSON text of `to_json`.

        The loaded specification scores, calibrates and grades every sample
        exactly as the one that was saved. A file written by a newer version
        of the format is rejected rather than read in part, and so is a JSON
        object that repeats a key, which ``json.loads`` would otherwise read
        as its last value.

        Parameters
        ----------
        text : str or bytes
            JSON text, for example from ``Path(...).read_text(encoding="utf-8")``.

        Returns
        -------
        ScorecardSpec
            The scorecard specification, equal to the one that wrote `text`.

        Raises
        ------
        ValueError
            If `text` is neither a str nor bytes, is not valid JSON, repeats
            a key of an object, contains ``NaN`` or ``Infinity``, or does not
            describe a valid specification; see `from_dict`.

        See Also
        --------
        to_json : Write the scorecard specification.
        from_dict : Rebuild the scorecard specification from a dict.

        Examples
        --------
        >>> import pandas as pd
        >>> from kivi.scorecard import ScorecardSpec
        >>> from kivi.spec import fit_spec
        >>> frame = pd.DataFrame({"age": [23, 31, 45, 52]})
        >>> binning = fit_spec(frame, [1, 0, 0, 1], bins=2)
        >>> card = ScorecardSpec.from_model(
        ...     binning, {"age": 0.8}, -1.1, pdo=20, base_score=600, base_odds=50
        ... )
        >>> loaded = ScorecardSpec.from_json(card.to_json())
        >>> loaded == card
        True
        >>> loaded.score(frame).equals(card.score(frame))
        True
        """
        if not isinstance(text, (str, bytes, bytearray)):
            raise ValueError(
                f"text must be JSON text as a str or bytes, got {type(text).__name__}; read a "
                'file first, for example with Path(...).read_text(encoding="utf-8")'
            )
        data = json.loads(
            text,
            object_pairs_hook=_validation.unique_keys,
            parse_constant=_validation.reject_constant,
        )
        return cls.from_dict(data)

    def _group_positions(self, frame: pd.DataFrame) -> dict[str, npt.NDArray[np.intp]]:
        """Return the row of each value's group in its column's table, -1 for an unseen value.

        Only columns with unseen points have unseen values; the other policies raise or name
        a group, as in BinningSpec.woe_transform.
        """
        positions = self.__dict__["_positions"].woe_transform(frame)
        return {
            name: positions[name].to_numpy(dtype=float).astype(np.intp)
            for name in self.binning.columns
        }

    def _point_columns(self, frame: pd.DataFrame) -> dict[str, npt.NDArray[np.float64]]:
        """Return the points of every column, by the position of each value's group."""
        positions = self._group_positions(frame)
        unseen: dict[str, float] = self.__dict__["_unseen"]
        columns = {}
        for name, points in zip(self.binning.columns, self.__dict__["_points"], strict=True):
            # The position -1 stands for an unseen value of a column with unseen points.
            lookup = np.append(points, unseen.get(name, np.nan))
            columns[name] = lookup[positions[name]]
        return columns

    def _calibrated(self, score: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the probability of each score, calibrated if the specification has one."""
        fitted = self.calibration
        if fitted is not None and fitted["kind"] == "isotonic":
            return calibration.isotonic_predict(score, fitted["edges"], fitted["probability"])
        factor, offset = self.__dict__["_scaling"]
        log_odds = (offset - score) / factor
        slope, intercept = 1.0, 0.0
        if fitted is not None and fitted["kind"] == "shift":
            intercept = fitted["shift"]
        elif fitted is not None:
            slope, intercept = fitted["slope"], fitted["intercept"]
        return calibration.platt_predict(log_odds, slope, intercept)


def _policy(policy: object) -> typing.Literal["raise", "group", "woe"]:
    """Name the kind of an unseen policy of a BinningSpec column."""
    if isinstance(policy, str):
        return "raise"
    return "group" if isinstance(policy, abc.Mapping) else "woe"


def _woe_policies(binning: spec.BinningSpec) -> dict[str, float]:
    """Return the unseen WOE of each column whose unseen policy is a WOE."""
    return {
        name: float(typing.cast(float, binning.unseen[name]))
        for name in binning.columns
        if _policy(binning.unseen[name]) == "woe"
    }


def _positions(binning: spec.BinningSpec) -> spec.BinningSpec:
    """Return the binning with the row position of each group in place of its WOE.

    A column whose unseen policy is a WOE gets the position -1 for unseen groups; the other
    policies stay, so that a policy naming a group gives that group's position.
    """
    tables, unseen = {}, {}
    for name in binning.columns:
        table = binning.tables[name]
        tables[name] = table.assign(woe=np.arange(len(table), dtype=float))
        policy = binning.unseen[name]
        unseen[name] = -1.0 if _policy(policy) == "woe" else policy
    return dataclasses.replace(binning, tables=tables, unseen=unseen)


def _checked_points(table: object, binning: spec.BinningSpec) -> list[npt.NDArray[np.float64]]:
    """Check a points table against the binning and return the points of each column."""
    if not isinstance(table, pd.DataFrame) or table.index.name != "variable":
        raise ValueError(f"table must be a points_table result, got {_describe(table)}")
    if len(table.columns) != 3 or set(table.columns) != {"bin", "woe", "points"}:
        raise ValueError(
            "table must have the columns 'bin', 'woe' and 'points' of a points_table result, "
            f"got {table.columns.tolist()}"
        )
    names = list(binning.columns)
    fitted = [binning.tables[name] for name in names]
    variables = table.index.tolist()
    unknown = [variable for variable in dict.fromkeys(variables) if variable not in names]
    if unknown:
        raise ValueError(
            f"table has rows for variables that binning lacks: {_validation.shown_list(unknown)}"
        )
    absent = [name for name in names if name not in variables]
    if absent:
        raise ValueError(
            f"table has no rows for the columns {_validation.shown_list(absent)} of binning; "
            "keep only the columns of the model with BinningSpec.select"
        )
    sizes = [len(groups) for groups in fitted]
    if variables != [name for name, size in zip(names, sizes, strict=True) for _ in range(size)]:
        for name, size in zip(names, sizes, strict=True):
            count = variables.count(name)
            if count != size:
                raise ValueError(
                    f"table has {count} row(s) for {_validation.shown(name)}, whose table in "
                    f"binning has {size}"
                )
        raise ValueError(
            "table must list the groups of each variable together, in the order of the columns "
            f"of binning {_validation.shown_list(names)}"
        )
    labels = table["bin"].to_numpy(dtype=object)
    woe = _validation.as_real(table["woe"], "woe of table")
    points = _validation.as_real(table["points"], "points of table")
    result, start = [], 0
    for name, groups in zip(names, fitted, strict=True):
        rows = slice(start, start + len(groups))
        start += len(groups)
        for label, group in zip(labels[rows], groups.index, strict=True):
            if not _same_label(label, group):
                raise ValueError(
                    f"bin of table must list the groups of binning in their order, got "
                    f"{_validation.shown(label)} where the table of {_validation.shown(name)} "
                    f"in binning has {_validation.shown(group)}"
                )
        expected = groups["woe"].to_numpy(dtype=float)
        differ = np.flatnonzero(woe[rows] != expected)
        if len(differ):
            row = differ[0]
            raise ValueError(
                f"woe of table must equal the WOE of binning, got {float(woe[rows][row])!r} "
                f"where binning has {float(expected[row])!r} for the group "
                f"{_validation.shown(groups.index[row])} of {_validation.shown(name)}; round only "
                "the points column"
            )
        values = points[rows] + 0.0
        invalid = np.flatnonzero(~np.isfinite(values))
        if len(invalid):
            row = invalid[0]
            raise ValueError(
                f"points of table must be finite, got {float(values[row])!r} for the group "
                f"{_validation.shown(groups.index[row])} of {_validation.shown(name)}"
            )
        values.flags.writeable = False
        result.append(values)
    return result


def _same_label(label: object, group: object) -> bool:
    """Whether a label of a points table names a group of a binning table."""
    if _validation.is_missing(label) or _validation.is_missing(group):
        return _validation.is_missing(label) and _validation.is_missing(group)
    # Groups are strings, numbers or intervals; comparing an array would compare its elements.
    if not isinstance(label, (str, numbers.Number, pd.Interval, np.generic)):
        return False
    if isinstance(label, (bool, np.bool_)) != isinstance(group, (bool, np.bool_)):
        return False
    return bool(label == group)


def _points_frame(
    binning: spec.BinningSpec, points: abc.Sequence[npt.NDArray[np.float64]]
) -> pd.DataFrame:
    """Return the points table of the binning's groups, with the given points per column."""
    names = list(binning.columns)
    fitted = [binning.tables[name] for name in names]
    variables = [name for name, groups in zip(names, fitted, strict=True) for _ in groups.index]
    labels = np.array([label for groups in fitted for label in groups.index] or [], dtype=object)
    table = pd.DataFrame(
        {
            "woe": np.concatenate([groups["woe"].to_numpy(dtype=float) for groups in fitted]),
            "points": np.concatenate(points),
        },
        index=pd.Index(variables, dtype=object, name="variable"),
    )
    table.insert(0, "bin", pd.array(labels, dtype=object))  # type: ignore[arg-type]  # pandas-stubs omit dtype=object
    return table.astype({"bin": object})


def _checked_unseen_points(
    unseen_points: object, binning: spec.BinningSpec
) -> dict[str, float] | None:
    """Check the points of unseen groups against the unseen policies of the binning."""
    wanted = list(_woe_policies(binning))
    if unseen_points is None:
        if wanted:
            raise ValueError(
                "unseen_points must give the points of unseen groups of the columns whose unseen "
                f"policy in binning is a WOE, got None for {_validation.shown_list(wanted)}"
            )
        return None
    if isinstance(unseen_points, pd.Series):
        if unseen_points.index.has_duplicates:
            repeated = unseen_points.index[unseen_points.index.duplicated()][0]
            raise ValueError(
                f"unseen_points must name each column once, got {_validation.shown(repeated)} twice"
            )
        given = dict(unseen_points.items())
    elif isinstance(unseen_points, abc.Mapping):
        given = dict(unseen_points.items())
    else:
        raise ValueError(
            "unseen_points must be a mapping or Series by column, got "
            f"{type(unseen_points).__name__}"
        )
    other = [name for name in given if name not in wanted]
    if other:
        raise ValueError(
            "unseen_points has points for columns whose unseen policy in binning is not a WOE: "
            f"{_validation.shown_list(other)}"
        )
    absent = [name for name in wanted if name not in given]
    if absent:
        raise ValueError(
            f"unseen_points lacks the points of {_validation.shown_list(absent)}, whose unseen "
            "policy in binning is a WOE"
        )
    result = {}
    for name in wanted:
        value = given[name]
        if not _is_finite(value):
            raise ValueError(
                f"unseen_points must be finite numbers, got {_validation.shown(value)} for "
                f"{_validation.shown(name)}"
            )
        result[name] = float(value) + 0.0
    return result or None


def _checked_calibration(value: object) -> dict[str, typing.Any] | None:
    """Check a calibration and return it with floats and read-only arrays."""
    if value is None:
        return None
    if not isinstance(value, abc.Mapping):
        raise ValueError(f"calibration must be a mapping with a kind, got {type(value).__name__}")
    kind = value.get("kind")
    if not isinstance(kind, str) or kind not in _CALIBRATIONS:
        raise ValueError(
            f"the kind of calibration must be one of {tuple(_CALIBRATIONS)}, got "
            f"{_validation.shown(kind)}"
        )
    keys = ["kind", *_CALIBRATIONS[kind]]
    if set(value) != set(keys):
        raise ValueError(
            f"a {kind} calibration must have the keys {_validation.shown_list(keys)}, got "
            f"{_validation.shown_list(list(value))}"
        )
    if kind != "isotonic":
        result: dict[str, typing.Any] = {"kind": kind}
        for name in _CALIBRATIONS[kind]:
            number = value[name]
            if not _is_finite(number):
                raise ValueError(
                    f"the {name} of a {kind} calibration must be a finite number, got "
                    f"{_validation.shown_argument(number)}"
                )
            result[name] = float(number) + 0.0
        if kind == "platt" and result["slope"] < 0:
            raise ValueError(
                "the slope of a platt calibration must not be negative, which would give higher "
                f"scores a higher probability, got {result['slope']!r}"
            )
        return result
    edges, probability = value["edges"], value["probability"]
    try:
        # isotonic_predict checks the bands; it gives no probability for no scores.
        calibration.isotonic_predict(np.empty(0), edges, probability)
    except ValueError as error:
        raise ValueError(f"isotonic calibration: {error}") from None
    edges = np.array(edges, dtype=float) + 0.0
    probability = np.array(probability, dtype=float) + 0.0
    rising = np.flatnonzero(np.diff(probability) > 0)
    if len(rising):
        band = rising[0]
        raise ValueError(
            "isotonic calibration: probability must not rise with the score, as isotonic_fit "
            "gives it with increasing=False, got "
            f"{float(probability[band])!r} followed by {float(probability[band + 1])!r}"
        )
    edges.flags.writeable = False
    probability.flags.writeable = False
    return {"kind": kind, "edges": edges, "probability": probability}


def _checked_scale(scale: object) -> pd.DataFrame:
    """Check a master scale and return a copy with Python labels in an index named grade."""
    if not isinstance(scale, pd.DataFrame):
        raise ValueError(f"scale must be a DataFrame from master_scale, got {type(scale).__name__}")
    if len(scale.columns) != 3 or set(scale.columns) != set(_SCALE_COLUMNS):
        raise ValueError(
            "scale must have the columns 'lower', 'probability' and 'upper' of a master_scale "
            f"result, got {scale.columns.tolist()}"
        )
    # assign_grades checks the labels and the bounds of the grades.
    calibration.assign_grades(np.empty(0), scale)
    probability = _validation.as_real(scale["probability"], "probability of scale")
    outside = np.flatnonzero(~((probability > 0) & (probability < 1)))
    if len(outside):
        raise ValueError(
            "probability of scale must be strictly between 0 and 1, got "
            f"{float(probability[outside[0]])!r}"
        )
    labels = [_grade_label(label) for label in scale.index]
    return pd.DataFrame(
        {
            "lower": scale["lower"].to_numpy(dtype=float) + 0.0,
            "probability": probability + 0.0,
            "upper": scale["upper"].to_numpy(dtype=float) + 0.0,
        },
        index=pd.Index(labels, name="grade"),
    )


def _grade_label(label: object) -> str | int | float | bool:
    """Return a grade label as the Python str, int, float or bool that JSON keeps."""
    if isinstance(label, (bool, np.bool_)):
        return bool(label)
    if isinstance(label, numbers.Integral):
        return int(label)
    if _is_finite(label):
        return float(label)
    if isinstance(label, str):
        return str(label)
    raise ValueError(
        "scale must be labelled by strings, integers, finite floats or booleans, got "
        f"{_validation.shown(label)} of type {type(label).__name__}"
    )


def _encoded(value: float) -> float | dict[str, str]:
    """Write an infinite number as a tagged object, which JSON allows."""
    if math.isinf(value):
        return {"float": "inf" if value > 0 else "-inf"}
    return value


def _calibration_json(fitted: abc.Mapping[str, typing.Any] | None) -> dict[str, typing.Any] | None:
    if fitted is None:
        return None
    if fitted["kind"] != "isotonic":
        return dict(fitted)
    return {
        "kind": "isotonic",
        "edges": [_encoded(float(edge)) for edge in fitted["edges"]],
        "probability": fitted["probability"].tolist(),
    }


def _scale_json(scale: pd.DataFrame | None) -> list[dict[str, typing.Any]] | None:
    if scale is None:
        return None
    columns = {name: scale[name].tolist() for name in _SCALE_COLUMNS}
    return [
        {"grade": label, **{name: values[row] for name, values in columns.items()}}
        for row, label in enumerate(scale.index.tolist())
    ]


def _decoded_points(points: object, binning: spec.BinningSpec) -> list[npt.NDArray[np.float64]]:
    if not isinstance(points, abc.Mapping):
        raise ValueError(f"points must be a mapping by column, got {_json_kind(points)}")
    _check_keys(points, set(binning.columns), "points")
    result = []
    for name in binning.columns:
        values = points[name]
        where = f"points of {_validation.shown(name)}"
        if not isinstance(values, list):
            raise ValueError(f"{where} must be a list, got {_json_kind(values)}")
        size = len(binning.tables[name])
        if len(values) != size:
            raise ValueError(
                f"{where} must have one value per group of its table in binning, {size}, got "
                f"{len(values)}"
            )
        result.append(np.array([_json_number(value, where) for value in values], dtype=float))
    return result


def _decoded_unseen_points(value: object) -> dict[str, typing.Any] | None:
    if value is None:
        return None
    if not isinstance(value, abc.Mapping):
        raise ValueError(f"unseen_points must be a mapping or null, got {_json_kind(value)}")
    return {name: _json_number(number, "unseen_points") for name, number in value.items()}


def _decoded_calibration(value: object) -> dict[str, typing.Any] | None:
    if value is None:
        return None
    if not isinstance(value, abc.Mapping):
        raise ValueError(f"calibration must be a mapping or null, got {_json_kind(value)}")
    result = dict(value)
    if result.get("kind") == "isotonic":
        for name in ("edges", "probability"):
            if name in result:
                values = result[name]
                if not isinstance(values, list):
                    raise ValueError(
                        f"{name} of the isotonic calibration must be a list, got "
                        f"{_json_kind(values)}"
                    )
                result[name] = [
                    _json_number(item, f"{name} of the isotonic calibration") for item in values
                ]
    else:
        for name, number in value.items():
            # Numbers are read here; anything else is left for the constructor to reject.
            if isinstance(number, (int, float)) and not isinstance(number, bool):
                result[name] = _json_number(number, f"the {name} of the calibration")
    return result


def _decoded_scale(value: object) -> pd.DataFrame | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"scale must be a non-empty list of grades or null, got {_json_kind(value)}"
        )
    labels: list[object] = []
    columns: dict[str, list[float]] = {name: [] for name in _SCALE_COLUMNS}
    for number, row in enumerate(value):
        where = f"scale[{number}]"
        if not isinstance(row, abc.Mapping):
            raise ValueError(f"{where} must be a mapping, got {_json_kind(row)}")
        _check_keys(row, {"grade", *_SCALE_COLUMNS}, where)
        label = row["grade"]
        if label is None or isinstance(label, (dict, list)):
            raise ValueError(
                f"the grade of {where} must be a string, number or boolean, got {_json_kind(label)}"
            )
        labels.append(label)
        for name in _SCALE_COLUMNS:
            columns[name].append(_json_number(row[name], f"{name} of {where}"))
    return pd.DataFrame(columns, index=pd.Index(labels, name="grade"))


def _json_number(value: object, where: str) -> float:
    """Read a JSON number, or an infinity written as a tagged object."""
    if isinstance(value, abc.Mapping) and set(value) == {"float"}:
        if value["float"] in ("inf", "-inf"):
            return float(value["float"])
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return float(value)
        except OverflowError:
            raise ValueError(
                f"{where} must be numbers that fit a float, got an integer too large for one"
            ) from None
    raise ValueError(f"{where} must be numbers, got {_validation.shown(value)}")


def _json_kind(value: object) -> str:
    """Name the kind of a JSON value for a message."""
    if value is None:
        return "null"
    if isinstance(value, abc.Mapping):
        return "an object"
    if isinstance(value, list):
        return "a list"
    return "a string" if isinstance(value, str) else type(value).__name__


def _check_keys(data: abc.Mapping[str, typing.Any], expected: set[str], where: str) -> None:
    missing = sorted(expected - set(data))
    unknown = sorted(set(data) - expected, key=str)
    if missing or unknown:
        problems = []
        if missing:
            problems.append(f"lacks {_validation.shown_list(missing)}")
        if unknown:
            problems.append(f"has unknown keys {_validation.shown_list(unknown)}")
        raise ValueError(f"{where} " + " and ".join(problems))


def _is_finite(value: object) -> typing.TypeGuard[float]:
    """Whether a value is a finite real number other than a boolean.

    An integer too large for a float is not finite, so that it raises a ValueError rather
    than the OverflowError of ``math.isfinite``.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@contextlib.contextmanager
def _fitting_floats(where: str) -> abc.Iterator[None]:
    """Turn the OverflowError of an integer too large for a float into a ValueError."""
    try:
        yield
    except OverflowError:
        raise ValueError(f"{where} must hold numbers that fit a float") from None


def _unseen_points(
    unseen_points: float | abc.Mapping[object, float] | pd.Series | None, variables: pd.Index
) -> npt.NDArray[np.float64]:
    """Return the unseen points of each variable, NaN for a variable without them."""
    if unseen_points is None:
        return np.full(len(variables), np.nan)
    if isinstance(unseen_points, (abc.Mapping, pd.Series)):
        by_variable = dict(unseen_points.items())
        if isinstance(unseen_points, pd.Series) and unseen_points.index.has_duplicates:
            repeated = unseen_points.index[unseen_points.index.duplicated()][0]
            raise ValueError(
                f"unseen_points must name each variable once, got {_validation.shown(repeated)} "
                "twice"
            )
    else:
        by_variable = dict.fromkeys(variables, unseen_points)
    result = np.full(len(variables), np.nan)
    for position, variable in enumerate(variables):
        if variable not in by_variable:
            continue
        value = by_variable[variable]
        if not _is_finite(value):
            raise ValueError(
                f"unseen_points must be finite numbers, got {_validation.shown(value)} for "
                f"{_validation.shown(variable)}"
            )
        result[position] = float(value)
    return result


def _model_points(
    coefficients: abc.Mapping[object, float] | pd.Series,
    intercept: float,
    pdo: float,
    base_score: float,
    base_odds: float,
) -> tuple[pd.Series, float]:
    """Return the points per unit of WOE and the base points of each variable."""
    factor, offset = score_scaling(pdo=pdo, base_score=base_score, base_odds=base_odds)
    if not _is_finite(intercept):
        raise ValueError(
            f"intercept must be a finite number, got {_validation.shown_argument(intercept)}"
        )
    if isinstance(coefficients, abc.Mapping):
        coefficients = pd.Series(dict(coefficients), dtype=object)
    elif not isinstance(coefficients, pd.Series):
        kind = type(coefficients).__name__  # type: ignore[unreachable]  # checked at runtime
        raise ValueError(f"coefficients must be a mapping or Series by variable, got {kind}")
    if coefficients.empty:
        raise ValueError("coefficients must name at least one variable, got none")
    if coefficients.index.has_duplicates:
        repeated = coefficients.index[coefficients.index.duplicated()][0]
        raise ValueError(
            f"coefficients must name each variable once, got {_validation.shown(repeated)} twice"
        )
    for variable, value in zip(coefficients.index, coefficients.tolist(), strict=True):
        if not _is_finite(value):
            raise ValueError(
                f"coefficients must be finite numbers, got {_validation.shown(value)} for "
                f"{_validation.shown(variable)}"
            )
    beta = pd.Series(coefficients.to_numpy(dtype=float), index=coefficients.index)
    return factor * beta, (offset - factor * intercept) / len(beta)


def _woe_column(table: pd.DataFrame | pd.Series, variable: object) -> pd.Series:
    try:
        woe = _validation.woe_values(table)
    except ValueError as error:
        raise ValueError(f"invalid table for {_validation.shown(variable)}: {error}") from None
    values = _validation.float_values(woe, f"the table of {_validation.shown(variable)}")
    invalid = ~np.isfinite(values)
    if invalid.any():
        raise ValueError(
            f"the table of {_validation.shown(variable)} must have finite WOE values, got "
            f"{float(values[invalid][0])!r} for group {_validation.shown(woe.index[invalid][0])}"
        )
    return woe


def _check_variables(variables: pd.Index, available: abc.Container[object], name: str) -> None:
    """Require an entry of ``available`` for each variable of the coefficients."""
    if "intercept" in variables and "intercept" not in available:
        # As in the coef column of logistic_regression, whose intercept row is named so.
        raise ValueError("coefficients has an 'intercept' entry; drop it and pass it as intercept")
    _check_present(variables, available, name)


def _check_present(variables: pd.Index, available: abc.Container[object], name: str) -> None:
    missing = [variable for variable in variables if variable not in available]
    if missing:
        examples = ", ".join(_validation.shown(variable) for variable in missing[:5])
        raise ValueError(f"{name} has no entry for {len(missing)} variable(s): {examples}")


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
        return (
            f"a DataFrame indexed by {_validation.shown(value.index.name)} "
            f"with columns {value.columns.tolist()}"
        )
    return type(value).__name__
