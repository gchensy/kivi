"""Monitoring of a score and its variables over time.

``monitoring_table`` gives one row per period with the population, the
event rate, the discrimination and the stability of a score, the report
that is otherwise written as a loop over the periods. ``stability_table``
gives the PSI of already-grouped variables in every period, for the
characteristic analysis of a scorecard.

As in ``kivi.metrics``, the score is higher for a higher event probability,
such as a PD, so negate scorecard points before passing them in.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._counts import divergence
from ._validation import as_series, as_weights, check_smoothing, first_value
from .metrics import _band_edges, ks, roc_auc

__all__ = ["monitoring_table", "stability_table"]


def monitoring_table(
    score: npt.ArrayLike,
    target: npt.ArrayLike | None,
    period: npt.ArrayLike,
    *,
    reference: object,
    bins: int | npt.ArrayLike = 10,
    smoothing: float = 0.5,
    weights: npt.ArrayLike | None = None,
    reference_weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Monitor the population, event rate, discrimination and stability of a score.

    Each row summarizes one period, such as a month or a quarter: how many
    observations it has, how many events among those with a known outcome,
    how well the score separates the two classes and how far the
    distribution of the score has moved from a reference sample.

    Parameters
    ----------
    score : array_like
        Finite numeric scores that are higher for a higher event probability,
        such as a PD. Negate scorecard points, which are higher for a lower
        risk.
    target : array_like or None
        Binary labels, with 1 indicating the event, and missing values for
        outcomes that are not known yet, such as those of loans that have not
        reached the end of the outcome window. With None, every outcome is
        unknown and the table only has the columns ``total`` and ``psi``. If
        `score` and `target` are both pandas Series, they must have identical
        indices; otherwise they are matched by position.
    period : array_like
        Period of each observation, such as the month of application, without
        missing values, matched like `target`. Periods are sorted, and
        categorical periods follow the order of their categories.
    reference : scalar, tuple or array_like
        The sample that the distribution of each period is compared with:
        either the label of one period, such as the first month after the
        development sample, or the scores of a reference sample, such as the
        development sample itself, as a list, an array or a Series. A string
        that names one period of datetime periods, such as ``"2024-01"`` for
        monthly dates, selects that period.
    bins : int or array_like, default 10
        Number of quantile bands of the reference scores, or increasing band
        edges. An integer is the same as passing
        ``kivi.binning.bin_edges(reference_score, bins=bins)``, with the
        weights of the reference as ``weights``, so tied scores can leave
        fewer bands than requested. Every period is compared on the same
        bands.
    smoothing : float, default 0.5
        Positive replacement for a zero count when calculating the PSI, as in
        `kivi.stability.psi_table`, in units of weight with `weights`.
    weights : array_like, optional
        Non-negative weight of each observation, such as the inverse of its
        sampling rate, matched like `target`. Counts become sums of weights,
        and the discrimination and the PSI are weighted. On the same band
        edges, integer weights give the same table as repeating rows.
    reference_weights : array_like, optional
        Non-negative weight of each observation of a reference sample passed
        as `reference`, matched like the sample. A reference period takes its
        weights from `weights`.

    Returns
    -------
    pandas.DataFrame
        One row per period, indexed by ``period``, with the columns described
        in the Notes.

    Raises
    ------
    ValueError
        If `score` or `period` is empty, an input is not one-dimensional, the
        inputs have different lengths or are Series with different indices,
        `score` is not real and numeric or has a missing or infinite value,
        `target` has a value other than 0, 1 and missing, or `period` has a
        missing value.
        Also if `reference` is neither a period of `period` with observations
        nor a non-empty sample of finite scores, `reference_weights` is given
        with a reference period, `bins` is neither a positive integer nor
        increasing edges, a score lies outside the edges, `smoothing` is not
        finite and positive, or a weight array is not numeric, does not give
        each observation one finite, non-negative weight with a positive
        total, or has an index other than that of its sample.

    Warns
    -----
    UserWarning
        If `bins` is an integer and tied reference scores leave fewer bands
        than requested, as in `kivi.binning.bin_edges`.

    See Also
    --------
    stability_table : The PSI of each variable of the score in each period.
    kivi.metrics.roc_auc : The AUC of one sample.
    kivi.stability.psi_table : The PSI of one sample and its contributions.

    Notes
    -----
    The columns are:

    - ``total``: observations in the period, or the sum of their weights,
      including those whose outcome is not known yet;
    - ``known``: observations with a known outcome;
    - ``event``: events among them;
    - ``event_rate``: ``event`` divided by ``known``, NaN without known
      outcomes;
    - ``auc``, ``gini`` and ``ks``: `kivi.metrics.roc_auc`,
      `kivi.metrics.gini` and `kivi.metrics.ks` of the known outcomes of
      the period, NaN unless both classes are present;
    - ``psi``: the population stability index of the period's scores
      against the reference scores, on the bands described under `bins`.

    Without a target the columns are ``total`` and ``psi``. Each row equals
    what the metric functions and ``psi_table(pandas.cut(reference_score,
    edges), pandas.cut(period_score, edges))`` give for the period alone,
    with the weights of the two samples as ``reference_weights`` and
    ``current_weights``.
    For a band :math:`i` with shares :math:`q_i` of the reference scores and
    :math:`p_i` of the period's scores, after smoothing,

    .. math::

       \mathrm{PSI} = \sum_i (p_i - q_i) \ln \frac{p_i}{q_i},

    where bands empty in both samples are left out. The reference period, if
    `reference` names one, has a PSI of 0.

    A period of a categorical `period` without observations, or whose
    weights sum to zero, keeps its row with zero counts and NaN in the other
    columns, so that tables of different samples line up. Other periods
    exist only through their observations, and those whose weights sum to
    zero are left out, as repeated rows would leave them out.

    The discrimination of the most recent periods usually rests on few known
    outcomes, and the PSI of a small period mostly reflects sampling noise,
    so read both together with ``known`` and ``total``.

    Examples
    --------
    >>> import numpy as np
    >>> from kivi.monitoring import monitoring_table
    >>> score = [0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.6, 0.7, 0.5, 0.6, 0.8, 0.9]
    >>> target = [0, 0, 1, 1, 0, 1, 0, 1, 0, np.nan, 1, np.nan]
    >>> period = ["2024-01"] * 4 + ["2024-02"] * 4 + ["2024-03"] * 4
    >>> table = monitoring_table(score, target, period, reference="2024-01", bins=2)
    >>> table[["total", "known", "event", "event_rate"]]
             total  known  event  event_rate
    period
    2024-01      4      4      2         0.5
    2024-02      4      4      2         0.5
    2024-03      4      2      1         0.5
    >>> table[["auc", "gini", "ks", "psi"]].round(4)
              auc  gini   ks     psi
    period
    2024-01  1.00   1.0  1.0  0.0000
    2024-02  0.75   0.5  0.5  0.2747
    2024-03  1.00   1.0  1.0  0.8087

    The March outcomes of two observations are not known yet, so its
    discrimination rests on the other two.
    """
    _check_indices({"score": score, "target": target, "period": period, "weights": weights})
    check_smoothing(smoothing)
    raw_score = score
    score, period = as_series(score, "score"), as_series(period, "period")
    if len(score) != len(period):
        raise ValueError(
            f"score and period must have the same length, got {len(score)} and {len(period)}"
        )
    if not len(score):
        raise ValueError("score and period must not be empty")
    values = _finite_scores(score, "score")
    mass = as_weights(weights, raw_score, of="score")
    known, event = (None, None) if target is None else _outcomes(target, len(score))
    code, periods, categorical = _periods(period)
    totals = _period_sums(code, len(periods), mass)

    position = _reference_period(reference, periods)
    if position is None:
        reference_values, reference_mass = _reference_scores(reference, reference_weights)
    else:
        if reference_weights is not None:
            raise ValueError(
                "reference_weights must not be given with a reference period, whose weights "
                f"come from weights; got the period {reference!r}"
            )
        _check_reference_period(reference, totals[position], mass)
        rows = code == position
        reference_values = values[rows]
        reference_mass = None if mass is None else mass[rows]
    # Called here, not in a helper, so that the warning for tied quantiles points at the caller.
    edges = _band_edges(reference_values, bins, reference_mass)
    band = _bands(values, edges, "score")
    reference_band = _bands(reference_values, edges, "reference")
    psi = _period_psi(
        code, band, mass, reference_band, reference_mass, len(periods), len(edges) - 1, smoothing
    )

    columns = {"total": totals}
    if target is not None:
        labels = event.astype(np.intp)
        columns["known"] = _period_sums(code[known], len(periods), _subset(mass, known))
        columns["event"] = _period_sums(code[event], len(periods), _subset(mass, event))
        columns["event_rate"] = _ratio(columns["event"], columns["known"])
        auc, ks_value = _discrimination(code, values, labels, known, mass, len(periods))
        columns.update(auc=auc, gini=2 * auc - 1, ks=ks_value)
    columns["psi"] = psi
    table = pd.DataFrame(columns, index=periods)
    if categorical or mass is None:
        return table
    return table.iloc[np.flatnonzero(totals > 0)]


def stability_table(
    groups: pd.DataFrame,
    period: npt.ArrayLike,
    *,
    reference: object,
    smoothing: float = 0.5,
    weights: npt.ArrayLike | None = None,
    reference_weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Monitor the population stability of grouped variables over time.

    Each entry is the PSI of one variable in one period against a reference
    sample, as `kivi.stability.psi_table` computes it for the period alone.
    With the groups of a scorecard's variables this is the variable part of
    a stability report, and the variables that move explain a shift of the
    score that `monitoring_table` shows.

    Parameters
    ----------
    groups : pandas.DataFrame
        One column per variable, holding labels or values binned using the
        same edges in every period, such as the result of
        ``kivi.spec.BinningSpec.cut``. Missing values form a group, and
        groups observed in only one sample are included.
    period : array_like
        Period of each row of `groups`, without missing values. Periods are
        sorted, and categorical periods follow the order of their categories.
        A Series must have the index of `groups`.
    reference : scalar, tuple or pandas.DataFrame
        The sample that each period is compared with: either the label of one
        period, as for `monitoring_table`, or the groups of a reference
        sample, such as the development sample, with every column of
        `groups`. Other columns of a reference DataFrame are ignored.
    smoothing : float, default 0.5
        Positive replacement for a zero count when calculating shares, in
        units of weight with `weights`.
    weights : array_like, optional
        Non-negative weight of each row of `groups`, matched like `period`.
        Counts become sums of weights, so integer weights give the same table
        as repeating rows.
    reference_weights : array_like, optional
        Non-negative weight of each row of a reference sample passed as
        `reference`. A reference period takes its weights from `weights`.

    Returns
    -------
    pandas.DataFrame
        One row per period, indexed by ``period``, and one column per
        variable, with the columns of `groups` and the column index named
        ``variable``. Each entry is the PSI of the variable in the period.

    Raises
    ------
    ValueError
        If `groups` is not a DataFrame, has no rows, no columns, repeated
        column names or more than one level of column names, `period` is not
        one-dimensional, does not have one value per row or has a
        missing value, or a Series argument has an index other than that of
        `groups`. Also if `reference` is neither a period of `period` with
        observations nor a non-empty DataFrame with every column of `groups`,
        `reference_weights` is given with a reference period, `smoothing` is
        not finite and positive, or a weight array is not numeric, does not
        give each row one finite, non-negative weight with a positive total,
        or has an index other than that of its sample.

    See Also
    --------
    monitoring_table : The population, discrimination and PSI of the score.
    kivi.stability.psi_table : The PSI of one sample and its contributions.
    kivi.stability.csi_table : The change of a variable's average points.

    Notes
    -----
    For a group :math:`i` of a variable, with shares :math:`q_i` of the
    reference sample and :math:`p_i` of the period, after smoothing,

    .. math::

       \mathrm{PSI} = \sum_i (p_i - q_i) \ln \frac{p_i}{q_i},

    over the groups observed in the reference sample or in the period. The
    reference period, if `reference` names one, has a PSI of 0.

    A period of a categorical `period` without observations, or whose
    weights sum to zero, keeps its row with NaN entries. Other periods exist
    only through their rows, and those whose weights sum to zero are left
    out, as repeated rows would leave them out.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.monitoring import stability_table
    >>> groups = pd.DataFrame({
    ...     "housing": ["own", "own", "rent", "rent", "own", "rent", "rent", "rent"],
    ...     "region": ["north", "south", "north", "south", "north", "north", "south", None],
    ... })
    >>> period = ["2024-01"] * 4 + ["2024-02"] * 4
    >>> stability_table(groups, period, reference="2024-01").round(4)
    variable  housing  region
    period
    2024-01    0.0000   0.000
    2024-02    0.2747   0.231

    One value of the February region is missing, a group that January does
    not have.
    """
    if not isinstance(groups, pd.DataFrame):
        raise ValueError(f"groups must be a DataFrame, got {type(groups).__name__}")
    if not len(groups.columns):
        raise ValueError("groups must have at least one column")
    if isinstance(groups.columns, pd.MultiIndex):
        raise ValueError(
            f"groups must have one level of column names, got {groups.columns.nlevels}"
        )
    if groups.columns.has_duplicates:
        repeated = groups.columns[groups.columns.duplicated()][0]
        raise ValueError(f"groups must have unique column names, got {repeated!r} twice")
    if not len(groups):
        raise ValueError("groups must have at least one row")
    _check_indices({"groups": groups, "period": period, "weights": weights})
    check_smoothing(smoothing)
    period = as_series(period, "period")
    if len(period) != len(groups):
        raise ValueError(
            f"period must have one value per row of groups, got {len(period)} for {len(groups)}"
        )
    mass = as_weights(weights, groups, of="groups")
    code, periods, categorical = _periods(period)
    totals = _period_sums(code, len(periods), mass)

    position = _reference_period(reference, periods)
    if position is None:
        if not isinstance(reference, pd.DataFrame):
            raise ValueError(
                "reference must be a period of period or a DataFrame of reference groups, "
                f"got {type(reference).__name__}"
            )
        absent = groups.columns.difference(reference.columns, sort=False)
        if len(absent):
            raise ValueError(
                f"reference must have every column of groups, got no column {absent[0]!r}"
            )
        if reference.columns.has_duplicates:
            repeated = reference.columns[reference.columns.duplicated()][0]
            raise ValueError(f"reference must have unique column names, got {repeated!r} twice")
        if not len(reference):
            raise ValueError("reference must have at least one row")
        if isinstance(reference_weights, pd.Series) and not reference_weights.index.equals(
            reference.index
        ):
            raise ValueError("reference_weights must have the same index as reference")
        reference_mass = as_weights(reference_weights, reference, "reference_weights", "reference")
        reference_rows = None
    else:
        if reference_weights is not None:
            raise ValueError(
                "reference_weights must not be given with a reference period, whose weights "
                f"come from weights; got the period {reference!r}"
            )
        _check_reference_period(reference, totals[position], mass)
        reference_rows = code == position
        reference_mass = _subset(mass, reference_rows)

    result = np.empty((len(periods), len(groups.columns)))
    for column_number, column in enumerate(groups.columns):
        current = groups[column]
        if reference_rows is None:
            levels, reference_level = _shared_levels(reference[column], current)
        else:
            levels = _levels(current)
            reference_level = levels[reference_rows]
        n_levels = int(max(levels.max(), reference_level.max())) + 1
        result[:, column_number] = _period_psi(
            code, levels, mass, reference_level, reference_mass, len(periods), n_levels, smoothing
        )
    table = pd.DataFrame(result, index=periods, columns=groups.columns.rename("variable"))
    if categorical or mass is None:
        return table
    return table.iloc[np.flatnonzero(totals > 0)]


def _check_indices(inputs: dict) -> None:
    """Require every Series or DataFrame among the inputs to have the index of the first one."""
    indexed = {
        name: values
        for name, values in inputs.items()
        if isinstance(values, (pd.Series, pd.DataFrame))
    }
    names = list(indexed)
    for name in names[1:]:
        if not indexed[name].index.equals(indexed[names[0]].index):
            raise ValueError(f"{names[0]} and {name} must have identical indices")


def _finite_scores(values: pd.Series, name: str) -> np.ndarray:
    """Finite float scores; booleans are accepted, as in ``kivi.metrics``."""
    dtype = values.dtype
    if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_complex_dtype(dtype):
        raise ValueError(f"{name} must be real and numeric, got dtype {dtype}")
    array = values.to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(array)
    if not finite.all():
        raise ValueError(
            f"{name} must be finite, got {(~finite).sum()} missing or infinite value(s) "
            f"such as {float(array[~finite][0])!r}"
        )
    return array


def _outcomes(target, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks of the known outcomes and of the events; missing labels are unknown."""
    target = as_series(target, "target")
    if len(target) != length:
        raise ValueError(
            f"target must have one label per value of score, got {len(target)} for {length}"
        )
    missing = target.isna().to_numpy(dtype=bool)
    # For nullable dtypes pandas 2.0 returns an object array, which ~ would negate as integers.
    invalid = ~(target.isin([0, 1]).to_numpy(dtype=bool) | missing)
    if invalid.any():
        shown = first_value(target[invalid])
        raise ValueError(f"target must contain only 0, 1 and missing values, got {shown!r}")
    event = np.zeros(length, dtype=bool)
    event[~missing] = target[~missing].to_numpy(dtype=float) == 1
    return ~missing, event


def _periods(period: pd.Series) -> tuple[np.ndarray, pd.Index, bool]:
    """Code of each observation's period, the periods in order, and whether they are categorical.

    Categorical periods are all the categories, in their order, observed or
    not; other periods are the distinct values, sorted.
    """
    if isinstance(period.dtype, pd.CategoricalDtype):
        code = period.cat.codes.to_numpy(dtype=np.intp)
        # From codes, which does not hash the categories, as building from labels would.
        n_categories = len(period.cat.categories)
        labels = pd.Categorical.from_codes(np.arange(n_categories), dtype=period.dtype)
        periods = pd.CategoricalIndex(labels, name="period")
        categorical = True
    else:
        code, periods = pd.factorize(period, sort=True)
        periods = pd.Index(periods, name="period")
        categorical = False
    missing = code < 0
    if missing.any():
        raise ValueError(f"period must not have missing values, got {missing.sum()}")
    return code, periods, categorical


def _reference_period(reference, periods: pd.Index) -> int | None:
    """Position of the reference period, or None if `reference` is a sample.

    A scalar or a tuple is a period label; anything else is a sample.
    """
    if not (isinstance(reference, tuple) or np.ndim(reference) == 0):
        return None
    if isinstance(reference, np.ndarray):
        reference = reference.item()
    try:
        position = periods.get_loc(reference)
    except (KeyError, TypeError, ValueError, pd.errors.InvalidIndexError):
        position = None
    if isinstance(position, slice):
        # A date string selects the periods of a DatetimeIndex that it covers.
        matched = range(len(periods))[position]
        if len(matched) > 1:
            raise ValueError(
                f"reference must name one period of period, got {reference!r}, which "
                f"matches {len(matched)} periods"
            )
        position = matched[0] if len(matched) else None
    if not isinstance(position, (int, np.integer)):
        shown = ", ".join(repr(label) for label in periods[:3].tolist())
        raise ValueError(
            f"reference must be a period of period or a reference sample, got {reference!r}, "
            f"which is not one of the periods {shown}{', ...' if len(periods) > 3 else ''}"
        )
    return int(position)


def _check_reference_period(reference, total: float, mass: np.ndarray | None) -> None:
    if not total > 0:
        detail = "" if mass is None else " with positive weight"
        raise ValueError(f"reference period {reference!r} has no observations{detail}")


def _reference_scores(reference, reference_weights) -> tuple[np.ndarray, np.ndarray | None]:
    """Finite scores and weights of an explicit reference sample."""
    raw_reference = reference
    reference = as_series(reference, "reference")
    if not len(reference):
        raise ValueError("reference must not be empty")
    dtype = reference.dtype
    if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_complex_dtype(dtype):
        raise ValueError(
            f"reference must be real and numeric scores or a period label, got dtype {dtype}; "
            "pass a period label as a scalar, not in a list"
        )
    values = _finite_scores(reference, "reference")
    mass = as_weights(reference_weights, raw_reference, "reference_weights", "reference")
    return values, mass


def _bands(values: np.ndarray, edges: np.ndarray, name: str) -> np.ndarray:
    """Band of each score, for bands closed on the right as in ``pandas.cut``."""
    band = np.searchsorted(edges, values, side="left") - 1
    outside = (band < 0) | (band >= len(edges) - 1)
    if outside.any():
        raise ValueError(
            f"{name} has {outside.sum()} value(s) outside the edges in bins, such as "
            f"{float(values[outside][0])!r}; use infinite outer edges"
        )
    return band


def _levels(groups: pd.Series) -> np.ndarray:
    """Group code of each value, as ``kivi._counts.group_counts`` factorizes groups."""
    ordered = isinstance(groups.dtype, pd.CategoricalDtype)
    return pd.factorize(groups, sort=ordered, use_na_sentinel=False)[0]


def _shared_levels(reference: pd.Series, current: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Group codes of the current and the reference values, from one factorization of both."""
    # pd.concat hashes interval categories, which warns on NumPy < 2.2, so samples binned with
    # the same edges are joined by their codes, as in kivi.stability.psi_table.
    categorical = [isinstance(values.dtype, pd.CategoricalDtype) for values in (reference, current)]
    if all(categorical) and current.dtype == reference.dtype:
        codes = np.concatenate([reference.cat.codes, current.cat.codes])
        both = pd.Series(pd.Categorical.from_codes(codes, dtype=current.dtype))
    elif any(categorical):
        # Concatenating other categoricals also hashes their categories, so compare the labels
        # as objects, by equality, as pd.concat does for categoricals of different categories.
        labels = [values.astype(object).to_numpy() for values in (reference, current)]
        both = pd.Series(np.concatenate(labels), dtype=object)
    else:
        both = pd.concat([reference, current], ignore_index=True)
    levels = _levels(both)
    return levels[len(reference) :], levels[: len(reference)]


def _period_sums(code: np.ndarray, n_periods: int, mass: np.ndarray | None) -> np.ndarray:
    """Count the observations, or sum their weights, in each period."""
    return np.bincount(code, weights=mass, minlength=n_periods)


def _period_psi(
    code: np.ndarray,
    level: np.ndarray,
    mass: np.ndarray | None,
    reference_level: np.ndarray,
    reference_mass: np.ndarray | None,
    n_periods: int,
    n_levels: int,
    smoothing: float,
) -> np.ndarray:
    """PSI of each period's groups against the reference groups; NaN for an empty period."""
    current = np.bincount(code * n_levels + level, weights=mass, minlength=n_periods * n_levels)
    current = current.reshape(n_periods, n_levels)
    reference = np.bincount(reference_level, weights=reference_mass, minlength=n_levels)
    psi = np.full(n_periods, np.nan)
    for number, counts in enumerate(current):
        if not counts.sum() > 0:
            continue
        both = np.column_stack([reference, counts])
        # Groups empty in both samples are left out, as psi_table never sees them.
        both = both[both.sum(axis=1) > 0]
        psi[number] = divergence(both, smoothing)[2].sum()
    return psi


def _discrimination(
    code: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    known: np.ndarray,
    mass: np.ndarray | None,
    n_periods: int,
) -> tuple[np.ndarray, np.ndarray]:
    """AUC and KS of the known outcomes of each period; NaN unless both classes are present."""
    auc, ks_value = np.full(n_periods, np.nan), np.full(n_periods, np.nan)
    rows = np.flatnonzero(known)
    rows = rows[np.argsort(code[rows], kind="stable")]
    bounds = np.searchsorted(code[rows], np.arange(n_periods + 1))
    for number in range(n_periods):
        selected = rows[bounds[number] : bounds[number + 1]]
        period_mass = _subset(mass, selected)
        classes = np.bincount(labels[selected], weights=period_mass, minlength=2)
        if not (classes > 0).all():
            continue
        auc[number] = roc_auc(values[selected], labels[selected], weights=period_mass)
        ks_value[number] = ks(values[selected], labels[selected], weights=period_mass)
    return auc, ks_value


def _subset(mass: np.ndarray | None, rows: np.ndarray) -> np.ndarray | None:
    return None if mass is None else mass[rows]


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> npt.NDArray[np.float64]:
    """Elementwise ratio that is NaN, without a warning, where the denominator is 0."""
    result = np.full(len(numerator), np.nan)
    return np.divide(numerator, denominator, out=result, where=denominator > 0)
