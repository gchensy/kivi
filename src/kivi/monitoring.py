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

from . import _counts, _edges, _validation, metrics

__all__ = ["monitoring_table", "stability_table"]


def monitoring_table(
    score: npt.ArrayLike,
    period: npt.ArrayLike,
    target: npt.ArrayLike | None = None,
    *,
    reference_period: object = None,
    reference_score: npt.ArrayLike | None = None,
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
    period : array_like
        Period of each observation, such as the month of application, without
        missing values. Periods are sorted, and categorical periods follow the
        order of their categories. If `score` and `period` are both pandas
        Series, they must have identical indices; otherwise they are matched
        by position.
    target : array_like, optional
        Binary labels, with 1 indicating the event, and missing values for
        outcomes that are not known yet, such as those of loans that have not
        reached the end of the outcome window, matched like `period`. Without
        `target`, every outcome is unknown and the table only has the columns
        ``total`` and ``psi``.
    reference_period : object, optional
        The label of the period that every period is compared with, such as
        the first month after the development sample. For datetime periods
        and pandas periods, a string that names exactly one period, such as
        ``"2024-01"`` for months, selects it. Give either `reference_period`
        or `reference_score`.
    reference_score : array_like, optional
        The scores of a reference sample that every period is compared with,
        such as the development sample, with any index. Pass the scores of
        the reference rows, such as ``score[mask]``, rather than a boolean
        mask, which is rejected.
    bins : int or array_like, default 10
        Number of quantile bands of the reference scores, or increasing band
        edges. An integer is the same as passing
        ``kivi.binning.bin_edges(reference, bins=bins)`` for the reference
        scores, with their weights as ``weights``, so tied scores can leave
        fewer bands than requested. Every period is compared on the same
        bands.
    smoothing : float, default 0.5
        Positive replacement for a zero count when calculating the PSI, as in
        `kivi.stability.psi_table`, in units of weight with `weights`.
    weights : array_like, optional
        Non-negative weight of each observation, such as the inverse of its
        sampling rate, matched like `period`. Counts become sums of weights,
        and the discrimination and the PSI are weighted. On the same band
        edges, integer weights give the same table as repeating rows.
    reference_weights : array_like, optional
        Non-negative weight of each score of `reference_score`, matched like
        it. A reference period takes its weights from `weights`.

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
        missing value. Also if neither or both of `reference_period` and
        `reference_score` are given, `reference_period` is not one period with
        observations, `reference_score` is empty, boolean or not finite and
        numeric, `reference_weights` is given with `reference_period`, `bins`
        is neither a positive integer nor increasing edges, a score lies
        outside the edges, `smoothing` is not finite and positive, or a weight
        array is not numeric, does not give each observation one finite,
        non-negative weight with a positive total, or has an index other than
        that of its sample.

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
    what the metric functions and ``psi_table(pandas.cut(reference, edges),
    pandas.cut(period_score, edges))`` give for the period alone, with the
    weights of the two samples as ``reference_weights`` and
    ``current_weights``. For a band :math:`i` with shares :math:`q_i` of the
    reference scores and :math:`p_i` of the period's scores, after
    smoothing,

    .. math::

       \mathrm{PSI} = \sum_i (p_i - q_i) \ln \frac{p_i}{q_i},

    where bands empty in both samples are left out. The reference period has
    a PSI of 0.

    A category of a categorical `period` without observations, or whose
    weights sum to zero, keeps its row with zero counts and NaN in the other
    columns, so that monitoring tables of different samples line up. Here
    the rule differs from `kivi.vintage.vintage_table` and
    `kivi.calibration.calibration_table`, which leave such categories out.
    Other periods exist only through their observations, and those whose
    weights sum to zero are left out, as repeated rows would leave them out.

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
    >>> table = monitoring_table(score, period, target, reference_period="2024-01", bins=2)
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
    _validation.shared_index(
        {"score": score, "period": period, "target": target, "weights": weights}, frames=True
    )
    _one_reference(reference_period, reference_score, "reference_score")
    _validation.check_smoothing(smoothing, booleans=True)
    raw_score = score
    score, period = _validation.as_series(score, "score"), _validation.as_series(period, "period")
    if len(score) != len(period):
        raise ValueError(
            f"score and period must have the same length, got {len(score)} and {len(period)}"
        )
    if not len(score):
        raise ValueError("score and period must not be empty")
    values = _validation.finite_scores(score, "score")
    mass = _validation.as_weights(weights, raw_score, of="score")
    known, event = (
        (None, None) if target is None else _validation.known_outcomes(target, len(score), "score")
    )
    code, periods, categorical = _periods(period)
    totals = _period_sums(code, len(periods), mass)

    if reference_score is None:
        position = _reference_position(reference_period, periods, "reference_score")
        _check_reference_period(reference_period, totals[position], mass, reference_weights)
        rows = code == position
        reference_values, reference_mass = values[rows], _subset(mass, rows)
    else:
        reference_values, reference_mass = _reference_scores(reference_score, reference_weights)
    # Called here, not in a helper, so that the warning for tied quantiles points at the caller.
    edges = _edges.band_edges(reference_values, bins, reference_mass)
    band = _bands(values, edges, "score")
    reference_band = _bands(reference_values, edges, "reference_score")
    psi = _period_psi(
        code, band, mass, reference_band, reference_mass, len(periods), len(edges) - 1, smoothing
    )

    columns = {"total": totals}
    if target is not None:
        labels = event.astype(np.intp)  # type: ignore[union-attr]  # set with target
        columns["known"] = _period_sums(code[known], len(periods), _subset(mass, known))  # type: ignore[arg-type]  # set with target
        columns["event"] = _period_sums(code[event], len(periods), _subset(mass, event))  # type: ignore[arg-type]  # set with target
        columns["event_rate"] = _counts.ratio(columns["event"], columns["known"])
        auc, ks_value = _discrimination(code, values, labels, known, mass, len(periods))  # type: ignore[arg-type]  # set with target
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
    reference_period: object = None,
    reference_groups: pd.DataFrame | None = None,
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
    reference_period : object, optional
        The label of the period that every period is compared with, as in
        `monitoring_table`. Give either `reference_period` or
        `reference_groups`.
    reference_groups : pandas.DataFrame, optional
        The groups of a reference sample that every period is compared with,
        such as the development sample, with every column of `groups` and any
        index. Its other columns are ignored.
    smoothing : float, default 0.5
        Positive replacement for a zero count when calculating shares, in
        units of weight with `weights`.
    weights : array_like, optional
        Non-negative weight of each row of `groups`, matched like `period`.
        Counts become sums of weights, so integer weights give the same table
        as repeating rows.
    reference_weights : array_like, optional
        Non-negative weight of each row of `reference_groups`. A reference
        period takes its weights from `weights`.

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
        one-dimensional, does not have one value per row or has a missing
        value, or a Series argument has an index other than that of `groups`.
        Also if neither or both of `reference_period` and `reference_groups`
        are given, `reference_period` is not one period with observations,
        `reference_groups` is not a DataFrame with rows, unique column names
        and every column of `groups`, `reference_weights` is given with
        `reference_period`, `smoothing` is not finite and positive, or a
        weight array is not numeric, does not give each row one finite,
        non-negative weight with a positive total, or has an index other than
        that of its sample.

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
    reference period has a PSI of 0.

    A category of a categorical `period` without observations, or whose
    weights sum to zero, keeps its row with NaN entries, as in
    `monitoring_table`. Other periods exist only through their rows, and
    those whose weights sum to zero are left out, as repeated rows would
    leave them out.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.monitoring import stability_table
    >>> groups = pd.DataFrame({
    ...     "housing": ["own", "own", "rent", "rent", "own", "rent", "rent", "rent"],
    ...     "region": ["north", "south", "north", "south", "north", "north", "south", None],
    ... })
    >>> period = ["2024-01"] * 4 + ["2024-02"] * 4
    >>> stability_table(groups, period, reference_period="2024-01").round(4)
    variable  housing  region
    period
    2024-01    0.0000   0.000
    2024-02    0.2747   0.231

    One value of the February region is missing, a group that January does
    not have.
    """
    if not isinstance(groups, pd.DataFrame):
        raise ValueError(f"groups must be a DataFrame, got {type(groups).__name__}")
    _check_columns(groups, "groups")
    if not len(groups):
        raise ValueError("groups must have at least one row")
    _validation.shared_index({"groups": groups, "period": period, "weights": weights}, frames=True)
    _one_reference(reference_period, reference_groups, "reference_groups")
    _validation.check_smoothing(smoothing, booleans=True)
    period = _validation.as_series(period, "period")
    if len(period) != len(groups):
        raise ValueError(
            f"period must have one value per row of groups, got {len(period)} for {len(groups)}"
        )
    mass = _row_weights(weights, groups, "weights", "groups")
    code, periods, categorical = _periods(period)
    totals = _period_sums(code, len(periods), mass)

    if reference_groups is None:
        position = _reference_position(reference_period, periods, "reference_groups")
        _check_reference_period(reference_period, totals[position], mass, reference_weights)
        reference_rows = code == position
        reference_mass = _subset(mass, reference_rows)
    else:
        if not isinstance(reference_groups, pd.DataFrame):
            raise ValueError(
                f"reference_groups must be a DataFrame, got {type(reference_groups).__name__}"
            )
        _check_columns(reference_groups, "reference_groups")
        absent = groups.columns.difference(reference_groups.columns, sort=False)
        if len(absent):
            raise ValueError(
                "reference_groups must have every column of groups, got no column "
                f"{_validation.shown(absent)}"
            )
        if not len(reference_groups):
            raise ValueError("reference_groups must have at least one row")
        if isinstance(reference_weights, pd.Series) and not reference_weights.index.equals(
            reference_groups.index
        ):
            raise ValueError("reference_groups and reference_weights must have identical indices")
        reference_mass = _row_weights(
            reference_weights, reference_groups, "reference_weights", "reference_groups"
        )
        reference_rows = None

    result = np.empty((len(periods), len(groups.columns)))
    for column_number, column in enumerate(groups.columns):
        current = groups[column]
        if reference_rows is None:
            levels, reference_level = _shared_levels(reference_groups[column], current)  # type: ignore[index]  # given when reference_rows is None
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


def _one_reference(reference_period: object, sample: object, sample_name: str) -> None:
    """Require exactly one of a reference period and a reference sample."""
    if reference_period is None and sample is None:
        raise ValueError(f"pass reference_period or {sample_name} to give the reference")
    if reference_period is not None and sample is not None:
        raise ValueError(f"pass reference_period or {sample_name}, not both")


def _check_columns(frame: pd.DataFrame, name: str) -> None:
    """Require one level of unique column names, and at least one column."""
    if not len(frame.columns):
        raise ValueError(f"{name} must have at least one column")
    if isinstance(frame.columns, pd.MultiIndex):
        raise ValueError(f"{name} must have one level of column names, got {frame.columns.nlevels}")
    if frame.columns.has_duplicates:
        repeated = _validation.shown(frame.columns[frame.columns.duplicated()])
        raise ValueError(f"{name} must have unique column names, got {repeated} twice")


def _row_weights(
    weights: npt.ArrayLike | None, frame: pd.DataFrame, name: str, of: str
) -> np.ndarray | None:
    """Weights of the rows of a DataFrame, checked as in ``kivi._validation.as_weights``."""
    if weights is not None:
        length = len(_validation.as_series(weights, name))
        if length != len(frame):
            raise ValueError(
                f"{name} must have one weight per row of {of}, got {length} for {len(frame)}"
            )
    return _validation.as_weights(weights, frame, name, of)


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
        periods: pd.Index = pd.CategoricalIndex(labels, name="period")
        categorical = True
    else:
        code, periods = pd.factorize(period, sort=True)
        periods = pd.Index(periods, name="period")
        categorical = False
    missing = code < 0
    if missing.any():
        raise ValueError(f"period must not have missing values, got {missing.sum()}")
    return code, periods, categorical


def _reference_position(label: object, periods: pd.Index, sample_name: str) -> int:
    """Position of the reference period `label` among `periods`."""
    if isinstance(label, np.ndarray) and label.ndim == 0:
        label = label.item()
    if not isinstance(label, tuple) and np.ndim(label) != 0:  # type: ignore[arg-type]  # takes any object
        raise ValueError(
            f"reference_period must be one period label, got a {type(label).__name__}; pass "
            f"the reference sample as {sample_name}"
        )
    try:
        position = periods.get_loc(label)
    except (KeyError, TypeError, ValueError, pd.errors.InvalidIndexError):
        position = None
    if isinstance(position, slice):
        # A date string selects the periods of a DatetimeIndex that it covers.
        matched = range(len(periods))[position]
        if len(matched) > 1:
            raise ValueError(
                f"reference_period must name one period, got {_validation.shown_argument(label)}, "
                f"which matches {len(matched)} periods"
            )
        position = matched[0] if len(matched) else None
    if not isinstance(position, (int, np.integer)):
        shown = ", ".join(repr(value) for value in periods[:3].tolist())
        raise ValueError(
            f"reference_period must be one of the periods {shown}"
            f"{', ...' if len(periods) > 3 else ''}, got {_validation.shown_argument(label)}"
        )
    return int(position)


def _check_reference_period(
    label: object,
    total: float,
    mass: np.ndarray | None,
    reference_weights: npt.ArrayLike | None,
) -> None:
    """Require a reference period with observations, and no reference weights of its own."""
    if reference_weights is not None:
        raise ValueError(
            "reference_weights must not be given with reference_period, whose weights come "
            "from weights"
        )
    if not total > 0:
        detail = "" if mass is None else " with positive weight"
        raise ValueError(
            f"reference_period {_validation.shown_argument(label)} has no observations{detail}"
        )


def _reference_scores(
    reference_score: npt.ArrayLike, reference_weights: npt.ArrayLike | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Finite scores and weights of an explicit reference sample."""
    raw_reference = reference_score
    reference_score = _validation.as_series(reference_score, "reference_score")
    if not len(reference_score):
        raise ValueError("reference_score must not be empty")
    if pd.api.types.is_bool_dtype(reference_score.dtype):
        # A boolean reference is almost always a row mask, such as month == first_month.
        raise ValueError(
            f"reference_score must be real and numeric, got dtype {reference_score.dtype}; "
            "pass the scores of the reference rows, such as score[mask], not the mask"
        )
    values = _validation.finite_scores(reference_score, "reference_score")
    mass = _validation.as_weights(
        reference_weights, raw_reference, "reference_weights", "reference_score"
    )
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
    levels = _levels(_counts.concat_groups(reference, current))
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
    current: np.ndarray = np.bincount(
        code * n_levels + level, weights=mass, minlength=n_periods * n_levels
    )
    current = current.reshape(n_periods, n_levels)
    reference = np.bincount(reference_level, weights=reference_mass, minlength=n_levels)
    psi = np.full(n_periods, np.nan)
    for number, counts in enumerate(current):
        if not counts.sum() > 0:
            continue
        both = np.column_stack([reference, counts])
        # Groups empty in both samples are left out, as psi_table never sees them.
        both = both[both.sum(axis=1) > 0]
        psi[number] = _counts.divergence(both, smoothing)[2].sum()
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
        auc[number] = metrics.roc_auc(values[selected], labels[selected], weights=period_mass)
        ks_value[number] = metrics.ks(values[selected], labels[selected], weights=period_mass)
    return auc, ks_value


def _subset(mass: np.ndarray | None, rows: np.ndarray) -> np.ndarray | None:
    return None if mass is None else mass[rows]
