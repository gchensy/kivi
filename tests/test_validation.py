"""Inputs that describe the same observations must have identical indices when they are pandas.

Each case lists, for one public function, the arguments that describe the same
rows. Any two of them given as pandas objects with different indices must
raise, whichever they are and whatever the others are; arrays, and pandas
objects sharing one index, are matched by position. The last tests cover how
``kivi._validation.as_real`` reads the numbers that several modules share.
"""

import dataclasses
import inspect
import itertools

import numpy as np
import pandas as pd
import pytest

from kivi import (
    _validation,
    binning,
    calibration,
    capital,
    datasets,
    expert,
    features,
    lgd,
    lifetime,
    metrics,
    monitoring,
    reject_inference,
    rules,
    scorecard,
    selection,
    spec,
    stability,
    strategy,
    univariate,
    vintage,
    woe,
)

ROWS = 8
SCORE = [0.1, 0.4, 0.35, 0.8, 0.2, 0.7, 0.55, 0.9]
NEW = [0.2, 0.3, 0.6, 0.7, 0.1, 0.8, 0.4, 0.9]
TARGET = [0, 0, 1, 1, 0, 1, 0, 1]
WEIGHTS = [1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0]
GROUPS = ["a", "a", "b", "b", "a", "b", "a", "b"]
VALUES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
PERIOD = ["p1"] * 4 + ["p2"] * 4
FRAME = {"x": VALUES, "g": GROUPS}
RULES = {"high": [0, 1, 1, 0, 0, 1, 0, 1], "low": [1, 0, 0, 0, 1, 1, 0, 0]}

# Each case: the function, its arguments that describe the same rows, and its other options.
# A dict stands for a DataFrame, or for the array of its values.
CASES = {
    "bin_edges": (binning.bin_edges, {"values": VALUES, "weights": WEIGHTS}, {"bins": 2}),
    "chimerge_edges": (
        binning.chimerge_edges,
        {"values": VALUES, "target": TARGET, "weights": WEIGHTS},
        {},
    ),
    "tree_edges": (
        binning.tree_edges,
        {"values": VALUES, "target": TARGET, "weights": WEIGHTS},
        {},
    ),
    "ks_edges": (
        binning.ks_edges,
        {"values": VALUES, "target": TARGET, "weights": WEIGHTS},
        {},
    ),
    "monotonic_edges": (
        binning.monotonic_edges,
        {"values": VALUES, "target": TARGET, "weights": WEIGHTS},
        {},
    ),
    "category_groups": (binning.category_groups, {"values": GROUPS, "weights": WEIGHTS}, {}),
    "chimerge_groups": (
        binning.chimerge_groups,
        {"values": GROUPS, "target": TARGET, "weights": WEIGHTS},
        {},
    ),
    "woe_table": (woe.woe_table, {"groups": GROUPS, "target": TARGET, "weights": WEIGHTS}, {}),
    # Counts per group: the rows are the groups.
    "woe": (woe.woe, {"event_counts": WEIGHTS, "non_event_counts": VALUES}, {}),
    "iv": (woe.iv, {"event_counts": WEIGHTS, "non_event_counts": VALUES}, {}),
    "psi": (stability.psi, {"reference_counts": WEIGHTS, "current_counts": VALUES}, {}),
    "roc_auc": (metrics.roc_auc, {"score": SCORE, "target": TARGET, "weights": WEIGHTS}, {}),
    "gini": (metrics.gini, {"score": SCORE, "target": TARGET, "weights": WEIGHTS}, {}),
    "ks": (metrics.ks, {"score": SCORE, "target": TARGET, "weights": WEIGHTS}, {}),
    "roc_table": (metrics.roc_table, {"score": SCORE, "target": TARGET, "weights": WEIGHTS}, {}),
    "roc_auc_interval": (
        metrics.roc_auc_interval,
        {"score": SCORE, "target": TARGET, "weights": WEIGHTS},
        {},
    ),
    "delong_test": (
        metrics.delong_test,
        {"current": SCORE, "new": NEW, "target": TARGET, "weights": WEIGHTS},
        {},
    ),
    "gains_table": (
        metrics.gains_table,
        {"score": SCORE, "target": TARGET, "weights": WEIGHTS},
        {"bins": 2},
    ),
    "brier_score": (
        metrics.brier_score,
        {"probability": SCORE, "target": TARGET, "weights": WEIGHTS},
        {},
    ),
    "expected_maximum_profit": (
        metrics.expected_maximum_profit,
        {"score": SCORE, "target": TARGET, "weights": WEIGHTS},
        {},
    ),
    "generalized_auc": (
        metrics.generalized_auc,
        {"estimated": [1, 1, 2, 2, 3, 3, 1, 2], "realized": VALUES, "weights": WEIGHTS},
        {},
    ),
    "calibration_table": (
        calibration.calibration_table,
        {"groups": GROUPS, "target": TARGET, "probability": SCORE, "weights": WEIGHTS},
        {},
    ),
    "spiegelhalter_test": (
        calibration.spiegelhalter_test,
        {"probability": SCORE, "target": TARGET, "weights": WEIGHTS},
        {},
    ),
    "brier_decomposition": (
        calibration.brier_decomposition,
        {"groups": GROUPS, "target": TARGET, "probability": SCORE, "weights": WEIGHTS},
        {},
    ),
    "backtest_table": (
        calibration.backtest_table,
        {"groups": GROUPS, "realized": SCORE, "estimated": NEW, "weights": WEIGHTS},
        {},
    ),
    # Counts per grade: the rows are the grades of the scale.
    "concentration_test": (
        calibration.concentration_test,
        {"counts": VALUES, "initial_counts": WEIGHTS},
        {},
    ),
    "platt_fit": (
        calibration.platt_fit,
        {"score": SCORE, "target": TARGET, "weights": WEIGHTS},
        {},
    ),
    "isotonic_fit": (
        calibration.isotonic_fit,
        {"score": SCORE, "target": TARGET, "weights": WEIGHTS},
        {},
    ),
    "logistic_regression": (
        selection.logistic_regression,
        {"X": {"x": SCORE}, "target": TARGET, "weights": WEIGHTS},
        {},
    ),
    "stepwise": (selection.stepwise, {"X": {"x": SCORE}, "target": TARGET, "weights": WEIGHTS}, {}),
    "selection_path": (
        selection.selection_path,
        {"X": {"x": SCORE}, "target": TARGET, "weights": WEIGHTS},
        {},
    ),
    # The validation sample describes other rows, but its X and target share one index. The
    # development X is an array, so that the validation X may be an array or a DataFrame.
    "selection_path-validation": (
        lambda X, target: selection.selection_path(
            np.array(SCORE)[:, None], TARGET, validation=(X, target)
        ),
        {"X": {"x": NEW}, "target": TARGET},
        {},
    ),
    "cutoff_table": (
        strategy.cutoff_table,
        {
            "score": SCORE,
            "target": TARGET,
            "loss": [10.0] * ROWS,
            "revenue": VALUES,
            "weights": WEIGHTS,
        },
        {"cutoffs": [0.5]},
    ),
    "confusion_table": (
        strategy.confusion_table,
        {"score": SCORE, "target": TARGET, "weights": WEIGHTS},
        {"cutoffs": [0.5]},
    ),
    "swap_set": (
        strategy.swap_set,
        {"current": SCORE, "new": NEW, "target": TARGET, "weights": WEIGHTS},
        {"current_cutoff": 0.5, "new_cutoff": 0.5},
    ),
    "rule_table": (
        rules.rule_table,
        {"rules": RULES, "target": TARGET, "weights": WEIGHTS},
        {},
    ),
    "rule_set_table": (
        rules.rule_set_table,
        {"rules": RULES, "target": TARGET, "weights": WEIGHTS},
        {},
    ),
    "expected_loss_table": (
        strategy.expected_loss_table,
        {
            "groups": GROUPS,
            "probability": SCORE,
            "lgd": [0.45] * ROWS,
            "ead": [100.0] * ROWS,
            "weights": WEIGHTS,
        },
        {},
    ),
    "monitoring_table": (
        monitoring.monitoring_table,
        {"score": SCORE, "period": PERIOD, "target": TARGET, "weights": WEIGHTS},
        {"reference_period": "p1", "bins": 2},
    ),
    "stability_table": (
        monitoring.stability_table,
        {"groups": {"g": GROUPS}, "period": PERIOD, "weights": WEIGHTS},
        {"reference_period": "p1"},
    ),
    "workout_lgd": (
        lgd.workout_lgd,
        {
            "loan": list("abcdefgh"),
            "default_date": ["2023-01-31"] * ROWS,
            "ead": VALUES,
            "rate": [0.05, 0.06, 0.05, 0.07, 0.05, 0.06, 0.05, 0.08],
        },
        {
            "flow_loan": list("abcdefgh"),
            "flow_date": ["2023-07-31"] * ROWS,
            "flow_amount": [0.5, -0.1, 1.5, 2.0, -0.2, 3.0, 3.5, 4.0],
        },
    ),
    "conversion_factor": (
        lgd.conversion_factor,
        {"ead": VALUES, "drawn": [0.5] * ROWS, "limit": [4.0] * ROWS},
        {},
    ),
    "lgd_table": (lgd.lgd_table, {"groups": GROUPS, "lgd": SCORE, "weights": WEIGHTS}, {}),
    "fractional_regression": (
        lgd.fractional_regression,
        {"X": {"x": VALUES}, "response": SCORE, "weights": WEIGHTS},
        {},
    ),
    "survival_table": (
        lifetime.survival_table,
        {
            "duration": [1, 2, 2, 3, 3, 4, 4, 4],
            "event": [1, 0, 1, 0, 0, 1, 0, 0],
            "competing": [0, 1, 0, 0, 1, 0, 0, 0],
            "entry": [0, 0, 1, 0, 2, 0, 1, 3],
            "weights": WEIGHTS,
        },
        {},
    ),
    "asset_correlation": (
        capital.asset_correlation,
        {"probability": SCORE, "turnover": [10.0, 20.0, 30.0, 40.0, 5.0, 8.0, 60.0, 50.0]},
        {"asset_class": "corporate"},
    ),
    "capital_requirement": (
        capital.capital_requirement,
        {
            "probability": SCORE,
            "lgd": [0.45] * ROWS,
            "maturity": [1.0, 2.5, 5.0, 3.0, 2.0, 1.5, 4.0, 2.5],
            "turnover": [10.0, 20.0, 30.0, 40.0, 5.0, 8.0, 60.0, 50.0],
        },
        {"asset_class": "corporate"},
    ),
    "capital_requirement-correlation": (
        capital.capital_requirement,
        {"probability": SCORE, "lgd": [0.45] * ROWS, "correlation": [0.12] * 4 + [0.2] * 4},
        {"asset_class": "mortgage"},
    ),
    "conditional_pd": (
        capital.conditional_pd,
        {"probability": SCORE, "correlation": [0.12] * ROWS, "factor": VALUES},
        {},
    ),
    "conditional_pd-level": (
        capital.conditional_pd,
        {"probability": SCORE, "correlation": [0.12] * ROWS, "level": NEW},
        {},
    ),
    "systematic_factor": (
        capital.systematic_factor,
        {"default_rate": NEW, "probability": SCORE, "correlation": [0.12] * ROWS},
        {},
    ),
    "most_prudent_pd": (
        capital.most_prudent_pd,
        {"event": [0, 1, 0, 2, 0, 1, 3, 2], "total": [100] * ROWS},
        {},
    ),
    "stage_table": (
        lifetime.stage_table,
        {
            "stage": [1, 2, 3, 1, 1, 2, 3, 1],
            "exposure": [100.0] * ROWS,
            "allowance": [1.0] * ROWS,
            "weights": WEIGHTS,
        },
        {},
    ),
    # The four inputs of the schedule share one index; the payments describe other rows.
    "delinquency_panel": (
        vintage.delinquency_panel,
        {
            "loan": ["a"] * 4 + ["b"] * 4,
            "due_date": ["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15"] * 2,
            "amount_due": [100.0] * ROWS,
            "principal": [90.0] * ROWS,
        },
        {
            "payment_loan": ["a", "b"],
            "payment_date": ["2024-01-15", "2024-03-01"],
            "payment_amount": [100.0, 150.0],
            "end": "2024-05",
        },
    ),
    # The payments share one index too; a wrapper takes them as its positional arguments.
    "delinquency_panel-payments": (
        lambda payment_loan, payment_date, payment_amount: vintage.delinquency_panel(
            ["a", "a", "b"],
            ["2024-01-15", "2024-02-15", "2024-01-20"],
            [100.0] * 3,
            payment_loan,
            payment_date,
            payment_amount,
            end="2024-04",
        ),
        {
            "payment_loan": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "payment_date": ["2024-01-10"] * 4 + ["2024-02-10"] * 4,
            "payment_amount": [10.0] * ROWS,
        },
        {},
    ),
    "vintage_table": (
        vintage.vintage_table,
        {
            "loan": [1, 1, 2, 2, 3, 3, 4, 4],
            "cohort": ["c1"] * 4 + ["c2"] * 4,
            "month_on_book": [1, 2] * 4,
            "bad": [0, 0, 0, 1, 0, 0, 0, 0],
            "weights": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        },
        {},
    ),
    "transition_matrix": (
        vintage.transition_matrix,
        {"before": GROUPS, "after": ["a", "b", "b", "a", "a", "b", "b", "b"], "weights": WEIGHTS},
        {},
    ),
    "roll_rates": (
        vintage.roll_rates,
        {"before": GROUPS, "after": ["a", "b", "b", "a", "a", "b", "b", "b"], "weights": WEIGHTS},
        {"states": ["a", "b"]},
    ),
    "migration_bandwidth": (
        vintage.migration_bandwidth,
        {"before": GROUPS, "after": ["a", "b", "b", "a", "a", "b", "x", "b"], "weights": WEIGHTS},
        {"grades": ["a", "b"]},
    ),
    "migration_test": (
        vintage.migration_test,
        {"before": GROUPS, "after": ["a", "b", "b", "a", "a", "b", "x", "b"], "weights": WEIGHTS},
        {"grades": ["a", "b"]},
    ),
    "fit_spec": (spec.fit_spec, {"frame": FRAME, "target": TARGET, "weights": WEIGHTS}, {}),
    "profile_table": (
        univariate.profile_table,
        {"frame": FRAME, "target": TARGET, "weights": WEIGHTS},
        {},
    ),
}
DEFAULT = pd.RangeIndex(ROWS)
OTHER = pd.Index(range(100, 100 + ROWS))


def _is_frame(case: str, name: str) -> bool:
    """Whether the argument is always a DataFrame, which always has an index."""
    return name in ("frame", "rules") or (case == "stability_table" and name == "groups")


def _call(case: str, indices: dict):
    """Call the function of a case: pandas inputs on ``indices`` by name, arrays otherwise."""
    function, inputs, options = CASES[case]
    parameters = inspect.signature(function).parameters
    positional, keywords = [], {}
    for name, values in inputs.items():
        index = indices.get(name, DEFAULT if _is_frame(case, name) else None)
        if isinstance(values, dict):
            frame = pd.DataFrame(values, index=DEFAULT if index is None else index)
            argument = frame if index is not None else frame.to_numpy(dtype=float)
        elif index is None:
            argument = np.asarray(values)
        else:
            argument = pd.Series(values, index=index)
        if parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            positional.append(argument)
        else:
            keywords[name] = argument
    return function(*positional, **keywords, **options)


def _pairs():
    for case, (_, inputs, _) in CASES.items():
        for first, second in itertools.permutations(inputs, 2):
            yield pytest.param(case, first, second, id=f"{case}-{first}-{second}")


@pytest.mark.parametrize("case,first,second", list(_pairs()))
def test_two_pandas_inputs_with_other_indices_raise(case, first, second):
    indices = {first: DEFAULT, second: OTHER}
    # The message names the argument whose index differs, such as "target and weights must
    # have identical indices".
    message = rf"(?=.*\b{second}\b).*identical indices"
    with pytest.raises(ValueError, match=message):
        _call(case, indices)


@pytest.mark.parametrize("case", CASES)
def test_pandas_inputs_on_one_index_and_arrays_are_matched_by_position(case):
    _, inputs, _ = CASES[case]
    expected = _call(case, {})
    for size in range(1, len(inputs) + 1):
        for names in itertools.combinations(inputs, size):
            # DataFrames always have an index, so they share the one of the other inputs.
            indices = {name: OTHER for name in inputs if name in names or _is_frame(case, name)}
            _assert_same(_call(case, indices), expected)


def test_target_and_weights_series_on_other_indices_raise_with_an_array_score():
    score = np.array([1.0, 2, 3, 4])
    target = pd.Series([0, 1, 0, 1], index=[5, 6, 7, 8])
    weights = pd.Series([1.0] * 4)
    with pytest.raises(ValueError, match="^target and weights must have identical indices$"):
        metrics.roc_auc(score, target, weights=weights)
    assert metrics.roc_auc(score, target, weights=weights.set_axis(target.index)) == 0.75


def _assert_same(result, expected):
    if isinstance(result, pd.DataFrame):
        # The variables of a model fitted on an array are numbered, not named, both in the index
        # of logistic_regression and in the variable column of selection_path.
        drop = result.index.name is None or result.index[0] == "intercept"
        result, expected = (
            table.reset_index(drop=drop).drop(columns="variable", errors="ignore")
            for table in (result, expected)
        )
        pd.testing.assert_frame_equal(result, expected)
    elif isinstance(result, pd.Index):
        assert len(result) == len(expected)
    elif isinstance(result, pd.Series | np.ndarray):
        np.testing.assert_allclose(
            np.asarray(result, dtype=float), np.asarray(expected, dtype=float)
        )
    elif isinstance(result, tuple):
        for item, other in zip(result, expected, strict=True):
            _assert_same(item, other)
    elif isinstance(result, float):
        assert result == pytest.approx(expected)
    else:
        assert result == expected


# --- as_real ------------------------------------------------------------------------------

STRING_DTYPE = "(object|str)"  # pandas 3 reads a list of strings as dtype str


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0.5, pd.NA], [0.5, np.nan]),
        ([0.5, None, np.nan], [0.5, np.nan, np.nan]),
        (np.array([1, pd.NA, 2.5], dtype=object), [1.0, np.nan, 2.5]),
        (pd.Series([0.25, pd.NA], dtype=object), [0.25, np.nan]),
        (pd.Series([0.25, None], dtype="Float64"), [0.25, np.nan]),
        (pd.Series([None, None], dtype=object), [np.nan, np.nan]),
        (np.array([np.float32(0.5), np.int64(3)], dtype=object), [0.5, 3.0]),
    ],
)
def test_as_real_reads_numbers_and_missing_values_of_any_dtype(values, expected):
    np.testing.assert_array_equal(_validation.as_real(values, "x"), expected)


def test_as_real_keeps_the_shape_of_object_arrays():
    values = np.array([[0.1, None], [pd.NA, 2]], dtype=object)
    np.testing.assert_array_equal(_validation.as_real(values, "x"), [[0.1, np.nan], [np.nan, 2]])
    assert _validation.as_real(np.array([], dtype=object), "x").shape == (0,)


@pytest.mark.parametrize(
    "values, message",
    [
        (["0.1", "0.2"], f"got dtype {STRING_DTYPE} with the value '0.1'$"),
        ([0.5, pd.NA, "a"], "got dtype object with the value 'a'$"),
        (np.array([0.5, b"a"], dtype=object), "got dtype object with the value b'a'$"),
        (pd.Series(["a"], dtype="string"), "got dtype string with the value 'a'$"),
        (np.array([[0.5, "b"]], dtype=object), "got dtype object with the value 'b'$"),
        (np.array([[1, 2], [3]], dtype=object), r"got dtype object with the value \[1, 2\]$"),
        ([1, True], "got dtype object with the value True$"),
        ([True, False], "got dtype bool$"),
        (np.array([1 + 2j]), "got dtype complex128$"),
        (None, "got dtype object$"),
        (pd.NA, "got dtype object$"),
    ],
)
def test_as_real_shows_the_element_that_is_not_a_number(values, message):
    with pytest.raises(ValueError, match=f"^x must be real and numeric, {message}"):
        _validation.as_real(values, "x")


def test_as_real_reads_booleans_only_when_asked():
    for values in ([True, False], [True, None], np.array([1, False], dtype=object)):
        result = _validation.as_real(values, "x", booleans=True)
        np.testing.assert_array_equal(result, pd.Series(values, dtype=object).astype(float))
    np.testing.assert_array_equal(
        _validation.as_real([True, None], "x", booleans=True), [1, np.nan]
    )


SMOOTHED = {
    "woe_table": lambda smoothing: woe.woe_table(GROUPS, TARGET, smoothing=smoothing),
    "woe": lambda smoothing: woe.woe([0, 2, 3], [3, 1, 0], smoothing=smoothing),
    "iv": lambda smoothing: woe.iv([0, 2, 3], [3, 1, 0], smoothing=smoothing),
    "psi_table": lambda smoothing: stability.psi_table(GROUPS[:4], GROUPS[4:], smoothing=smoothing),
    "psi": lambda smoothing: stability.psi([0, 2, 3], [3, 1, 0], smoothing=smoothing),
    "chimerge_groups": lambda smoothing: binning.chimerge_groups(
        GROUPS, TARGET, smoothing=smoothing
    ),
    "monotonic_edges": lambda smoothing: binning.monotonic_edges(
        VALUES, TARGET, smoothing=smoothing
    ),
    "fit_grade_pd": lambda smoothing: calibration.fit_grade_pd(
        [0, 3, 5], [400, 500, 300], smoothing=smoothing
    ),
    "fit_spec": lambda smoothing: spec.fit_spec(
        pd.DataFrame(FRAME), TARGET, bins=2, smoothing=smoothing
    ),
    "BinningSpec": lambda smoothing: dataclasses.replace(
        spec.fit_spec(pd.DataFrame(FRAME), TARGET, bins=2), smoothing=smoothing
    ),
    "monitoring_table": lambda smoothing: monitoring.monitoring_table(
        SCORE, PERIOD, TARGET, reference_period="p1", bins=2, smoothing=smoothing
    ),
    "stability_table": lambda smoothing: monitoring.stability_table(
        pd.DataFrame({"g": GROUPS}), PERIOD, reference_period="p1", smoothing=smoothing
    ),
}


def test_every_public_smoothing_argument_is_checked():
    modules = (binning, calibration, capital, lgd, lifetime, metrics, monitoring, reject_inference)
    modules += (scorecard, selection, spec, stability, strategy, univariate, vintage, woe, datasets)
    modules += (expert, features, rules)
    names = set()
    for module in modules:
        for name in module.__all__:
            item = getattr(module, name)
            # Exceptions have no signature of their own.
            if isinstance(item, type) and issubclass(item, Exception):
                continue
            if "smoothing" in inspect.signature(item).parameters:
                names.add(name)
    assert names == set(SMOOTHED)


@pytest.mark.parametrize("case", SMOOTHED)
@pytest.mark.parametrize(
    "smoothing,shown",
    [
        ("1", "'1'"),
        (None, "None"),
        ([0.5, 0.5], r"\[0.5, 0.5\]"),
        (np.array(0.5), r"array\(0.5\)"),
        (np.float64(np.nan), "nan"),
        (np.inf, "inf"),
        (0, "0"),
        (np.int64(-1), "-1"),
        (False, "False"),
    ],
)
def test_smoothing_must_be_a_finite_positive_number(case, smoothing, shown):
    with pytest.raises(ValueError, match=f"^smoothing must be finite and positive, got {shown}$"):
        SMOOTHED[case](smoothing)


@pytest.mark.parametrize("case", SMOOTHED)
@pytest.mark.parametrize("flag", [True, np.True_])
def test_smoothing_true_raises(case, flag):
    # kivi 0.9 read True as 1 in nine of these functions, with a FutureWarning.
    with pytest.raises(ValueError, match="^smoothing must be finite and positive, got True$"):
        SMOOTHED[case](flag)


@pytest.mark.parametrize(
    "call,message",
    [
        (
            lambda: binning.bin_edges(VALUES, bins=np.int64(0)),
            "bins must be a positive integer, got 0",
        ),
        (
            lambda: binning.chimerge_edges(VALUES, TARGET, alpha=np.float64(1.5)),
            "alpha must be None or between 0 and 1, got 1.5",
        ),
        (
            lambda: metrics.gains_table(SCORE, TARGET, bins=np.int64(0)),
            "bins must be a positive integer or at least two increasing edges, got 0",
        ),
        (
            lambda: metrics.gains_table(SCORE, TARGET, bins=pd.Series([0.5, 0.1])),
            r"bins must be a positive integer or at least two increasing edges, got \[0.5, 0.1\]",
        ),
        (
            lambda: strategy.cutoff_table(SCORE, TARGET, cutoffs=np.array([0.5, 0.1])),
            r"cutoffs must be a positive integer or a sequence of increasing numbers, got "
            r"\[0.5, 0.1\]",
        ),
        (
            lambda: strategy.confusion_table(SCORE, TARGET, cutoffs=np.array([0.5, 0.1])),
            r"cutoffs must be a positive integer or a sequence of increasing numbers, got "
            r"\[0.5, 0.1\]",
        ),
        (
            lambda: binning.ks_edges(VALUES, TARGET, min_bin_share=np.float64(0.6)),
            "min_bin_share must be between 0 and 0.5, got 0.6",
        ),
        (
            lambda: expert.ahp_weights([[1.0, 2.0], [0.5, 1.0]], rtol=np.float64(-1)),
            "rtol must be finite and non-negative, got -1.0",
        ),
        (
            lambda: vintage.delinquency_panel(
                ["a"], ["2024-01-15"], [1.0], [], [], [], end="2024-02", tolerance=np.float64(-1)
            ),
            "tolerance must be finite and non-negative, got -1.0",
        ),
        (
            lambda: metrics.roc_auc_interval(SCORE, TARGET, level=np.float64(2)),
            "level must be a number between 0 and 1, got 2.0",
        ),
        (
            lambda: lifetime.transition_pd(
                pd.DataFrame([[0.9, 0.1], [0.0, 1.0]], index=["A", "D"], columns=["A", "D"]),
                horizons=np.int64(0),
            ),
            "horizons must be a positive integer or a sequence of increasing positive integers, "
            "got 0",
        ),
    ],
)
def test_messages_show_numpy_scalars_as_python_numbers(call, message):
    with pytest.raises(ValueError, match=f"^{message}$"):
        call()


@pytest.mark.parametrize("rng", [1.5, "a", -1, [1.5]])
def test_rng_must_be_accepted_by_default_rng(rng):
    from kivi.datasets import make_applications, make_loan_panel, make_transactions

    applications = make_applications(n_applicants=10, include_risk=True, rng=0)
    message = "^rng must be None, an integer, a SeedSequence or a Generator, got "
    for call in (
        lambda: make_applications(n_applicants=10, rng=rng),
        lambda: make_loan_panel(n_loans=10, n_months=2, rng=rng),
        lambda: make_transactions(applications, application_date="2024-06-30", rng=rng),
    ):
        with pytest.raises(ValueError, match=message):
            call()


@pytest.mark.parametrize(
    "rng", [None, 0, np.random.SeedSequence(0), np.random.default_rng(0), [1, 2]]
)
def test_rng_accepts_what_default_rng_accepts(rng):
    assert isinstance(_validation.as_generator(rng), np.random.Generator)


def test_shown_stands_a_series_or_index_for_its_first_value():
    assert _validation.shown(pd.Series([np.int64(2), 3])) == "2"
    assert _validation.shown(pd.Index(["a", "b"])) == "'a'"
    assert _validation.shown(np.float32(0.5)) == "0.5"
    assert _validation.shown_argument(pd.Series([1, 2])) == "[1, 2]"
    assert _validation.shown_argument(np.array([0.5])) == "[0.5]"
    assert _validation.shown_argument(3) == "3"
    assert _validation.shown(pd.Series([], dtype=float)) == "[]"
    assert _validation.shown(pd.Index([])) == "[]"
    assert _validation.shown_argument(pd.Series([], dtype=float)) == "[]"
    assert _validation.shown_argument(np.array(0.5)) == "array(0.5)"


def test_check_count_returns_an_int():
    assert _validation.check_count(np.int64(3), "bins") == 3
    assert type(_validation.check_count(np.int64(3), "bins")) is int
    with pytest.raises(ValueError, match="^n must be a positive integer, got True$"):
        _validation.check_count(np.True_, "n")


@pytest.mark.parametrize(
    "target,message",
    [
        ([0, 2], "target must contain only 0 and 1, got 2"),
        ([0, None], "target must contain only 0 and 1, got None"),
        ([0, np.nan], "target must contain only 0 and 1, got nan"),
        (["0", "1"], "target must contain only 0 and 1, got '0'"),
        ([1, 1], "target must contain both 0 and 1, got only 1"),
        ([], "target must contain both 0 and 1, got no labels"),
    ],
)
def test_binary_labels_show_the_label_at_fault(target, message):
    with pytest.raises(ValueError, match=f"^{message}$"):
        _validation.binary_labels(pd.Series(target, dtype=object if target else float))


def test_binary_labels_without_both_classes():
    labels = _validation.binary_labels(pd.Series([1.0, 1.0]), both_classes=False)
    assert labels.tolist() == [1, 1]
    assert labels.dtype == np.intp


def _option_cases():
    """Each public option check, as a call that passes the option ``o``."""
    frame = pd.DataFrame(FRAME)
    X = pd.DataFrame({"x": VALUES, "y": SCORE})
    table = woe.woe_table(GROUPS, TARGET)
    scale = {"pdo": 20, "base_score": 600, "base_odds": 50}
    card = scorecard.points_table({"g": table}, {"g": 1.0}, 0.0, **scale)
    points = scorecard.woe_to_points(
        pd.DataFrame({"g": woe.woe_transform(GROUPS, table)}), {"g": 1.0}, 0.0, **scale
    )
    matrix = pd.DataFrame([[0.9, 0.1], [0.0, 1.0]], index=["A", "D"], columns=["A", "D"])
    event, total, rates = [1, 2, 3], [10, 20, 30], [0.1, 0.1, 0.1]
    comparison = [[1.0, 2.0], [0.5, 1.0]]
    events = pd.DataFrame(
        {"c": [1, 1], "t": pd.to_datetime(["2024-01-10", "2024-02-10"]), "a": [1.0, 2.0]}
    )
    observations = pd.DataFrame({"c": [1, 2], "d": pd.to_datetime(["2024-03-31"] * 2)})
    keys = {"id": "c", "time": "t", "observation_time": "d"}
    panel = pd.DataFrame(
        {
            "c": [1, 1],
            "m": pd.PeriodIndex(["2024-01", "2024-02"], freq="M"),
            "s": pd.Categorical(["current", "30-59"], ["current", "30-59"], ordered=True),
        }
    )
    panel_keys = {"id": "c", "month": "m", "status": "s", "observation_time": "d"}
    group = {"time": "t"}
    applications = datasets.make_applications(n_applicants=2, include_risk=True, rng=0)
    defaults = datasets.make_defaults(n_defaults=5, rng=0)
    return {
        # binning
        "bin_edges-bins": lambda o: binning.bin_edges(VALUES, bins=o),
        "bin_edges-method": lambda o: binning.bin_edges(VALUES, method=o),
        "chimerge_edges-max_bins": lambda o: binning.chimerge_edges(VALUES, TARGET, max_bins=o),
        "chimerge_edges-alpha": lambda o: binning.chimerge_edges(VALUES, TARGET, alpha=o),
        "chimerge_edges-initial_bins": lambda o: binning.chimerge_edges(
            VALUES, TARGET, initial_bins=o
        ),
        "tree_edges-max_bins": lambda o: binning.tree_edges(VALUES, TARGET, max_bins=o),
        "tree_edges-min_bin_share": lambda o: binning.tree_edges(VALUES, TARGET, min_bin_share=o),
        "tree_edges-criterion": lambda o: binning.tree_edges(VALUES, TARGET, criterion=o),
        "ks_edges-max_bins": lambda o: binning.ks_edges(VALUES, TARGET, max_bins=o),
        "ks_edges-min_bin_share": lambda o: binning.ks_edges(VALUES, TARGET, min_bin_share=o),
        "monotonic_edges-max_bins": lambda o: binning.monotonic_edges(VALUES, TARGET, max_bins=o),
        "monotonic_edges-direction": lambda o: binning.monotonic_edges(VALUES, TARGET, direction=o),
        "monotonic_edges-initial_bins": lambda o: binning.monotonic_edges(
            VALUES, TARGET, initial_bins=o
        ),
        "monotonic_edges-smoothing": lambda o: binning.monotonic_edges(VALUES, TARGET, smoothing=o),
        "category_groups-min_share": lambda o: binning.category_groups(GROUPS, min_share=o),
        "category_groups-min_count": lambda o: binning.category_groups(GROUPS, min_count=o),
        "category_groups-rare_label": lambda o: binning.category_groups(GROUPS, rare_label=o),
        "chimerge_groups-max_groups": lambda o: binning.chimerge_groups(
            GROUPS, TARGET, max_groups=o
        ),
        "chimerge_groups-alpha": lambda o: binning.chimerge_groups(GROUPS, TARGET, alpha=o),
        "chimerge_groups-min_group_share": lambda o: binning.chimerge_groups(
            GROUPS, TARGET, min_group_share=o
        ),
        "chimerge_groups-smoothing": lambda o: binning.chimerge_groups(GROUPS, TARGET, smoothing=o),
        "group_categories-unseen": lambda o: binning.group_categories(
            ["c"], {"a": "A", "b": "B"}, unseen=o
        ),
        # woe and stability
        "woe-smoothing": lambda o: woe.woe([1, 2], [2, 1], smoothing=o),
        "iv-smoothing": lambda o: woe.iv([1, 2], [2, 1], smoothing=o),
        "woe_table-smoothing": lambda o: woe.woe_table(GROUPS, TARGET, smoothing=o),
        "woe_transform-unseen": lambda o: woe.woe_transform(["c"], table, unseen=o),
        "psi-smoothing": lambda o: stability.psi([1, 2], [2, 1], smoothing=o),
        "psi_table-smoothing": lambda o: stability.psi_table(GROUPS, GROUPS, smoothing=o),
        "csi_table-unseen_points": lambda o: stability.csi_table(
            GROUPS, GROUPS, pd.Series({"a": 1.0, "b": 2.0}), unseen_points=o
        ),
        # metrics
        "roc_auc_interval-level": lambda o: metrics.roc_auc_interval(SCORE, TARGET, level=o),
        "delong_test-level": lambda o: metrics.delong_test(SCORE, NEW, TARGET, level=o),
        "generalized_auc-level": lambda o: metrics.generalized_auc(
            [1, 1, 2, 3], [1, 2, 2, 3], level=o
        ),
        # calibration
        "platt_predict-slope": lambda o: calibration.platt_predict(SCORE, o, 0.0),
        "platt_predict-intercept": lambda o: calibration.platt_predict(SCORE, 1.0, o),
        "isotonic_fit-increasing": lambda o: calibration.isotonic_fit(SCORE, TARGET, increasing=o),
        "herfindahl_index-normalize": lambda o: calibration.herfindahl_index([1, 2], normalize=o),
        "concentration_test-initial_cv": lambda o: calibration.concentration_test(
            [1, 2], initial_cv=o
        ),
        "hosmer_lemeshow-ddof": lambda o: calibration.hosmer_lemeshow(event, total, rates, ddof=o),
        "master_scale-boundary": lambda o: calibration.master_scale([0.01, 0.1], boundary=o),
        "prior_correction-sample_rate": lambda o: calibration.prior_correction(
            SCORE, sample_rate=o, population_rate=0.1
        ),
        "corrected_intercept-intercept": lambda o: calibration.corrected_intercept(
            o, sample_rate=0.5, population_rate=0.1
        ),
        "corrected_intercept-population_rate": lambda o: calibration.corrected_intercept(
            0.0, sample_rate=0.5, population_rate=o
        ),
        "fit_grade_pd-smoothing": lambda o: calibration.fit_grade_pd(event, total, smoothing=o),
        # capital
        "asset_correlation-asset_class": lambda o: capital.asset_correlation(0.01, asset_class=o),
        "asset_correlation-turnover_range": lambda o: capital.asset_correlation(
            0.01, asset_class="corporate", turnover=10.0, turnover_range=o
        ),
        "asset_correlation-financial": lambda o: capital.asset_correlation(
            0.01, asset_class="corporate", financial=o
        ),
        "capital_requirement-level": lambda o: capital.capital_requirement(
            0.01, 0.45, asset_class="corporate", level=o
        ),
        "capital_requirement-scaling": lambda o: capital.capital_requirement(
            0.01, 0.45, asset_class="corporate", scaling=o
        ),
        "most_prudent_pd-level": lambda o: capital.most_prudent_pd([0], [10], level=o),
        "most_prudent_pd-correlation": lambda o: capital.most_prudent_pd([0], [10], correlation=o),
        # lifetime
        "convert_pd-kind": lambda o: lifetime.convert_pd([[0.1, 0.2]], kind=o, to="marginal"),
        "convert_pd-to": lambda o: lifetime.convert_pd([[0.1, 0.2]], kind="cumulative", to=o),
        "transition_pd-default_state": lambda o: lifetime.transition_pd(
            matrix, horizons=2, default_state=o
        ),
        "expected_credit_loss-horizon": lambda o: lifetime.expected_credit_loss(
            [[0.1, 0.1]], 0.5, 100.0, rate=0.0, horizon=o
        ),
        "significant_increase-relative": lambda o: lifetime.significant_increase(
            [0.2], [0.1], relative=o, absolute=None
        ),
        "significant_increase-absolute": lambda o: lifetime.significant_increase(
            [0.2], [0.1], relative=None, absolute=o
        ),
        "assign_stages-stage_2_days": lambda o: lifetime.assign_stages([False], stage_2_days=o),
        "assign_stages-stage_3_days": lambda o: lifetime.assign_stages([False], stage_3_days=o),
        # reject inference
        "fuzzy_augmentation-factor": lambda o: reject_inference.fuzzy_augmentation([0.2], factor=o),
        "parceling-factor": lambda o: reject_inference.parceling(
            SCORE, TARGET, [0.3], bins=[0, 0.5, 1], factor=o
        ),
        "parceling-expected": lambda o: reject_inference.parceling(
            SCORE, TARGET, [0.3], bins=[0, 0.5, 1], expected=o
        ),
        "parceling-rng": lambda o: reject_inference.parceling(
            SCORE, TARGET, [0.3], bins=[0, 0.5, 1], rng=o
        ),
        # scorecard
        "score_scaling-pdo": lambda o: scorecard.score_scaling(pdo=o, base_score=600, base_odds=50),
        "score_scaling-base_score": lambda o: scorecard.score_scaling(
            pdo=20, base_score=o, base_odds=50
        ),
        "score_scaling-base_odds": lambda o: scorecard.score_scaling(
            pdo=20, base_score=600, base_odds=o
        ),
        "points_table-intercept": lambda o: scorecard.points_table(
            {"g": table}, {"g": 1.0}, o, **scale
        ),
        "reason_codes-top": lambda o: scorecard.reason_codes(points, card, top=o),
        "reason_codes-atol": lambda o: scorecard.reason_codes(points, card, atol=o),
        # selection
        "correlation_filter-threshold": lambda o: selection.correlation_filter(X, threshold=o),
        "vif_filter-threshold": lambda o: selection.vif_filter(X, threshold=o),
        "stepwise-p_enter": lambda o: selection.stepwise(X, TARGET, p_enter=o),
        "stepwise-p_remove": lambda o: selection.stepwise(X, TARGET, p_remove=o),
        "stepwise-direction": lambda o: selection.stepwise(X, TARGET, direction=o),
        "stepwise-positive": lambda o: selection.stepwise(X, TARGET, positive=o),
        # expert
        "ahp_weights-method": lambda o: expert.ahp_weights(comparison, method=o),
        "ahp_weights-rtol": lambda o: expert.ahp_weights(comparison, rtol=o),
        "ahp_consistency-rtol": lambda o: expert.ahp_consistency(comparison, rtol=o),
        # features
        "window_features-id": lambda o: features.window_features(
            events, observations, **(keys | {"id": o})
        ),
        "window_features-time": lambda o: features.window_features(
            events, observations, **(keys | {"time": o})
        ),
        "window_features-observation_time": lambda o: features.window_features(
            events, observations, **(keys | {"observation_time": o})
        ),
        "window_features-windows": lambda o: features.window_features(
            events, observations, **keys, windows=o
        ),
        "window_features-aggregations": lambda o: features.window_features(
            events, observations, **keys, aggregations=o
        ),
        "window_features-ratios": lambda o: features.window_features(
            events, observations, **keys, ratios=o
        ),
        "window_features-closed": lambda o: features.window_features(
            events, observations, **keys, closed=o
        ),
        "window_features-prefix": lambda o: features.window_features(
            events, observations, **keys, prefix=o
        ),
        "panel_features-id": lambda o: features.panel_features(
            panel, observations, **(panel_keys | {"id": o})
        ),
        "panel_features-month": lambda o: features.panel_features(
            panel, observations, **(panel_keys | {"month": o})
        ),
        "panel_features-status": lambda o: features.panel_features(
            panel, observations, **(panel_keys | {"status": o})
        ),
        "panel_features-observation_time": lambda o: features.panel_features(
            panel, observations, **(panel_keys | {"observation_time": o})
        ),
        "panel_features-windows": lambda o: features.panel_features(
            panel, observations, **panel_keys, windows=o
        ),
        "panel_features-bad": lambda o: features.panel_features(
            panel, observations, **panel_keys, bad=o
        ),
        "panel_features-prefix": lambda o: features.panel_features(
            panel, observations, **panel_keys, prefix=o
        ),
        "FeatureSpec-id": lambda o: features.FeatureSpec(o, "d", events=[group]),
        "FeatureSpec-observation_time": lambda o: features.FeatureSpec("c", o, events=[group]),
        "FeatureSpec-events": lambda o: features.FeatureSpec("c", "d", events=o),
        "FeatureSpec-ratios": lambda o: features.FeatureSpec("c", "d", events=[group], ratios=o),
        "FeatureSpec-panel": lambda o: features.FeatureSpec("c", "d", panel=o),
        "FeatureSpec-kivi_version": lambda o: features.FeatureSpec(
            "c", "d", events=[group], kivi_version=o
        ),
        # spec
        "fit_spec-method": lambda o: spec.fit_spec(frame, TARGET, method=o),
        "fit_spec-bins": lambda o: spec.fit_spec(frame, TARGET, bins=o),
        "fit_spec-unseen": lambda o: spec.fit_spec(frame, TARGET, bins=2, unseen=o),
        "fit_spec-smoothing": lambda o: spec.fit_spec(frame, TARGET, bins=2, smoothing=o),
        # strategy, univariate, monitoring, vintage
        "swap_set-current_cutoff": lambda o: strategy.swap_set(
            SCORE, NEW, current_cutoff=o, new_cutoff=0.5
        ),
        "swap_set-new_cutoff": lambda o: strategy.swap_set(
            SCORE, NEW, current_cutoff=0.5, new_cutoff=o
        ),
        "profile_table-bins": lambda o: univariate.profile_table(frame, bins=o),
        "expected_maximum_profit-no_loss_share": lambda o: metrics.expected_maximum_profit(
            SCORE, TARGET, no_loss_share=o
        ),
        "expected_maximum_profit-full_loss_share": lambda o: metrics.expected_maximum_profit(
            SCORE, TARGET, full_loss_share=o
        ),
        "expected_maximum_profit-roi": lambda o: metrics.expected_maximum_profit(
            SCORE, TARGET, roi=o
        ),
        "monitoring_table-smoothing": lambda o: monitoring.monitoring_table(
            SCORE, PERIOD, reference_period="p1", smoothing=o
        ),
        "stability_table-smoothing": lambda o: monitoring.stability_table(
            pd.DataFrame({"g": GROUPS}), PERIOD, reference_period="p1", smoothing=o
        ),
        "transition_matrix-normalize": lambda o: vintage.transition_matrix(
            GROUPS, GROUPS, normalize=o
        ),
        "delinquency_panel-end": lambda o: vintage.delinquency_panel(
            GROUPS[:2], ["2024-01-15"] * 2, [1.0, 1.0], [], [], [], end=o
        ),
        "delinquency_panel-tolerance": lambda o: vintage.delinquency_panel(
            GROUPS[:2], ["2024-01-15"] * 2, [1.0, 1.0], [], [], [], end="2024-02", tolerance=o
        ),
        # datasets
        "make_applications-n_applicants": lambda o: datasets.make_applications(n_applicants=o),
        "make_applications-rng": lambda o: datasets.make_applications(n_applicants=10, rng=o),
        "make_applications-include_risk": lambda o: datasets.make_applications(
            n_applicants=10, include_risk=o
        ),
        "make_transactions-n_months": lambda o: datasets.make_transactions(
            applications, application_date="2024-06-30", n_months=o
        ),
        "make_transactions-rng": lambda o: datasets.make_transactions(
            applications, application_date="2024-06-30", rng=o
        ),
        "make_loan_panel-n_loans": lambda o: datasets.make_loan_panel(n_loans=o),
        "make_loan_panel-n_months": lambda o: datasets.make_loan_panel(n_loans=10, n_months=o),
        "make_loan_panel-rng": lambda o: datasets.make_loan_panel(n_loans=10, rng=o),
        "make_loan_panel-prepayment": lambda o: datasets.make_loan_panel(n_loans=10, prepayment=o),
        "make_defaults-n_defaults": lambda o: datasets.make_defaults(n_defaults=o),
        "make_defaults-rng": lambda o: datasets.make_defaults(n_defaults=10, rng=o),
        "make_recoveries-rng": lambda o: datasets.make_recoveries(defaults, rng=o),
        # lgd
        "lgd_table-floor": lambda o: lgd.lgd_table(GROUPS, SCORE, floor=o),
    }


OPTIONS = _option_cases()
# Options for which a Series of two increasing numbers is valid: only the empty one raises.
SEQUENCES = {
    "gains_table-bins": lambda o: metrics.gains_table(SCORE, TARGET, bins=o),
    "cutoff_table-cutoffs": lambda o: strategy.cutoff_table(SCORE, TARGET, cutoffs=o),
    "confusion_table-cutoffs": lambda o: strategy.confusion_table(SCORE, TARGET, cutoffs=o),
    "transition_pd-horizons": lambda o: lifetime.transition_pd(
        pd.DataFrame([[0.9, 0.1], [0.0, 1.0]], index=["A", "D"], columns=["A", "D"]),
        horizons=o,
    ),
    "monitoring_table-bins": lambda o: monitoring.monitoring_table(
        SCORE, PERIOD, reference_period="p1", bins=o
    ),
    # The positions 1 and 2 of a model matrix with three columns.
    "selection_path-order": lambda o: selection.selection_path(
        np.column_stack([SCORE, NEW, VALUES]), TARGET, order=o
    ),
    "migration_bandwidth-grades": lambda o: vintage.migration_bandwidth(
        [1.0, 2.0, 2.0], [2.0, 1.0, 2.0], grades=o
    ),
    "migration_test-grades": lambda o: vintage.migration_test(
        [1.0, 2.0, 2.0], [2.0, 1.0, 2.0], grades=o
    ),
}


@pytest.mark.parametrize("case", list(OPTIONS) + list(SEQUENCES))
def test_an_empty_series_option_is_shown_as_an_empty_list(case):
    call = OPTIONS.get(case) or SEQUENCES[case]
    with pytest.raises(ValueError, match=r"got \[\]"):
        call(pd.Series([], dtype=float))


@pytest.mark.parametrize("case", OPTIONS)
def test_a_series_option_is_shown_whole(case):
    with pytest.raises(ValueError, match=r"got \[1\.0, 2\.0\]"):
        OPTIONS[case](pd.Series([1.0, 2.0]))
