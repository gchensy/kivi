"""Inputs that describe the same observations must have identical indices when they are pandas.

Each case lists, for one public function, the arguments that describe the same
rows. Any two of them given as pandas objects with different indices must
raise, whichever they are and whatever the others are; arrays, and pandas
objects sharing one index, are matched by position. The last tests cover how
``kivi._validation.as_real`` reads the numbers that several modules share.
"""

import inspect
import itertools

import numpy as np
import pandas as pd
import pytest

from kivi import (
    _validation,
    binning,
    calibration,
    lifetime,
    metrics,
    monitoring,
    selection,
    spec,
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
    "calibration_table": (
        calibration.calibration_table,
        {"groups": GROUPS, "target": TARGET, "probability": SCORE, "weights": WEIGHTS},
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
    "cutoff_table": (
        strategy.cutoff_table,
        {"score": SCORE, "target": TARGET, "loss": [10.0] * ROWS, "weights": WEIGHTS},
        {"cutoffs": [0.5]},
    ),
    "swap_set": (
        strategy.swap_set,
        {"current": SCORE, "new": NEW, "target": TARGET, "weights": WEIGHTS},
        {"current_cutoff": 0.5, "new_cutoff": 0.5},
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
    return name == "frame" or (case == "stability_table" and name == "groups")


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
        # The variables of a model fitted on an array are numbered, not named.
        drop = result.index.name is None
        pd.testing.assert_frame_equal(
            result.reset_index(drop=drop), expected.reset_index(drop=drop)
        )
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
    "chimerge_groups": lambda smoothing: binning.chimerge_groups(
        GROUPS, TARGET, smoothing=smoothing
    ),
    "monotonic_edges": lambda smoothing: binning.monotonic_edges(
        VALUES, TARGET, smoothing=smoothing
    ),
    "fit_spec": lambda smoothing: spec.fit_spec(
        pd.DataFrame(FRAME), TARGET, bins=2, smoothing=smoothing
    ),
    "stability_table": lambda smoothing: monitoring.stability_table(
        pd.DataFrame({"g": GROUPS}), PERIOD, reference_period="p1", smoothing=smoothing
    ),
}


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
def test_smoothing_true_keeps_the_behavior_of_each_function(case):
    # Until 0.9 decides on booleans, the counting functions read True as 1.
    if case in ("chimerge_groups", "fit_spec"):
        with pytest.raises(ValueError, match="^smoothing must be finite and positive, got True$"):
            SMOOTHED[case](True)
    else:
        expected, result = SMOOTHED[case](1), SMOOTHED[case](True)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(result, expected)
        else:
            pd.testing.assert_frame_equal(result, expected)


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
    from kivi.datasets import make_applications, make_loan_panel

    message = "^rng must be None, an integer, a SeedSequence or a Generator, got "
    for call in (
        lambda: make_applications(n_applicants=10, rng=rng),
        lambda: make_loan_panel(n_loans=10, n_months=2, rng=rng),
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
