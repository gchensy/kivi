"""The deprecations of 0.9: each warns once, at the caller's line, and keeps its result.

kivi 1.0 removes them; delete each test together with its deprecation.
"""

import numpy as np
import pandas as pd
import pytest

from kivi import binning, calibration, monitoring, stability, woe
from kivi.datasets import Dataset, load_bank

TARGET = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
# Group "c" has no non-events and group "d" no events, so the smoothing changes the result.
GROUPS = ["a", "a", "b", "b", "d", "c", "a", "b", "c", "d"]
VALUES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
SCORE = [0.1, 0.4, 0.35, 0.8, 0.2, 0.7, 0.55, 0.9, 0.95, 0.05]
PERIOD = ["p1"] * 5 + ["p2"] * 5


def single_warning_here(record):
    """Check that a call gave exactly one warning, attributed to this file."""
    assert len(record) == 1
    assert record[0].filename == __file__


def assert_same(result, expected):
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(result, expected)
    else:
        np.testing.assert_array_equal(result, expected)


SMOOTHED = {
    "woe_table": lambda smoothing: woe.woe_table(GROUPS, TARGET, smoothing=smoothing),
    "woe": lambda smoothing: woe.woe([0, 2, 3], [3, 1, 0], smoothing=smoothing),
    "iv": lambda smoothing: woe.iv([0, 2, 3], [3, 1, 0], smoothing=smoothing),
    "psi_table": lambda smoothing: stability.psi_table(GROUPS[:5], GROUPS[5:], smoothing=smoothing),
    "psi": lambda smoothing: stability.psi([0, 2, 3], [3, 1, 0], smoothing=smoothing),
    "monotonic_edges": lambda smoothing: binning.monotonic_edges(
        VALUES, TARGET, initial_bins=5, smoothing=smoothing
    ),
    "fit_grade_pd": lambda smoothing: calibration.fit_grade_pd(
        [0, 3, 5], [400, 500, 300], smoothing=smoothing
    ),
    "monitoring_table": lambda smoothing: monitoring.monitoring_table(
        SCORE, PERIOD, TARGET, reference_period="p1", bins=3, smoothing=smoothing
    ),
    "stability_table": lambda smoothing: monitoring.stability_table(
        pd.DataFrame({"g": GROUPS}), PERIOD, reference_period="p1", smoothing=smoothing
    ),
}


def test_every_function_that_reads_smoothing_true_is_checked():
    assert len(SMOOTHED) == 9


@pytest.mark.parametrize("case", SMOOTHED)
@pytest.mark.parametrize("flag", [True, np.True_])
def test_smoothing_true_warns_and_counts_as_one(case, flag):
    expected = SMOOTHED[case](1)
    message = (
        r"^smoothing=True is deprecated and counts as 1; kivi 1\.0 raises a ValueError\. "
        r"Pass smoothing=1 for the same result"
    )
    with pytest.warns(FutureWarning, match=message) as record:
        result = SMOOTHED[case](flag)
    single_warning_here(record)
    assert_same(result, expected)


@pytest.mark.parametrize("case", ["woe_table", "psi", "fit_grade_pd", "stability_table"])
def test_smoothing_false_still_raises_without_a_warning(case):
    with pytest.raises(ValueError, match="^smoothing must be finite and positive, got False$"):
        SMOOTHED[case](False)


COUNTS = {
    "woe": (woe.woe, "non_event_counts and event_counts"),
    "iv": (woe.iv, "non_event_counts and event_counts"),
    "psi": (stability.psi, "reference_counts and current_counts"),
}


@pytest.mark.parametrize("case", COUNTS)
def test_count_series_with_different_indices_warn_and_match_by_position(case):
    function, names = COUNTS[case]
    first = pd.Series([1, 9], index=["a", "b"])
    second = pd.Series([9, 1], index=["b", "a"])
    expected = function(first.to_numpy(), second.to_numpy())
    message = (
        f"^{names} are Series with different indices, which are matched by position; "
        r"kivi 1\.0 raises a ValueError\. Pass Series with identical indices, or arrays"
    )
    with pytest.warns(FutureWarning, match=message) as record:
        result = function(first, second)
    single_warning_here(record)
    assert_same(result, expected)


@pytest.mark.parametrize("case", COUNTS)
def test_count_series_with_identical_indices_or_arrays_do_not_warn(case):
    function = COUNTS[case][0]
    first = pd.Series([1, 9], index=["a", "b"])
    second = pd.Series([9, 1], index=["a", "b"])
    expected = function([1, 9], [9, 1])
    for pair in ((first, second), (first, [9, 1]), ([1, 9], second), (first, second.to_numpy())):
        assert_same(function(*pair), expected)


def test_count_series_of_different_lengths_raise_without_a_warning():
    with pytest.raises(ValueError, match="must have the same length, got 3 and 2"):
        woe.woe(pd.Series([1, 2]), pd.Series([1, 2, 3], index=[5, 6, 7]))


def test_count_warning_comes_once_with_the_smoothing_warning():
    first = pd.Series([1, 9], index=["a", "b"])
    second = pd.Series([9, 1], index=["b", "a"])
    with pytest.warns(FutureWarning) as record:
        woe.iv(first, second, smoothing=True)
    assert len(record) == 2
    assert all(warning.filename == __file__ for warning in record)
    assert "different indices" in str(record[0].message)
    assert "smoothing=True" in str(record[1].message)


def test_bank_data_warns_and_returns_load_bank():
    message = (
        r"^Dataset\.bank_data\(\) is deprecated and is removed in kivi 1\.0; "
        r"use kivi\.datasets\.load_bank\(\), which returns the same frame$"
    )
    with pytest.warns(FutureWarning, match=message) as record:
        data = Dataset.bank_data()
    single_warning_here(record)
    pd.testing.assert_frame_equal(data, load_bank())


def test_isotonic_predict_accepts_probabilities_with_a_warning():
    edges, probability = [-np.inf, 0.5, np.inf], [0.1, 0.9]
    expected = calibration.isotonic_predict([0.2, 0.7], edges, probability)
    message = (
        r"^the probabilities argument of isotonic_predict is deprecated and is removed in "
        r"kivi 1\.0; pass probability instead$"
    )
    with pytest.warns(FutureWarning, match=message) as record:
        result = calibration.isotonic_predict([0.2, 0.7], edges, probabilities=probability)
    single_warning_here(record)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(
        calibration.isotonic_predict([0.2, 0.7], edges, probability=probability), expected
    )


def test_isotonic_predict_needs_exactly_one_name_for_the_probabilities():
    edges = [-np.inf, 0.5, np.inf]
    with pytest.raises(TypeError, match="got both probability and its deprecated name"):
        calibration.isotonic_predict([0.2], edges, [0.1, 0.9], probabilities=[0.1, 0.9])
    with pytest.raises(TypeError, match="missing 1 required argument: 'probability'"):
        calibration.isotonic_predict([0.2], edges)
