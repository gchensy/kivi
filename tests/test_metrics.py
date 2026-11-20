import numpy as np
import pandas as pd
import pytest

from kivi.binning import bin_edges
from kivi.datasets import Dataset
from kivi.metrics import brier_score, gains_table, gini, ks, roc_auc

# Events score 0.4, 0.7 and 0.9; non-events score 0.1, 0.4 and 0.8.
TARGET = [0, 0, 1, 0, 1, 1]
SCORE = [0.1, 0.4, 0.4, 0.8, 0.7, 0.9]

GAINS_TARGET = [0, 0, 1, 0, 1, 1, 0, 0]
GAINS_SCORE = [1, 2, 3, 3, 4, 5, 1, 2]

COLUMNS = [
    "count",
    "event",
    "event_rate",
    "lift",
    "cum_count_share",
    "cum_event_share",
    "cum_non_event_share",
    "ks",
    "cum_lift",
]


def test_roc_auc_counts_tied_pairs_as_half():
    # Of the 9 (event, non-event) pairs, 6 are ordered correctly, 1 is tied at 0.4 and 2 are not.
    assert roc_auc(TARGET, SCORE) == pytest.approx(6.5 / 9)
    assert gini(TARGET, SCORE) == pytest.approx(2 * 6.5 / 9 - 1)


def test_roc_auc_matches_average_ranks_and_pair_counts():
    rng = np.random.default_rng(1)
    target = rng.integers(0, 2, 400)
    score = rng.integers(0, 15, 400) + target
    ranks = pd.Series(score).rank(method="average").to_numpy()
    n_event, n_non_event = target.sum(), len(target) - target.sum()
    expected = (ranks[target == 1].sum() - n_event * (n_event + 1) / 2) / (n_event * n_non_event)
    assert roc_auc(target, score) == pytest.approx(expected)
    gap = score[target == 1][:, None] - score[target == 0][None, :]
    assert roc_auc(target, score) == pytest.approx(np.mean((gap > 0) + 0.5 * (gap == 0)))


def test_roc_auc_on_bank_duration_matches_pair_counts():
    bank = Dataset.bank_data()
    events = bank.loc[bank["target"] == 1, "duration"].to_numpy()
    non_events = bank.loc[bank["target"] == 0, "duration"].to_numpy()
    gap = events[:, None] - non_events[None, :]
    expected = np.mean((gap > 0) + 0.5 * (gap == 0))
    assert roc_auc(bank["target"], bank["duration"]) == pytest.approx(expected)


def test_ranking_measures_at_the_extremes():
    target = [0, 0, 1, 1]
    assert roc_auc(target, [1, 2, 3, 4]) == 1.0
    assert gini(target, [1, 2, 3, 4]) == 1.0
    assert ks(target, [1, 2, 3, 4]) == 1.0
    assert roc_auc(target, [4, 3, 2, 1]) == 0.0
    assert gini(target, [4, 3, 2, 1]) == -1.0
    assert ks(target, [4, 3, 2, 1]) == 1.0
    assert roc_auc(target, [5, 5, 5, 5]) == 0.5
    assert gini(target, [5, 5, 5, 5]) == 0.0
    assert ks(target, [5, 5, 5, 5]) == 0.0


def test_scorecard_points_must_be_negated():
    # Points that fall as the score rises rank the customers in the opposite order.
    points = np.array([720, 650, 650, 560, 590, 510])
    assert roc_auc(TARGET, points) == pytest.approx(1 - 6.5 / 9)
    assert roc_auc(TARGET, -points) == pytest.approx(6.5 / 9)
    assert gini(TARGET, points) == pytest.approx(-gini(TARGET, -points))
    assert ks(TARGET, points) == pytest.approx(ks(TARGET, -points))


def test_ks_is_largest_gap_between_class_distributions():
    # At scores 0.1, 0.4, 0.7, 0.8 and 0.9 the event CDF is 0, 1/3, 2/3, 2/3, 1 and the
    # non-event CDF is 1/3, 2/3, 2/3, 1, 1.
    assert ks(TARGET, SCORE) == pytest.approx(1 / 3)
    rng = np.random.default_rng(2)
    target = rng.integers(0, 2, 300)
    score = rng.integers(0, 10, 300) + 2 * target
    gaps = [
        abs(np.mean(score[target == 1] <= t) - np.mean(score[target == 0] <= t))
        for t in np.unique(score)
    ]
    assert ks(target, score) == pytest.approx(max(gaps))


def test_gains_table_hand_calculation():
    table = gains_table(GAINS_TARGET, GAINS_SCORE, bins=[-np.inf, 2, 3, np.inf])
    assert list(table.columns) == COLUMNS
    assert table.index.name == "bin"
    np.testing.assert_array_equal(table.index.left, [3, 2, -np.inf])
    np.testing.assert_array_equal(table.index.right, [np.inf, 3, 2])
    assert table.index.closed == "right"
    np.testing.assert_array_equal(table["count"], [2, 2, 4])
    np.testing.assert_array_equal(table["event"], [2, 1, 0])
    assert table["count"].dtype.kind == table["event"].dtype.kind == "i"
    # 3 events and 5 non-events in 8 observations: an event rate of 3/8.
    np.testing.assert_allclose(table["event_rate"], [1, 1 / 2, 0])
    np.testing.assert_allclose(table["lift"], [8 / 3, 4 / 3, 0])
    np.testing.assert_allclose(table["cum_count_share"], [2 / 8, 4 / 8, 1])
    np.testing.assert_allclose(table["cum_event_share"], [2 / 3, 1, 1])
    np.testing.assert_allclose(table["cum_non_event_share"], [0, 1 / 5, 1])
    np.testing.assert_allclose(table["ks"], [2 / 3, 4 / 5, 0])
    np.testing.assert_allclose(table["cum_lift"], [8 / 3, 2, 1])


def test_gains_table_quantile_bands_keep_ties_together():
    # The quantiles at 1/3 and 2/3 of 1, 1, 2, 2, 3, 3, 4, 5 are the tied scores 2 and 3.
    by_count = gains_table(GAINS_TARGET, GAINS_SCORE, bins=3)
    by_edges = gains_table(GAINS_TARGET, GAINS_SCORE, bins=[-np.inf, 2, 3, np.inf])
    pd.testing.assert_frame_equal(by_count, by_edges)
    # The median is 1, so all four 1s form the lower band instead of an even split.
    halves = gains_table([0, 0, 1, 0, 1, 1], [1, 1, 1, 1, 2, 3], bins=2)
    np.testing.assert_array_equal(halves.index.right, [np.inf, 1])
    np.testing.assert_array_equal(halves["count"], [2, 4])
    np.testing.assert_array_equal(halves["event"], [2, 1])


def test_gains_table_quantile_edges_follow_bin_edges():
    rng = np.random.default_rng(3)
    score = rng.normal(size=200)
    target = rng.integers(0, 2, 200)
    table = gains_table(target, score)
    np.testing.assert_array_equal(table.index.right[::-1], bin_edges(score, bins=10)[1:])
    np.testing.assert_array_equal(table["count"], [20] * 10)


def test_gains_table_omits_empty_bands():
    # The interpolated quartiles of 0, 0, 0, 10 are 0, 0 and 2.5, which leaves (0, 2.5] empty.
    table = gains_table([0, 0, 0, 1], [0, 0, 0, 10], bins=4)
    np.testing.assert_array_equal(table.index.right, [np.inf, 0])
    np.testing.assert_array_equal(table.index.left, [2.5, -np.inf])
    np.testing.assert_array_equal(table["count"], [1, 3])
    explicit = gains_table([0, 1], [1, 9], bins=[-np.inf, 0, 5, 7, np.inf])
    np.testing.assert_array_equal(explicit.index.left, [7, 0])
    np.testing.assert_array_equal(explicit["count"], [1, 1])


def test_gains_table_ks_reaches_ks_with_one_band_per_score():
    rng = np.random.default_rng(4)
    target = rng.integers(0, 2, 300)
    score = rng.integers(0, 12, 300) + 3 * target
    distinct = np.unique(score)
    table = gains_table(target, score, bins=np.r_[-np.inf, distinct[:-1], np.inf])
    assert len(table) == len(distinct)
    assert table["ks"].max() == pytest.approx(ks(target, score))
    assert gains_table(target, score, bins=3)["ks"].max() <= ks(target, score) + 1e-12


def test_gains_table_applies_development_edges_to_another_sample():
    edges = [-np.inf, 0.3, 0.6, np.inf]
    table = gains_table([0, 1, 1, 0, 1], [0.05, 0.95, 0.65, 0.4, 0.2], bins=edges)
    np.testing.assert_array_equal(table["count"], [2, 1, 2])
    np.testing.assert_array_equal(table["event"], [2, 0, 1])


def test_brier_score_hand_calculation():
    # Squared errors 0.01, 0.09, 0.16 and 0.04.
    assert brier_score([0, 0, 1, 1], [0.1, 0.3, 0.6, 0.8]) == pytest.approx(0.075)
    # A constant forecast of the event rate r scores r (1 - r).
    assert brier_score([0, 0, 0, 1], [0.25] * 4) == pytest.approx(0.25 * 0.75)
    assert brier_score([1, 0], [1, 0]) == 0.0
    # A sample with one class still has a Brier score.
    assert brier_score([0, 0], [0.0, 0.5]) == pytest.approx(0.125)


def test_series_must_share_index_and_other_inputs_match_by_position():
    target = pd.Series([0, 1, 0, 1], index=[10, 11, 12, 13])
    score = pd.Series([0.2, 0.9, 0.4, 0.6], index=[10, 11, 12, 13])
    assert roc_auc(target, score) == 1.0
    assert roc_auc(target, score.to_numpy()) == 1.0
    assert roc_auc(target.to_list(), score) == 1.0
    with pytest.raises(ValueError, match="target and score must have identical indices"):
        ks(target, score.set_axis([0, 1, 2, 3]))
    with pytest.raises(ValueError, match="target and probability must have identical indices"):
        brier_score(target, score.sort_index(ascending=False))


def test_nullable_and_boolean_inputs():
    target = pd.Series([0, 1, 0, 1], dtype="Int64")
    score = pd.Series([0.2, 0.9, 0.4, 0.6], dtype="Float64")
    assert roc_auc(target, score) == 1.0
    assert roc_auc(target.astype(bool), score) == 1.0
    assert brier_score(target, score.astype(float)) == pytest.approx(0.0925)
    with pytest.raises(ValueError, match=r"score must be finite, got 1 missing .* such as nan"):
        roc_auc(target, pd.Series([0.2, None, 0.4, 0.6], dtype="Float64"))


def test_inputs_are_not_modified():
    target = pd.Series([0, 1, 0, 1, 1])
    score = pd.Series([0.3, 0.1, 0.2, 0.4, 0.4])
    target_before, score_before = target.copy(), score.copy()
    for metric in (roc_auc, gini, ks, gains_table, brier_score):
        metric(target, score)
    pd.testing.assert_series_equal(target, target_before)
    pd.testing.assert_series_equal(score, score_before)


@pytest.mark.parametrize("metric", [roc_auc, gini, ks, gains_table])
@pytest.mark.parametrize(
    "target,score,message",
    [
        ([[0, 1]], [0.1, 0.2], "target must be one-dimensional, got 2 dimensions"),
        ([0, 1], [[0.1, 0.2]], "score must be one-dimensional, got 2 dimensions"),
        ([0, 1], [0.1], "target and score must have the same length, got 2 and 1"),
        ([0, 2, 1], [0.1, 0.2, 0.3], "target must contain only 0 and 1, got 2"),
        ([0, None, 1], [0.1, 0.2, 0.3], "target must contain only 0 and 1, got nan"),
        (["no", "yes"], [0.1, 0.2], "target must contain only 0 and 1, got 'no'"),
        ([1, 1], [0.1, 0.2], "target must contain both 0 and 1, got only 1"),
        ([0, 0], [0.1, 0.2], "target must contain both 0 and 1, got only 0"),
        ([], [], "target and score must not be empty"),
        ([0, 1], ["a", "b"], "score must be numeric, got dtype"),
        (
            [0, 1, 0],
            [0.1, np.nan, 0.2],
            r"score must be finite, got 1 missing or infinite value\(s\) such as nan",
        ),
        (
            [0, 1, 0],
            [np.inf, -np.inf, 0.2],
            r"score must be finite, got 2 missing or infinite value\(s\) such as inf",
        ),
    ],
)
def test_ranking_measures_reject_invalid_inputs(metric, target, score, message):
    with pytest.raises(ValueError, match=message):
        metric(target, score)


@pytest.mark.parametrize(
    "target,probability,message",
    [
        ([0, 1], [0.5, 1.5], "probability must be between 0 and 1, got 1.5"),
        ([0, 1], [-0.1, 0.5], "probability must be between 0 and 1, got -0.1"),
        ([0, 1], [np.nan, 0.5], r"probability must be finite, got 1 missing"),
        ([0, 2], [0.1, 0.2], "target must contain only 0 and 1, got 2"),
        ([0, 1], [0.1], "target and probability must have the same length, got 2 and 1"),
        ([], [], "target and probability must not be empty"),
    ],
)
def test_brier_score_rejects_invalid_inputs(target, probability, message):
    with pytest.raises(ValueError, match=message):
        brier_score(target, probability)


@pytest.mark.parametrize("bins", [0, -1, True, 2.5, [1], [3, 2, 1], [0, 0, 1], [0, np.nan, 1]])
def test_gains_table_rejects_invalid_bins(bins):
    message = r"bins must be a positive integer or at least two increasing edges, got "
    with pytest.raises(ValueError, match=message):
        gains_table([0, 1], [0.1, 0.2], bins=bins)


def test_gains_table_rejects_scores_outside_the_edges():
    message = r"score has 2 value\(s\) outside the edges in bins, such as 0.0; use infinite"
    with pytest.raises(ValueError, match=message):
        gains_table([0, 1, 0, 1], [0.0, 0.5, 3.0, 1.0], bins=[0, 1, 2])
