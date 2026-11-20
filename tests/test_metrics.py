import numpy as np
import pandas as pd
import pytest

from kivi.binning import bin_edges
from kivi.datasets import Dataset
from kivi.metrics import brier_score, gains_table, gini, ks, roc_auc
from kivi.stability import psi

# Events score 0.4, 0.7 and 0.9; non-events score 0.1, 0.4 and 0.8.
SCORE = [0.1, 0.4, 0.4, 0.8, 0.7, 0.9]
TARGET = [0, 0, 1, 0, 1, 1]

GAINS_SCORE = [1, 2, 3, 3, 4, 5, 1, 2]
GAINS_TARGET = [0, 0, 1, 0, 1, 1, 0, 0]

COLUMNS = [
    "non_event",
    "event",
    "total",
    "event_rate",
    "lift",
    "cum_non_event_share",
    "cum_event_share",
    "cum_total_share",
    "ks",
    "cum_lift",
]


def test_roc_auc_counts_tied_pairs_as_half():
    # Of the 9 (event, non-event) pairs, 6 are ordered correctly, 1 is tied at 0.4 and 2 are not.
    assert roc_auc(SCORE, TARGET) == pytest.approx(6.5 / 9)
    assert gini(SCORE, TARGET) == pytest.approx(2 * 6.5 / 9 - 1)


def test_roc_auc_matches_average_ranks_and_pair_counts():
    rng = np.random.default_rng(1)
    target = rng.integers(0, 2, 400)
    score = rng.integers(0, 15, 400) + target
    ranks = pd.Series(score).rank(method="average").to_numpy()
    n_event, n_non_event = target.sum(), len(target) - target.sum()
    expected = (ranks[target == 1].sum() - n_event * (n_event + 1) / 2) / (n_event * n_non_event)
    assert roc_auc(score, target) == pytest.approx(expected)
    gap = score[target == 1][:, None] - score[target == 0][None, :]
    assert roc_auc(score, target) == pytest.approx(np.mean((gap > 0) + 0.5 * (gap == 0)))


def test_roc_auc_on_bank_duration_matches_pair_counts():
    bank = Dataset.bank_data()
    events = bank.loc[bank["target"] == 1, "duration"].to_numpy()
    non_events = bank.loc[bank["target"] == 0, "duration"].to_numpy()
    gap = events[:, None] - non_events[None, :]
    expected = np.mean((gap > 0) + 0.5 * (gap == 0))
    assert roc_auc(bank["duration"], bank["target"]) == pytest.approx(expected)


def test_ranking_measures_at_the_extremes():
    target = [0, 0, 1, 1]
    assert roc_auc([1, 2, 3, 4], target) == 1.0
    assert gini([1, 2, 3, 4], target) == 1.0
    assert ks([1, 2, 3, 4], target) == 1.0
    assert roc_auc([4, 3, 2, 1], target) == 0.0
    assert gini([4, 3, 2, 1], target) == -1.0
    assert ks([4, 3, 2, 1], target) == 1.0
    assert roc_auc([5, 5, 5, 5], target) == 0.5
    assert gini([5, 5, 5, 5], target) == 0.0
    assert ks([5, 5, 5, 5], target) == 0.0


def test_scorecard_points_must_be_negated():
    # Points that fall as the score rises rank the customers in the opposite order.
    points = np.array([720, 650, 650, 560, 590, 510])
    assert roc_auc(points, TARGET) == pytest.approx(1 - 6.5 / 9)
    assert roc_auc(-points, TARGET) == pytest.approx(6.5 / 9)
    assert gini(points, TARGET) == pytest.approx(-gini(-points, TARGET))
    assert ks(points, TARGET) == pytest.approx(ks(-points, TARGET))


def test_ks_is_largest_gap_between_class_distributions():
    # At scores 0.1, 0.4, 0.7, 0.8 and 0.9 the event CDF is 0, 1/3, 2/3, 2/3, 1 and the
    # non-event CDF is 1/3, 2/3, 2/3, 1, 1.
    assert ks(SCORE, TARGET) == pytest.approx(1 / 3)
    rng = np.random.default_rng(2)
    target = rng.integers(0, 2, 300)
    score = rng.integers(0, 10, 300) + 2 * target
    gaps = [
        abs(np.mean(score[target == 1] <= t) - np.mean(score[target == 0] <= t))
        for t in np.unique(score)
    ]
    assert ks(score, target) == pytest.approx(max(gaps))


def test_ks_can_come_from_a_reversed_region():
    # Scores 1 to 10 in order, with events at 3, 4 and 10. The events have 2, 2 and 7 of the
    # 7 non-events below them, so AUC = 11/21 and Gini = 1/21: the ranking is barely right.
    target = [0, 0, 1, 1, 0, 0, 0, 0, 0, 1]
    score = list(range(1, 11))
    assert gini(score, target) == pytest.approx(1 / 21)
    # In the expected direction the largest gap is F_0 - F_1 = 1 - 2/3 = 7/21 at score 9, but at
    # score 4, F_1 - F_0 = 2/3 - 2/7 = 8/21 in the reversed direction, and KS takes that one.
    assert ks(score, target) == pytest.approx(8 / 21)
    table = gains_table(score, target, bins=np.r_[-np.inf, np.arange(1, 10), np.inf])
    assert table["ks"].max() == pytest.approx(8 / 21)
    signed = table["cum_event_share"] - table["cum_non_event_share"]
    assert signed.max() == pytest.approx(7 / 21)


def test_imbalanced_target():
    # One event among 1,000 observations scored 0 to 999; 900 of the 999 non-events score lower.
    target = np.zeros(1000, dtype=int)
    target[900] = 1
    score = np.arange(1000)
    assert roc_auc(score, target) == pytest.approx(900 / 999)
    assert ks(score, target) == pytest.approx(900 / 999)
    table = gains_table(score, target, bins=4)
    np.testing.assert_array_equal(table["total"], [250] * 4)
    np.testing.assert_array_equal(table["event"], [1, 0, 0, 0])
    np.testing.assert_allclose(table["lift"], [4, 0, 0, 0])
    np.testing.assert_allclose(table["cum_lift"], [4, 2, 4 / 3, 1])
    assert brier_score(np.full(1000, 0.001), target) == pytest.approx(0.001 * 0.999)


def test_gains_table_hand_calculation():
    table = gains_table(GAINS_SCORE, GAINS_TARGET, bins=[-np.inf, 2, 3, np.inf])
    assert list(table.columns) == COLUMNS
    assert table.index.name == "bin"
    np.testing.assert_array_equal(table.index.left, [3, 2, -np.inf])
    np.testing.assert_array_equal(table.index.right, [np.inf, 3, 2])
    assert table.index.closed == "right"
    np.testing.assert_array_equal(table["non_event"], [0, 1, 4])
    np.testing.assert_array_equal(table["event"], [2, 1, 0])
    np.testing.assert_array_equal(table["total"], [2, 2, 4])
    assert {table[c].dtype.kind for c in ["non_event", "event", "total"]} == {"i"}
    # 3 events and 5 non-events in 8 observations: an event rate of 3/8.
    np.testing.assert_allclose(table["event_rate"], [1, 1 / 2, 0])
    np.testing.assert_allclose(table["lift"], [8 / 3, 4 / 3, 0])
    np.testing.assert_allclose(table["cum_non_event_share"], [0, 1 / 5, 1])
    np.testing.assert_allclose(table["cum_event_share"], [2 / 3, 1, 1])
    np.testing.assert_allclose(table["cum_total_share"], [2 / 8, 4 / 8, 1])
    np.testing.assert_allclose(table["ks"], [2 / 3, 4 / 5, 0])
    np.testing.assert_allclose(table["cum_lift"], [8 / 3, 2, 1])


def test_gains_table_integer_bins_are_bin_edges():
    # The quantiles at 1/3 and 2/3 of 1, 1, 2, 2, 3, 3, 4, 5 are the tied scores 2 and 3.
    by_count = gains_table(GAINS_SCORE, GAINS_TARGET, bins=3)
    by_edges = gains_table(GAINS_SCORE, GAINS_TARGET, bins=[-np.inf, 2, 3, np.inf])
    pd.testing.assert_frame_equal(by_count, by_edges)
    rng = np.random.default_rng(3)
    score = rng.normal(size=200)
    table = gains_table(score, rng.integers(0, 2, 200))
    np.testing.assert_array_equal(table.index.right[::-1], bin_edges(score, bins=10)[1:])
    np.testing.assert_array_equal(table["total"], [20] * 10)


def test_gains_table_warns_when_ties_leave_fewer_bands():
    # A PD floor: four of ten scores are 0.01. The quantiles at 0 and 0.2 are both 0.01 and
    # merge, as in bin_edges, which leaves four bands with the floor in the lowest one.
    score = [0.01, 0.01, 0.01, 0.01, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9]
    target = [0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
    message = "tied edges leave 4 of the 5 requested bins"
    with pytest.warns(UserWarning, match=message):
        table = gains_table(score, target, bins=5)
    with pytest.warns(UserWarning, match=message):
        edges = bin_edges(score, bins=5)
    pd.testing.assert_frame_equal(table, gains_table(score, target, bins=edges))
    np.testing.assert_allclose(table.index.right[::-1], [0.124, 0.38, 0.64, np.inf])
    np.testing.assert_array_equal(table["total"], [2, 2, 2, 4])
    np.testing.assert_array_equal(table["event"], [2, 1, 1, 1])
    # The median of 1, 1, 1, 1, 2, 3 equals the minimum, so a single band remains.
    with pytest.warns(UserWarning, match="tied edges leave 1 of the 2 requested bins"):
        single = gains_table([1, 1, 1, 1, 2, 3], [0, 0, 1, 0, 1, 1], bins=2)
    np.testing.assert_array_equal(single["total"], [6])
    np.testing.assert_allclose(single[["lift", "ks", "cum_lift"]].to_numpy(), [[1, 0, 1]])


def test_gains_table_keeps_empty_bands():
    # Bands from the top: (10, inf] and (5, 7] and (-inf, 0] are empty.
    table = gains_table([1, 9], [0, 1], bins=[-np.inf, 0, 5, 7, 10, np.inf])
    np.testing.assert_array_equal(table.index.left, [10, 7, 5, 0, -np.inf])
    np.testing.assert_array_equal(table["total"], [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(table["event"], [0, 1, 0, 0, 0])
    np.testing.assert_allclose(table["event_rate"], [np.nan, 1, np.nan, 0, np.nan])
    np.testing.assert_allclose(table["lift"], [np.nan, 2, np.nan, 0, np.nan])
    np.testing.assert_allclose(table["cum_total_share"], [0, 1 / 2, 1 / 2, 1, 1])
    np.testing.assert_allclose(table["ks"], [0, 1, 1, 0, 0])
    np.testing.assert_allclose(table["cum_lift"], [np.nan, 2, 2, 1, 1])


def test_tables_cut_on_the_same_edges_line_up():
    edges = [-np.inf, 0.2, 0.5, 0.8, np.inf]
    development = gains_table(
        [0.1, 0.3, 0.6, 0.9, 0.1, 0.3, 0.6, 0.9], [0, 1, 0, 1, 0, 1, 0, 1], bins=edges
    )
    # No current score falls in (0.5, 0.8].
    current = gains_table([0.1, 0.3, 0.9, 0.95], [0, 1, 0, 1], bins=edges)
    assert current.index.equals(development.index)
    np.testing.assert_array_equal(development["total"], [2, 2, 2, 2])
    np.testing.assert_array_equal(current["total"], [2, 0, 1, 1])
    # psi replaces the empty current band with the default smoothing of 0.5.
    q = np.full(4, 1 / 4)
    p = np.array([2, 0.5, 1, 1]) / 4.5
    expected = np.sum((p - q) * np.log(p / q))
    assert psi(development["total"], current["total"]) == pytest.approx(expected)


def test_gains_table_on_unsorted_series_with_a_custom_index():
    index = [50, 10, 40, 20, 30]
    target = pd.Series([1, 0, 1, 0, 0], index=index)
    score = pd.Series([0.9, 0.1, 0.7, 0.4, 0.2], index=index)
    edges = [-np.inf, 0.3, 0.6, np.inf]
    table = gains_table(score, target, bins=edges)
    np.testing.assert_array_equal(table["total"], [2, 1, 2])
    np.testing.assert_array_equal(table["event"], [2, 0, 0])
    sorted_input = gains_table(score.sort_index(), target.sort_index(), bins=edges)
    pd.testing.assert_frame_equal(table, sorted_input)
    pd.testing.assert_frame_equal(
        table, gains_table(score.to_numpy(), target.to_list(), bins=edges)
    )


def test_gains_table_ks_reaches_ks_with_one_band_per_score():
    rng = np.random.default_rng(4)
    target = rng.integers(0, 2, 300)
    score = rng.integers(0, 12, 300) + 3 * target
    distinct = np.unique(score)
    table = gains_table(score, target, bins=np.r_[-np.inf, distinct[:-1], np.inf])
    assert len(table) == len(distinct)
    assert table["ks"].max() == pytest.approx(ks(score, target))
    assert gains_table(score, target, bins=3)["ks"].max() <= ks(score, target) + 1e-12


def test_brier_score_hand_calculation():
    # Squared errors 0.01, 0.09, 0.16 and 0.04.
    assert brier_score([0.1, 0.3, 0.6, 0.8], [0, 0, 1, 1]) == pytest.approx(0.075)
    # A constant forecast of the event rate r scores r (1 - r).
    assert brier_score([0.25] * 4, [0, 0, 0, 1]) == pytest.approx(0.25 * 0.75)
    assert brier_score([1, 0], [1, 0]) == 0.0
    # A sample with one class still has a Brier score.
    assert brier_score([0.0, 0.5], [0, 0]) == pytest.approx(0.125)


def test_series_must_share_index_and_other_inputs_match_by_position():
    target = pd.Series([0, 1, 0, 1], index=[10, 11, 12, 13])
    score = pd.Series([0.2, 0.9, 0.4, 0.6], index=[10, 11, 12, 13])
    assert roc_auc(score, target) == 1.0
    assert roc_auc(score.to_numpy(), target) == 1.0
    assert roc_auc(score, target.to_list()) == 1.0
    with pytest.raises(ValueError, match="score and target must have identical indices"):
        ks(score.set_axis([0, 1, 2, 3]), target)
    with pytest.raises(ValueError, match="probability and target must have identical indices"):
        brier_score(score.sort_index(ascending=False), target)


def test_nullable_and_boolean_inputs():
    target = pd.Series([0, 1, 0, 1], dtype="Int64")
    score = pd.Series([0.2, 0.9, 0.4, 0.6], dtype="Float64")
    assert roc_auc(score, target) == 1.0
    assert roc_auc(score, target.astype(bool)) == 1.0
    assert brier_score(score.astype(float), target) == pytest.approx(0.0925)
    with pytest.raises(ValueError, match=r"score must be finite, got 1 missing .* such as nan"):
        roc_auc(pd.Series([0.2, None, 0.4, 0.6], dtype="Float64"), target)


def test_inputs_are_not_modified():
    target = pd.Series([0, 1, 0, 1, 1])
    score = pd.Series([0.3, 0.1, 0.2, 0.4, 0.4])
    target_before, score_before = target.copy(), score.copy()
    for metric in (roc_auc, gini, ks, brier_score):
        metric(score, target)
    gains_table(score, target, bins=2)
    pd.testing.assert_series_equal(target, target_before)
    pd.testing.assert_series_equal(score, score_before)


@pytest.mark.parametrize("metric", [roc_auc, gini, ks, gains_table])
@pytest.mark.parametrize(
    "target,score,message",
    [
        ([[0, 1]], [0.1, 0.2], "target must be one-dimensional, got 2 dimensions"),
        ([0, 1], [[0.1, 0.2]], "score must be one-dimensional, got 2 dimensions"),
        ([0, 1], [0.1], "score and target must have the same length, got 1 and 2"),
        ([0, 2, 1], [0.1, 0.2, 0.3], "target must contain only 0 and 1, got 2"),
        ([0, None, 1], [0.1, 0.2, 0.3], "target must contain only 0 and 1, got nan"),
        (["no", "yes"], [0.1, 0.2], "target must contain only 0 and 1, got 'no'"),
        ([1, 1], [0.1, 0.2], "target must contain both 0 and 1, got only 1"),
        ([0, 0], [0.1, 0.2], "target must contain both 0 and 1, got only 0"),
        ([], [], "score and target must not be empty"),
        ([0, 1], ["a", "b"], "score must be numeric, got dtype"),
        ([0, 1, 0, 1], np.array([1 + 1j, 2, 3, 4]), "score must be real, got dtype complex128"),
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
        metric(score, target)


@pytest.mark.parametrize(
    "target,probability,message",
    [
        ([0, 1], [0.5, 1.5], "probability must be between 0 and 1, got 1.5"),
        ([0, 1], [-0.1, 0.5], "probability must be between 0 and 1, got -0.1"),
        ([0, 1], [np.nan, 0.5], r"probability must be finite, got 1 missing"),
        ([0, 1], np.array([0.1 + 0j, 0.5]), "probability must be real, got dtype complex128"),
        ([0, 2], [0.1, 0.2], "target must contain only 0 and 1, got 2"),
        ([0, 1], [0.1], "probability and target must have the same length, got 1 and 2"),
        ([], [], "probability and target must not be empty"),
    ],
)
def test_brier_score_rejects_invalid_inputs(target, probability, message):
    with pytest.raises(ValueError, match=message):
        brier_score(probability, target)


@pytest.mark.parametrize("bins", [0, -1, True, 2.5, [1], [3, 2, 1], [0, 0, 1], [0, np.nan, 1]])
def test_gains_table_rejects_invalid_bins(bins):
    message = r"bins must be a positive integer or at least two increasing edges, got "
    with pytest.raises(ValueError, match=message):
        gains_table([0.1, 0.2], [0, 1], bins=bins)


def test_gains_table_rejects_scores_outside_the_edges():
    message = r"score has 2 value\(s\) outside the edges in bins, such as 0.0; use infinite"
    with pytest.raises(ValueError, match=message):
        gains_table([0.0, 0.5, 3.0, 1.0], [0, 1, 0, 1], bins=[0, 1, 2])


def test_weighted_ranking_measures_match_hand_calculation():
    weights = [1, 2, 1, 1, 3, 1]
    # Events 0.4 (1), 0.7 (3), 0.9 (1); non-events 0.1 (1), 0.4 (2), 0.8 (1). Weighted pairs
    # ordered correctly: 1 + 0.5 * 2 for the event at 0.4, 3 + 6 for 0.7 and 4 for 0.9, of 5 * 4.
    assert roc_auc(SCORE, TARGET, weights=weights) == pytest.approx(15 / 20)
    assert gini(SCORE, TARGET, weights=weights) == pytest.approx(0.5)
    # Cumulative shares at 0.1, 0.4, 0.7, 0.8: events 0, 1/5, 4/5, 4/5; non-events 1/4, 3/4, ...
    assert ks(SCORE, TARGET, weights=weights) == pytest.approx(3 / 4 - 1 / 5)
    # Squared errors 0.01, 0.09, 0.16 and 0.04 with weights 1, 0, 2 and 1.
    brier = brier_score([0.1, 0.3, 0.6, 0.8], [0, 0, 1, 1], weights=[1, 0, 2, 1])
    assert brier == pytest.approx((0.01 + 2 * 0.16 + 0.04) / 4)


def test_integer_weights_equal_repeated_rows():
    data = Dataset.bank_data()
    score, target = data["duration"].to_numpy(float), data["target"].to_numpy()
    weights = np.random.default_rng(11).integers(0, 4, len(data))
    rows = np.repeat(np.arange(len(data)), weights)
    for metric in (roc_auc, gini, ks):
        assert metric(score, target, weights=weights) == pytest.approx(
            metric(score[rows], target[rows]), rel=1e-12
        )
    probability = np.clip(score / score.max(), 0, 1)
    assert brier_score(probability, target, weights=weights) == pytest.approx(
        brier_score(probability[rows], target[rows]), rel=1e-12
    )
    edges = bin_edges(score, bins=5)
    weighted = gains_table(score, target, bins=edges, weights=weights)
    pd.testing.assert_frame_equal(
        weighted, gains_table(score[rows], target[rows], bins=edges), check_dtype=False
    )
    assert weighted["total"].dtype == float


def test_zero_weights_drop_rows():
    weights = np.array([1, 0, 1, 1, 0, 1])
    kept = weights > 0
    score, target = np.array(SCORE), np.array(TARGET)
    for metric in (roc_auc, ks):
        assert metric(score, target, weights=weights) == pytest.approx(
            metric(score[kept], target[kept])
        )


def test_gains_table_weighted_quantile_bands():
    score = [1, 2, 3, 4, 5, 6]
    target = [0, 0, 1, 0, 1, 1]
    weights = [3, 1, 1, 1, 1, 3]
    edges = bin_edges(score, bins=2, weights=weights)
    np.testing.assert_array_equal(edges, [-np.inf, 3, np.inf])
    table = gains_table(score, target, bins=2, weights=weights)
    assert table.index.right.tolist() == [np.inf, 3]
    assert table["total"].tolist() == [5.0, 5.0] and table["event"].tolist() == [4.0, 1.0]
    np.testing.assert_allclose(table["lift"], [0.8 / 0.5, 0.2 / 0.5])


def test_weights_match_like_target():
    score = pd.Series([0.2, 0.9, 0.4, 0.6], index=[10, 11, 12, 13])
    target = pd.Series([0, 1, 0, 1], index=[10, 11, 12, 13])
    weights = pd.Series([1.0, 2.0, 1.0, 1.0], index=[10, 11, 12, 13])
    assert roc_auc(score, target, weights=weights) == 1.0
    assert roc_auc(score, target, weights=weights.to_numpy()) == 1.0
    with pytest.raises(ValueError, match="weights must have the same index as score"):
        roc_auc(score, target, weights=weights.reset_index(drop=True))
    nullable = pd.Series([1, 2, 1, 1], index=score.index, dtype="Int64")
    assert ks(score, target, weights=nullable) == 1.0


@pytest.mark.parametrize("metric", [roc_auc, gini, ks, gains_table])
@pytest.mark.parametrize(
    "weights,message",
    [
        ([1, 1, 1, 0, 0, 0], "weights must give both target classes a positive total"),
        ([0, 0, 0, 0, 0, 0], "weights must have a finite, positive total, got 0.0"),
        ([1, 1, 1, 1, 1, -1], "weights must be finite and non-negative"),
        ([1, 1, 1, 1, 1, np.nan], "weights must be finite and non-negative"),
        ([1, 1], "weights must have one weight per value of score, got 2 for 6"),
        (["a"] * 6, "weights must be numeric, got dtype"),
    ],
)
def test_invalid_weights(metric, weights, message):
    target = [0, 0, 0, 1, 1, 1]
    with pytest.raises(ValueError, match=message):
        metric([1, 2, 3, 4, 5, 6], target, weights=weights)


def test_brier_score_weights_need_a_positive_total_only():
    assert brier_score([0.2, 0.4], [0, 0], weights=[1, 0]) == pytest.approx(0.04)
    with pytest.raises(ValueError, match="weights must have a finite, positive total"):
        brier_score([0.2, 0.4], [0, 1], weights=[0, 0])
