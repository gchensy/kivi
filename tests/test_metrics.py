import numpy as np
import pandas as pd
import pytest

from kivi._stats import normal_quantile
from kivi.binning import bin_edges
from kivi.datasets import Dataset
from kivi.metrics import (
    brier_score,
    delong_test,
    gains_table,
    gini,
    ks,
    roc_auc,
    roc_auc_interval,
    roc_table,
)
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
    with pytest.warns(UserWarning, match=message) as caught:
        table = gains_table(score, target, bins=5)
    # gains_table has no special argument, and the warning points at this call.
    assert str(caught[0].message).endswith("pass fewer bins or explicit edges")
    assert caught[0].filename == __file__
    with pytest.warns(UserWarning, match=message) as caught:
        edges = bin_edges(score, bins=5)
    assert str(caught[0].message).endswith("consider passing it in special")
    assert caught[0].filename == __file__
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
    for metric in (roc_auc, gini, ks, brier_score, roc_table, roc_auc_interval):
        metric(score, target)
    gains_table(score, target, bins=2)
    delong_test(score, score.iloc[::-1].to_numpy(), target)
    pd.testing.assert_series_equal(target, target_before)
    pd.testing.assert_series_equal(score, score_before)


@pytest.mark.parametrize("metric", [roc_auc, gini, ks, gains_table, roc_table, roc_auc_interval])
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


@pytest.mark.parametrize("metric", [roc_auc, gini, ks, gains_table, roc_table, roc_auc_interval])
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


@pytest.mark.parametrize("scale", [1e-300, 1e-170, 1e160, 1e300])
def test_ranking_measures_do_not_depend_on_the_weight_scale(scale):
    weights = np.array([1, 2, 1, 1, 3, 1]) * scale
    assert roc_auc(SCORE, TARGET, weights=weights) == pytest.approx(15 / 20)
    assert gini(SCORE, TARGET, weights=weights) == pytest.approx(0.5)
    assert ks(SCORE, TARGET, weights=weights) == pytest.approx(3 / 4 - 1 / 5)


# Hanley and McNeil (1982), Table 1: ratings 1 (definitely normal) to 5 (definitely abnormal) of
# 58 normal and 51 abnormal CT images. Stata's manual ([R] roctab) reports an area of 0.8932 with
# the DeLong standard error 0.0307 and the 95% interval 0.83295 to 0.95339.
HM_RATING = np.tile(np.arange(1, 6), 2)
HM_DISEASE = np.repeat([0, 1], 5)
HM_COUNT = np.array([33, 6, 6, 11, 2, 3, 2, 2, 11, 33])

# DeLong, DeLong and Clarke-Pearson (1988): albumin, total protein and K-G score of 43 of their 49
# patients, as printed in the SAS/STAT example "Comparing Receiver Operating Characteristic
# Curves" of PROC LOGISTIC (six patients with a missing test are left out there). Each row is
# alb, tp, totscore, popind; SAS models popind = 0 and replaces totscore by 10 - totscore.
DELONG_DATA = np.array(
    [
        [3.0, 5.8, 10, 0], [3.2, 6.3, 5, 1], [3.9, 6.8, 3, 1], [2.8, 4.8, 6, 0],
        [3.2, 5.8, 3, 1], [0.9, 4.0, 5, 0], [2.5, 5.7, 8, 0], [1.6, 5.6, 5, 1],
        [3.8, 5.7, 5, 1], [3.7, 6.7, 6, 1], [3.2, 5.4, 4, 1], [3.8, 6.6, 6, 1],
        [4.1, 6.6, 5, 1], [3.6, 5.7, 5, 1], [4.3, 7.0, 4, 1], [3.6, 6.7, 4, 0],
        [2.3, 4.4, 6, 1], [4.2, 7.6, 4, 0], [4.0, 6.6, 6, 0], [3.5, 5.8, 6, 1],
        [3.8, 6.8, 7, 1], [3.0, 4.7, 8, 0], [4.5, 7.4, 5, 1], [3.7, 7.4, 5, 1],
        [3.1, 6.6, 6, 1], [4.1, 8.2, 6, 1], [4.3, 7.0, 5, 1], [4.3, 6.5, 4, 1],
        [3.2, 5.1, 5, 1], [2.6, 4.7, 6, 1], [3.3, 6.8, 6, 0], [1.7, 4.0, 7, 0],
        [3.7, 6.1, 5, 1], [3.3, 6.3, 7, 1], [4.2, 7.7, 6, 1], [3.5, 6.2, 5, 1],
        [2.9, 5.7, 9, 0], [2.1, 4.8, 7, 1], [2.8, 6.2, 8, 0], [4.0, 7.0, 7, 1],
        [3.3, 5.7, 6, 1], [3.7, 6.9, 5, 1], [3.6, 6.6, 5, 1],
    ]
)  # fmt: skip


def _pairs(score, target, weights=None):
    """Brute-force placement values: compare every event with every non-event."""
    score, target = np.asarray(score, dtype=float), np.asarray(target)
    weights = np.ones(len(score)) if weights is None else np.asarray(weights, dtype=float)
    event = target == 1
    x, y, wx, wy = score[event], score[~event], weights[event], weights[~event]
    psi_ = (x[:, None] > y[None, :]) + 0.5 * (x[:, None] == y[None, :])
    return psi_ @ wy / wy.sum(), wx @ psi_ / wx.sum(), wx, wy


def _brute_variance(first, second, target, weights=None):
    """DeLong's covariance matrix of two AUCs, from O(n^2) placement values."""
    v10_a, v01_a, wx, wy = _pairs(first, target, weights)
    v10_b, v01_b, _, _ = _pairs(second, target, weights)

    def cov(u, v, w):
        return w @ ((u - w @ u / w.sum()) * (v - w @ v / w.sum())) / (w.sum() - 1)

    s10 = np.array([[cov(a, b, wx) for b in (v10_a, v10_b)] for a in (v10_a, v10_b)])
    s01 = np.array([[cov(a, b, wy) for b in (v01_a, v01_b)] for a in (v01_a, v01_b)])
    return s10 / wx.sum() + s01 / wy.sum()


def _trapezoid(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def test_roc_table_hand_calculation():
    table = roc_table(SCORE, TARGET)
    assert table.index.name == "threshold"
    assert table.index.tolist() == [np.inf, 0.9, 0.8, 0.7, 0.4, 0.1]
    assert table.columns.tolist() == [
        "cum_non_event",
        "cum_event",
        "false_positive_rate",
        "true_positive_rate",
        "precision",
    ]
    # The tie at 0.4 adds an event and a non-event in one step.
    assert table["cum_non_event"].tolist() == [0, 0, 1, 1, 2, 3]
    assert table["cum_event"].tolist() == [0, 1, 1, 2, 3, 3]
    assert table["cum_event"].dtype == np.int64
    np.testing.assert_allclose(table["false_positive_rate"], [0, 0, 1 / 3, 1 / 3, 2 / 3, 1])
    np.testing.assert_allclose(table["true_positive_rate"], [0, 1 / 3, 1 / 3, 2 / 3, 1, 1])
    np.testing.assert_allclose(table["precision"], [np.nan, 1, 1 / 2, 2 / 3, 3 / 5, 1 / 2])


def test_roc_table_area_is_roc_auc_and_largest_gap_is_ks():
    rng = np.random.default_rng(5)
    for trial in range(20):
        target = rng.integers(0, 2, 200)
        score = rng.integers(0, 2 + trial, 200) + target * rng.integers(0, 3, 200)
        weights = rng.integers(0, 4, 200) if trial % 2 else None
        table = roc_table(score, target, weights=weights)
        area = _trapezoid(table["false_positive_rate"], table["true_positive_rate"])
        assert area == pytest.approx(roc_auc(score, target, weights=weights), rel=1e-12)
        gap = (table["true_positive_rate"] - table["false_positive_rate"]).abs()
        assert gap.max() == pytest.approx(ks(score, target, weights=weights), rel=1e-12)
        assert table["false_positive_rate"].iloc[-1] == 1.0
        assert table["true_positive_rate"].iloc[-1] == 1.0


def test_roc_table_matches_cumulative_gains_with_one_band_per_score():
    score, target = np.array(GAINS_SCORE), np.array(GAINS_TARGET)
    gains = gains_table(score, target, bins=np.r_[-np.inf, np.unique(score)[:-1], np.inf])
    table = roc_table(score, target).iloc[1:]
    np.testing.assert_allclose(table["true_positive_rate"], gains["cum_event_share"])
    np.testing.assert_allclose(table["false_positive_rate"], gains["cum_non_event_share"])


def test_roc_table_integer_weights_equal_repeated_rows():
    data = Dataset.bank_data()
    score, target = data["balance"].to_numpy(float), data["target"].to_numpy()
    weights = np.random.default_rng(12).integers(0, 4, len(data))
    rows = np.repeat(np.arange(len(data)), weights)
    weighted = roc_table(score, target, weights=weights)
    pd.testing.assert_frame_equal(weighted, roc_table(score[rows], target[rows]), check_dtype=False)
    assert weighted["cum_event"].dtype == float


def test_roc_table_leaves_out_scores_of_zero_weight():
    table = roc_table(SCORE, TARGET, weights=[1, 1, 1, 0, 1, 1])
    assert table.index.tolist() == [np.inf, 0.9, 0.7, 0.4, 0.1]
    assert table["cum_non_event"].tolist() == [0, 0, 0, 1, 2]


def test_roc_auc_interval_hand_calculation():
    # Event placements 1/2, 2/3 and 1 (the event at 0.4 ties one non-event); non-event placements
    # 1, 5/6 and 1/3. S10 = 7/108 and S01 = 13/108, so Var = (7/108 + 13/108) / 3 = 5/81.
    interval = roc_auc_interval(SCORE, TARGET)
    assert interval.index.tolist() == ["auc", "gini"]
    assert interval.columns.tolist() == ["estimate", "se", "lower", "upper"]
    se = np.sqrt(5) / 9
    z = 1.959963984540054
    np.testing.assert_allclose(interval.loc["auc"], [6.5 / 9, se, 6.5 / 9 - z * se, 1.0])
    np.testing.assert_allclose(
        interval.loc["gini"], [4 / 9, 2 * se, 2 * (6.5 / 9 - z * se) - 1, 1.0]
    )
    assert interval.loc["auc", "estimate"] == roc_auc(SCORE, TARGET)


def test_roc_auc_interval_level():
    narrow = roc_auc_interval(SCORE, TARGET, level=0.5)
    se = np.sqrt(5) / 9
    np.testing.assert_allclose(
        narrow.loc["auc", ["lower", "upper"]], 6.5 / 9 + np.array([-1, 1]) * 0.6744897501960817 * se
    )
    # The lower bound is cut off at 0 as the upper is at 1.
    wide = roc_auc_interval([4, 3, 2, 1], [0, 0, 1, 1], level=0.99)
    assert wide.loc["auc"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert wide.loc["gini", ["estimate", "lower", "upper"]].tolist() == [-1.0, -1.0, -1.0]


def test_roc_auc_interval_matches_brute_force_with_ties_and_weights():
    rng = np.random.default_rng(7)
    for trial in range(40):
        n = int(rng.integers(5, 150))
        target = rng.integers(0, 2, n)
        target[:2], target[2:4] = 0, 1
        score = rng.normal(size=n) + target if trial % 3 == 0 else rng.integers(0, 6, n) + target
        weights = None if trial % 2 else rng.integers(1, 4, n).astype(float)
        interval = roc_auc_interval(score, target, weights=weights)
        variance = _brute_variance(score, score, target, weights)[0, 0]
        assert interval.loc["auc", "se"] ** 2 == pytest.approx(variance, rel=1e-10)
        assert interval.loc["auc", "estimate"] == roc_auc(score, target, weights=weights)


def test_roc_auc_interval_integer_weights_equal_repeated_rows():
    data = Dataset.bank_data()
    score, target = data["duration"].to_numpy(float), data["target"].to_numpy()
    weights = np.random.default_rng(13).integers(0, 4, len(data))
    rows = np.repeat(np.arange(len(data)), weights)
    pd.testing.assert_frame_equal(
        roc_auc_interval(score, target, weights=weights),
        roc_auc_interval(score[rows], target[rows]),
        rtol=1e-12,
    )


def test_roc_auc_interval_matches_hanley_mcneil_example_in_stata():
    weighted = roc_auc_interval(HM_RATING, HM_DISEASE, weights=HM_COUNT)
    np.testing.assert_allclose(weighted.loc["auc"], [0.8932, 0.0307, 0.83295, 0.95339], atol=5e-5)
    np.testing.assert_allclose(
        weighted.loc["auc", ["lower", "upper"]], [0.83295, 0.95339], atol=5e-6
    )
    rows = np.repeat(np.arange(len(HM_COUNT)), HM_COUNT)
    pd.testing.assert_frame_equal(weighted, roc_auc_interval(HM_RATING[rows], HM_DISEASE[rows]))


def _delong_example():
    alb, tp, totscore, popind = DELONG_DATA.T
    # The event is popind = 0, and all three measures fall as its probability rises.
    return -alb, -tp, -(10 - totscore), (popind == 0).astype(int)


def test_roc_auc_interval_matches_delong_example_in_sas():
    albumin, protein, kg_score, event = _delong_example()
    expected = {
        "albumin": (albumin, [0.7366, 0.0927, 0.5549, 0.9182]),
        "K-G score": (kg_score, [0.7258, 0.1028, 0.5243, 0.9273]),
        "total protein": (protein, [0.6478, 0.1000, 0.4518, 0.8439]),
    }
    for score, values in expected.values():
        interval = roc_auc_interval(score, event)
        np.testing.assert_allclose(interval.loc["auc"], values, atol=5e-5)
        # SAS reports Somers' D, the Gini coefficient, as 2 AUC - 1.
        assert interval.loc["gini", "estimate"] == pytest.approx(2 * values[0] - 1, abs=1e-4)


def test_delong_test_matches_delong_example_in_sas():
    albumin, protein, kg_score, event = _delong_example()
    # SAS's contrasts with the K-G score as the reference: estimate, standard error, the 95% Wald
    # limits, chi-square and p-value.
    for new, values in [
        (albumin, [0.0108, 0.0953, -0.1761, 0.1976, 0.0127, 0.9102]),
        (protein, [-0.0780, 0.1046, -0.2830, 0.1271, 0.5554, 0.4561]),
    ]:
        test = delong_test(kg_score, new, event)
        np.testing.assert_allclose(
            [*test[["difference", "se", "lower", "upper"]], test["z"] ** 2, test["p_value"]],
            values,
            atol=5e-5,
        )
        assert test["current_auc"] == roc_auc(kg_score, event)
        assert test["new_auc"] == roc_auc(new, event)


def test_delong_test_matches_brute_force_covariance():
    rng = np.random.default_rng(8)
    for trial in range(40):
        n = int(rng.integers(5, 150))
        target = rng.integers(0, 2, n)
        target[:2], target[2:4] = 0, 1
        current = rng.integers(0, 5, n) + target
        new = current + rng.integers(0, 3, n) * target if trial % 2 else rng.normal(size=n)
        weights = None if trial % 3 else rng.integers(1, 4, n)
        test = delong_test(current, new, target, weights=weights)
        covariance = _brute_variance(current, new, target, weights)
        variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
        assert test["se"] ** 2 == pytest.approx(variance, rel=1e-10, abs=1e-15)
        difference = roc_auc(new, target, weights=weights) - roc_auc(
            current, target, weights=weights
        )
        assert test["difference"] == difference


def test_delong_test_integer_weights_equal_repeated_rows():
    data = Dataset.bank_data()
    target = data["target"].to_numpy()
    current, new = data["duration"].to_numpy(float), data["balance"].to_numpy(float)
    weights = np.random.default_rng(14).integers(0, 4, len(data))
    rows = np.repeat(np.arange(len(data)), weights)
    pd.testing.assert_series_equal(
        delong_test(current, new, target, weights=weights),
        delong_test(current[rows], new[rows], target[rows]),
        rtol=1e-12,
    )


def test_delong_test_hand_calculation_and_symmetry():
    # The new score separates the classes perfectly, so all its placement values are 1 and the
    # variance of the difference is that of the current score, 5/81.
    new = [0.2, 0.3, 0.6, 0.5, 0.7, 0.9]
    test = delong_test(SCORE, new, TARGET)
    assert test.index.tolist() == [
        "current_auc",
        "new_auc",
        "difference",
        "se",
        "lower",
        "upper",
        "z",
        "p_value",
    ]
    se = np.sqrt(5) / 9
    z = 1.959963984540054
    np.testing.assert_allclose(
        test[["current_auc", "new_auc", "difference", "se", "z"]],
        [6.5 / 9, 1, 2.5 / 9, se, 2.5 / 9 / se],
    )
    np.testing.assert_allclose(test[["lower", "upper"]], 2.5 / 9 + np.array([-z, z]) * se)
    reversed_ = delong_test(new, SCORE, TARGET)
    np.testing.assert_allclose(reversed_[["difference", "z"]], -test[["difference", "z"]])
    assert reversed_["se"] == test["se"] and reversed_["p_value"] == test["p_value"]


def test_delong_test_without_variance():
    # Scores that order every (event, non-event) pair in the same way have equal AUCs and
    # identical placement values: 0 / 0 gives NaN.
    same = delong_test(SCORE, 2 * np.array(SCORE) + 1, TARGET)
    assert same["difference"] == 0.0 and same["se"] == 0.0
    assert np.isnan(same["z"]) and np.isnan(same["p_value"])
    assert same["lower"] == same["upper"] == 0.0
    # A constant score puts every placement value at 1/2 and a perfect one at 1, so the
    # difference is certain.
    certain = delong_test([1, 1, 1, 1], [1, 2, 3, 4], [0, 0, 1, 1])
    assert certain["difference"] == 0.5 and certain["se"] == 0.0
    assert certain["z"] == np.inf and certain["p_value"] == 0.0


def test_delong_test_matches_series_by_index_or_position():
    index = [10, 11, 12, 13, 14, 15]
    current = pd.Series(SCORE, index=index)
    new = pd.Series([0.2, 0.3, 0.6, 0.5, 0.7, 0.9], index=index)
    target = pd.Series(TARGET, index=index)
    expected = delong_test(SCORE, new.to_numpy(), TARGET)
    pd.testing.assert_series_equal(delong_test(current, new, target), expected)
    pd.testing.assert_series_equal(delong_test(current, new.to_list(), TARGET), expected)
    with pytest.raises(ValueError, match="current and new must have identical indices"):
        delong_test(current, new.reset_index(drop=True), target)
    with pytest.raises(ValueError, match="current and target must have identical indices"):
        delong_test(current, new, target.reset_index(drop=True))
    with pytest.raises(ValueError, match="new and target must have identical indices"):
        delong_test(SCORE, new, target.reset_index(drop=True))
    weights = pd.Series(1.0, index=range(6))
    with pytest.raises(ValueError, match="weights must have the same index as current"):
        delong_test(current, new, target, weights=weights)


@pytest.mark.parametrize(
    "current,new,target,message",
    [
        ([0.1, 0.2], [0.1, 0.2, 0.3], [0, 1, 0], "current and target must have the same length"),
        ([0.1, 0.2, 0.3], [0.1, 0.2], [0, 1, 0], "new and target must have the same length"),
        ([0.1, np.nan, 0.3], [1, 2, 3], [0, 1, 0], r"current must be finite, got 1 missing"),
        ([1, 2, 3], ["a", "b", "c"], [0, 1, 0], "new must be numeric, got dtype"),
        ([1, 2, 3], [1, 2, 3], [0, 0, 0], "target must contain both 0 and 1, got only 0"),
    ],
)
def test_delong_test_rejects_invalid_inputs(current, new, target, message):
    with pytest.raises(ValueError, match=message):
        delong_test(current, new, target)


@pytest.mark.parametrize("metric", ["interval", "test"])
def test_standard_errors_need_two_observations_of_each_class(metric):
    def call(score, target, **kwargs):
        if metric == "interval":
            return roc_auc_interval(score, target, **kwargs)
        return delong_test(score, score[::-1], target, **kwargs)

    message = (
        r"target must contain at least two events and two non-events for a standard error, "
        r"got 1 event\(s\) and 3 non-event\(s\)"
    )
    with pytest.raises(ValueError, match=message):
        call([1, 2, 3, 4], [0, 0, 1, 0])
    # Frequency weights: a class needs a total weight above 1.
    message = r"weights must give each target class a total above 1 .* got 2.0 for 0 and 1.0 for 1"
    with pytest.raises(ValueError, match=message):
        call([1, 2, 3, 4], [0, 0, 1, 1], weights=[1, 1, 1, 0])
    call([1, 2, 3, 4], [0, 0, 1, 1], weights=[1, 1, 1.5, 0.5])


@pytest.mark.parametrize("level", [0, 1, 1.5, -0.1, np.nan, True, "0.95", None, 95])
@pytest.mark.parametrize("metric", ["interval", "test"])
def test_invalid_level(metric, level):
    with pytest.raises(ValueError, match="level must be a number between 0 and 1, got "):
        if metric == "interval":
            roc_auc_interval(SCORE, TARGET, level=level)
        else:
            delong_test(SCORE, SCORE, TARGET, level=level)


def test_level_close_to_one_keeps_its_precision():
    interval = roc_auc_interval(HM_RATING, HM_DISEASE, weights=HM_COUNT, level=1 - 1e-12)
    se = interval.loc["auc", "se"]
    # scipy.stats.norm.isf((1 - level) / 2) = 7.130509892879273, while the rounded
    # norm.ppf(0.5 + level / 2) would give 7.130494613066504.
    expected = interval.loc["auc", "estimate"] - 7.130509892879273 * se
    assert interval.loc["auc", "lower"] == pytest.approx(expected, rel=1e-12)


def test_standard_errors_on_the_bank_data_with_large_weights():
    # Frequency weights of 1e300 stand for an enormous sample, whose standard error vanishes.
    data = Dataset.bank_data().iloc[:200]
    weights = np.full(len(data), 1e300)
    interval = roc_auc_interval(data["duration"], data["target"], weights=weights)
    assert interval.loc["auc", "se"] == pytest.approx(0, abs=1e-140)
    assert interval.loc["auc", "estimate"] == pytest.approx(
        roc_auc(data["duration"], data["target"]), rel=1e-12
    )


@pytest.mark.parametrize(
    "probability,expected",
    [
        (0.975, 1.959963984540054),
        (0.995, 2.5758293035489004),
        (0.95, 1.6448536269514722),
        (0.3, -0.5244005127080409),
        (1e-10, -6.361340902404056),
        (1e-300, -37.0470962993612),
        (0.5, 0.0),
    ],
)
def test_normal_quantile_matches_reference_values(probability, expected):
    # Reference values from scipy.stats.norm.ppf.
    assert normal_quantile(probability) == pytest.approx(expected, rel=1e-14, abs=1e-300)
    if probability > 1e-3:  # 1 - probability rounds a tiny probability away
        assert normal_quantile(1 - probability) == pytest.approx(-expected, rel=1e-12)


@pytest.mark.parametrize("probability", [0.0, 1.0, -0.5, np.nan])
def test_normal_quantile_rejects_probabilities_outside_the_open_interval(probability):
    with pytest.raises(ValueError, match="probability must be between 0 and 1"):
        normal_quantile(probability)


@pytest.mark.parametrize(
    "current,new,target,weights",
    [
        # Two non-events change places: every (event, non-event) pair keeps its order.
        ([0, 1, 2, 3, 4], [0, 1, 3, 2, 4], [0, 1, 0, 0, 1], [1.5, 0.8, 2.3, 2.9, 2.1]),
        # Two events change places, and the rounded AUCs differ while the placements agree.
        ([0, 1, 2, 3], [0, 2, 3, 1], [0, 1, 1, 1], [5, 0.3, 0.7, 0.3]),
    ],
)
def test_delong_test_ignores_rounding_of_fractional_weights(current, new, target, weights):
    test = delong_test(current, new, target, weights=weights)
    assert test["difference"] == 0.0 and test["se"] == 0.0
    assert test["lower"] == test["upper"] == 0.0
    assert np.isnan(test["z"]) and np.isnan(test["p_value"])


def test_delong_test_finds_no_difference_when_classes_reorder_internally():
    # Swapping two adjacent scores of the same class keeps every (event, non-event) pair in
    # order, so whatever the weights, the test must not report a difference.
    rng = np.random.default_rng(21)
    checked = 0
    for _ in range(400):
        n = int(rng.integers(6, 30))
        target = np.r_[0, 0, 1, 1, rng.integers(0, 2, n - 4)]
        current = rng.permutation(n).astype(float)
        k = int(rng.integers(0, n - 1))
        a, b = np.argsort(current)[k : k + 2]
        if target[a] != target[b]:
            continue
        new = current.copy()
        new[[a, b]] = current[[b, a]]
        test = delong_test(current, new, target, weights=rng.uniform(0.5, 5, n))
        assert test["difference"] == 0.0 and np.isnan(test["p_value"])
        checked += 1
    assert checked > 100


def test_roc_auc_interval_bounds_stay_ordered_when_the_auc_rounds_past_one():
    # The weighted AUC rounds to 1 + 2**-52 here, and its standard error is 0.
    interval = roc_auc_interval([0, 2, 3, 1], [0, 1, 1, 1], weights=[5.0, 0.3, 0.7, 0.3])
    assert interval.loc["auc", "estimate"] == pytest.approx(1.0, abs=1e-15)
    assert interval.loc["auc", "lower"] == interval.loc["auc", "upper"] == 1.0
    assert interval.loc["gini", "lower"] == interval.loc["gini", "upper"] == 1.0


def test_one_weighted_row_per_class_is_enough():
    # A row of weight 3 stands for three observations.
    interval = roc_auc_interval([1, 2, 3], [0, 0, 1], weights=[1, 1, 3])
    rows = roc_auc_interval([1, 2, 3, 3, 3], [0, 0, 1, 1, 1])
    pd.testing.assert_frame_equal(interval, rows)


def test_level_is_checked_as_a_float():
    from fractions import Fraction

    with pytest.raises(ValueError, match=r"level must be a number between 0 and 1, got Fraction"):
        roc_auc_interval(SCORE, TARGET, level=Fraction(1) - Fraction(1, 10**30))
    fraction = roc_auc_interval(SCORE, TARGET, level=Fraction(19, 20))
    pd.testing.assert_frame_equal(fraction, roc_auc_interval(SCORE, TARGET))
    numpy_level = roc_auc_interval(SCORE, TARGET, level=np.float32(0.5))
    assert numpy_level.loc["auc", "lower"] > roc_auc_interval(SCORE, TARGET).loc["auc", "lower"]


def test_normal_quantile_far_in_the_tail():
    # scipy.stats.norm.ppf(1e-310); the approximation alone is accurate to about 1e-9.
    assert normal_quantile(1e-310) == pytest.approx(-37.663060331949524, rel=1e-8)
