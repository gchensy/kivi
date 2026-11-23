import math

import numpy as np
import pandas as pd
import pytest

from kivi.binning import (
    _pair_chi2,
    bin_edges,
    chimerge_edges,
    cut,
    ks_edges,
    monotonic_edges,
    tree_edges,
)
from kivi.datasets import load_bank
from kivi.metrics import gini, ks
from kivi.woe import woe_table


def repeated(groups):
    """Values and labels from (value, non_event count, event count) triples."""
    values, target = [], []
    for value, non_event, event in groups:
        values += [value] * (non_event + event)
        target += [0] * non_event + [1] * event
    return pd.Series(values, dtype=float), pd.Series(target)


def chi2_2x2(a, b, c, d):
    """Pearson chi-square of the table [[a, b], [c, d]] without continuity correction."""
    n = a + b + c + d
    return n * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))


@pytest.fixture(scope="module")
def reference():
    return load_bank().iloc[:3000]


# Event rates 0, 0.5 and 0.9 for the values 1, 2 and 3.
CHI_VALUES, CHI_TARGET = repeated([(1, 10, 0), (2, 5, 5), (3, 1, 9)])


def test_chi_square_of_adjacent_pairs_matches_hand_calculation():
    counts = np.array([[10, 0], [5, 5], [1, 9]])
    # Pair 1: expected [[7.5, 2.5], [7.5, 2.5]]; pair 2: expected [[3, 7], [3, 7]].
    np.testing.assert_allclose(_pair_chi2(counts), [20 / 3, 80 / 21])
    assert _pair_chi2(counts)[1] == pytest.approx(chi2_2x2(5, 5, 1, 9))
    # A class absent from both bins contributes nothing instead of dividing by zero.
    np.testing.assert_equal(_pair_chi2(np.array([[4, 0], [6, 0]])), [0.0])


def test_chimerge_merges_the_least_significant_pair_first():
    # 80/21 = 3.81 is below the 5% critical value 3.84 (p = 0.051), 20/3 is above it.
    assert math.erfc(math.sqrt(80 / 21 / 2)) == pytest.approx(0.0510, abs=1e-4)
    np.testing.assert_equal(
        chimerge_edges(CHI_VALUES, CHI_TARGET, alpha=0.1), [-np.inf, 1, 2, np.inf]
    )
    np.testing.assert_equal(
        chimerge_edges(CHI_VALUES, CHI_TARGET, alpha=0.05), [-np.inf, 1, np.inf]
    )
    np.testing.assert_equal(
        chimerge_edges(CHI_VALUES, CHI_TARGET, alpha=None), [-np.inf, 1, 2, np.inf]
    )
    np.testing.assert_equal(
        chimerge_edges(CHI_VALUES, CHI_TARGET, max_bins=2, alpha=None), [-np.inf, 1, np.inf]
    )
    np.testing.assert_equal(chimerge_edges(CHI_VALUES, CHI_TARGET, max_bins=1), [-np.inf, np.inf])


def test_chimerge_starts_from_quantile_bins_when_values_are_many():
    values = pd.Series(np.arange(40, dtype=float))
    target = pd.Series([0] * 20 + [1] * 20)
    # Four quantile bins with edges 9.75, 19.5 and 29.25 are pure, so only 19.5 survives.
    np.testing.assert_equal(chimerge_edges(values, target, initial_bins=4), [-np.inf, 19.5, np.inf])


TREE_VALUES = pd.Series(np.arange(1, 13, dtype=float))
TREE_TARGET = pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1])


def entropy_gain(left, right):
    def total(events, n):
        p = events / n
        return 0.0 if p in (0, 1) else -n * (p * math.log(p) + (1 - p) * math.log(1 - p))

    events, n = left[0] + right[0], left[1] + right[1]
    return total(events, n) - total(*left) - total(*right)


@pytest.mark.parametrize("criterion", ["entropy", "gini"])
def test_tree_first_split_matches_hand_calculation(criterion):
    # Cutting after 4 leaves (0 of 4) and (6 of 8): gain 3.82, ahead of cutting after 6
    # with (1 of 6) and (5 of 6): gain 2.91. The Gini index agrees.
    assert entropy_gain((0, 4), (6, 8)) == pytest.approx(3.819, abs=1e-3)
    assert entropy_gain((1, 6), (5, 6)) == pytest.approx(2.911, abs=1e-3)
    edges = tree_edges(TREE_VALUES, TREE_TARGET, max_bins=2, criterion=criterion)
    np.testing.assert_equal(edges, [-np.inf, 4, np.inf])


def test_tree_grows_best_first_and_respects_the_minimum_share():
    # The second split goes to the impure right bin (6 of 8), between 8 and 9.
    np.testing.assert_equal(
        tree_edges(TREE_VALUES, TREE_TARGET, max_bins=3), [-np.inf, 4, 9, np.inf]
    )
    np.testing.assert_equal(
        tree_edges(TREE_VALUES, TREE_TARGET, max_bins=4), [-np.inf, 4, 8, 9, np.inf]
    )
    # A 10% minimum (1.2 observations) rules out the single-value bin (8, 9].
    np.testing.assert_equal(
        tree_edges(TREE_VALUES, TREE_TARGET, max_bins=4, min_bin_share=0.1),
        [-np.inf, 4, 6, 9, np.inf],
    )
    # With half the sample required on each side, only the cut after 6 is allowed.
    np.testing.assert_equal(
        tree_edges(TREE_VALUES, TREE_TARGET, min_bin_share=0.5), [-np.inf, 6, np.inf]
    )


def test_tree_stops_at_pure_bins():
    values = pd.Series([1, 2, 3, 4])
    np.testing.assert_equal(
        tree_edges(values, pd.Series([0, 0, 1, 1]), max_bins=5), [-np.inf, 2, np.inf]
    )


def cumulative_gaps(values, target):
    """KS of every cut, after each distinct value but the last, as in ``metrics.ks``."""
    events = target.groupby(values).sum().cumsum() / target.sum()
    non_events = (1 - target).groupby(values).sum().cumsum() / (1 - target).sum()
    return (events - non_events).abs().iloc[:-1]


def test_ks_first_split_matches_hand_calculation():
    # Cumulative events after 1..11: 0 0 0 0 1 1 2 3 3 4 5, non-events 1 2 3 4 4 5 5 5 6 6 6,
    # of 6 each, so the KS of a cut is |events - non-events| / 6.
    gaps = cumulative_gaps(TREE_VALUES, TREE_TARGET)
    np.testing.assert_allclose(gaps, np.array([1, 2, 3, 4, 3, 4, 3, 2, 3, 2, 1]) / 6)
    assert ks(TREE_VALUES, TREE_TARGET) == pytest.approx(4 / 6)
    # The largest gap, 4/6, is reached after 4 and after 6; the tie goes to the smaller value.
    np.testing.assert_equal(ks_edges(TREE_VALUES, TREE_TARGET, max_bins=2), [-np.inf, 4, np.inf])
    # With 40% (4.8 rows) on each side, the cuts after 5, 6 and 7 remain: 3/6, 4/6 and 3/6.
    np.testing.assert_equal(
        ks_edges(TREE_VALUES, TREE_TARGET, min_bin_share=0.4), [-np.inf, 6, np.inf]
    )
    np.testing.assert_equal(
        ks_edges(TREE_VALUES, TREE_TARGET, min_bin_share=0.5), [-np.inf, 6, np.inf]
    )


def test_ks_splits_each_bin_at_its_own_largest_gap():
    # (5, 12] holds 6 events and 2 non-events; |2 e - 6 n| for e events and n non-events
    # left of each cut is 2, 4, 2, 0, 6, 4, 2 after 5..11, largest after 9 (KS 6/12).
    np.testing.assert_equal(ks_edges(TREE_VALUES, TREE_TARGET, max_bins=3), [-np.inf, 4, 9, np.inf])
    # (4, 9] holds 3 events and 2 non-events: |2 e - 3 n| is 2, 1, 1, 3 after 5..8.
    np.testing.assert_equal(
        ks_edges(TREE_VALUES, TREE_TARGET, max_bins=4), [-np.inf, 4, 8, 9, np.inf]
    )
    # A 10% minimum (1.2 rows) leaves (4, 9] only the cuts after 6 and 7, tied at 1.
    np.testing.assert_equal(
        ks_edges(TREE_VALUES, TREE_TARGET, max_bins=4, min_bin_share=0.1),
        [-np.inf, 4, 6, 9, np.inf],
    )
    edges = ks_edges(TREE_VALUES, TREE_TARGET, max_bins=12, min_bin_share=0)
    np.testing.assert_equal(edges, [-np.inf, 4, 5, 6, 8, 9, np.inf])


# (non_event, event) counts at 1..4: 9 non-events and 6 events in all.
KS_VALUES, KS_TARGET = repeated([(1, 0, 3), (2, 2, 2), (3, 2, 1), (4, 5, 0)])


def test_ks_splits_the_bin_that_ranks_the_most_pairs_apart():
    # The first cut, after 2, has the largest |9 e - 6 n|: 27, 33 and 30 after 1, 2 and 3.
    np.testing.assert_equal(ks_edges(KS_VALUES, KS_TARGET, max_bins=2), [-np.inf, 2, np.inf])
    # (-inf, 2] has 5 events and 2 non-events in 7 rows; its cut after 1 has KS
    # |3/5 - 0/2| = 0.6 and ranks 0.6 * 5 * 2 = 6 pairs of an event and a non-event apart.
    # (2, inf] has 1 event and 7 non-events in 8 rows; its cut after 3 has the larger KS
    # |1/1 - 2/7| = 5/7 but ranks only 5/7 * 1 * 7 = 5 pairs apart.
    left, right = [-np.inf, 1, 2, np.inf], [-np.inf, 2, 3, np.inf]
    np.testing.assert_equal(ks_edges(KS_VALUES, KS_TARGET, max_bins=3), left)
    # Each pair adds 1 / (6 * 9) to the Gini of the bins ranked by event rate: 33/54 for
    # the two bins, 39/54 after the split on the left and 38/54 after the one on the right.
    for edges, expected in ((left, 39), (right, 38), ([-np.inf, 2, np.inf], 33)):
        codes = pd.cut(KS_VALUES, edges, labels=False)
        rate = KS_TARGET.groupby(codes).transform("mean")
        assert gini(rate, KS_TARGET) == pytest.approx(expected / 54)
    np.testing.assert_equal(ks_edges(KS_VALUES, KS_TARGET, max_bins=4), [-np.inf, 1, 2, 3, np.inf])


def test_ks_tie_between_bins_goes_to_the_first_bin():
    values, target = range(1, 9), [0, 1, 0, 0, 1, 1, 0, 1]
    # The first cut, after 4, ranks 4 |e - n| = 8 pairs apart. (-inf, 4] holds 1 event and
    # 3 non-events and its cut after 2 ranks |3 e - n| = 2 pairs apart; (4, inf] holds 3 events
    # and 1 non-event and its cut after 6 ranks |e - 3 n| = 2 pairs apart as well.
    np.testing.assert_equal(
        ks_edges(values, target, max_bins=2, min_bin_share=0), [-np.inf, 4, np.inf]
    )
    np.testing.assert_equal(
        ks_edges(values, target, max_bins=3, min_bin_share=0), [-np.inf, 2, 4, np.inf]
    )
    np.testing.assert_equal(
        ks_edges(values, target, max_bins=4, min_bin_share=0), [-np.inf, 2, 4, 6, np.inf]
    )


def test_ks_stops_at_pure_bins_and_at_zero_gaps():
    values = pd.Series([1, 2, 3, 4])
    np.testing.assert_equal(
        ks_edges(values, pd.Series([0, 0, 1, 1]), max_bins=5), [-np.inf, 2, np.inf]
    )
    # Both values have the same event rate, so no cut separates the classes.
    np.testing.assert_equal(ks_edges([1, 1, 2, 2], [0, 1, 0, 1]), [-np.inf, np.inf])
    # Event shares 0.2 / 0.7 and non-event shares 0.6 / 2.1 are both 2/7, but the sums of
    # weights leave a gap of about 1e-16, which gives no cut either.
    weights = [0.2, 0.6, 0.5, 1.5]
    np.testing.assert_equal(
        ks_edges([1, 1, 2, 2], [1, 0, 1, 0], weights=weights), [-np.inf, np.inf]
    )
    # Weights of 0.1 leave the right bin of events only with non-event weight 0, not -1e-16
    # as totals minus events would, so it is not cut either.
    for weights in ([0.1] * 6, [1] * 6, None):
        edges = ks_edges(
            [1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 1, 1], max_bins=4, min_bin_share=0, weights=weights
        )
        np.testing.assert_equal(edges, [-np.inf, 3, np.inf])
    # A single value, or a minimum share no cut can meet, leaves one bin.
    np.testing.assert_equal(ks_edges([5, 5, 5], [0, 1, 1]), [-np.inf, np.inf])
    np.testing.assert_equal(
        ks_edges([1, 2, 2, 2], [0, 1, 1, 1], min_bin_share=0.5), [-np.inf, np.inf]
    )


@pytest.mark.parametrize("min_bin_share", [0, 0.05, 0.1, 0.2])
@pytest.mark.parametrize("scale", [0.1, 1 / 7, 3.7])
def test_ks_edges_do_not_change_when_the_weights_are_scaled(min_bin_share, scale):
    rng = np.random.default_rng(3)
    for _ in range(25):
        values = rng.integers(0, 8, 20).astype(float)
        target = np.r_[0, 1, rng.integers(0, 2, 18)]
        weights = rng.integers(1, 5, 20)
        expected = ks_edges(
            np.repeat(values, weights), np.repeat(target, weights), min_bin_share=min_bin_share
        )
        scaled = ks_edges(values, target, weights=weights * scale, min_bin_share=min_bin_share)
        np.testing.assert_equal(scaled, expected)


def test_ks_side_at_exactly_the_minimum_share_is_allowed():
    # 0.07 * 100 is 7.000000000000001 in floating point; the 7 events are still 7% of the rows.
    values, target = np.arange(1, 101), np.r_[np.ones(7, int), np.zeros(93, int)]
    np.testing.assert_equal(ks_edges(values, target, min_bin_share=0.07), [-np.inf, 7, np.inf])
    np.testing.assert_equal(
        ks_edges(values, target, min_bin_share=0.07, weights=np.full(100, 0.1)),
        [-np.inf, 7, np.inf],
    )
    np.testing.assert_equal(ks_edges(values, target, min_bin_share=0.08), [-np.inf, 8, np.inf])


def test_ks_with_min_bin_share_measured_in_weight():
    values = np.arange(1, 11)
    target = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    weights = np.r_[10, np.ones(9)]
    # The first value carries 10 of the 19 units of weight, enough for a bin of its own.
    edges = ks_edges(values, target, weights=weights, min_bin_share=0.3)
    np.testing.assert_equal(edges, [-np.inf, 1, np.inf])
    repeated = ks_edges(
        np.repeat(values, [10] + [1] * 9), np.repeat(target, [10] + [1] * 9), min_bin_share=0.3
    )
    np.testing.assert_equal(edges, repeated)
    # Counted in rows, the first bin needs three of the ten values.
    np.testing.assert_equal(ks_edges(values, target, min_bin_share=0.3), [-np.inf, 3, np.inf])


# Event rates 0.1, 0.3, 0.2 and 0.4: the pair (0.3, 0.2) breaks an increasing trend.
MONO_VALUES, MONO_TARGET = repeated([(1, 9, 1), (2, 7, 3), (3, 8, 2), (4, 6, 4)])


def test_monotonic_pools_adjacent_violators():
    edges = monotonic_edges(MONO_VALUES, MONO_TARGET, max_bins=None, direction="increasing")
    np.testing.assert_equal(edges, [-np.inf, 1, 3, np.inf])
    table = woe_table(pd.cut(MONO_VALUES, edges), MONO_TARGET).sort_index()
    assert table.event.tolist() == [1, 5, 4] and table.non_event.tolist() == [9, 15, 6]
    # WOE = ln((event / 10) / (non_event / 30)).
    np.testing.assert_allclose(
        table.woe, np.log(np.array([1, 5, 4]) / 10 / (np.array([9, 15, 6]) / 30))
    )
    assert table.woe.is_monotonic_increasing
    # A decreasing trend pools everything; "auto" keeps the direction with the larger IV.
    np.testing.assert_equal(
        monotonic_edges(MONO_VALUES, MONO_TARGET, direction="decreasing"), [-np.inf, np.inf]
    )
    np.testing.assert_equal(monotonic_edges(MONO_VALUES, MONO_TARGET), edges)


def test_monotonic_merges_by_chi_square_beyond_max_bins():
    # Of the pairs (1/10, 5/20) and (5/20, 4/10), the second has the smaller chi-square.
    assert chi2_2x2(9, 1, 15, 5) == pytest.approx(0.9375)
    assert chi2_2x2(15, 5, 6, 4) == pytest.approx(5 / 7)
    np.testing.assert_equal(
        monotonic_edges(MONO_VALUES, MONO_TARGET, max_bins=2), [-np.inf, 1, np.inf]
    )


def test_monotonic_uses_the_smoothed_woe():
    # Raw event rates 0/2 < 1/11 rise, but with a zero count replaced by 0.5 the WOE falls.
    values, target = repeated([(1, 2, 0), (2, 10, 1), (3, 1, 3)])
    edges = monotonic_edges(values, target, direction="increasing")
    np.testing.assert_equal(edges, [-np.inf, 2, np.inf])
    assert woe_table(pd.cut(values, edges), target).sort_index().woe.is_monotonic_increasing


@pytest.mark.parametrize("fit", [chimerge_edges, tree_edges, ks_edges, monotonic_edges])
def test_missing_values_are_omitted_and_inputs_are_kept(fit):
    values = pd.concat([TREE_VALUES, pd.Series([np.nan, np.nan], index=[20, 21])])
    target = pd.concat([TREE_TARGET, pd.Series([1, 1], index=[20, 21])])
    before = values.copy(), target.copy()
    np.testing.assert_equal(fit(values, target), fit(TREE_VALUES, TREE_TARGET))
    pd.testing.assert_series_equal(values, before[0])
    pd.testing.assert_series_equal(target, before[1])


@pytest.mark.parametrize("fit", [chimerge_edges, tree_edges, ks_edges, monotonic_edges])
def test_special_values_are_left_out_with_their_labels(fit):
    # Codes at both ends of the range would change every fit if they were counted.
    values = pd.concat([TREE_VALUES, pd.Series([-1, -1, -9, 999, np.nan], index=range(20, 25))])
    target = pd.concat([TREE_TARGET, pd.Series([0, 0, 1, 0, 1], index=range(20, 25))])
    before = values.copy(), target.copy()
    expected = fit(TREE_VALUES, TREE_TARGET)
    for special in (
        [-1, -9, 999],
        (999.0, -1.0, -9.0),
        np.array([-9, -1, 999, -1]),
        pd.Index([-1, -9, 999]),
    ):
        np.testing.assert_equal(fit(values, target, special=special), expected)
    assert not np.array_equal(fit(values, target), expected)
    np.testing.assert_equal(
        fit(values.tolist(), target.to_numpy(), special=[-1, -9, 999]), expected
    )
    np.testing.assert_equal(fit(TREE_VALUES, TREE_TARGET, special=[]), expected)
    pd.testing.assert_series_equal(values, before[0])
    pd.testing.assert_series_equal(target, before[1])
    # A label such as "N/A" in an otherwise numeric column is a special value as well.
    labelled = pd.Series(["N/A", *TREE_VALUES, "N/A"], dtype=object)
    np.testing.assert_equal(fit(labelled, [0, *TREE_TARGET, 1], special=["N/A"]), expected)
    infinite = pd.Series([-np.inf, *TREE_VALUES])
    np.testing.assert_equal(fit(infinite, [1, *TREE_TARGET], special=[-np.inf]), expected)


def test_chimerge_with_a_special_code_merges_only_the_real_values():
    # Ten events at -1 form a pure bin of their own unless -1 is passed in special.
    values, target = repeated([(-1, 0, 10), (1, 10, 0), (2, 5, 5), (3, 1, 9)])
    np.testing.assert_equal(chimerge_edges(values, target, alpha=0.1), [-np.inf, -1, 1, 2, np.inf])
    np.testing.assert_equal(
        chimerge_edges(values, target, alpha=0.1, special=[-1]), [-np.inf, 1, 2, np.inf]
    )
    np.testing.assert_equal(
        chimerge_edges(values, target, alpha=0.05, special=[-1]), [-np.inf, 1, np.inf]
    )
    groups = cut(values, [-np.inf, 1, 2, np.inf], special=[-1])
    table = woe_table(groups, target)
    assert table.index.tolist() == [-1, *groups.cat.categories[1:]]
    assert table.non_event.tolist() == [0, 10, 5, 1] and table.event.tolist() == [10, 0, 5, 9]


def test_tree_minimum_share_counts_only_the_fitted_values():
    # Twelve non-events at -1 double the sample. Without special, half of 24 rows is 12,
    # and the only allowed cut isolates -1; with special, half of the 12 fitted rows is 6.
    values = pd.concat([TREE_VALUES, pd.Series([-1.0] * 12)], ignore_index=True)
    target = pd.concat([TREE_TARGET, pd.Series([0] * 12)], ignore_index=True)
    np.testing.assert_equal(tree_edges(values, target, min_bin_share=0.5), [-np.inf, -1, np.inf])
    np.testing.assert_equal(
        tree_edges(values, target, min_bin_share=0.5, special=[-1]), [-np.inf, 6, np.inf]
    )
    np.testing.assert_equal(
        tree_edges(values, target, max_bins=2, special=[-1]), [-np.inf, 4, np.inf]
    )


def test_monotonic_trend_excludes_the_special_group():
    # The code 999 has event rate 0.1. Counted as the top value it breaks the rising trend
    # and pools everything above 1; left out, the bins match the MONO example.
    values, target = repeated([(1, 9, 1), (2, 7, 3), (3, 8, 2), (4, 6, 4), (999, 9, 1)])
    np.testing.assert_equal(
        monotonic_edges(values, target, max_bins=None, direction="increasing"), [-np.inf, 1, np.inf]
    )
    edges = monotonic_edges(values, target, max_bins=None, direction="increasing", special=[999])
    np.testing.assert_equal(edges, [-np.inf, 1, 3, np.inf])
    np.testing.assert_equal(monotonic_edges(values, target, max_bins=None, special=[999]), edges)
    table = woe_table(cut(values, edges, special=[999]), target)
    assert table.index[0] == 999
    assert table.non_event.tolist() == [9, 9, 15, 6] and table.event.tolist() == [1, 1, 5, 4]
    # The shares include the special group: WOE = ln((event / 11) / (non_event / 39)).
    expected = np.log(np.array([1, 1, 5, 4]) / 11 / (np.array([9, 9, 15, 6]) / 39))
    np.testing.assert_allclose(table.woe, expected)
    # The special group shares the event rate of the first bin but is not merged into it.
    assert table.woe.iloc[1:].is_monotonic_increasing
    assert table.woe.iloc[0] == pytest.approx(table.woe.iloc[1])


# Fitted on the 525 contacted customers of the reference sample; -1 marks the other 2475.
PDAYS_EDGES = {
    chimerge_edges: [-np.inf, 105, 175.6, 188, 371, np.inf],
    tree_edges: [-np.inf, 100, 177, 192, 370, np.inf],
    ks_edges: [-np.inf, 100, 177, 192, 370, np.inf],
    monotonic_edges: [-np.inf, 97, 105, 188, 211.2, np.inf],
}


@pytest.mark.parametrize("fit", [chimerge_edges, tree_edges, ks_edges, monotonic_edges])
def test_bank_pdays_code_is_fitted_separately(reference, fit):
    pdays, target = reference["pdays"], reference["target"]
    is_special = pdays == -1
    assert is_special.sum() == 2475
    assert target[~is_special].value_counts().sort_index().tolist() == [405, 120]
    edges = fit(pdays, target, special=[-1])
    np.testing.assert_allclose(edges, PDAYS_EDGES[fit])
    np.testing.assert_equal(edges, fit(pdays[~is_special], target[~is_special]))
    contacted = pdays[~is_special]
    if fit in (tree_edges, ks_edges):
        assert set(edges[1:-1]) <= set(contacted)
    else:
        assert set(edges[1:-1]) <= set(np.quantile(contacted, np.linspace(0, 1, 21)))
    table = woe_table(cut(pdays, edges, special=[-1]), target)
    assert table.index[0] == -1 and len(table) == len(edges)
    assert table.loc[-1, ["non_event", "event"]].tolist() == [2247, 228]
    # Cross-tabulate the codes: crosstab hashes interval categories, which warns on NumPy < 2.2.
    intervals = pd.crosstab(pd.cut(contacted, edges).cat.codes, target[~is_special])
    np.testing.assert_equal(table.iloc[1:][["non_event", "event"]].to_numpy(), intervals.to_numpy())
    assert table.iloc[1:][["non_event", "event"]].sum().tolist() == [405, 120]
    if fit is monotonic_edges:
        assert table.woe.iloc[1:].is_monotonic_decreasing


@pytest.mark.parametrize("fit", [chimerge_edges, monotonic_edges])
def test_without_special_the_bank_pdays_code_shares_a_bin(reference, fit):
    pdays = reference["pdays"]
    edges = fit(pdays, reference["target"])
    assert edges[1] == 96
    first = pdays[pdays <= edges[1]]
    assert (first == -1).sum() == 2475 and (first > -1).sum() == 78


@pytest.mark.parametrize("fit", [chimerge_edges, tree_edges, ks_edges, monotonic_edges])
@pytest.mark.parametrize("column", ["age", "balance", "duration", "campaign", "pdays"])
def test_edges_on_bank_data_have_infinite_ends_and_at_most_max_bins(reference, fit, column):
    edges = fit(reference[column], reference["target"], max_bins=4)
    assert edges[0] == -np.inf and edges[-1] == np.inf
    assert np.diff(edges).min() > 0
    assert 1 <= len(edges) - 1 <= 4
    groups = pd.cut(reference[column], edges)
    assert groups.notna().all() and groups.value_counts().min() > 0


@pytest.mark.parametrize("column", ["age", "balance", "duration", "campaign", "day"])
def test_chimerge_on_bank_data_keeps_only_significant_neighbours(reference, column):
    edges = chimerge_edges(reference[column], reference["target"], max_bins=6, alpha=0.01)
    table = woe_table(pd.cut(reference[column], edges), reference["target"]).sort_index()
    # Each bin is paired with the next one, so the shifted columns are one shorter.
    pairs = zip(table.non_event, table.event, table.non_event[1:], table.event[1:], strict=False)
    for statistic in (chi2_2x2(*cells) for cells in pairs):
        assert math.erfc(math.sqrt(statistic / 2)) < 0.01


@pytest.mark.parametrize("criterion", ["entropy", "gini"])
@pytest.mark.parametrize("column", ["age", "balance", "duration", "day"])
def test_tree_on_bank_data_respects_the_minimum_share(reference, column, criterion):
    edges = tree_edges(
        reference[column], reference["target"], max_bins=6, min_bin_share=0.1, criterion=criterion
    )
    counts = pd.cut(reference[column], edges).value_counts()
    assert counts.min() >= 0.1 * len(reference)
    assert set(edges[1:-1]) <= set(reference[column])


@pytest.mark.parametrize("column", ["age", "balance", "duration", "campaign", "day", "previous"])
def test_monotonic_on_bank_data_gives_a_strict_woe_trend(reference, column):
    edges = monotonic_edges(reference[column], reference["target"], max_bins=None)
    woe = woe_table(pd.cut(reference[column], edges), reference["target"]).sort_index().woe
    steps = np.diff(woe.to_numpy())
    assert (steps > 0).all() or (steps < 0).all() or len(woe) == 1


@pytest.mark.parametrize("column", ["age", "balance", "duration", "day"])
def test_ks_on_bank_data_respects_the_minimum_share(reference, column):
    edges = ks_edges(reference[column], reference["target"], max_bins=6, min_bin_share=0.1)
    assert len(edges) - 1 == 6
    counts = pd.cut(reference[column], edges).value_counts()
    assert counts.min() >= 0.1 * len(reference)
    assert set(edges[1:-1]) <= set(reference[column])


@pytest.mark.parametrize("column", ["age", "balance", "duration", "campaign", "day"])
def test_ks_first_cut_keeps_the_ks_of_the_variable(reference, column):
    values, target = reference[column], reference["target"]
    edges = ks_edges(values, target, max_bins=2, min_bin_share=0)
    two_bins = pd.cut(values, edges, labels=False)
    # Two bins have a Gini equal to their KS, and the cut keeps the KS of the values.
    assert ks(two_bins, target) == pytest.approx(ks(values, target), rel=1e-12)
    assert abs(gini(two_bins, target)) == pytest.approx(ks(values, target), rel=1e-12)


def test_ks_bins_of_bank_balance(reference):
    balance, target = reference["balance"], reference["target"]
    edges = ks_edges(balance, target)
    np.testing.assert_equal(edges, [-np.inf, 99, 888, 3276, 6046, np.inf])
    # 888 is the KS cut of the whole sample, and the next cuts follow by the pairs ranked apart.
    np.testing.assert_equal(ks_edges(balance, target, max_bins=2), [-np.inf, 888, np.inf])
    np.testing.assert_equal(ks_edges(balance, target, max_bins=3), [-np.inf, 99, 888, np.inf])
    table = woe_table(pd.cut(balance, edges), target).sort_index()
    assert table.total.min() >= 0.05 * len(balance)
    assert table.event.tolist() == [56, 123, 106, 47, 16]
    assert table.non_event.tolist() == [767, 996, 587, 168, 134]
    # Not monotonic: the WOE rises over the first four bins and falls in the top one.
    assert table.woe.iloc[:4].is_monotonic_increasing and table.woe.iloc[4] < table.woe.iloc[3]
    quantile = woe_table(pd.cut(balance, bin_edges(balance, bins=5)), target)
    assert table.iv.sum() > quantile.iv.sum()


def test_balance_woe_rises_under_every_method(reference):
    balance, target = reference["balance"], reference["target"]
    for edges in (chimerge_edges(balance, target), monotonic_edges(balance, target)):
        woe = woe_table(pd.cut(balance, edges), target).sort_index().woe
        assert woe.is_monotonic_increasing and len(woe) > 1
    assert len(tree_edges(balance, target)) - 1 == 5


@pytest.mark.parametrize("fit", [chimerge_edges, tree_edges, ks_edges, monotonic_edges])
def test_array_like_inputs_are_matched_by_position(fit):
    expected = fit(TREE_VALUES, TREE_TARGET)
    np.testing.assert_equal(fit(TREE_VALUES.tolist(), TREE_TARGET.to_numpy()), expected)
    np.testing.assert_equal(fit(TREE_VALUES, TREE_TARGET.tolist()), expected)
    shifted = TREE_TARGET.set_axis(TREE_TARGET.index + 100)
    np.testing.assert_equal(fit(TREE_VALUES.to_numpy(), shifted), expected)
    with_missing = [*TREE_VALUES, None]
    np.testing.assert_equal(fit(with_missing, [*TREE_TARGET, 1]), expected)


@pytest.mark.parametrize("fit", [chimerge_edges, tree_edges, ks_edges, monotonic_edges])
@pytest.mark.parametrize(
    "values,target,message",
    [
        (pd.Series([], dtype=float), pd.Series([], dtype=int), "^values have no finite obs"),
        (
            pd.Series(["a", "b"]),
            pd.Series([0, 1]),
            "values must be numbers or special values, got 'a'",
        ),
        (
            pd.Series([1.0, np.inf]),
            pd.Series([0, 1]),
            "values must be finite or special values, got inf",
        ),
        (pd.Series([1, 2]), pd.Series([0, 2]), "only 0 and 1"),
        (pd.Series([1, 2]), pd.Series([0, None]), "only 0 and 1"),
        (pd.Series([1, 2]), pd.Series([1, 1]), "both 0 and 1"),
        (pd.Series([1, 2, None]), pd.Series([0, 0, 1]), "both 0 and 1 where values are observed"),
        (pd.Series([1, 2]), pd.Series([0, 1], index=[1, 0]), "identical indices"),
        ([1, 2, 3], [0, 1], "same length, got 3 and 2"),
        (np.ones((2, 2)), [0, 1], "values must be one-dimensional, got 2 dimensions"),
        ([1, 2], np.ones((2, 2)), "target must be one-dimensional, got 2 dimensions"),
    ],
)
def test_invalid_supervised_inputs(fit, values, target, message):
    with pytest.raises(ValueError, match=message):
        fit(values, target)


@pytest.mark.parametrize("fit", [chimerge_edges, tree_edges, ks_edges, monotonic_edges])
@pytest.mark.parametrize(
    "values,target,special,message",
    [
        ([1, 2], [0, 1], -1, "special must be one-dimensional, got 0 dimensions"),
        ([1, 2], [0, 1], [-1, np.nan], "special must not contain missing values"),
        (
            [-1, -1, None],
            [0, 1, 1],
            [-1],
            "no finite observations apart from missing and special values",
        ),
        (
            [1, 2, np.inf],
            [0, 1, 1],
            [-1],
            "^values must be finite or special values, got inf; pass infinite codes in special",
        ),
        ([1, 2, -1], [0, 0, 1], [-1], "both 0 and 1 where values are observed and not special"),
        (["N/A", 1, 2], [0, 0, 1], [-1], "values must be numbers or special values, got 'N/A'"),
        ([1, 2, -1], [0, 1, 2], [-1], "only 0 and 1"),
    ],
)
def test_invalid_special_values(fit, values, target, special, message):
    with pytest.raises(ValueError, match=message):
        fit(values, target, special=special)


@pytest.mark.parametrize(
    "fit,options,message",
    [
        (chimerge_edges, {"max_bins": 0}, "max_bins must be a positive integer, got 0"),
        (chimerge_edges, {"max_bins": True}, "max_bins must be a positive integer, got True"),
        (chimerge_edges, {"initial_bins": 2.5}, "initial_bins must be a positive integer, got 2.5"),
        (chimerge_edges, {"alpha": 0}, "alpha must be None or between 0 and 1, got 0"),
        (chimerge_edges, {"alpha": 1.5}, "alpha must be None or between 0 and 1, got 1.5"),
        (tree_edges, {"max_bins": 0}, "max_bins must be a positive integer, got 0"),
        (tree_edges, {"min_bin_share": -0.1}, "min_bin_share must be between 0 and 0.5, got -0.1"),
        (tree_edges, {"min_bin_share": 0.6}, "min_bin_share must be between 0 and 0.5, got 0.6"),
        (tree_edges, {"criterion": "log_loss"}, "'entropy' or 'gini', got 'log_loss'"),
        (ks_edges, {"max_bins": 0}, "max_bins must be a positive integer, got 0"),
        (ks_edges, {"max_bins": 2.0}, "max_bins must be a positive integer, got 2.0"),
        (ks_edges, {"min_bin_share": -0.1}, "min_bin_share must be between 0 and 0.5, got -0.1"),
        (ks_edges, {"min_bin_share": 0.6}, "min_bin_share must be between 0 and 0.5, got 0.6"),
        (ks_edges, {"min_bin_share": True}, "min_bin_share must be between 0 and 0.5, got True"),
        (ks_edges, {"min_bin_share": "0.1"}, "min_bin_share must be between 0 and 0.5, got '0.1'"),
        (ks_edges, {"min_bin_share": np.nan}, "min_bin_share must be between 0 and 0.5, got nan"),
        (monotonic_edges, {"max_bins": 0}, "max_bins must be a positive integer, got 0"),
        (monotonic_edges, {"initial_bins": 0}, "initial_bins must be a positive integer, got 0"),
        (monotonic_edges, {"direction": "up"}, "'increasing' or 'decreasing', got 'up'"),
        (monotonic_edges, {"smoothing": 0}, "smoothing must be finite and positive, got 0"),
    ],
)
def test_invalid_supervised_options(fit, options, message):
    with pytest.raises(ValueError, match=message):
        fit(TREE_VALUES, TREE_TARGET, **options)


@pytest.mark.parametrize("fit", [chimerge_edges, tree_edges, ks_edges, monotonic_edges])
def test_integer_weights_match_repeated_rows(fit):
    values = np.arange(1, 13)
    target = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1])
    weights = np.array([1, 2, 1, 3, 1, 1, 2, 1, 1, 2, 1, 1])
    repeated = fit(np.repeat(values, weights), np.repeat(target, weights))
    np.testing.assert_equal(fit(values, target, weights=weights), repeated)


def test_zero_weight_rows_are_left_out_with_their_labels():
    values = np.arange(1, 13)
    target = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1])
    weights = np.r_[np.ones(12), 0.0]
    padded_values, padded_target = np.r_[values, 1000], np.r_[target, 1]
    for fit in (chimerge_edges, tree_edges, ks_edges, monotonic_edges):
        np.testing.assert_equal(
            fit(padded_values, padded_target, weights=weights), fit(values, target)
        )


def test_tree_min_bin_share_is_measured_in_weight():
    values = np.arange(1, 11)
    target = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    weights = np.r_[10, np.ones(9)]
    # The first value carries 10 of the 19 units of weight, so it can form a bin of its own
    # under min_bin_share=0.3 although it is one row of ten.
    edges = tree_edges(values, target, weights=weights, min_bin_share=0.3)
    np.testing.assert_equal(edges, [-np.inf, 1, np.inf])
    repeated = tree_edges(
        np.repeat(values, [10] + [1] * 9), np.repeat(target, [10] + [1] * 9), min_bin_share=0.3
    )
    np.testing.assert_equal(edges, repeated)
    # Counted in rows, the first bin needs three of the ten values.
    np.testing.assert_equal(tree_edges(values, target, min_bin_share=0.3), [-np.inf, 3, np.inf])


def test_supervised_weights_need_matching_index_and_weighted_classes():
    values = pd.Series(np.arange(1, 13))
    target = pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="values and weights must have identical indices"):
        chimerge_edges(values, target, weights=pd.Series(np.ones(12), index=np.arange(1, 13)))
    with pytest.raises(ValueError, match="observed with a positive weight"):
        tree_edges(values, target, weights=target.to_numpy(dtype=float))


def test_special_codes_and_weights_in_supervised_binning():
    values = np.r_[np.arange(1, 13), -1, -1]
    target = np.r_[[0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1], 1, 0]
    weights = np.r_[[1, 2, 1, 3, 1, 1, 2, 1, 1, 2, 1, 1], 7, 7]
    for fit in (chimerge_edges, tree_edges, ks_edges, monotonic_edges):
        np.testing.assert_equal(
            fit(values, target, special=[-1], weights=weights),
            fit(values[:12], target[:12], weights=weights[:12]),
        )


def test_nullable_values_with_special_codes_in_supervised_binning():
    values = pd.Series([*range(1, 13), -1, None], dtype="Int64")
    target = pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0])
    expected = chimerge_edges(np.arange(1, 13), target[:12].to_numpy())
    np.testing.assert_equal(chimerge_edges(values, target, special=[-1]), expected)
