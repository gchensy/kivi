import itertools
import math
import re

import numpy as np
import pandas as pd
import pytest

from kivi.binning import (
    _fine_bins,
    _labeled,
    _pair_chi2,
    bin_edges,
    chimerge_edges,
    cut,
    ks_edges,
    monotonic_edges,
    optimal_edges,
    tree_edges,
)
from kivi.datasets import load_bank
from kivi.metrics import gini, ks
from kivi.woe import woe_table, woe_trend


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


SUPERVISED = [chimerge_edges, tree_edges, ks_edges, monotonic_edges, optimal_edges]


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


def test_chimerge_merges_a_small_bin_into_the_more_similar_neighbour():
    # The middle value holds 3 of 23 rows, below 20%. Its chi-square with the left bin is
    # smaller, so the two merge; without min_bin_share it keeps a bin of its own.
    values, target = repeated([(1, 10, 0), (2, 2, 1), (3, 0, 10)])
    assert chi2_2x2(10, 0, 2, 1) == pytest.approx(13 / 3.6)
    assert chi2_2x2(2, 1, 0, 10) == pytest.approx(5200 / 660)
    np.testing.assert_equal(chimerge_edges(values, target, alpha=None), [-np.inf, 1, 2, np.inf])
    np.testing.assert_equal(
        chimerge_edges(values, target, alpha=None, min_bin_share=0.2), [-np.inf, 2, np.inf]
    )
    # With one event more, the right neighbour is the more similar one.
    values, target = repeated([(1, 10, 0), (2, 1, 2), (3, 0, 10)])
    assert chi2_2x2(1, 2, 0, 10) < chi2_2x2(10, 0, 1, 2)
    np.testing.assert_equal(
        chimerge_edges(values, target, alpha=None, min_bin_share=0.2), [-np.inf, 1, np.inf]
    )
    # Equal statistics merge into the left neighbour.
    values, target = repeated([(1, 10, 0), (2, 1, 1), (3, 0, 10)])
    assert chi2_2x2(10, 0, 1, 1) == pytest.approx(chi2_2x2(1, 1, 0, 10), rel=1e-15)
    np.testing.assert_equal(
        chimerge_edges(values, target, alpha=None, min_bin_share=0.1), [-np.inf, 2, np.inf]
    )


def test_small_bins_merge_smallest_first_and_at_the_ends_into_their_one_neighbour():
    # Shares 2/30, 3/30, 20/30 and 5/30 with a 12% minimum (3.6 rows): the first value, the
    # smallest, joins its only neighbour, which then holds 5 rows like the last value.
    values, target = repeated([(1, 1, 1), (2, 2, 1), (3, 10, 10), (4, 1, 4)])
    np.testing.assert_equal(chimerge_edges(values, target, alpha=None), [-np.inf, 1, 2, 3, np.inf])
    np.testing.assert_equal(
        chimerge_edges(values, target, alpha=None, min_bin_share=0.12), [-np.inf, 2, 3, np.inf]
    )
    # Reversed, the last value joins its only neighbour.
    np.testing.assert_equal(
        chimerge_edges(-values, target, alpha=None, min_bin_share=0.12),
        [-np.inf, -4, -3, np.inf],
    )


def test_monotonic_merges_small_bins_before_pooling_violators():
    # Each value holds 10 of the 40 rows. A share of exactly 25% is allowed; with 30%, the
    # first value joins the second, the third joins that bin, whose rate it shares (a
    # chi-square of 0), and the last value joins them too.
    options = {"max_bins": None, "direction": "increasing"}
    expected = monotonic_edges(MONO_VALUES, MONO_TARGET, **options)
    np.testing.assert_equal(expected, [-np.inf, 1, 3, np.inf])
    np.testing.assert_equal(
        monotonic_edges(MONO_VALUES, MONO_TARGET, min_bin_share=0.25, **options), expected
    )
    assert chi2_2x2(16, 4, 8, 2) == 0
    np.testing.assert_equal(
        monotonic_edges(MONO_VALUES, MONO_TARGET, min_bin_share=0.3, **options),
        [-np.inf, np.inf],
    )


@pytest.mark.parametrize("column", ["age", "balance", "campaign", "day", "duration"])
@pytest.mark.parametrize("fit", [chimerge_edges, monotonic_edges])
def test_min_bin_share_holds_on_bank_data(reference, fit, column):
    values, target = reference[column], reference["target"]
    edges = fit(values, target, min_bin_share=0.05)
    counts = pd.cut(values, edges).value_counts()
    assert len(counts) == 1 or counts.min() >= 0.05 * len(values)
    # The default merges no bin for its size.
    np.testing.assert_equal(fit(values, target, min_bin_share=0), fit(values, target))


# (non_event, event) per value: event rates 0.2, 0.4, 0.6 and 0.8, 20 events of 40 rows.
RISING_VALUES, RISING_TARGET = repeated([(1, 8, 2), (2, 6, 4), (3, 4, 6), (4, 2, 8)])
# Event rates 0.8, 0.2, 0.2 and 0.8: a U shape.
U_VALUES, U_TARGET = repeated([(1, 2, 8), (2, 8, 2), (3, 8, 2), (4, 2, 8)])


def information_value(values, target, edges):
    return woe_table(pd.cut(values, edges), target)["iv"].sum()


def test_optimal_maximizes_the_information_value_by_hand():
    # Shares of the 20 events and 20 non-events: (0.1, 0.4), (0.2, 0.3), (0.3, 0.2), (0.4, 0.1).
    ends = 0.3 * math.log(4)
    middle = 0.1 * math.log(1.5)
    edges = optimal_edges(RISING_VALUES, RISING_TARGET, min_bin_share=0)
    np.testing.assert_equal(edges, [-np.inf, 1, 2, 3, np.inf])
    assert information_value(RISING_VALUES, RISING_TARGET, edges) == pytest.approx(
        2 * ends + 2 * middle, rel=1e-14
    )
    # Two bins: the middle cut gives 0.8 ln(7/3), the others 0.3 ln 4 + 0.3 ln 1.5.
    assert 0.8 * math.log(7 / 3) > ends + 0.3 * math.log(1.5)
    np.testing.assert_equal(
        optimal_edges(RISING_VALUES, RISING_TARGET, max_bins=2), [-np.inf, 2, np.inf]
    )
    # Three bins: an end value on its own (0.3 ln 4) beats a middle one (0.1 ln 1.5), and
    # of the two equal choices the lower edge comes first.
    three = optimal_edges(RISING_VALUES, RISING_TARGET, max_bins=3)
    np.testing.assert_equal(three, [-np.inf, 1, 3, np.inf])
    assert information_value(RISING_VALUES, RISING_TARGET, three) == pytest.approx(
        2 * ends, rel=1e-14
    )


def test_optimal_respects_the_share_and_the_events_of_every_bin():
    # 30% of 40 rows is 12, so no value can form a bin alone (10 rows).
    np.testing.assert_equal(
        optimal_edges(RISING_VALUES, RISING_TARGET, min_bin_share=0.3), [-np.inf, 2, np.inf]
    )
    # The values 1 and 4 hold only 2 events or 2 non-events, below 3 of each.
    np.testing.assert_equal(
        optimal_edges(RISING_VALUES, RISING_TARGET, min_bin_events=3), [-np.inf, 2, np.inf]
    )
    np.testing.assert_equal(
        optimal_edges(RISING_VALUES, RISING_TARGET, min_bin_events=2),
        [-np.inf, 1, 2, 3, np.inf],
    )
    # 10 events are needed on both sides of a cut, so only one bin is left.
    np.testing.assert_equal(
        optimal_edges(RISING_VALUES, RISING_TARGET, min_bin_events=10.5), [-np.inf, np.inf]
    )
    # A bin without events would have an infinite WOE, so the default needs one of each.
    values, target = repeated([(1, 10, 0), (2, 5, 5), (3, 1, 9)])
    np.testing.assert_equal(optimal_edges(values, target), [-np.inf, 2, np.inf])
    table = woe_table(pd.cut(values, optimal_edges(values, target, max_bins=3)), target)
    assert table.event.min() >= 1 and table.non_event.min() >= 1


def test_optimal_fits_each_trend():
    np.testing.assert_equal(
        optimal_edges(U_VALUES, U_TARGET, trend="u_shaped"), [-np.inf, 1, 3, np.inf]
    )
    np.testing.assert_equal(optimal_edges(U_VALUES, U_TARGET, trend="any"), [-np.inf, 1, 3, np.inf])
    # (18, 12) then (2, 8) rises, (2, 8) then (18, 12) falls: both 0.3 ln 1.5 + 0.3 ln 4.
    np.testing.assert_equal(
        optimal_edges(U_VALUES, U_TARGET, trend="increasing"), [-np.inf, 3, np.inf]
    )
    np.testing.assert_equal(
        optimal_edges(U_VALUES, U_TARGET, trend="decreasing"), [-np.inf, 1, np.inf]
    )
    for edges in ([-np.inf, 1, np.inf], [-np.inf, 3, np.inf]):
        assert information_value(U_VALUES, U_TARGET, edges) == pytest.approx(
            0.3 * math.log(1.5) + 0.3 * math.log(4), rel=1e-14
        )
    # "auto" prefers increasing on the tie; an inverted U can only be monotonic here, and of
    # the two equal binnings the one with the lower edge comes first.
    np.testing.assert_equal(optimal_edges(U_VALUES, U_TARGET), [-np.inf, 3, np.inf])
    np.testing.assert_equal(
        optimal_edges(U_VALUES, U_TARGET, trend="inverted_u"), [-np.inf, 1, np.inf]
    )
    # The U shape has the IV 1.2 ln 4, and the reversed sample has an inverted U.
    u_edges = optimal_edges(U_VALUES, U_TARGET, trend="u_shaped")
    assert information_value(U_VALUES, U_TARGET, u_edges) == pytest.approx(1.2 * math.log(4))
    assert woe_trend(woe_table(pd.cut(U_VALUES, u_edges), U_TARGET)) == "u_shaped"
    peak = optimal_edges(U_VALUES, 1 - U_TARGET, trend="inverted_u")
    np.testing.assert_equal(peak, u_edges)
    assert woe_trend(woe_table(pd.cut(U_VALUES, peak), 1 - U_TARGET)) == "inverted_u"


def test_optimal_ties_go_to_fewer_bins_then_to_lower_edges():
    # Splitting the middle of the U into its two equal halves adds nothing to the IV.
    four = woe_table(pd.cut(U_VALUES, [-np.inf, 1, 2, 3, np.inf]), U_TARGET)["iv"].sum()
    assert four == pytest.approx(1.2 * math.log(4), rel=1e-14)
    np.testing.assert_equal(optimal_edges(U_VALUES, U_TARGET, trend="any"), [-np.inf, 1, 3, np.inf])
    # Proportional neighbours have the same WOE, so two bins carry the IV of four.
    values, target = repeated([(1, 2, 1), (2, 4, 2), (3, 1, 2), (4, 2, 4)])
    np.testing.assert_equal(
        optimal_edges(values, target, trend="any", min_bin_share=0), [-np.inf, 2, np.inf]
    )
    # The rates 1/4, 1/2 and 3/4 cut after 1 or after 2 give the same IV by symmetry.
    values, target = repeated([(1, 3, 1), (2, 2, 2), (3, 1, 3)])
    assert information_value(values, target, [-np.inf, 1, np.inf]) == pytest.approx(
        information_value(values, target, [-np.inf, 2, np.inf]), rel=1e-12
    )
    for trend in ("increasing", "any", "auto"):
        np.testing.assert_equal(
            optimal_edges(values, target, max_bins=2, trend=trend, min_bin_share=0),
            [-np.inf, 1, np.inf],
        )


def test_optimal_with_a_single_initial_bin_or_no_signal():
    np.testing.assert_equal(optimal_edges([5, 5, 5, 5], [0, 1, 0, 1]), [-np.inf, np.inf])
    values, target = repeated([(1, 2, 2), (2, 3, 3), (3, 1, 1)])
    for trend in ("auto", "any", "u_shaped"):
        np.testing.assert_equal(optimal_edges(values, target, trend=trend), [-np.inf, np.inf])


TIE = 1e-10


def exhaustive_edges(
    values, target, *, max_bins, min_bin_share, min_bin_events, trend, initial_bins, weights=None
):
    """The edges of optimal_edges by trying every binning of the initial bins."""
    if trend == "auto":
        fits = [
            exhaustive_edges(
                values,
                target,
                max_bins=max_bins,
                min_bin_share=min_bin_share,
                min_bin_events=min_bin_events,
                trend=shape,
                initial_bins=initial_bins,
                weights=weights,
            )
            for shape in ("increasing", "decreasing")
        ]
        return fits[0] if fits[0][1] >= fits[1][1] - TIE * fits[1][1] else fits[1]
    observed, labels, mass = _labeled(values, target, None, weights)
    interior, counts = _fine_bins(observed, labels, initial_bins, mass)
    total = counts.sum(axis=0)
    found = []
    for size in range(min(max_bins, len(counts))):
        for cuts in itertools.combinations(range(1, len(counts)), size):
            bounds = [0, *cuts, len(counts)]
            bins = np.array([counts[a:b].sum(axis=0) for a, b in itertools.pairwise(bounds)])
            if (bins.sum(axis=1) < min_bin_share * total.sum() * (1 - TIE)).any():
                continue
            if (bins < min_bin_events * (1 - TIE)).any():
                continue
            event, non_event = bins[:, 1] / total[1], bins[:, 0] / total[0]
            woe = np.log(event / non_event)
            change = np.diff(woe)
            scale = np.maximum(np.abs(woe[:-1]), np.abs(woe[1:]))
            steps = "".join(
                "+" if step > 0 else "-" if step < 0 else "="
                for step in np.where(np.abs(change) > TIE * scale, np.sign(change), 0)
            )
            allowed = {
                "increasing": r"\+*",
                "decreasing": r"-*",
                "u_shaped": r"-*\+*",
                "inverted_u": r"\+*-*",
                "any": r"[-+=]*",
            }[trend]
            if re.fullmatch(allowed, steps):
                found.append((((event - non_event) * woe).sum(), cuts))
    best = max(iv for iv, _ in found)
    cuts = min((len(cuts), cuts) for iv, cuts in found if iv >= best - TIE * best)[1]
    return np.r_[-np.inf, interior[np.array(cuts, dtype=int) - 1], np.inf], best


@pytest.mark.parametrize("weighting", ["none", "integer", "fractional"])
def test_optimal_matches_an_exhaustive_search(weighting):
    rng = np.random.default_rng({"none": 0, "integer": 1, "fractional": 2}[weighting])
    checked = 0
    for _ in range(40):
        size = int(rng.integers(20, 80))
        values = rng.integers(0, int(rng.integers(3, 12)), size).astype(float)
        shape = rng.normal(size=3)
        scaled = (values - values.mean()) / 3
        logit = shape[0] + shape[1] * scaled + shape[2] * scaled**2
        target = (rng.random(size) < 1 / (1 + np.exp(-logit))).astype(int)
        if target.min() == target.max():
            continue
        weights = {
            "none": None,
            "integer": rng.integers(1, 4, size),
            "fractional": rng.random(size) * 3,
        }[weighting]
        options = {
            "max_bins": int(rng.integers(1, 6)),
            "min_bin_share": float(rng.choice([0, 0.05, 0.1, 0.2])),
            "min_bin_events": float(rng.choice([0.5, 1, 2])),
            "initial_bins": int(rng.integers(3, 10)),
        }
        mass = np.ones(size) if weights is None else weights
        if min((mass * target).sum(), (mass * (1 - target)).sum()) < 2:
            continue
        for trend in ("auto", "increasing", "decreasing", "u_shaped", "inverted_u", "any"):
            expected, _ = exhaustive_edges(values, target, trend=trend, weights=weights, **options)
            actual = optimal_edges(values, target, trend=trend, weights=weights, **options)
            np.testing.assert_equal(actual, expected)
            checked += 1
    assert checked >= 150


@pytest.mark.parametrize("trend", ["auto", "increasing", "u_shaped", "inverted_u", "any"])
def test_optimal_integer_weights_match_repeated_rows(trend):
    rng = np.random.default_rng(3)
    values = rng.integers(0, 15, 200).astype(float)
    target = (rng.random(200) < 0.1 + 0.05 * np.abs(values - 7) / 7).astype(int)
    weights = rng.integers(1, 5, 200)
    options = {"trend": trend, "max_bins": 6, "min_bin_share": 0.03, "min_bin_events": 5}
    weighted = optimal_edges(values, target, weights=weights, **options)
    repeated_rows = optimal_edges(np.repeat(values, weights), np.repeat(target, weights), **options)
    np.testing.assert_equal(weighted, repeated_rows)
    assert len(weighted) >= 3
    # min_bin_events and min_bin_share are measured in weight.
    groups = pd.cut(values, weighted, labels=False)
    events = np.bincount(groups, weights=target * weights)
    assert events.min() >= 5
    assert np.bincount(groups, weights=weights).min() >= 0.03 * weights.sum()


ALLOWED_TRENDS = {
    "auto": {"increasing", "decreasing", "flat"},
    "increasing": {"increasing", "flat"},
    "decreasing": {"decreasing", "flat"},
    "u_shaped": {"u_shaped", "increasing", "decreasing", "flat"},
    "inverted_u": {"inverted_u", "increasing", "decreasing", "flat"},
    "any": {"u_shaped", "inverted_u", "increasing", "decreasing", "flat", "irregular"},
}


@pytest.mark.parametrize("trend", list(ALLOWED_TRENDS))
@pytest.mark.parametrize("column", ["age", "balance", "duration", "campaign", "day", "pdays"])
def test_optimal_constraints_hold_on_bank_data(reference, column, trend):
    values, target = reference[column], reference["target"]
    options = {"max_bins": 6, "min_bin_share": 0.04, "min_bin_events": 10}
    special = [-1] if column == "pdays" else None
    edges = optimal_edges(values, target, trend=trend, special=special, **options)
    table = woe_table(cut(values, edges, special=special), target)
    intervals = table.loc[[isinstance(label, pd.Interval) for label in table.index]]
    fitted = values.ne(-1).sum() if special else len(values)
    assert 1 <= len(intervals) <= 6
    assert intervals.total.min() >= 0.04 * fitted
    assert intervals.event.min() >= 10 and intervals.non_event.min() >= 10
    assert woe_trend(table) in ALLOWED_TRENDS[trend]
    assert set(edges[1:-1]) <= set(_fine_bins(*_labeled(values, target, special)[:2], 20)[0])


@pytest.mark.parametrize("column", ["age", "balance", "duration", "campaign", "day"])
def test_optimal_is_at_least_as_good_as_the_greedy_merges(reference, column):
    # Both merge the same initial bins, so a greedy binning that meets the constraints is one
    # of the candidates of optimal_edges.
    values, target = reference[column], reference["target"]
    for direction in ("increasing", "decreasing"):
        greedy = monotonic_edges(values, target, direction=direction, min_bin_share=0.05)
        table = woe_table(pd.cut(values, greedy), target)
        if table.event.min() >= 1 and table.non_event.min() >= 1:
            best = optimal_edges(values, target, trend=direction)
            assert information_value(values, target, best) >= table.iv.sum() * (1 - 1e-12)
    greedy = chimerge_edges(values, target, alpha=None, min_bin_share=0.05)
    table = woe_table(pd.cut(values, greedy), target)
    if table.event.min() >= 1 and table.non_event.min() >= 1:
        best = optimal_edges(values, target, trend="any")
        assert information_value(values, target, best) >= table.iv.sum() * (1 - 1e-12)


def test_optimal_with_many_initial_bins(reference):
    balance, target = reference["balance"], reference["target"]
    edges = optimal_edges(balance, target, initial_bins=200, max_bins=8, trend="inverted_u")
    table = woe_table(pd.cut(balance, edges), target)
    assert len(table) == 8 and table.total.min() >= 0.05 * len(balance)
    assert woe_trend(table) in ALLOWED_TRENDS["inverted_u"]
    assert information_value(balance, target, edges) >= information_value(
        balance, target, optimal_edges(balance, target, max_bins=8, trend="inverted_u")
    )


# Computed once with optbinning 1.1.0 (ortools 9.15) in a throwaway environment with Python
# 3.12, not a test dependency; see test_optimal_matches_optbinning.
OPTBINNING = {
    ("age", "default", "increasing"): ([44, 51, 56], 0.012214730096840485),
    ("age", "default", "decreasing"): ([29], 0.01808901426737997),
    ("age", "default", "u_shaped"): ([29, 38, 42, 56], 0.03978343656179151),
    ("age", "default", "inverted_u"): ([29], 0.01808901426737997),
    ("age", "default", "any"): ([29, 38, 42, 56], 0.03978343656179151),
    ("age", "strict", "increasing"): ([59], 0.0552142285815414),
    ("age", "strict", "decreasing"): ([27, 29], 0.01809265889399122),
    ("age", "strict", "u_shaped"): ([27, 29, 38, 41, 44, 59], 0.08401659419680273),
    ("age", "strict", "inverted_u"): ([59], 0.0552142285815414),
    ("age", "strict", "any"): ([29, 33, 35, 38, 42, 53, 59], 0.08965784945413659),
    ("balance", "default", "increasing"): ([68.75, 125.7, 895.7, 1136.6], 0.12408159732442757),
    ("balance", "default", "decreasing"): ([6047.3], 0.0004512929891320046),
    ("balance", "default", "u_shaped"): ([-186.15, 68.75, 125.7, 895.7], 0.1276293576569535),
    ("balance", "default", "inverted_u"): ([68.75, 895.7, 3873.2, 6047.3], 0.14315114951283497),
    ("balance", "default", "any"): ([68.75, 895.7, 3873.2, 6047.3], 0.14315114951283497),
    ("balance", "strict", "increasing"): (
        [68.75, 197, 461.5, 895.7, 1136.6, 1905.2, 3873.2],
        0.123486091962317,
    ),
    ("balance", "strict", "decreasing"): ([], 0.0),
    ("balance", "strict", "u_shaped"): (
        [68.75, 197, 461.5, 895.7, 1136.6, 1905.2, 3873.2],
        0.123486091962317,
    ),
    ("balance", "strict", "inverted_u"): (
        [68.75, 197, 461.5, 895.7, 1136.6, 1445.75],
        0.1248380794569659,
    ),
    ("balance", "strict", "any"): (
        [68.75, 197, 263.6, 461.5, 895.7, 1445.75, 1905.2],
        0.13339415179835712,
    ),
    ("campaign", "default", "increasing"): ([], 0.0),
    ("campaign", "default", "decreasing"): ([2, 4], 0.03071202087983213),
    ("campaign", "default", "u_shaped"): ([2, 4], 0.03071202087983213),
    ("campaign", "default", "inverted_u"): ([2, 4], 0.03071202087983213),
    ("campaign", "default", "any"): ([2, 3, 4], 0.03912939229168381),
    ("campaign", "strict", "increasing"): ([], 0.0),
    ("campaign", "strict", "decreasing"): ([2, 4], 0.03071202087983213),
    ("campaign", "strict", "u_shaped"): ([2, 4], 0.03071202087983213),
    ("campaign", "strict", "inverted_u"): ([2, 4], 0.03071202087983213),
    ("campaign", "strict", "any"): ([2, 3, 4], 0.03912939229168381),
    ("day", "default", "increasing"): ([9], 0.0010600767672298205),
    ("day", "default", "decreasing"): ([3, 5, 18, 28], 0.03642373813783335),
    ("day", "default", "u_shaped"): ([3, 5, 7, 9], 0.07119693470599456),
    ("day", "default", "inverted_u"): ([3, 5, 18, 28], 0.03642373813783335),
    ("day", "default", "any"): ([5, 9, 18, 20], 0.0949187003972995),
    ("day", "strict", "increasing"): ([9], 0.0010600767672298205),
    ("day", "strict", "decreasing"): ([3, 5, 18, 28], 0.03642373813783335),
    ("day", "strict", "u_shaped"): ([3, 5, 9], 0.0678122551678442),
    ("day", "strict", "inverted_u"): ([3, 5, 18, 28], 0.03642373813783335),
    ("day", "strict", "any"): ([3, 5, 9, 12, 18, 20, 28], 0.11580240918489501),
    ("pdays", "default", "increasing"): ([96], 0.0685863370076607),
    ("pdays", "default", "decreasing"): ([316.05], 3.593499140530299e-05),
    ("pdays", "default", "u_shaped"): ([96], 0.0685863370076607),
    ("pdays", "default", "inverted_u"): ([96, 316.05], 0.09246938428842186),
    ("pdays", "default", "any"): ([96, 316.05], 0.09246938428842186),
    ("pdays", "strict", "increasing"): ([96], 0.0685863370076607),
    ("pdays", "strict", "decreasing"): ([], 0.0),
    ("pdays", "strict", "u_shaped"): ([96], 0.0685863370076607),
    ("pdays", "strict", "inverted_u"): ([96, 181], 0.07984714785466075),
    ("pdays", "strict", "any"): ([96, 181], 0.07984714785466075),
}
OPTBINNING_SETTINGS = {
    "default": {"max_bins": 5, "min_bin_share": 0.05, "min_bin_events": 1},
    "strict": {"max_bins": 8, "min_bin_share": 0.02, "min_bin_events": 20},
}


@pytest.mark.parametrize(("column", "setting", "trend"), list(OPTBINNING))
def test_optimal_matches_optbinning(reference, column, setting, trend):
    """Compare with the optimal binning of optbinning on the same initial bins.

    The expected edges and IVs were computed once with optbinning 1.1.0, which needs
    Python >= 3.11 and NumPy >= 2.0.2 and is therefore no test dependency::

        import numpy as np
        from optbinning import OptimalBinning
        from kivi.binning import _fine_bins, _labeled
        from kivi.datasets import load_bank

        reference = load_bank().iloc[:3000]
        x, y = reference[column].to_numpy(float), reference["target"].to_numpy()
        interior, _ = _fine_bins(*_labeled(x, y)[:2], 20)
        monotonic = {"increasing": "ascending", "decreasing": "descending",
                     "u_shaped": "valley", "inverted_u": "peak", "any": None}[trend]
        binning = OptimalBinning(
            dtype="numerical", solver=solver,  # "cp" and "mip" agree
            user_splits=np.nextafter(interior, np.inf),
            max_n_bins=max_bins, min_bin_size=min_bin_share,
            min_bin_n_event=min_bin_events, min_bin_n_nonevent=min_bin_events,
            monotonic_trend=monotonic,
            min_event_rate_diff=0 if monotonic is None else 1e-9,
        ).fit(x, y)
        np.nextafter(binning.splits, -np.inf)  # the interior edges
        binning.binning_table.build().loc["Totals", "IV"]

    optbinning cuts below each split (``x < s``) where ``pandas.cut`` cuts above each edge
    (``x <= e``), hence ``nextafter``. None of these columns has an initial bin without
    events or without non-events, which optbinning would merge before it optimizes. With
    ``min_event_rate_diff=0`` its presolve for a monotonic trend forbids a bin to end where
    the event rate of the next initial bin falls, which here misses the increasing optimum
    of ``age``; the tiny difference turns the presolve off without changing the problem,
    since kivi's trends are strict.
    """
    values, target = reference[column], reference["target"]
    interior, iv = OPTBINNING[column, setting, trend]
    edges = optimal_edges(values, target, trend=trend, **OPTBINNING_SETTINGS[setting])
    np.testing.assert_allclose(edges, [-np.inf, *interior, np.inf], rtol=1e-12, atol=0)
    assert information_value(values, target, edges) == pytest.approx(iv, rel=1e-12, abs=1e-15)


def test_optbinning_presolve_misses_the_increasing_age_optimum(reference):
    # optbinning's default (min_event_rate_diff=0) returned the edges 44 and 51, with an IV
    # of 0.00861, lower than that of optimal_edges, although the binning is allowed.
    age, target = reference["age"], reference["target"]
    presolved = [-np.inf, 44, 51, np.inf]
    table = woe_table(pd.cut(age, presolved), target)
    assert table.total.min() >= 0.05 * len(age) and woe_trend(table) == "increasing"
    assert table.iv.sum() == pytest.approx(0.008611798532111671, rel=1e-12)
    assert information_value(age, target, optimal_edges(age, target, trend="increasing")) > (
        table.iv.sum()
    )


@pytest.mark.parametrize("fit", SUPERVISED)
def test_missing_values_are_omitted_and_inputs_are_kept(fit):
    values = pd.concat([TREE_VALUES, pd.Series([np.nan, np.nan], index=[20, 21])])
    target = pd.concat([TREE_TARGET, pd.Series([1, 1], index=[20, 21])])
    before = values.copy(), target.copy()
    np.testing.assert_equal(fit(values, target), fit(TREE_VALUES, TREE_TARGET))
    pd.testing.assert_series_equal(values, before[0])
    pd.testing.assert_series_equal(target, before[1])


@pytest.mark.parametrize("fit", SUPERVISED)
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
    # 97 and 188 would leave bins below 5% of the 525 values.
    optimal_edges: [-np.inf, 105, 211.2, np.inf],
}


@pytest.mark.parametrize("fit", SUPERVISED)
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
    if fit in (monotonic_edges, optimal_edges):
        assert table.woe.iloc[1:].is_monotonic_decreasing


@pytest.mark.parametrize("fit", [chimerge_edges, monotonic_edges])
def test_without_special_the_bank_pdays_code_shares_a_bin(reference, fit):
    pdays = reference["pdays"]
    edges = fit(pdays, reference["target"])
    assert edges[1] == 96
    first = pdays[pdays <= edges[1]]
    assert (first == -1).sum() == 2475 and (first > -1).sum() == 78


@pytest.mark.parametrize("fit", SUPERVISED)
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


@pytest.mark.parametrize("fit", SUPERVISED)
def test_array_like_inputs_are_matched_by_position(fit):
    expected = fit(TREE_VALUES, TREE_TARGET)
    np.testing.assert_equal(fit(TREE_VALUES.tolist(), TREE_TARGET.to_numpy()), expected)
    np.testing.assert_equal(fit(TREE_VALUES, TREE_TARGET.tolist()), expected)
    shifted = TREE_TARGET.set_axis(TREE_TARGET.index + 100)
    np.testing.assert_equal(fit(TREE_VALUES.to_numpy(), shifted), expected)
    with_missing = [*TREE_VALUES, None]
    np.testing.assert_equal(fit(with_missing, [*TREE_TARGET, 1]), expected)


@pytest.mark.parametrize("fit", SUPERVISED)
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


@pytest.mark.parametrize("fit", SUPERVISED)
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
        (
            chimerge_edges,
            {"min_bin_share": 0.6},
            "min_bin_share must be between 0 and 0.5, got 0.6",
        ),
        (monotonic_edges, {"min_bin_share": -1}, "min_bin_share must be between 0 and 0.5, got -1"),
        (optimal_edges, {"max_bins": 0}, "max_bins must be a positive integer, got 0"),
        (optimal_edges, {"max_bins": None}, "max_bins must be a positive integer, got None"),
        (optimal_edges, {"initial_bins": 0}, "initial_bins must be a positive integer, got 0"),
        (optimal_edges, {"min_bin_share": 0.6}, "min_bin_share must be between 0 and 0.5, got 0.6"),
        (
            optimal_edges,
            {"min_bin_share": True},
            "min_bin_share must be between 0 and 0.5, got True",
        ),
        (
            optimal_edges,
            {"min_bin_events": 0},
            "min_bin_events must be a finite, positive number, got 0",
        ),
        (optimal_edges, {"min_bin_events": -1}, "min_bin_events must be a finite, positive number"),
        (optimal_edges, {"min_bin_events": np.inf}, "positive number, got inf"),
        (optimal_edges, {"min_bin_events": np.nan}, "positive number, got nan"),
        (optimal_edges, {"min_bin_events": True}, "positive number, got True"),
        (optimal_edges, {"min_bin_events": "1"}, "positive number, got '1'"),
        (
            optimal_edges,
            {"trend": "monotonic"},
            "trend must be one of 'auto', 'increasing', 'decreasing', 'u_shaped', 'inverted_u', "
            "'any', got 'monotonic'",
        ),
        (optimal_edges, {"trend": None}, "trend must be one of .*, got None"),
        (optimal_edges, {"trend": "flat"}, "trend must be one of .*, got 'flat'"),
        (
            optimal_edges,
            {"min_bin_events": 7},
            "min_bin_events must not exceed the events of the fitted values, got 7 and 6 events",
        ),
    ],
)
def test_invalid_supervised_options(fit, options, message):
    with pytest.raises(ValueError, match=message):
        fit(TREE_VALUES, TREE_TARGET, **options)


@pytest.mark.parametrize("fit", SUPERVISED)
def test_integer_weights_match_repeated_rows(fit):
    values = np.arange(1, 13)
    target = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1])
    weights = np.array([1, 2, 1, 3, 1, 1, 2, 1, 1, 2, 1, 1])
    repeated = fit(np.repeat(values, weights), np.repeat(target, weights))
    np.testing.assert_equal(fit(values, target, weights=weights), repeated)


def test_optimal_needs_enough_events_and_non_events_in_weight():
    values, target = repeated([(1, 1, 9), (2, 2, 1)])
    with pytest.raises(
        ValueError, match="exceed the non-events of the fitted values, got 4 and 3 non-events"
    ):
        optimal_edges(values, target, min_bin_events=4)
    weights = np.where(target == 1, 1.0, 0.25)
    with pytest.raises(
        ValueError, match="exceed the non-events of the fitted values, got 1 and 0.75"
    ):
        optimal_edges(values, target, weights=weights)
    # Ten events of weight 0.1 make one unit, although their sum rounds below 1.
    weights = np.full(20, 0.1)
    assert np.bincount(np.zeros(10, dtype=int), weights=weights[:10])[0] < 1
    values, target = repeated([(1, 10, 10)])
    np.testing.assert_equal(optimal_edges(values, target, weights=weights), [-np.inf, np.inf])


def test_zero_weight_rows_are_left_out_with_their_labels():
    values = np.arange(1, 13)
    target = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1])
    weights = np.r_[np.ones(12), 0.0]
    padded_values, padded_target = np.r_[values, 1000], np.r_[target, 1]
    for fit in SUPERVISED:
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
    for fit in SUPERVISED:
        np.testing.assert_equal(
            fit(values, target, special=[-1], weights=weights),
            fit(values[:12], target[:12], weights=weights[:12]),
        )


def test_nullable_values_with_special_codes_in_supervised_binning():
    values = pd.Series([*range(1, 13), -1, None], dtype="Int64")
    target = pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0])
    expected = chimerge_edges(np.arange(1, 13), target[:12].to_numpy())
    np.testing.assert_equal(chimerge_edges(values, target, special=[-1]), expected)
