import math

import numpy as np
import pandas as pd
import pytest

from kivi.binning import _pair_chi2, chimerge_edges, cut, monotonic_edges, tree_edges
from kivi.datasets import Dataset
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
    return Dataset.bank_data().iloc[:3000]


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
    np.testing.assert_equal(chimerge_edges(CHI_VALUES, CHI_TARGET, alpha=0.1), [-np.inf, 1, 2, np.inf])
    np.testing.assert_equal(chimerge_edges(CHI_VALUES, CHI_TARGET, alpha=0.05), [-np.inf, 1, np.inf])
    np.testing.assert_equal(chimerge_edges(CHI_VALUES, CHI_TARGET, alpha=None), [-np.inf, 1, 2, np.inf])
    np.testing.assert_equal(chimerge_edges(CHI_VALUES, CHI_TARGET, max_bins=2, alpha=None),
                            [-np.inf, 1, np.inf])
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
    np.testing.assert_equal(tree_edges(TREE_VALUES, TREE_TARGET, max_bins=3), [-np.inf, 4, 9, np.inf])
    np.testing.assert_equal(tree_edges(TREE_VALUES, TREE_TARGET, max_bins=4), [-np.inf, 4, 8, 9, np.inf])
    # A 10% minimum (1.2 observations) rules out the single-value bin (8, 9].
    np.testing.assert_equal(tree_edges(TREE_VALUES, TREE_TARGET, max_bins=4, min_bin_share=0.1),
                            [-np.inf, 4, 6, 9, np.inf])
    # With half the sample required on each side, only the cut after 6 is allowed.
    np.testing.assert_equal(tree_edges(TREE_VALUES, TREE_TARGET, min_bin_share=0.5), [-np.inf, 6, np.inf])


def test_tree_stops_at_pure_bins():
    values = pd.Series([1, 2, 3, 4])
    np.testing.assert_equal(tree_edges(values, pd.Series([0, 0, 1, 1]), max_bins=5), [-np.inf, 2, np.inf])


# Event rates 0.1, 0.3, 0.2 and 0.4: the pair (0.3, 0.2) breaks an increasing trend.
MONO_VALUES, MONO_TARGET = repeated([(1, 9, 1), (2, 7, 3), (3, 8, 2), (4, 6, 4)])


def test_monotonic_pools_adjacent_violators():
    edges = monotonic_edges(MONO_VALUES, MONO_TARGET, max_bins=None, direction="increasing")
    np.testing.assert_equal(edges, [-np.inf, 1, 3, np.inf])
    table = woe_table(pd.cut(MONO_VALUES, edges), MONO_TARGET).sort_index()
    assert table.event.tolist() == [1, 5, 4] and table.non_event.tolist() == [9, 15, 6]
    # WOE = ln((event / 10) / (non_event / 30)).
    np.testing.assert_allclose(table.woe, np.log(np.array([1, 5, 4]) / 10 / (np.array([9, 15, 6]) / 30)))
    assert table.woe.is_monotonic_increasing
    # A decreasing trend pools everything; "auto" keeps the direction with the larger IV.
    np.testing.assert_equal(monotonic_edges(MONO_VALUES, MONO_TARGET, direction="decreasing"),
                            [-np.inf, np.inf])
    np.testing.assert_equal(monotonic_edges(MONO_VALUES, MONO_TARGET), edges)


def test_monotonic_merges_by_chi_square_beyond_max_bins():
    # Of the pairs (1/10, 5/20) and (5/20, 4/10), the second has the smaller chi-square.
    assert chi2_2x2(9, 1, 15, 5) == pytest.approx(0.9375)
    assert chi2_2x2(15, 5, 6, 4) == pytest.approx(5 / 7)
    np.testing.assert_equal(monotonic_edges(MONO_VALUES, MONO_TARGET, max_bins=2), [-np.inf, 1, np.inf])


def test_monotonic_uses_the_smoothed_woe():
    # Raw event rates 0/2 < 1/11 rise, but with a zero count replaced by 0.5 the WOE falls.
    values, target = repeated([(1, 2, 0), (2, 10, 1), (3, 1, 3)])
    edges = monotonic_edges(values, target, direction="increasing")
    np.testing.assert_equal(edges, [-np.inf, 2, np.inf])
    assert woe_table(pd.cut(values, edges), target).sort_index().woe.is_monotonic_increasing


@pytest.mark.parametrize("fit", [chimerge_edges, tree_edges, monotonic_edges])
def test_missing_values_are_omitted_and_inputs_are_kept(fit):
    values = pd.concat([TREE_VALUES, pd.Series([np.nan, np.nan], index=[20, 21])])
    target = pd.concat([TREE_TARGET, pd.Series([1, 1], index=[20, 21])])
    before = values.copy(), target.copy()
    np.testing.assert_equal(fit(values, target), fit(TREE_VALUES, TREE_TARGET))
    pd.testing.assert_series_equal(values, before[0])
    pd.testing.assert_series_equal(target, before[1])


@pytest.mark.parametrize("fit", [chimerge_edges, tree_edges, monotonic_edges])
def test_special_values_are_left_out_with_their_labels(fit):
    # Codes at both ends of the range would change every fit if they were counted.
    values = pd.concat([TREE_VALUES, pd.Series([-1, -1, -9, 999, np.nan], index=range(20, 25))])
    target = pd.concat([TREE_TARGET, pd.Series([0, 0, 1, 0, 1], index=range(20, 25))])
    before = values.copy(), target.copy()
    expected = fit(TREE_VALUES, TREE_TARGET)
    for special in ([-1, -9, 999], (999.0, -1.0, -9.0), np.array([-9, -1, 999, -1]), pd.Index([-1, -9, 999])):
        np.testing.assert_equal(fit(values, target, special=special), expected)
    assert not np.array_equal(fit(values, target), expected)
    np.testing.assert_equal(fit(values.tolist(), target.to_numpy(), special=[-1, -9, 999]), expected)
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
    np.testing.assert_equal(chimerge_edges(values, target, alpha=0.1, special=[-1]),
                            [-np.inf, 1, 2, np.inf])
    np.testing.assert_equal(chimerge_edges(values, target, alpha=0.05, special=[-1]),
                            [-np.inf, 1, np.inf])
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
    np.testing.assert_equal(tree_edges(values, target, min_bin_share=0.5, special=[-1]),
                            [-np.inf, 6, np.inf])
    np.testing.assert_equal(tree_edges(values, target, max_bins=2, special=[-1]), [-np.inf, 4, np.inf])


def test_monotonic_trend_excludes_the_special_group():
    # The code 999 has event rate 0.1. Counted as the top value it breaks the rising trend
    # and pools everything above 1; left out, the bins match the MONO example.
    values, target = repeated([(1, 9, 1), (2, 7, 3), (3, 8, 2), (4, 6, 4), (999, 9, 1)])
    np.testing.assert_equal(monotonic_edges(values, target, max_bins=None, direction="increasing"),
                            [-np.inf, 1, np.inf])
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
    monotonic_edges: [-np.inf, 97, 105, 188, 211.2, np.inf],
}


@pytest.mark.parametrize("fit", [chimerge_edges, tree_edges, monotonic_edges])
def test_bank_pdays_code_is_fitted_separately(reference, fit):
    pdays, target = reference["pdays"], reference["target"]
    is_special = pdays == -1
    assert is_special.sum() == 2475
    assert target[~is_special].value_counts().sort_index().tolist() == [405, 120]
    edges = fit(pdays, target, special=[-1])
    np.testing.assert_allclose(edges, PDAYS_EDGES[fit])
    np.testing.assert_equal(edges, fit(pdays[~is_special], target[~is_special]))
    contacted = pdays[~is_special]
    if fit is tree_edges:
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


@pytest.mark.parametrize("fit", [chimerge_edges, tree_edges, monotonic_edges])
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
    pairs = zip(table.non_event, table.event, table.non_event[1:], table.event[1:])
    for statistic in (chi2_2x2(*cells) for cells in pairs):
        assert math.erfc(math.sqrt(statistic / 2)) < 0.01


@pytest.mark.parametrize("criterion", ["entropy", "gini"])
@pytest.mark.parametrize("column", ["age", "balance", "duration", "day"])
def test_tree_on_bank_data_respects_the_minimum_share(reference, column, criterion):
    edges = tree_edges(reference[column], reference["target"], max_bins=6, min_bin_share=0.1,
                       criterion=criterion)
    counts = pd.cut(reference[column], edges).value_counts()
    assert counts.min() >= 0.1 * len(reference)
    assert set(edges[1:-1]) <= set(reference[column])


@pytest.mark.parametrize("column", ["age", "balance", "duration", "campaign", "day", "previous"])
def test_monotonic_on_bank_data_gives_a_strict_woe_trend(reference, column):
    edges = monotonic_edges(reference[column], reference["target"], max_bins=None)
    woe = woe_table(pd.cut(reference[column], edges), reference["target"]).sort_index().woe
    steps = np.diff(woe.to_numpy())
    assert (steps > 0).all() or (steps < 0).all() or len(woe) == 1


def test_balance_woe_rises_under_every_method(reference):
    balance, target = reference["balance"], reference["target"]
    for edges in (chimerge_edges(balance, target), monotonic_edges(balance, target)):
        woe = woe_table(pd.cut(balance, edges), target).sort_index().woe
        assert woe.is_monotonic_increasing and len(woe) > 1
    assert len(tree_edges(balance, target)) - 1 == 5


@pytest.mark.parametrize("fit", [chimerge_edges, tree_edges, monotonic_edges])
def test_array_like_inputs_are_matched_by_position(fit):
    expected = fit(TREE_VALUES, TREE_TARGET)
    np.testing.assert_equal(fit(TREE_VALUES.tolist(), TREE_TARGET.to_numpy()), expected)
    np.testing.assert_equal(fit(TREE_VALUES, TREE_TARGET.tolist()), expected)
    shifted = TREE_TARGET.set_axis(TREE_TARGET.index + 100)
    np.testing.assert_equal(fit(TREE_VALUES.to_numpy(), shifted), expected)
    with_missing = [*TREE_VALUES, None]
    np.testing.assert_equal(fit(with_missing, [*TREE_TARGET, 1]), expected)


@pytest.mark.parametrize("fit", [chimerge_edges, tree_edges, monotonic_edges])
@pytest.mark.parametrize("values,target,message", [
    (pd.Series([], dtype=float), pd.Series([], dtype=int), "finite observations"),
    (pd.Series(["a", "b"]), pd.Series([0, 1]), "Unable to parse string"),
    (pd.Series([1.0, np.inf]), pd.Series([0, 1]), "finite observations"),
    (pd.Series([1, 2]), pd.Series([0, 2]), "only 0 and 1"),
    (pd.Series([1, 2]), pd.Series([0, None]), "only 0 and 1"),
    (pd.Series([1, 2]), pd.Series([1, 1]), "both 0 and 1"),
    (pd.Series([1, 2, None]), pd.Series([0, 0, 1]), "both 0 and 1 where values are observed"),
    (pd.Series([1, 2]), pd.Series([0, 1], index=[1, 0]), "identical indices"),
    ([1, 2, 3], [0, 1], "same length, got 3 and 2"),
    (np.ones((2, 2)), [0, 1], "values must be one-dimensional, got 2 dimensions"),
    ([1, 2], np.ones((2, 2)), "target must be one-dimensional, got 2 dimensions"),
])
def test_invalid_supervised_inputs(fit, values, target, message):
    with pytest.raises(ValueError, match=message):
        fit(values, target)


@pytest.mark.parametrize("fit", [chimerge_edges, tree_edges, monotonic_edges])
@pytest.mark.parametrize("values,target,special,message", [
    ([1, 2], [0, 1], -1, "special must be one-dimensional, got 0 dimensions"),
    ([1, 2], [0, 1], [-1, np.nan], "special must not contain missing values"),
    ([-1, -1, None], [0, 1, 1], [-1], "no finite observations apart from missing and special values"),
    ([1, 2, np.inf], [0, 1, 1], [-1], "finite observations only; pass infinite codes in special"),
    ([1, 2, -1], [0, 0, 1], [-1], "both 0 and 1 where values are observed and not special"),
    (["N/A", 1, 2], [0, 0, 1], [-1], "Unable to parse string"),
    ([1, 2, -1], [0, 1, 2], [-1], "only 0 and 1"),
])
def test_invalid_special_values(fit, values, target, special, message):
    with pytest.raises(ValueError, match=message):
        fit(values, target, special=special)


@pytest.mark.parametrize("fit,options,message", [
    (chimerge_edges, {"max_bins": 0}, "max_bins must be a positive integer, got 0"),
    (chimerge_edges, {"max_bins": True}, "max_bins must be a positive integer, got True"),
    (chimerge_edges, {"initial_bins": 2.5}, "initial_bins must be a positive integer, got 2.5"),
    (chimerge_edges, {"alpha": 0}, "alpha must be None or between 0 and 1, got 0"),
    (chimerge_edges, {"alpha": 1.5}, "alpha must be None or between 0 and 1, got 1.5"),
    (tree_edges, {"max_bins": 0}, "max_bins must be a positive integer, got 0"),
    (tree_edges, {"min_bin_share": -0.1}, "min_bin_share must be between 0 and 0.5, got -0.1"),
    (tree_edges, {"min_bin_share": 0.6}, "min_bin_share must be between 0 and 0.5, got 0.6"),
    (tree_edges, {"criterion": "log_loss"}, "'entropy' or 'gini', got 'log_loss'"),
    (monotonic_edges, {"max_bins": 0}, "max_bins must be a positive integer, got 0"),
    (monotonic_edges, {"initial_bins": 0}, "initial_bins must be a positive integer, got 0"),
    (monotonic_edges, {"direction": "up"}, "'increasing' or 'decreasing', got 'up'"),
    (monotonic_edges, {"smoothing": 0}, "smoothing must be finite and positive, got 0"),
])
def test_invalid_supervised_options(fit, options, message):
    with pytest.raises(ValueError, match=message):
        fit(TREE_VALUES, TREE_TARGET, **options)
