import math

import numpy as np
import pandas as pd
import pytest

from kivi.binning import (
    _merge_categories,
    _odds,
    _pair_chi2,
    category_groups,
    chimerge_groups,
    group_categories,
)
from kivi.datasets import load_bank
from kivi.woe import woe_table, woe_transform


def repeated(groups, dtype=None):
    """Labels and targets from (category, non_event count, event count) triples."""
    values, target = [], []
    for value, non_event, event in groups:
        values += [value] * (non_event + event)
        target += [0] * non_event + [1] * event
    return pd.Series(values, dtype=dtype), pd.Series(target)


def chi2_2x2(a, b, c, d):
    """Pearson chi-square of the table [[a, b], [c, d]] without continuity correction."""
    n = a + b + c + d
    return n * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))


def p_value(statistic):
    return math.erfc(math.sqrt(statistic / 2))


@pytest.fixture(scope="module")
def reference():
    return load_bank().iloc[:3000]


# Event rates 0.2, 0.4, 0.8 and 0.5, ten observations each; sorted by WOE: A, B, D, C.
RATES = [("A", 8, 2), ("B", 6, 4), ("C", 2, 8), ("D", 5, 5)]
VALUES, TARGET = repeated(RATES)
# Rare categories: a 4, b 3, c 1 and d 1 of the 9 observed values.
RARE = pd.Series(["a", "b", "a", "c", "b", "a", None, "d", "b", "a"])


# category_groups ----------------------------------------------------------------------------


def test_rare_categories_are_pooled_by_share():
    assert category_groups(RARE, min_share=0.2) == {"a": "a", "b": "b", "c": "other", "d": "other"}
    # b has 3 / 9 = 0.333 of the values, so it is rare only above that share.
    assert category_groups(RARE, min_share=0.34)["b"] == "other"
    assert category_groups(RARE, min_share=0.0) == {"a": "a", "b": "b", "c": "c", "d": "d"}
    assert set(category_groups(RARE, min_share=1.0).values()) == {"other"}
    # The default pools categories below 5%; with 9 values every category has more.
    assert category_groups(RARE) == {"a": "a", "b": "b", "c": "c", "d": "d"}


def test_a_share_equal_to_min_share_is_not_rare():
    values = ["a", "a", "b", "c"]  # Shares 0.5, 0.25 and 0.25.
    assert category_groups(values, min_share=0.25) == {"a": "a", "b": "b", "c": "c"}
    assert category_groups(values, min_count=1) == {"a": "a", "b": "b", "c": "c"}
    assert category_groups(values, min_count=1.5)["b"] == "other"


def test_rare_categories_are_pooled_by_count():
    groups = category_groups(RARE, min_share=0, min_count=3)
    assert groups == {"a": "a", "b": "b", "c": "other", "d": "other"}
    assert category_groups(RARE, min_share=0, min_count=4)["b"] == "other"
    # A category is rare when it is below either limit.
    assert category_groups(RARE, min_share=0.4, min_count=2)["b"] == "other"


def test_rare_categories_come_last_and_keep_their_order():
    values = pd.Series(["c", "a", "b", "a", "a", "d", "a", "b"])
    assert list(category_groups(values, min_share=0.2).items()) == [
        ("a", "a"),
        ("b", "b"),
        ("c", "other"),
        ("d", "other"),
    ]
    ordered = values.astype(pd.CategoricalDtype(["d", "b", "a", "c", "unused"]))
    assert list(category_groups(ordered, min_share=0.2)) == ["b", "a", "d", "c"]


def test_rare_label_can_be_chosen():
    assert category_groups(RARE, min_share=0.2, rare_label="rare")["c"] == "rare"
    assert category_groups(RARE, min_share=0.2, rare_label=-1)["d"] == -1
    # The label may be one of the rare categories, which then keeps its name.
    assert category_groups(RARE, min_share=0.2, rare_label="c")["d"] == "c"
    # Without rare categories the label is not used, so it may be any label.
    assert category_groups(RARE, min_share=0, rare_label="a")["a"] == "a"
    with pytest.raises(ValueError, match="rare_label 'a' is the label of a category that is not"):
        category_groups(RARE, min_share=0.2, rare_label="a")
    with pytest.raises(ValueError, match="rare_label 'x' is the label of .* or of a special code"):
        category_groups(RARE, min_share=0.2, special=["x"], rare_label="x")


def test_special_codes_keep_their_own_group_and_are_left_out_of_the_shares():
    # Without d, the shares are a 4 / 8, b 3 / 8 and c 1 / 8.
    groups = category_groups(RARE, min_share=0.13, special=["d", "unknown"])
    assert list(groups.items()) == [
        ("d", "d"),
        ("unknown", "unknown"),
        ("a", "a"),
        ("b", "b"),
        ("c", "other"),
    ]
    assert category_groups(RARE, min_share=0.12, special=["d"])["c"] == "c"


def test_weights_are_counted_and_integer_weights_repeat_rows():
    weights = np.array([1, 2, 1, 3, 0, 1, 5, 2, 1, 1])
    # Weighted counts: a 4, b 3 (one b row has zero weight), c 3 and d 2, of 12.
    groups = category_groups(RARE, min_share=0.2, weights=weights)
    assert groups == {"a": "a", "b": "b", "c": "c", "d": "other"}
    repeated_rows = RARE.repeat(weights).reset_index(drop=True)
    for kwargs in ({"min_share": 0.2}, {"min_share": 0, "min_count": 3}, {"min_share": 0.3}):
        assert category_groups(RARE, weights=weights, **kwargs) == category_groups(
            repeated_rows, **kwargs
        )
    # A category whose rows all have zero weight is not in the grouping, and rows with zero
    # weight do not change the order of the categories.
    assert "d" not in category_groups(RARE, weights=np.where(RARE == "d", 0, 1))
    assert list(category_groups(["b", "a", "b"], min_share=0, weights=[0, 1, 1])) == ["a", "b"]


def test_numeric_codes_never_match_booleans():
    # True == 1, but a boolean is not a code, as in cut.
    values = pd.Series([True, False, True, True])
    with pytest.raises(ValueError, match="special must not match booleans, got 1, which equals"):
        category_groups(values, min_share=0, special=[1])
    with pytest.raises(ValueError, match="got 0.0, which equals the value False"):
        chimerge_groups(values, [1, 0, 0, 1], special=[0.0])
    assert category_groups(values, min_share=0, special=["x"]) == {
        "x": "x",
        True: True,
        False: False,
    }


def test_labels_are_python_scalars_and_inputs_are_kept():
    values = np.array([3, 1, 3, 3, 2])
    groups = category_groups(values, min_share=0.3)
    assert groups == {3: 3, 1: "other", 2: "other"}
    assert all(type(category) is int for category in groups)
    series = RARE.copy()
    category_groups(series, min_share=0.2)
    pd.testing.assert_series_equal(series, RARE)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"values": [["a"]]}, "values must be one-dimensional, got 2 dimensions"),
        ({"values": [None, np.nan]}, "values have no categories to fit apart from missing values"),
        (
            {"values": ["a", None], "special": ["a"]},
            "values have no categories to fit apart from missing and special values",
        ),
        (
            {"values": ["a", "b"], "special": ["a"], "weights": [1, 0]},
            "apart from missing, special and zero-weight values",
        ),
        ({"min_share": -0.1}, "min_share must be between 0 and 1, got -0.1"),
        ({"min_share": 1.5}, "min_share must be between 0 and 1, got 1.5"),
        ({"min_share": True}, "min_share must be between 0 and 1, got True"),
        ({"min_share": "0.1"}, "min_share must be between 0 and 1, got '0.1'"),
        ({"min_count": -1}, "min_count must be a finite, non-negative number, got -1"),
        ({"min_count": np.inf}, "min_count must be a finite, non-negative number, got inf"),
        ({"min_count": False}, "min_count must be a finite, non-negative number, got False"),
        ({"rare_label": None}, "rare_label must be a string or a number, got None"),
        ({"rare_label": np.nan}, "rare_label must be a string or a number, got nan"),
        ({"rare_label": True}, "rare_label must be a string or a number, got True"),
        ({"rare_label": ["x"]}, r"rare_label must be a string or a number, got \['x'\]"),
        ({"special": [None]}, "special must not contain missing values"),
        ({"special": [True]}, "special must not contain booleans, got True"),
        ({"min_share": np.float64(2)}, r"min_share must be between 0 and 1, got 2.0$"),
        ({"rare_label": np.array([1])}, r"rare_label must be a string or a number, got"),
        ({"weights": [1, 2]}, "weights must have one weight per value of values, got 2 for 10"),
        ({"weights": [-1] * 10}, "weights must be finite and non-negative"),
    ],
)
def test_invalid_rare_grouping(kwargs, message):
    kwargs = {"values": RARE, **kwargs}
    with pytest.raises(ValueError, match=message):
        category_groups(**kwargs)


# chimerge_groups ----------------------------------------------------------------------------


def test_chi_square_of_the_sorted_categories_matches_hand_calculation():
    counts = np.array([[8, 2], [6, 4], [5, 5], [2, 8]])  # A, B, D and C, in WOE order.
    expected = [chi2_2x2(8, 2, 6, 4), chi2_2x2(6, 4, 5, 5), chi2_2x2(5, 5, 2, 8)]
    np.testing.assert_allclose(_pair_chi2(counts), expected)
    np.testing.assert_allclose(expected, [20 / 21, 20 / 99, 180 / 91])


def test_chimerge_merges_the_most_similar_neighbours_in_woe_order():
    # B and D have the smallest chi-square, 20 / 99.
    assert chimerge_groups(VALUES, TARGET, max_groups=3, alpha=None) == {
        "A": "A",
        "B": "B, D",
        "D": "B, D",
        "C": "C",
    }
    # Then A and (B, D): 30 * 50**2 / (10 * 20 * 19 * 11) = 1.794, below 3.326 for (B, D) and C.
    assert chi2_2x2(8, 2, 11, 9) == pytest.approx(1.79426, abs=1e-5)
    assert chi2_2x2(11, 9, 2, 8) == pytest.approx(3.32579, abs=1e-5)
    two = {"A": "A, B, D", "B": "A, B, D", "D": "A, B, D", "C": "C"}
    assert chimerge_groups(VALUES, TARGET, max_groups=2, alpha=None) == two
    assert set(chimerge_groups(VALUES, TARGET, max_groups=1, alpha=None).values()) == {"A, B, D, C"}
    assert chimerge_groups(VALUES, TARGET, max_groups=4, alpha=None) == {
        "A": "A",
        "B": "B",
        "D": "D",
        "C": "C",
    }


def test_chimerge_merges_until_every_pair_is_significant():
    # p = 0.180 for A and (B, D), and 0.0175 for (A, B, D) and C.
    assert p_value(chi2_2x2(8, 2, 11, 9)) == pytest.approx(0.1805, abs=1e-4)
    assert p_value(chi2_2x2(19, 11, 2, 8)) == pytest.approx(0.0175, abs=1e-4)
    assert chimerge_groups(VALUES, TARGET, max_groups=5) == {
        "A": "A, B, D",
        "B": "A, B, D",
        "D": "A, B, D",
        "C": "C",
    }
    assert chimerge_groups(VALUES, TARGET, max_groups=5, alpha=0.19)["A"] == "A"
    assert chimerge_groups(VALUES, TARGET, max_groups=5, alpha=0.01)["C"] == "A, B, D, C"


def test_small_groups_are_merged_first_with_the_more_similar_neighbour():
    # E (1, 1) ties with D in WOE and follows it; its share 2 / 42 is below 0.05.
    values, target = repeated([*RATES, ("E", 1, 1)])
    groups = chimerge_groups(values, target, alpha=None, min_group_share=0.05)
    assert groups == {"A": "A", "B": "B", "D": "D, E", "E": "D, E", "C": "C"}
    assert chimerge_groups(values, target, alpha=None, min_group_share=0.04)["E"] == "E"
    # G (3, 0) has the lowest WOE and can only join A; F (0, 3) the highest and only C.
    values, target = repeated([*RATES, ("F", 0, 3), ("G", 3, 0)])
    groups = chimerge_groups(values, target, alpha=None, min_group_share=0.08)
    assert groups == {"G": "G, A", "A": "G, A", "B": "B", "D": "D", "C": "C, F", "F": "C, F"}
    # Only groups below the share are merged: G (5, 0) has 5 / 48 of the values.
    values, target = repeated([*RATES, ("F", 0, 3), ("G", 5, 0)])
    groups = chimerge_groups(values, target, alpha=None, min_group_share=0.08)
    assert groups["G"] == "G" and groups["F"] == "C, F"
    # H (1, 3) sits between B and D in WOE and is closer to D.
    values, target = repeated([("A", 8, 2), ("B", 4, 1), ("H", 1, 3), ("D", 1, 4)])
    assert chi2_2x2(1, 3, 1, 4) < chi2_2x2(4, 1, 1, 3)
    groups = chimerge_groups(values, target, alpha=None, min_group_share=0.2)
    assert groups == {"A": "A", "B": "B", "H": "H, D", "D": "H, D"}
    # With equally similar neighbours, the group joins the one before it: P ties with D and Q.
    values, target = repeated([*RATES, ("P", 1, 1), ("Q", 2, 2)])
    groups = chimerge_groups(values, target, alpha=None, min_group_share=0.1)
    assert groups["P"] == groups["Q"] == "D, P, Q"


def test_groups_follow_the_smoothed_woe():
    # X and Y have no events, so both get 0.5 events; Y, with more non-events, has the lower
    # WOE although both event rates are 0.
    values, target = repeated([("X", 30, 0), ("Y", 60, 0), ("Z", 20, 20)])
    assert list(chimerge_groups(values, target, max_groups=3, alpha=None, min_group_share=0)) == [
        "Y",
        "X",
        "Z",
    ]
    for seed in range(5):
        rng = np.random.default_rng(seed)
        labels = pd.Series(rng.choice(list("abcdefghij"), size=400))
        rate = labels.map(dict(zip("abcdefghij", rng.uniform(0, 0.5, 10), strict=True)))
        target = pd.Series((rng.uniform(size=400) < rate).astype(int))
        for smoothing in (0.5, 2.0):
            groups = chimerge_groups(labels, target, smoothing=smoothing, alpha=None)
            table = woe_table(group_categories(labels, groups), target, smoothing=smoothing)
            assert table.index.tolist() == list(dict.fromkeys(groups.values()))
            assert table["woe"].is_monotonic_increasing
            assert len(table) <= 5 and table["total_share"].min() >= 0.05


def test_ties_go_to_the_first_category_despite_rounding():
    # x, y and z have proportional counts, so the same odds 1; their WOE differs in the
    # last bit, which must not reorder them.
    values, target = repeated([("x", 3, 3), ("y", 1, 1), ("z", 7, 7), ("w", 10, 2)])
    groups = chimerge_groups(values, target, max_groups=4, alpha=None, min_group_share=0)
    assert list(groups) == ["w", "x", "y", "z"]
    # x-y and y-z both have a chi-square of 0, so the first pair merges.
    groups = chimerge_groups(values, target, max_groups=3, alpha=None, min_group_share=0)
    assert groups == {"w": "w", "x": "x, y", "y": "x, y", "z": "z"}
    # The pairs (a, b) and (b, c) mirror each other: equal chi-square, unequal floats.
    counts = np.array([[5, 2], [6, 6], [2, 5]])
    statistic = _pair_chi2(counts)
    assert statistic[0] == pytest.approx(statistic[1], rel=1e-12)
    values, target = repeated([("a", 5, 2), ("b", 6, 6), ("c", 2, 5)])
    groups = chimerge_groups(values, target, max_groups=2, alpha=None, min_group_share=0)
    assert groups == {"a": "a, b", "b": "a, b", "c": "c"}
    # A small group between two equally similar neighbours joins the one before it.
    values, target = repeated([("a", 50, 20), ("b", 3, 3), ("c", 20, 50)])
    groups = chimerge_groups(values, target, max_groups=3, alpha=None, min_group_share=0.1)
    assert groups["b"] == "a, b"


def _reference_merge(counts, max_groups, alpha, min_share, smoothing):
    """The merging rule computed from scratch after every merge, to check the incremental one."""

    def odds(rows):
        adjusted = np.where(rows == 0, smoothing, rows)
        return adjusted[:, 1] / adjusted[:, 0]

    def first_min(values):
        return int(np.flatnonzero(values <= values.min() * (1 + 1e-10))[0])

    counts = counts.astype(float)
    order = np.argsort(odds(counts), kind="stable")
    counts, members = counts[order], [[row] for row in order]
    while len(counts) > 1:
        totals = counts.sum(axis=1)
        statistic = _pair_chi2(counts)
        small = totals / totals.sum() < min_share
        if small.any():
            group = int(np.flatnonzero(small)[first_min(totals[small])])
            if group == 0:
                pair = 0
            elif group == len(counts) - 1:
                pair = group - 1
            else:
                left, right = statistic[group - 1], statistic[group]
                pair = group if right < left * (1 - 1e-10) else group - 1
        else:
            pair = first_min(statistic)
            if len(counts) <= max_groups and (
                alpha is None or math.erfc(math.sqrt(statistic[pair] / 2)) < alpha
            ):
                break
        counts[pair] += counts[pair + 1]
        counts = np.delete(counts, pair + 1, axis=0)
        members[pair : pair + 2] = [members[pair] + members[pair + 1]]
        order = np.argsort(odds(counts), kind="stable")
        counts, members = counts[order], [members[row] for row in order]
    return [sorted(group) for group in members]


@pytest.mark.parametrize("seed", range(40))
def test_incremental_merging_matches_a_full_recomputation(seed):
    rng = np.random.default_rng(seed)
    size = int(rng.integers(2, 40))
    counts = rng.integers(0, 12, size=(size, 2)) * rng.integers(1, 4, size=(size, 1))
    counts[counts.sum(axis=1) == 0] = [1, 0]
    if seed % 3 == 0:
        counts = counts * rng.uniform(0.5, 2.0, size=(size, 1))
    kwargs = {
        "max_groups": int(rng.integers(1, 8)),
        "alpha": [None, 0.05, 0.5][seed % 3],
        "min_share": [0, 0.02, 0.1][seed % 3 - 1],
        "smoothing": [0.5, 2.0][seed % 2],
    }
    expected = _reference_merge(counts, **kwargs)
    order = np.argsort(_odds(counts, kwargs["smoothing"]), kind="stable")
    merged = _merge_categories(counts[order], *kwargs.values())
    assert [sorted(int(order[row]) for row in group) for group in merged] == expected


def test_long_groups_are_labelled_with_their_first_categories():
    values, target = repeated([(name, 5, 5) for name in "abcdefg"])
    groups = chimerge_groups(values, target, max_groups=1, min_group_share=0)
    assert set(groups.values()) == {"a, b, c, d, ... (+3)"}
    values, target = repeated([(name, 5, 5) for name in "abcde"])
    groups = chimerge_groups(values, target, max_groups=1, min_group_share=0)
    assert set(groups.values()) == {"a, b, c, d, e"}


def test_group_labels_join_the_categories_in_woe_order():
    values, target = repeated([(3, 5, 5), (1, 9, 1), (2, 5, 5)])
    groups = chimerge_groups(values, target, max_groups=2, alpha=None, min_group_share=0)
    assert groups == {1: 1, 3: "3, 2", 2: "3, 2"}
    assert type(next(iter(groups))) is int
    # A merged label that is already the label of another group raises an error.
    values, target = repeated([("a, b", 9, 1), ("a", 5, 5), ("b", 5, 5)])
    with pytest.raises(ValueError, match="two groups would have the label 'a, b'; rename"):
        chimerge_groups(values, target, max_groups=2, alpha=None, min_group_share=0)


def test_chimerge_special_codes_are_left_out_with_their_labels():
    values, target = repeated([*RATES, ("S", 0, 30)])
    groups = chimerge_groups(values, target, max_groups=2, alpha=None, special=["S", "T"])
    assert list(groups.items())[:2] == [("S", "S"), ("T", "T")]
    assert {key: groups[key] for key in "ABCD"} == chimerge_groups(
        VALUES, TARGET, max_groups=2, alpha=None
    )
    # With special codes, the target must have both classes in the remaining rows.
    with pytest.raises(ValueError, match="where values are observed and not special$"):
        chimerge_groups(*repeated([("A", 3, 0), ("S", 0, 3)]), special=["S"])


def test_chimerge_missing_values_are_omitted_and_inputs_are_kept():
    values = pd.concat([VALUES, pd.Series([None, None])], ignore_index=True)
    target = pd.concat([TARGET, pd.Series([1, 1])], ignore_index=True)
    kept = values.copy()
    assert chimerge_groups(values, target, max_groups=3, alpha=None) == chimerge_groups(
        VALUES, TARGET, max_groups=3, alpha=None
    )
    pd.testing.assert_series_equal(values, kept)
    # Lists and arrays are matched by position.
    assert chimerge_groups(VALUES.tolist(), TARGET.to_numpy(), max_groups=3) == chimerge_groups(
        VALUES, TARGET, max_groups=3
    )


def test_chimerge_integer_weights_repeat_rows():
    rng = np.random.default_rng(7)
    labels = pd.Series(rng.choice(list("abcdefg"), size=120))
    target = pd.Series(rng.integers(0, 2, size=120))
    weights = rng.integers(0, 4, size=120)
    repeated_labels = labels.repeat(weights).reset_index(drop=True)
    repeated_target = target.repeat(weights).reset_index(drop=True)
    for kwargs in ({}, {"alpha": None}, {"max_groups": 3, "min_group_share": 0.15}):
        weighted = chimerge_groups(labels, target, weights=weights, **kwargs)
        assert list(weighted.items()) == list(
            chimerge_groups(repeated_labels, repeated_target, **kwargs).items()
        )
    with pytest.raises(ValueError, match="where values are observed with a positive weight$"):
        chimerge_groups(VALUES, TARGET, weights=TARGET)


def test_bank_categories_merge_into_groups_with_increasing_woe(reference):
    target = reference["target"]
    for column in ("job", "education", "marital", "month", "contact", "poutcome"):
        groups = chimerge_groups(reference[column], target)
        table = woe_table(group_categories(reference[column], groups), target)
        assert len(table) <= 5 and table["total_share"].min() >= 0.05
        assert table["woe"].is_monotonic_increasing
        assert table["total"].sum() == len(reference)
    groups = chimerge_groups(reference["job"], target)
    assert groups["retired"] == groups["student"] == "retired, student"


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"max_groups": 0}, "max_groups must be a positive integer, got 0"),
        ({"max_groups": 2.0}, "max_groups must be a positive integer, got 2.0"),
        ({"alpha": 1}, "alpha must be None or between 0 and 1, got 1"),
        ({"min_group_share": 0.6}, "min_group_share must be between 0 and 0.5, got 0.6"),
        ({"min_group_share": None}, "min_group_share must be between 0 and 0.5, got None"),
        ({"smoothing": 0}, "smoothing must be finite and positive, got 0"),
        ({"smoothing": True}, "smoothing must be finite and positive, got True"),
        ({"target": TARGET[:5]}, "values and target must have identical indices"),
        ({"target": TARGET.to_numpy()[:5]}, "values and target must have the same length"),
        ({"target": TARGET.replace(1, 2)}, "target must contain only 0 and 1"),
        ({"target": TARGET.where(TARGET > 0)}, "target must contain only 0 and 1"),
        ({"target": TARGET * 0}, "target must contain both 0 and 1 where values are observed$"),
        ({"special": [np.nan]}, "special must not contain missing values"),
        ({"weights": TARGET[:5]}, "values and weights must have identical indices"),
    ],
)
def test_invalid_chimerge_grouping(kwargs, message):
    kwargs = {"values": VALUES, "target": TARGET, **kwargs}
    with pytest.raises(ValueError, match=message):
        chimerge_groups(**kwargs)


# group_categories ---------------------------------------------------------------------------

GROUPS = {"a": "a", "b": "b, c", "c": "b, c"}


def test_values_are_replaced_by_their_groups():
    values = pd.Series(["c", "a", None, "b"], index=[5, 6, 7, 8], name="x")
    grouped = group_categories(values, GROUPS)
    assert isinstance(grouped.dtype, pd.CategoricalDtype)
    assert grouped.index.equals(values.index) and grouped.name == "x"
    assert grouped.cat.categories.tolist() == ["a", "b, c"]
    assert grouped.cat.codes.tolist() == [1, 0, -1, 1]
    array = group_categories(["c", "a"], GROUPS)
    assert isinstance(array, pd.Categorical) and array.tolist() == ["b, c", "a"]
    assert group_categories(np.array(["a", "a"]), GROUPS).categories.tolist() == ["a", "b, c"]


def test_the_categories_follow_the_first_appearance_of_each_group():
    groups = {"z": 2, "y": 1, "x": 2, "w": "w"}
    grouped = group_categories(["w", "x", "y"], groups)
    assert grouped.categories.tolist() == [2, 1, "w"]
    assert grouped.codes.tolist() == [2, 0, 1]


def test_categories_are_matched_by_equality():
    groups = {1: "one", 2: "two"}
    assert group_categories([1.0, 2, np.int64(1)], groups).tolist() == ["one", "two", "one"]
    with pytest.raises(ValueError, match="such as '1'"):
        group_categories(["1"], groups)
    categorical = pd.Series(["b", "a"], dtype=pd.CategoricalDtype(["a", "b", "c"]))
    assert group_categories(categorical, {"a": 0, "b": 1}).tolist() == [1, 0]


def test_unseen_categories_raise_or_take_the_named_group():
    with pytest.raises(
        ValueError,
        match=r"values have 2 categories that groups does not map, such as 'e', 'd'; pass unseen",
    ):
        group_categories(["e", "a", "d", "e"], GROUPS)
    with pytest.raises(ValueError, match="values have 1 category that groups does not map"):
        group_categories(["d"], GROUPS)
    grouped = group_categories(["e", "a", None, "d"], GROUPS, unseen="b, c")
    assert grouped.tolist()[:2] == ["b, c", "a"] and pd.isna(grouped[2]) and grouped[3] == "b, c"
    rare = category_groups(RARE, min_share=0.2)
    assert group_categories(["new", "a"], rare, unseen="other").tolist() == ["other", "a"]


@pytest.mark.parametrize("unseen", ["d", None, np.nan, ["a"], "Raise"])
def test_unseen_must_be_raise_or_a_group(unseen):
    with pytest.raises(ValueError, match="unseen must be 'raise' or one of the groups, got"):
        group_categories(["a"], GROUPS, unseen=unseen)


def test_numpy_labels_become_python_labels():
    grouped = group_categories(np.array([1, 2]), {np.int64(1): np.str_("one"), 2: np.float64(2)})
    assert [type(label) for label in grouped.categories] == [str, float]
    assert category_groups(["a", "b"], min_share=0.6, rare_label=np.int64(0)) == {"a": 0, "b": 0}


def test_an_empty_grouping_only_accepts_missing_values():
    grouped = group_categories(pd.Series([None, np.nan]), {})
    assert grouped.isna().all() and grouped.cat.categories.empty
    with pytest.raises(ValueError, match="such as 'a'"):
        group_categories(["a"], {})


@pytest.mark.parametrize(
    "groups,message",
    [
        (pd.Series(GROUPS), "groups must be a mapping from category to group, got Series"),
        ([("a", "a")], "groups must be a mapping from category to group, got list"),
        ({None: "a"}, "got the category None; missing values always form their own group"),
        ({np.nan: "a"}, "got the category nan"),
        ({"a": None}, "got the group None"),
        ({"a": pd.NA}, "got the group <NA>"),
        ({"a": ["x"]}, r"non-missing scalar groups, got the group \['x'\]"),
        ({("a", "b"): "x"}, r"got the category \('a', 'b'\)"),
    ],
)
def test_invalid_groupings(groups, message):
    with pytest.raises(ValueError, match=message):
        group_categories(["a"], groups)


def test_groups_work_with_woe_table_and_woe_transform(reference):
    target = reference["target"]
    groups = chimerge_groups(reference["job"], target)
    table = woe_table(group_categories(reference["job"], groups), target)
    later = load_bank().iloc[3000:]
    encoded = woe_transform(group_categories(later["job"], groups), table)
    by_group = table["woe"].to_dict()
    expected = later["job"].map(lambda job: by_group[groups[job]])
    np.testing.assert_allclose(encoded, expected)
    # Grouping and binning the counts agree: the table counts every group of the grouping.
    totals = reference["job"].map(groups).value_counts()
    assert table["total"].to_dict() == totals.to_dict()


def test_the_mapping_and_the_values_are_not_changed():
    groups = dict(GROUPS)
    values = pd.Series(["a", "c"])
    group_categories(values, groups, unseen="a")
    assert groups == GROUPS and values.tolist() == ["a", "c"]
