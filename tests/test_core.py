import importlib
import importlib.metadata
import inspect
import re

import numpy as np
import pandas as pd
import pytest

import kivi
from kivi.binning import bin_edges, cut
from kivi.datasets import load_bank
from kivi.spec import BinningSpec, fit_spec
from kivi.stability import psi, psi_table
from kivi.woe import iv, woe, woe_table, woe_transform, woe_trend


def test_example_data():
    bank = load_bank()
    assert bank.shape == (4521, 17)
    assert bank.target.value_counts().to_dict() == {0: 4000, 1: 521}


@pytest.mark.parametrize("method", ["quantile", "uniform"])
def test_reference_edges_cover_future_values(method):
    edges = bin_edges(pd.Series([1, 2, None, 3]), bins=2, method=method)
    np.testing.assert_equal(edges, [-np.inf, 2.0, np.inf])
    assigned = pd.cut(pd.Series([-100, 2, 100, np.nan]), edges, labels=False)
    np.testing.assert_equal(assigned.to_numpy(), [0, 0, 1, np.nan])


def test_constant_reference_and_repeated_edges():
    with pytest.warns(UserWarning, match="leave 1 of the 5 requested bins"):
        np.testing.assert_equal(bin_edges(pd.Series([3, 3])), [-np.inf, np.inf])
    with pytest.warns(UserWarning, match="leave 3 of the 5 requested bins"):
        edges = bin_edges(pd.Series([0, 0, 0, 1, 2]), bins=5)
    np.testing.assert_allclose(edges, [-np.inf, 0.4, 1.2, np.inf])


def test_special_value_mass_is_reported():
    pdays = load_bank()["pdays"]
    with pytest.warns(UserWarning, match="value -1 makes up 82% of the fitted values"):
        np.testing.assert_equal(bin_edges(pdays), [-np.inf, np.inf])


@pytest.mark.parametrize(
    "values",
    [
        [1, 2, None, 3],
        (1.0, 2.0, np.nan, 3.0),
        np.array([1, 2, np.nan, 3]),
        pd.array([1, 2, None, 3], dtype="Int64"),
        pd.Series([1, 2, None, 3]),
    ],
)
def test_bin_edges_accepts_array_like(values):
    np.testing.assert_equal(bin_edges(values, bins=2), [-np.inf, 2.0, np.inf])


@pytest.mark.parametrize(
    "values,bins,method,message",
    [
        ([], 2, "quantile", "^values have no finite observations apart from missing and special"),
        ([None], 2, "quantile", "^values have no finite observations apart from missing and spec"),
        ([np.inf], 2, "quantile", "^values must be finite or special values, got inf; pass inf"),
        ([1], 0, "quantile", "positive integer, got 0"),
        ([1], True, "quantile", "positive integer, got True"),
        ([1], 2.0, "quantile", "positive integer, got 2.0"),
        ([1], 2, "median", "'quantile' or 'uniform', got 'median'"),
        (["a", "b"], 2, "quantile", "values must be numbers or special values, got 'a'"),
        (np.ones((2, 2)), 2, "quantile", "one-dimensional, got 2 dimensions"),
        (1.0, 2, "quantile", "one-dimensional, got 0 dimensions"),
    ],
)
def test_invalid_binning_inputs(values, bins, method, message):
    with pytest.raises(ValueError, match=message):
        bin_edges(values, bins=bins, method=method)


def test_special_values_are_left_out_of_the_fit():
    values = [-1, -1, -1, 10, 20, 30, 40, None]
    np.testing.assert_equal(bin_edges(values, bins=2, special=[-1]), [-np.inf, 25, np.inf])
    np.testing.assert_equal(
        bin_edges(values, bins=2, method="uniform", special=[-1, -1]), [-np.inf, 25, np.inf]
    )
    np.testing.assert_equal(
        bin_edges(["N/A", 1, 3, 5], bins=2, special=["N/A"]), [-np.inf, 3, np.inf]
    )
    np.testing.assert_equal(
        bin_edges([-np.inf, 1, 3, 5], bins=2, special=[-np.inf]), [-np.inf, 3, np.inf]
    )


def test_cut_gives_each_special_value_a_group():
    values = pd.Series([-1, 15, 35, None, -9, -1.0], index=list("abcdef"), name="x")
    edges = [-np.inf, 25, np.inf]
    groups = cut(values, edges, special=[-9, -1])
    assert groups.index.equals(values.index) and groups.name == "x"
    assert groups.cat.categories.tolist() == [
        -9,
        -1,
        pd.Interval(-np.inf, 25),
        pd.Interval(25, np.inf),
    ]
    np.testing.assert_equal(groups.cat.codes.to_numpy(), [1, 2, 3, -1, 0, 1])
    as_array = cut(values.tolist(), edges, special=[-9, -1])
    assert isinstance(as_array, pd.Categorical)
    np.testing.assert_equal(as_array.codes, groups.cat.codes.to_numpy())
    labels = cut(["N/A", 1, 30], edges, special=["N/A"])
    assert labels.categories[0] == "N/A" and labels.codes.tolist() == [0, 1, 2]


def test_cut_without_special_matches_pandas():
    values = pd.Series([1, 5, None, 9], index=[3, 2, 1, 0])
    edges = [-np.inf, 4, np.inf]
    pd.testing.assert_series_equal(cut(values, edges), pd.cut(values, edges))
    pd.testing.assert_extension_array_equal(
        cut(values.to_numpy(), edges), pd.cut(values.to_numpy(), edges)
    )


@pytest.mark.parametrize("infinite", [-np.inf, np.inf, "inf"])
def test_cut_rejects_infinite_values_that_are_not_special(infinite):
    # pandas.cut would leave -inf in no interval, where it counts as missing, and put inf in the
    # last interval.
    edges = bin_edges([1, 2, 3, 4], 2)
    expected = "-inf" if infinite == -np.inf else "inf"
    with pytest.raises(
        ValueError,
        match=f"values must be finite or special values, got {expected}; pass infinite codes",
    ):
        cut(pd.Series([1.0, infinite, None], dtype=object), edges)
    # A text "inf" matches the code inf after pandas.to_numeric, like other special values.
    groups = cut(pd.Series([infinite, 1.0, None], dtype=object), edges, special=[float(infinite)])
    np.testing.assert_equal(groups.cat.codes.to_numpy(), [0, 1, -1])
    assert groups.cat.categories[0] == float(infinite)


def test_cut_keeps_finite_values_outside_the_edges_missing():
    groups = cut([0.5, 1.5, 9.0], [1, 2, 3])
    np.testing.assert_equal(groups.codes, [-1, 0, -1])


def test_special_value_bins_on_bank_pdays():
    bank = load_bank()
    is_special = bank.pdays == -1
    edges = bin_edges(bank.pdays, bins=5, special=[-1])
    probabilities = np.linspace(0, 1, 6)[1:-1]
    np.testing.assert_equal(edges[1:-1], np.quantile(bank.pdays[~is_special], probabilities))
    result = woe_table(cut(bank.pdays, edges, special=[-1]), bank.target)
    special_counts = bank.target[is_special].value_counts().reindex([0, 1]).to_numpy()
    intervals = pd.cut(bank.pdays[~is_special], edges)
    # Cross-tabulate the codes: crosstab hashes interval categories, which warns on NumPy < 2.2.
    interval_counts = pd.crosstab(intervals.cat.codes, bank.target[~is_special])
    expected = np.vstack([special_counts, interval_counts.to_numpy()])
    observed = intervals.cat.categories[interval_counts.index].tolist()
    assert result.index[0] == -1 and result.index[1:].tolist() == observed
    np.testing.assert_equal(result[["non_event", "event"]].to_numpy(), expected)
    p, q = expected[:, 1] / expected[:, 1].sum(), expected[:, 0] / expected[:, 0].sum()
    assert result.iv.sum() == pytest.approx(((p - q) * np.log(p / q)).sum())
    # legacy FrequencyBins(bins=5, abnormal_vals=[-1]) reported an IV of 0.298906
    assert result.iv.sum() == pytest.approx(0.298906, abs=1e-6)


def test_numeric_strings_match_numeric_special_codes():
    # Columns read from a file as text: "-1" is the code -1, not the number -1.
    values = pd.Series(["-1", "15", "N/A", "35", None, "-1.0"])
    groups = cut(values, [-np.inf, 25, np.inf], special=[-1, "N/A"])
    assert groups.cat.categories[:2].tolist() == [-1, "N/A"]
    np.testing.assert_equal(groups.cat.codes.to_numpy(), [0, 2, 1, 3, -1, 0])
    numeric = pd.Series([-1, 15, -1, 35, None, -1.0])
    np.testing.assert_equal(
        bin_edges(values.mask(values == "N/A"), bins=2, special=[-1]),
        bin_edges(numeric, bins=2, special=[-1]),
    )
    np.testing.assert_equal(bin_edges(numeric, bins=2, special=[-1]), [-np.inf, 25, np.inf])


def test_special_codes_match_exactly():
    # Booleans are not the numbers 0 and 1 here, and integers are compared as integers.
    edges = [-np.inf, 0.5, np.inf]
    flags = cut(pd.Series([True, False]), edges, special=[0])
    np.testing.assert_equal(flags.cat.codes.to_numpy(), [2, 1])
    big = pd.Series([2**53 + 1, 2**53, 5], dtype=np.int64)
    groups = cut(big, [-np.inf, 10, np.inf], special=[float(2**53)])
    np.testing.assert_equal(groups.cat.codes.to_numpy(), [2, 0, 1])
    np.testing.assert_equal(
        cut(pd.Series([-1, 3], dtype="Int64"), edges, special=[-1.0]).cat.codes.to_numpy(), [0, 2]
    )


@pytest.mark.parametrize(
    "special,message",
    [
        ([np.nan], "must not contain missing values"),
        (-1, "special must be one-dimensional"),
        ([-1, True], "special must not contain booleans, got True"),
        ([1, True], "special must not contain booleans, got True"),
        ([0, False], "special must not contain booleans, got False"),
    ],
)
def test_invalid_special_values(special, message):
    with pytest.raises(ValueError, match=message):
        bin_edges([1, 2], special=special)
    with pytest.raises(ValueError, match=message):
        cut([1, 2], [-np.inf, np.inf], special=special)


@pytest.mark.parametrize(
    "values,edges,message",
    [
        (["a", 1], [-np.inf, np.inf], "values must be numbers or special values, got 'a'"),
        ([1], [3, 1], re.escape("edges must be increasing numbers, got [3, 1]")),
        ([1], [1, 1, 3], re.escape("edges must be increasing numbers, got [1, 1, 3]")),
        (
            [1],
            np.array([1.0, np.nan]),
            re.escape("edges must be increasing numbers, got [1.0, nan]"),
        ),
        ([1], ["a", "b"], re.escape("edges must be increasing numbers, got ['a', 'b']")),
        ([[1]], [-np.inf, np.inf], "values must be one-dimensional"),
    ],
)
def test_invalid_cut_inputs(values, edges, message):
    with pytest.raises(ValueError, match=message):
        cut(values, edges)


def test_cut_leaves_an_integer_number_of_bins_to_pandas():
    # cut documents edges only, but passes an integer on to pandas.cut, and with it the
    # message of pandas.cut when that number cannot be used.
    values = pd.Series([1.0, 3.0, 5.0])
    pd.testing.assert_series_equal(cut(values, 2), pd.cut(values, 2))
    with pytest.raises(ValueError, match="^`bins` should be a positive integer.$"):
        cut(values, 0)
    with pytest.raises(ValueError, match="^Cannot cut empty array$"):
        cut(pd.Series([], dtype=float), 3)


def test_all_special_reference_cannot_be_fitted():
    with pytest.raises(ValueError, match="no finite observations apart from missing and special"):
        bin_edges([-1, -1, None], special=[-1])


def test_woe_matches_hand_calculation():
    labels = pd.Series(["a", "a", "a", "b", "b", "b"])
    target = pd.Series([1, 1, 0, 1, 0, 0])
    result = woe_table(labels, target)
    np.testing.assert_allclose(result.woe, [np.log(2), -np.log(2)])
    assert result.iv.sum() == pytest.approx(2 * np.log(2) / 3)


def test_woe_counts_match_crosstab():
    bank = load_bank()
    result = woe_table(bank["job"], bank["target"])
    expected = pd.crosstab(bank["job"], bank["target"]).reindex(result.index)
    np.testing.assert_equal(result[["non_event", "event"]].to_numpy(), expected.to_numpy())
    assert (expected > 0).all(axis=None)
    p, q = expected[1] / expected[1].sum(), expected[0] / expected[0].sum()
    np.testing.assert_allclose(result.woe, np.log(p / q))
    assert result.iv.sum() == pytest.approx(((p - q) * np.log(p / q)).sum())


def test_woe_matches_arrays_by_position():
    groups = pd.Series(["a", "a", "a", "b", "b", "b"], index=[5, 4, 3, 2, 1, 0])
    target = [1, 1, 0, 1, 0, 0]
    expected = woe_table(pd.Series(groups.to_numpy()), pd.Series(target))
    pd.testing.assert_frame_equal(woe_table(groups, target), expected)
    pd.testing.assert_frame_equal(woe_table(groups.to_numpy(), np.array(target)), expected)
    intervals = pd.cut(np.array([1, 2, 3, 4]), [-np.inf, 2, np.inf])
    assert woe_table(intervals, [1, 0, 0, 1]).event.tolist() == [1, 1]


def test_categorical_groups_follow_category_order():
    groups = pd.cut(pd.Series([35, None, 5, 15, 35]), [-np.inf, 10, 20, np.inf])
    result = woe_table(groups, pd.Series([1, 0, 0, 1, 0]))
    assert result.index[:3].tolist() == groups.cat.categories.tolist()
    assert pd.isna(result.index[3])
    assert result.event.tolist() == [0, 1, 1, 0]
    assert result.non_event.tolist() == [1, 0, 1, 1]


def test_missing_categories_and_smoothing_preserve_counts():
    labels = pd.Series(pd.Categorical(["x", None, "y", "y"], categories=["unused", "x", "y"]))
    target = pd.Series([0, 1, 0, 0])
    result = woe_table(labels, target)
    assert result.index[:2].tolist() == ["x", "y"] and pd.isna(result.index[2])
    assert result.event.tolist() == [0, 0, 1]
    np.testing.assert_allclose(result.event_share, [0.25, 0.25, 0.5])
    assert result.non_event_share.sum() == pytest.approx(1)


@pytest.mark.parametrize(
    "target,message",
    [
        (pd.Series([0, 0]), "target must contain both 0 and 1, got only 0"),
        (pd.Series([0, 2]), "target must contain only 0 and 1, got 2"),
        (pd.Series([1, None]), "target must contain only 0 and 1, got nan"),
        (pd.Series([0, 1], index=[1, 0]), "identical indices"),
        ([0, 1, 1], "same length, got 2 and 3"),
        (np.array([[0, 1]]), "target must be one-dimensional"),
    ],
)
def test_invalid_target(target, message):
    with pytest.raises(ValueError, match=message):
        woe_table(pd.Series(["a", "b"]), target)


def test_psi_contributions_with_new_categories_and_missing_values():
    before = pd.Series(["A", "A", None])
    after = pd.Series(["B", None, None])
    result = psi_table(before, after)
    p = np.array([0.5, 2, 1]) / 3.5
    q = np.array([2, 1, 0.5]) / 3.5
    np.testing.assert_equal(result.reference.to_numpy(), [2, 1, 0])
    np.testing.assert_equal(result.current.to_numpy(), [0, 2, 1])
    np.testing.assert_allclose(result.psi, (p - q) * np.log(p / q))
    assert psi_table(before, before).psi.sum() == 0


def test_psi_keeps_category_order_and_absent_special_values():
    edges = [-np.inf, 2, np.inf]
    reference = cut([-1, -1, 1, 3], edges, special=[-1])
    current = cut([1, 3, 3, None], edges, special=[-1])
    result = psi_table(reference, current)
    assert result.index[:3].tolist() == [-1, pd.Interval(-np.inf, 2), pd.Interval(2, np.inf)]
    assert pd.isna(result.index[3])
    np.testing.assert_equal(result.reference.to_numpy(), [2, 1, 1, 0])
    np.testing.assert_equal(result.current.to_numpy(), [0, 1, 2, 1])
    q = np.array([2, 1, 1, 0.5]) / 4.5
    p = np.array([0.5, 1, 2, 1]) / 4.5
    np.testing.assert_allclose(result.psi, (p - q) * np.log(p / q))


def test_interval_bins_are_counted_without_hashing_their_dtype(monkeypatch):
    # Hashing interval categories with infinite edges warns on NumPy < 2.2.
    def unhashable(self):
        raise AssertionError("CategoricalDtype was hashed")

    edges = bin_edges(pd.Series([1, 2, 3, 4]), bins=2)
    before = pd.cut(pd.Series([1, 2, 3, np.nan]), edges)
    after = pd.cut(pd.Series([4, 4, 1]), edges)
    monkeypatch.setattr(pd.CategoricalDtype, "__hash__", unhashable)
    psi = psi_table(before, after)
    woe = woe_table(before, pd.Series([0, 1, 1, 0]))
    monkeypatch.undo()
    assert psi.index.dtype == before.dtype
    np.testing.assert_equal(psi.reference.to_numpy(), [2, 1, 1])
    np.testing.assert_equal(psi.current.to_numpy(), [1, 2, 0])
    np.testing.assert_equal(woe.event.to_numpy(), [1, 1, 0])
    np.testing.assert_equal(woe.non_event.to_numpy(), [1, 0, 1])


def test_psi_accepts_array_like():
    expected = psi_table(pd.Series(["A", "A", None]), pd.Series(["B", None, None]))
    pd.testing.assert_frame_equal(
        psi_table(["A", "A", None], np.array(["B", None, None])), expected
    )


@pytest.mark.parametrize(
    "reference,current,smoothing,message",
    [
        ([], [1], 0.5, "both contain observations"),
        ([1], [1], np.nan, "finite and positive, got nan"),
        ([1], [1], 0, "finite and positive, got 0"),
        ([[1]], [1], 0.5, "reference must be one-dimensional"),
    ],
)
def test_psi_validates_samples_and_smoothing(reference, current, smoothing, message):
    with pytest.raises(ValueError, match=message):
        psi_table(reference, current, smoothing=smoothing)


def test_version_falls_back_without_metadata(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    try:
        assert importlib.reload(kivi).__version__ == "0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(kivi)
    assert kivi.__version__ == importlib.metadata.version("kivi")


def test_woe_transform_returns_table_values():
    table = woe_table(pd.Series(["a", "a", "a", "b", "b", "b"]), pd.Series([1, 1, 0, 1, 0, 0]))
    new = pd.Series(["b", "a", "b"], index=[10, 11, 12], name="grade")
    expected = pd.Series([-np.log(2), np.log(2), -np.log(2)], index=[10, 11, 12], name="grade")
    pd.testing.assert_series_equal(woe_transform(new, table), expected)
    np.testing.assert_allclose(woe_transform(["a", "b"], table["woe"]), [np.log(2), -np.log(2)])
    assert isinstance(woe_transform(np.array(["a"]), table), np.ndarray)


def test_woe_transform_maps_intervals_and_missing_values():
    edges = [-np.inf, 4, np.inf]
    development = pd.cut(pd.Series([1, 5, None, 9, 5, 2]), edges)
    table = woe_table(development, pd.Series([1, 0, 1, 0, 1, 0]))
    # (-inf, 4]: 1 event, 1 non-event; (4, inf]: 1 and 2; missing: 1 and 0, smoothed to 0.5
    p = np.array([1, 1, 1]) / 3
    q = np.array([1, 2, 0.5]) / 3.5
    result = woe_transform(pd.cut(pd.Series([100, None, -100]), edges), table)
    np.testing.assert_allclose(result, np.log(p / q)[[1, 2, 0]])


def test_woe_transform_handles_unseen_groups():
    table = woe_table(pd.Series(["a", "a", "b", "b"]), pd.Series([1, 0, 0, 1]))
    with pytest.raises(ValueError, match=r"2 group\(s\) are not in table, such as 'c', nan"):
        woe_transform(["a", "c", None, "c"], table)
    np.testing.assert_equal(woe_transform(["a", "c", None], table, unseen=-1.5), [0.0, -1.5, -1.5])
    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match=f"'raise' or a finite number, got {value}"):
            woe_transform(["c"], table, unseen=value)
    edges = [-np.inf, 0, 10, np.inf]
    by_interval = woe_table(pd.cut(pd.Series([5, 15, 5, 15]), edges), pd.Series([1, 0, 0, 1]))
    with pytest.raises(
        ValueError, match=r"1 group\(s\) are not in table, such as Interval\(-inf, 0.0"
    ):
        woe_transform(pd.cut(pd.Series([-5, 5]), edges), by_interval)


@pytest.mark.parametrize(
    "woe,message",
    [
        (["x", 0.5], "table must hold numbers, got 'x'"),
        ([0.5, {"a": 1}], "table must hold numbers, got {'a': 1}"),
        ([0.5, pd.NA], "table must hold numbers, got <NA>"),
    ],
)
def test_woe_transform_names_a_table_value_that_is_not_a_number(woe, message):
    table = pd.Series(woe, index=["a", "b"], dtype=object)
    with pytest.raises(ValueError, match=f"^{message}$"):
        woe_transform(["a"], table)


def test_woe_transform_matches_numbers_by_equality_not_containment():
    values = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8])
    target = pd.Series([0, 0, 1, 0, 1, 1, 0, 1])
    edges = [-np.inf, 4.5, np.inf]
    table = woe_table(pd.cut(values, edges), target)
    # Raw numbers are not groups: each would otherwise get the WOE of the interval holding it.
    with pytest.raises(
        ValueError, match=r"2 group\(s\) are not in table, such as 1.0, 7.0; pass unseen"
    ):
        woe_transform(pd.Series([1.0, 7.0]), table)
    # A special code that the development table never saw is unseen, not binned.
    groups = cut(pd.Series([-1.0, 2.0, 7.0]), edges, special=[-1])
    with pytest.raises(ValueError, match=r"1 group\(s\) are not in table, such as -1;"):
        woe_transform(groups, table)
    np.testing.assert_allclose(
        woe_transform(groups, table, unseen=0.25), [0.25, -np.log(3), np.log(3)]
    )
    np.testing.assert_allclose(woe_transform(groups, table["woe"], unseen=0.25)[1:], table["woe"])
    # Equal labels still match across types, such as 1 and 1.0.
    by_number = woe_table(pd.Series([1, 1, 2, 2]), pd.Series([1, 0, 1, 1]))
    np.testing.assert_allclose(woe_transform([1.0, 2.0], by_number), by_number["woe"])


def test_woe_transform_on_bank_matches_lookup():
    bank = load_bank()
    development, monitoring = bank.iloc[:3000], bank.iloc[3000:]
    edges = bin_edges(development["age"], bins=5)
    table = woe_table(pd.cut(development["age"], edges), development["target"])
    result = woe_transform(pd.cut(monitoring["age"], edges), table)
    lookup = dict(zip(table.index, table["woe"], strict=True))
    expected = [lookup[interval] for interval in pd.cut(monitoring["age"], edges)]
    np.testing.assert_allclose(result.to_numpy(), expected)
    assert result.index.equals(monitoring.index)


@pytest.mark.parametrize(
    "groups,table,unseen,message",
    [
        (["a"], pd.DataFrame({"iv": [0.1]}, index=["a"]), "raise", "must have a woe column"),
        (["a"], pd.Series([0.1, 0.2], index=["a", "a"]), "raise", "one row per group"),
        (["a"], pd.Series([0.1, 0.2], index=[np.nan, None]), "raise", "one row per group"),
        (["a"], {"a": 0.1}, "raise", "woe_table result or a Series, got dict"),
        (["a"], pd.Series([0.1], index=["a"]), "zero", "'raise' or a finite number, got 'zero'"),
        (["a"], pd.Series([0.1], index=["a"]), True, "'raise' or a finite number, got True"),
        (["a"], pd.Series([0.1], index=["a"]), None, "'raise' or a finite number, got None"),
        (np.ones((2, 2)), pd.Series([0.1], index=["a"]), "raise", "groups must be one-dimensional"),
    ],
)
def test_woe_transform_validates_inputs(groups, table, unseen, message):
    with pytest.raises(ValueError, match=message):
        woe_transform(groups, table, unseen=unseen)


def test_count_functions_match_the_tables():
    groups = pd.Series(["a", "a", "a", "b", "b", "b", "c"])
    target = pd.Series([1, 1, 0, 1, 0, 0, 1])
    table = woe_table(groups, target)
    np.testing.assert_allclose(woe(table.event, table.non_event), table.woe)
    assert iv(table.event, table.non_event) == pytest.approx(table.iv.sum())
    stability = psi_table(pd.Series(["x", "x", "y"]), pd.Series(["y", "y", "z"]))
    assert psi(stability.reference, stability.current) == pytest.approx(stability.psi.sum())


def test_woe_of_count_series_is_a_series_on_their_index():
    table = woe_table(pd.Series(["a", "a", "a", "b", "b", "c"]), pd.Series([1, 1, 0, 1, 0, 0]))
    pd.testing.assert_series_equal(woe(table.event, table.non_event), table.woe)
    result = woe(table.event.to_numpy(), table.non_event)
    pd.testing.assert_series_equal(result, table.woe)
    pd.testing.assert_series_equal(woe(table.event, table.non_event.to_list()), table.woe)
    assert isinstance(woe(table.event.to_list(), table.non_event.to_numpy()), np.ndarray)


@pytest.mark.parametrize(
    "function,names",
    [
        (woe, "event_counts and non_event_counts"),
        (iv, "event_counts and non_event_counts"),
        (psi, "reference_counts and current_counts"),
    ],
)
def test_count_series_with_different_indices_raise(function, names):
    # kivi 0.9 matched them by position with a FutureWarning.
    first = pd.Series([1, 9], index=["a", "b"])
    second = pd.Series([3, 5], index=["b", "a"])
    with pytest.raises(ValueError, match=f"^{names} must have identical indices$"):
        function(first, second)
    np.testing.assert_allclose(
        function(*first.align(second, fill_value=0)), function([1, 9], [5, 3])
    )


def test_count_functions_match_hand_calculation():
    np.testing.assert_allclose(woe([2, 1], [1, 2]), [np.log(2), -np.log(2)])
    assert iv([2, 1], [1, 2]) == pytest.approx(2 * np.log(2) / 3)
    q = np.array([1, 1]) / 2
    np.testing.assert_allclose(woe([0, 2], [1, 1]), np.log(np.array([0.5, 2]) / 2.5 / q))
    np.testing.assert_allclose(woe([0, 2], [1, 1], smoothing=1), np.log(np.array([1, 2]) / 3 / q))
    assert psi([1.5, 2.5], [3.0, 5.0]) == 0
    p, r = np.array([1, 3]) / 4, np.array([2, 2]) / 4
    assert psi([2, 2], [1, 3]) == pytest.approx(((p - r) * np.log(p / r)).sum())


@pytest.mark.parametrize(
    "first,second,message",
    [
        ([1, 2], [1], "must have the same length"),
        ([[1, 2]], [1, 2], "must be one-dimensional, got 2 dimensions"),
        ([1, -1], [1, 2], "finite, non-negative counts"),
        ([1, np.nan], [1, 2], "finite, non-negative counts"),
        ([0, 0], [1, 2], "must contain a positive count"),
        (["a", "b"], [1, 2], "must contain numeric counts"),
    ],
)
def test_count_functions_validate_counts(first, second, message):
    for function in (woe, iv, psi):
        with pytest.raises(ValueError, match=message):
            function(first, second)
    with pytest.raises(ValueError, match="smoothing must be finite and positive, got 0"):
        psi([1, 2], [1, 2], smoothing=0)


def test_method_is_keyword_only():
    method = inspect.signature(bin_edges).parameters["method"]
    assert method.kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match="positional argument"):
        bin_edges([1, 2, 3, 4], 2, "uniform")


def test_evaluate_alias_is_removed():
    with pytest.raises(ModuleNotFoundError, match="kivi.evaluate"):
        import kivi.evaluate  # noqa: F401


def test_woe_table_reports_totals_shares_and_event_rates():
    groups = pd.Series(["a", "a", "b", "b", "c"])
    target = pd.Series([1, 0, 0, 0, 1])
    result = woe_table(groups, target)
    assert result.columns.tolist() == [
        "non_event",
        "event",
        "total",
        "non_event_share",
        "event_share",
        "total_share",
        "event_rate",
        "woe",
        "iv",
    ]
    assert result.total.tolist() == [2, 2, 1]
    np.testing.assert_allclose(result.total_share, [0.4, 0.4, 0.2])
    np.testing.assert_allclose(result.event_rate, [0.5, 0.0, 1.0])
    # The smoothed class shares are unchanged by the new columns.
    np.testing.assert_allclose(result.event_share, [0.4, 0.2, 0.4])


def test_integer_weights_match_repeated_rows():
    groups = pd.Series(["a", "a", "b", "b", "c"])
    target = pd.Series([1, 0, 1, 0, 1])
    weights = np.array([2, 1, 1, 3, 1])
    repeated = woe_table(
        pd.Series(np.repeat(groups.to_numpy(), weights)),
        pd.Series(np.repeat(target.to_numpy(), weights)),
    )
    weighted = woe_table(groups, target, weights=weights)
    pd.testing.assert_frame_equal(weighted, repeated, check_dtype=False)

    reference, current = pd.Series(["x", "x", "y"]), pd.Series(["y", "y", "z"])
    reference_weights, current_weights = [3, 1, 2], [1, 2, 4]
    repeated = psi_table(
        pd.Series(np.repeat(reference.to_numpy(), reference_weights)),
        pd.Series(np.repeat(current.to_numpy(), current_weights)),
    )
    weighted = psi_table(
        reference,
        current,
        reference_weights=reference_weights,
        current_weights=current_weights,
    )
    pd.testing.assert_frame_equal(weighted, repeated, check_dtype=False)


def test_reweighting_a_class_keeps_woe_and_changes_event_rates():
    groups = pd.Series(["a", "a", "a", "b", "b", "b"])
    target = pd.Series([1, 1, 0, 1, 0, 0])
    plain = woe_table(groups, target)
    weighted = woe_table(groups, target, weights=np.where(target == 0, 4.0, 1.0))
    np.testing.assert_allclose(weighted.woe, plain.woe)
    np.testing.assert_allclose(weighted.iv, plain.iv)
    # a: 2 events and 1 non-event weighted 4; b: 1 event and 2 non-events weighted 4.
    np.testing.assert_allclose(weighted.event_rate, [2 / 6, 1 / 9])
    np.testing.assert_allclose(weighted.total_share, [6 / 15, 9 / 15])


def test_zero_weight_groups_are_left_out():
    result = woe_table(["a", "b", "b"], [1, 0, 1], weights=[0, 1, 1])
    assert result.index.tolist() == ["b"]
    assert result.total.tolist() == [2.0]
    with pytest.raises(ValueError, match="positive total, got 1.0 for 0 and 0.0 for 1"):
        woe_table(["a", "b"], [1, 0], weights=[0, 1])
    stability = psi_table(["x", "y"], ["x", "z"], reference_weights=[1, 0], current_weights=[1, 0])
    assert stability.index.tolist() == ["x"]


@pytest.mark.parametrize(
    "weights,message",
    [
        ([1, 1], "weights must have one weight per value of .*, got 2 for 3"),
        ([1, -1, 1], "weights must be finite and non-negative, got -1.0"),
        ([1, np.nan, 1], "weights must be finite and non-negative, got nan"),
        ([1, 1, np.inf], "weights must be finite and non-negative, got inf"),
        ([0, 0, 0], "weights must have a finite, positive total, got 0.0"),
        ([[1, 1, 1]], "weights must be one-dimensional, got 2 dimensions"),
        (["a", "b", "c"], "weights must be numeric, got dtype "),
        (pd.Series([1, 1, 1], index=[3, 4, 5]), "(values|groups) and weights must have identical"),
    ],
)
def test_invalid_weights(weights, message):
    values = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=message):
        bin_edges(values, bins=2, weights=weights)
    with pytest.raises(ValueError, match=message):
        woe_table(values, pd.Series([0, 1, 1]), weights=weights)
    with pytest.raises(ValueError, match="reference_weights must have one weight per value"):
        psi_table(["x"], ["y"], reference_weights=[1, 2])


def test_weighted_quantile_edges_match_repeated_values():
    values = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
    weights = np.array([3, 1, 1, 2, 1, 1, 4, 1])
    probabilities = np.linspace(0, 1, 5)
    expected = np.quantile(np.repeat(values, weights), probabilities, method="inverted_cdf")
    np.testing.assert_equal(
        bin_edges(values, bins=4, weights=weights),
        np.r_[-np.inf, np.unique(expected)[1:-1], np.inf],
    )
    np.testing.assert_equal(
        bin_edges([1, 2, 3, 4], bins=2, weights=[1, 1, 1, 3]), [-np.inf, 3, np.inf]
    )
    uniform = bin_edges([0, 10, 20, 100], bins=2, method="uniform", weights=[1, 1, 1, 0])
    np.testing.assert_equal(uniform, [-np.inf, 10, np.inf])


@pytest.mark.skipif(
    np.lib.NumpyVersion(np.__version__) < "2.0.0", reason="np.quantile has weights since 2.0"
)
def test_weighted_quantile_edges_match_numpy():
    rng = np.random.default_rng(0)
    values, weights = rng.normal(size=500), rng.uniform(0.1, 3, size=500)
    expected = np.quantile(values, np.linspace(0, 1, 11), weights=weights, method="inverted_cdf")
    np.testing.assert_equal(bin_edges(values, bins=10, weights=weights)[1:-1], expected[1:-1])


def test_tied_edges_warning_uses_weighted_shares():
    with pytest.warns(UserWarning, match="the value 1 makes up 77% of the fitted values"):
        edges = bin_edges([1, 2, 3, 4], bins=4, weights=[10, 1, 1, 1])
    np.testing.assert_equal(edges, [-np.inf, np.inf])


def test_weighted_quantiles_select_cdf_steps_despite_rounding():
    # np.linspace(0, 1, 6)[3] is 0.6000000000000001, just above the CDF step at 6 / 10.
    expected = [-np.inf, 1, 3, 5, 7, np.inf]
    for scale in (1, 0.1, 1 / 3):
        np.testing.assert_equal(
            bin_edges(np.arange(10.0), bins=5, weights=np.full(10, scale)), expected
        )


def test_dropping_a_zero_weight_interval_group_does_not_hash_the_dtype(monkeypatch):
    def unhashable(self):
        raise AssertionError("CategoricalDtype was hashed")

    edges = [-np.inf, 2, 4, np.inf]
    groups = pd.cut(pd.Series([1, 2, 3, 4, 5, 6]), edges)
    monkeypatch.setattr(pd.CategoricalDtype, "__hash__", unhashable)
    woe = woe_table(groups, pd.Series([0, 1, 0, 1, 0, 1]), weights=[1, 1, 0, 0, 1, 1])
    psi = psi_table(groups, groups, reference_weights=[1, 1, 0, 0, 1, 1])
    monkeypatch.undo()
    assert len(woe) == 2 and woe.total.tolist() == [2.0, 2.0]
    np.testing.assert_equal(psi.reference.to_numpy(), [2, 0, 2])
    np.testing.assert_equal(psi.current.to_numpy(), [2, 2, 2])


def test_smoothing_is_in_units_of_weight():
    groups = pd.Series(["a", "a", "b", "b", "c", "c"])
    target = pd.Series([1, 0, 1, 0, 1, 1])  # group c has no non-events
    plain = woe_table(groups, target)
    scaled = woe_table(groups, target, weights=np.full(6, 1 / 6), smoothing=0.5 / 6)
    np.testing.assert_allclose(scaled.woe, plain.woe)
    unscaled = woe_table(groups, target, weights=np.full(6, 1 / 6))
    assert not np.allclose(unscaled.woe, plain.woe)


def test_special_values_and_weights_together():
    values = [-1, -1, 1, 2, 3, 4]
    edges = bin_edges(values, bins=2, special=[-1], weights=[5, 5, 1, 1, 1, 3])
    np.testing.assert_equal(edges, [-np.inf, 3, np.inf])
    with pytest.raises(ValueError, match="apart from missing, special and zero-weight values"):
        bin_edges([-1, 2], special=[-1], weights=[1, 0])


def test_weights_are_aligned_like_the_values():
    reference = pd.Series(["x", "x", "y"], index=[10, 11, 12])
    current = pd.Series(["y", "y", "z"])
    with pytest.raises(
        ValueError, match="reference and reference_weights must have identical indices"
    ):
        psi_table(reference, current, reference_weights=pd.Series([1, 1, 1]))
    by_position = psi_table(reference, current, reference_weights=[3, 1, 2])
    by_index = psi_table(
        reference, current, reference_weights=pd.Series([3, 1, 2], index=[10, 11, 12])
    )
    pd.testing.assert_frame_equal(by_position, by_index)
    # Only the current sample is weighted.
    repeated = psi_table(reference, pd.Series(np.repeat(current.to_numpy(), [1, 2, 4])))
    pd.testing.assert_frame_equal(
        psi_table(reference, current, current_weights=[1, 2, 4]), repeated, check_dtype=False
    )


@pytest.mark.parametrize(
    "call,message",
    [
        (lambda: woe_table(5, [0, 1], weights=[1, 1]), "groups must be one-dimensional"),
        (
            lambda: psi_table([], ["x"], reference_weights=[]),
            "reference and current must both contain observations",
        ),
        (lambda: bin_edges([1, 2, 3], weights=[True, False, True]), "got dtype bool"),
        (lambda: bin_edges([1, 2, 3], weights=[1e308, 1e308, 1]), "finite, positive total"),
        (
            lambda: bin_edges([1, 2, 3], weights=pd.array([1, None, 1], dtype="Float64")),
            "weights must be finite and non-negative",
        ),
    ],
)
def test_weight_validation_order_and_types(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_nullable_values_with_special_codes():
    values = pd.Series([1, -1, 5, None], dtype="Int64")
    groups = cut(values, [-np.inf, 2, np.inf], special=[-1])
    np.testing.assert_equal(groups.cat.codes.to_numpy(), [1, 0, 2, -1])
    np.testing.assert_equal(bin_edges(values, bins=2, special=[-1]), [-np.inf, 3, np.inf])


def interval_woe(values):
    """A Series of WOE values on consecutive intervals, as woe_transform accepts."""
    breaks = np.arange(len(values) + 1, dtype=float)
    return pd.Series(values, index=pd.IntervalIndex.from_breaks(breaks), name="woe")


@pytest.mark.parametrize(
    ("values", "trend"),
    [
        ([0.1, 0.2, 0.5], "increasing"),
        ([0.5, -0.2, -0.3], "decreasing"),
        ([0.3, -0.1, 0.2], "u_shaped"),
        ([-0.3, 0.1, 0.4, -0.2], "inverted_u"),
        ([0.1, 0.3, -0.2, 0.4], "irregular"),
        ([0.1, -0.1, 0.1, -0.1, 0.1], "irregular"),
        ([0.2, 0.2, 0.2], "flat"),
        ([0.0, 0.0], "flat"),
        ([0.7], "flat"),
        # Equal neighbours neither rise nor fall, so they break no trend.
        ([0.1, 0.1, 0.3], "increasing"),
        ([0.5, 0.5, 0.2, 0.2], "decreasing"),
        ([0.3, 0.1, 0.1, 0.1, 0.4], "u_shaped"),
        ([0.1, 0.3, 0.3, 0.1], "inverted_u"),
        ([0.1, 0.3, 0.3, 0.1, 0.1, 0.3], "irregular"),
    ],
)
def test_woe_trend_names_the_shape_of_the_woe(values, trend):
    assert woe_trend(interval_woe(values)) == trend
    assert woe_trend(interval_woe(values).to_frame()) == trend


def test_woe_trend_counts_neighbours_within_a_relative_1e_10_as_equal():
    assert woe_trend(interval_woe([2.0, 2.0 * (1 + 5e-11), 2.0])) == "flat"
    assert woe_trend(interval_woe([-2.0, -2.0 * (1 + 5e-11), -2.0, -1.0])) == "increasing"
    assert woe_trend(interval_woe([2.0, 2.0 * (1 + 5e-10), 2.0])) == "inverted_u"


def reference_trend(values):
    """The trend by its definition: split the bins into pieces that rise or fall."""

    def rises(part):
        return all(b >= a for a, b in zip(part, part[1:], strict=False))

    def falls(part):
        return all(b <= a for a, b in zip(part, part[1:], strict=False))

    if rises(values) and falls(values):
        return "flat"
    if rises(values):
        return "increasing"
    if falls(values):
        return "decreasing"
    for k in range(len(values)):
        if falls(values[: k + 1]) and rises(values[k:]):
            return "u_shaped"
        if rises(values[: k + 1]) and falls(values[k:]):
            return "inverted_u"
    return "irregular"


def test_woe_trend_matches_its_definition_on_random_tables_with_ties():
    rng = np.random.default_rng(0)
    mirror = {
        "increasing": "decreasing",
        "decreasing": "increasing",
        "u_shaped": "inverted_u",
        "inverted_u": "u_shaped",
        "irregular": "irregular",
        "flat": "flat",
    }
    seen = set()
    for _ in range(3000):
        # Few distinct values, so that equal neighbours are common.
        values = rng.integers(-2, 3, size=rng.integers(1, 9)) / 10
        trend = woe_trend(interval_woe(values))
        assert trend == reference_trend(values.tolist())
        assert woe_trend(interval_woe(-values)) == mirror[trend]
        seen.add(trend)
    assert seen == set(mirror)


def test_woe_trend_leaves_out_special_codes_and_missing_values():
    values = pd.Series([-1.0] * 4 + [1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5] + [np.nan] * 4)
    target = pd.Series([1, 1, 1, 0] + [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1] + [0, 0, 0, 0])
    groups = cut(values, [-np.inf, 2, 4, np.inf], special=[-1])
    table = woe_table(groups, target)
    assert table.index[0] == -1 and pd.isna(table.index[-1])
    woe_values = table["woe"].to_numpy()
    # The special group has the highest WOE and the missing group the lowest; the bins rise.
    assert woe_values[0] > woe_values[1] < woe_values[2] < woe_values[3] > woe_values[4]
    assert woe_trend(table) == "increasing"


def test_woe_trend_reads_the_tables_of_a_binning_spec():
    data = load_bank().iloc[:3000]
    spec = fit_spec(data[["pdays", "job"]], data["target"], special={"pdays": [-1]})
    edges = bin_edges(data["pdays"], bins=5, special=[-1])
    table = woe_table(cut(data["pdays"], edges, special=[-1]), data["target"])
    assert woe_trend(spec.tables["pdays"]) == woe_trend(table) == "irregular"
    assert woe_trend(BinningSpec.from_json(spec.to_json()).tables["pdays"]) == "irregular"
    with pytest.raises(ValueError, match="a categorical variable has no trend"):
        woe_trend(spec.tables["job"])


def test_woe_trend_leaves_the_table_alone():
    table = woe_table(pd.cut(pd.Series([1, 2, 3, 4]), [0, 2, 4]), pd.Series([0, 1, 1, 1]))
    copy = table.copy()
    woe_trend(table)
    pd.testing.assert_frame_equal(table, copy)


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ([0.1, 0.2], "table must be a woe_table result or a Series, got list"),
        (
            pd.DataFrame({"iv": [0.1]}, index=pd.IntervalIndex.from_breaks([0.0, 1.0])),
            "table must have a woe column, as returned by woe_table",
        ),
        (
            pd.Series([0.1, 0.2], index=pd.IntervalIndex.from_breaks([0.0, 1.0, 2.0])[[0, 0]]),
            "table must have one row per group",
        ),
        (
            woe_table(pd.Series(["a", "a", "b", "b"]), pd.Series([1, 0, 0, 1])),
            "table must have a row labeled with an interval, as the bins of a numeric variable "
            "are; a categorical variable has no trend, got the label 'a'",
        ),
        (interval_woe([0.1, np.nan]), "the woe of table must be finite, got nan"),
        (interval_woe([-np.inf, 0.1]), "the woe of table must be finite, got -inf"),
        (interval_woe(["0.1", "x"]), "table must hold numbers, got 'x'"),
    ],
)
def test_woe_trend_rejects_another_table(table, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        woe_trend(table)
