import importlib
import importlib.metadata
import inspect

import numpy as np
import pandas as pd
import pytest

import kivi
from kivi.binning import bin_edges, cut
from kivi.datasets import Dataset
from kivi.stability import psi, psi_table
from kivi.woe import iv, woe, woe_table, woe_transform


def test_example_data():
    bank = Dataset.bank_data()
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
    pdays = Dataset.bank_data()["pdays"]
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
        ([], 2, "quantile", "no finite observations apart from missing and special values"),
        ([None], 2, "quantile", "no finite observations apart from missing and special values"),
        ([np.inf], 2, "quantile", "finite observations only; pass infinite codes in special"),
        ([1], 0, "quantile", "positive integer, got 0"),
        ([1], True, "quantile", "positive integer, got True"),
        ([1], 2.0, "quantile", "positive integer, got 2.0"),
        ([1], 2, "median", "'quantile' or 'uniform', got 'median'"),
        (["a", "b"], 2, "quantile", "Unable to parse string"),
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


def test_special_value_bins_on_bank_pdays():
    bank = Dataset.bank_data()
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


@pytest.mark.parametrize(
    "special,message",
    [([np.nan], "must not contain missing values"), (-1, "special must be one-dimensional")],
)
def test_invalid_special_values(special, message):
    with pytest.raises(ValueError, match=message):
        bin_edges([1, 2], special=special)
    with pytest.raises(ValueError, match=message):
        cut([1, 2], [-np.inf, np.inf], special=special)


@pytest.mark.parametrize(
    "values,edges,message",
    [
        (["a", 1], [-np.inf, np.inf], "Unable to parse string"),
        ([1], [3, 1], "increase monotonically"),
        ([[1]], [-np.inf, np.inf], "values must be one-dimensional"),
    ],
)
def test_invalid_cut_inputs(values, edges, message):
    with pytest.raises(ValueError, match=message):
        cut(values, edges)


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
    bank = Dataset.bank_data()
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
        (pd.Series([0, 0]), "both 0 and 1"),
        (pd.Series([0, 2]), "both 0 and 1"),
        (pd.Series([1, None]), "no missing labels"),
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
    assert np.isnan(woe_transform(["c"], table, unseen=np.nan)).all()
    edges = [-np.inf, 0, 10, np.inf]
    by_interval = woe_table(pd.cut(pd.Series([5, 15, 5, 15]), edges), pd.Series([1, 0, 0, 1]))
    with pytest.raises(
        ValueError, match=r"1 group\(s\) are not in table, such as Interval\(-inf, 0.0"
    ):
        woe_transform(pd.cut(pd.Series([-5, 5]), edges), by_interval)


def test_woe_transform_on_bank_matches_lookup():
    bank = Dataset.bank_data()
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
        (["a"], pd.Series([0.1], index=["a"]), "zero", "'raise' or a number, got 'zero'"),
        (["a"], pd.Series([0.1], index=["a"]), True, "'raise' or a number, got True"),
        (["a"], pd.Series([0.1], index=["a"]), None, "'raise' or a number, got None"),
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


def test_positional_method_is_deprecated():
    with pytest.warns(FutureWarning, match="passing method to bin_edges by position"):
        edges = bin_edges([1, 2, 3, 4], 2, "uniform")
    np.testing.assert_equal(edges, bin_edges([1, 2, 3, 4], 2, method="uniform"))
    method = inspect.signature(bin_edges).parameters["method"]
    assert method.kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match="multiple values for 'method'"):
        bin_edges([1, 2], 2, "uniform", method="quantile")
    with pytest.raises(TypeError, match="positional arguments"):
        bin_edges([1, 2], 2, "uniform", None)


def test_evaluate_is_a_deprecated_alias_of_stability():
    import kivi.evaluate
    import kivi.stability

    with pytest.warns(FutureWarning, match="import psi_table from kivi.stability"):
        from kivi.evaluate import psi_table as old

    assert old is kivi.stability.psi_table
    with pytest.raises(AttributeError, match="has no attribute 'psi'"):
        kivi.evaluate.psi  # noqa: B018


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
