import importlib
import importlib.metadata

import numpy as np
import pandas as pd
import pytest

import kivi
from kivi.binning import bin_edges, cut
from kivi.datasets import Dataset
from kivi.evaluate import psi_table
from kivi.woe import woe_table


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
    [[1, 2, None, 3], (1.0, 2.0, np.nan, 3.0), np.array([1, 2, np.nan, 3]),
     pd.array([1, 2, None, 3], dtype="Int64"), pd.Series([1, 2, None, 3])],
)
def test_bin_edges_accepts_array_like(values):
    np.testing.assert_equal(bin_edges(values, bins=2), [-np.inf, 2.0, np.inf])


@pytest.mark.parametrize(
    "values,bins,method,message",
    [([], 2, "quantile", "no finite observations apart from missing and special values"),
     ([None], 2, "quantile", "no finite observations apart from missing and special values"),
     ([np.inf], 2, "quantile", "finite observations only; pass infinite codes in special"),
     ([1], 0, "quantile", "positive integer, got 0"),
     ([1], True, "quantile", "positive integer, got True"),
     ([1], 2.0, "quantile", "positive integer, got 2.0"),
     ([1], 2, "median", "'quantile' or 'uniform', got 'median'"),
     (["a", "b"], 2, "quantile", "Unable to parse string"),
     (np.ones((2, 2)), 2, "quantile", "one-dimensional, got 2 dimensions"),
     (1.0, 2, "quantile", "one-dimensional, got 0 dimensions")],
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
    np.testing.assert_equal(bin_edges(["N/A", 1, 3, 5], bins=2, special=["N/A"]), [-np.inf, 3, np.inf])
    np.testing.assert_equal(bin_edges([-np.inf, 1, 3, 5], bins=2, special=[-np.inf]), [-np.inf, 3, np.inf])


def test_cut_gives_each_special_value_a_group():
    values = pd.Series([-1, 15, 35, None, -9, -1.0], index=list("abcdef"), name="x")
    edges = [-np.inf, 25, np.inf]
    groups = cut(values, edges, special=[-9, -1])
    assert groups.index.equals(values.index) and groups.name == "x"
    assert groups.cat.categories.tolist() == [-9, -1, pd.Interval(-np.inf, 25), pd.Interval(25, np.inf)]
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
    interval_counts = pd.crosstab(pd.cut(bank.pdays[~is_special], edges), bank.target[~is_special])
    expected = np.vstack([special_counts, interval_counts.to_numpy()])
    assert result.index[0] == -1 and result.index[1:].tolist() == interval_counts.index.tolist()
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
    [(["a", 1], [-np.inf, np.inf], "Unable to parse string"),
     ([1], [3, 1], "increase monotonically"),
     ([[1]], [-np.inf, np.inf], "values must be one-dimensional")],
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
    [(pd.Series([0, 0]), "both 0 and 1"),
     (pd.Series([0, 2]), "both 0 and 1"),
     (pd.Series([1, None]), "no missing labels"),
     (pd.Series([0, 1], index=[1, 0]), "identical indices"),
     ([0, 1, 1], "same length, got 2 and 3"),
     (np.array([[0, 1]]), "target must be one-dimensional")],
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


def test_psi_accepts_array_like():
    expected = psi_table(pd.Series(["A", "A", None]), pd.Series(["B", None, None]))
    pd.testing.assert_frame_equal(psi_table(["A", "A", None], np.array(["B", None, None])), expected)


@pytest.mark.parametrize(
    "reference,current,smoothing,message",
    [([], [1], 0.5, "both contain observations"),
     ([1], [1], np.nan, "finite and positive, got nan"),
     ([1], [1], 0, "finite and positive, got 0"),
     ([[1]], [1], 0.5, "reference must be one-dimensional")],
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
