import numpy as np
import pandas as pd
import pytest

from kivi.binning import bin_edges
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
    np.testing.assert_equal(bin_edges(pd.Series([3, 3])), [-np.inf, np.inf])
    assert np.diff(bin_edges(pd.Series([0, 0, 0, 1, 2]), bins=5)).min() > 0


@pytest.mark.parametrize("values,bins", [([], 2), ([np.inf], 2), ([1], 0), ([1], True)])
def test_invalid_binning_inputs(values, bins):
    with pytest.raises(ValueError):
        bin_edges(pd.Series(values), bins=bins)


def test_woe_matches_hand_calculation():
    labels = pd.Series(["a", "a", "a", "b", "b", "b"])
    target = pd.Series([1, 1, 0, 1, 0, 0])
    result = woe_table(labels, target)
    np.testing.assert_allclose(result.woe, [np.log(2), -np.log(2)])
    assert result.iv.sum() == pytest.approx(2 * np.log(2) / 3)


def test_missing_categories_and_smoothing_preserve_counts():
    labels = pd.Series(pd.Categorical(["x", None, "y", "y"], categories=["unused", "x", "y"]))
    target = pd.Series([0, 1, 0, 0])
    result = woe_table(labels, target)
    assert len(result) == 3 and result.index.isna().sum() == 1
    assert result.event.tolist() == [0, 1, 0]
    np.testing.assert_allclose(result.event_share, [0.25, 0.5, 0.25])
    assert result.non_event_share.sum() == pytest.approx(1)


@pytest.mark.parametrize("target", [pd.Series([0, 0]), pd.Series([0, 2]), pd.Series([1, None]),
                                     pd.Series([0, 1], index=[1, 0])])
def test_invalid_target(target):
    with pytest.raises(ValueError):
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


def test_psi_validates_samples_and_smoothing():
    with pytest.raises(ValueError):
        psi_table(pd.Series([], dtype=float), pd.Series([1]))
    with pytest.raises(ValueError):
        psi_table(pd.Series([1]), pd.Series([1]), smoothing=np.nan)
