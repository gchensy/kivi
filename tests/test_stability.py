import numpy as np
import pandas as pd
import pytest

from kivi.datasets import Dataset
from kivi.scorecard import points_table, woe_to_points
from kivi.stability import csi_table, psi_table
from kivi.woe import woe_table, woe_transform

COLUMNS = ["reference", "current", "reference_share", "current_share", "points", "csi"]
SCALE = {"pdo": 20, "base_score": 600, "base_odds": 50}


def test_csi_matches_hand_calculation():
    reference = pd.Series(["x", "x", "y", "y"])
    current = pd.Series(["x", "y", "y", "y"])
    table = csi_table(reference, current, {"x": 40.0, "y": 25.0})
    assert table.columns.tolist() == COLUMNS
    assert table.index.name == "bin" and table.index.tolist() == ["x", "y"]
    assert table["reference"].tolist() == [2, 2] and table["current"].tolist() == [1, 3]
    np.testing.assert_allclose(table["reference_share"], [0.5, 0.5])
    np.testing.assert_allclose(table["current_share"], [0.25, 0.75])
    # (0.25 - 0.5) * 40 = -10 and (0.75 - 0.5) * 25 = 6.25.
    np.testing.assert_allclose(table["csi"], [-10.0, 6.25])
    # The average points fall from 32.5 to 28.75.
    assert table["csi"].sum() == pytest.approx(28.75 - 32.5)


def test_csi_matches_legacy_on_bank_data():
    data = Dataset.bank_data()
    reference, current = data.iloc[:3000], data.iloc[3000:]
    # The legacy CSI(score).Category gave the same contributions and -0.2651545036160423.
    points = {"primary": 30.0, "secondary": 42.5, "tertiary": 55.0, "unknown": 47.0}
    table = csi_table(reference["education"], current["education"], points)
    assert table["reference"].tolist() == [448, 1512, 915, 125]
    assert table["current"].tolist() == [230, 794, 435, 62]
    np.testing.assert_allclose(
        table["csi"], [0.056489, 0.766062, -1.045217, -0.042488], rtol=0, atol=1e-6
    )
    assert table["csi"].sum() == pytest.approx(-0.2651545036160423, rel=1e-12)
    # Legacy CSI(score).Continue with custom cut points gave -0.02596754385964073; it
    # required the current values to lie within the range of the reference values.
    edges = [-np.inf, 25, 35, 45, 55, 65, np.inf]
    age_points = pd.Series(
        [120.0, 137.1, 139.8, 136.4, 135.6, 113.4], index=pd.IntervalIndex.from_breaks(edges)
    )
    age = current["age"][current["age"] <= reference["age"].max()]
    table = csi_table(pd.cut(reference["age"], edges), pd.cut(age, edges), age_points)
    assert table["current"].tolist() == [31, 503, 468, 348, 137, 33]
    assert table["csi"].sum() == pytest.approx(-0.02596754385964073, rel=1e-12)


def test_group_missing_from_one_sample_counts_with_a_zero_share():
    reference = ["a", "a", "b", "c"]
    current = ["a", "b", "b", "d"]
    points = {"a": 10.0, "b": 20.0, "c": 30.0, "d": 5.0}
    table = csi_table(reference, current, points)
    assert table.index.tolist() == ["a", "b", "c", "d"]
    assert table["reference"].tolist() == [2, 1, 1, 0]
    assert table["current"].tolist() == [1, 2, 0, 1]
    np.testing.assert_allclose(table["reference_share"], [0.5, 0.25, 0.25, 0.0])
    np.testing.assert_allclose(table["current_share"], [0.25, 0.5, 0.0, 0.25])
    # -2.5 + 5 - 7.5 + 1.25. The legacy CSI dropped groups missing from one sample.
    np.testing.assert_allclose(table["csi"], [-2.5, 5.0, -7.5, 1.25])
    mean_reference = (10 + 10 + 20 + 30) / 4
    mean_current = (10 + 20 + 20 + 5) / 4
    assert table["csi"].sum() == pytest.approx(mean_current - mean_reference)


def test_csi_is_the_shift_of_the_average_points_of_a_scorecard():
    data = Dataset.bank_data()
    development, monitoring = data.iloc[:3000], data.iloc[3000:]
    edges = np.array([-np.inf, 25, 35, 45, 55, 65, np.inf])

    def to_groups(frame):
        groups = frame[["poutcome", "contact", "housing"]].copy()
        groups["age"] = pd.cut(frame["age"], edges)
        return groups

    dev_groups, mon_groups = to_groups(development), to_groups(monitoring)
    tables = {column: woe_table(dev_groups[column], development["target"]) for column in dev_groups}
    coefficients = {"poutcome": 0.899, "contact": 0.658, "housing": 0.691, "age": 0.714}
    table = points_table(tables, coefficients, -2.028, **SCALE)

    def score(groups):
        woe = {column: woe_transform(groups[column], tables[column]) for column in groups}
        return woe_to_points(pd.DataFrame(woe), coefficients, -2.028, **SCALE)

    dev_points, mon_points = score(dev_groups), score(mon_groups)
    csi = pd.Series(
        {
            column: csi_table(dev_groups[column], mon_groups[column], table.loc[[column]])[
                "csi"
            ].sum()
            for column in coefficients
        }
    )
    shift = mon_points.mean() - dev_points.mean()
    np.testing.assert_allclose(csi, shift[csi.index], rtol=1e-10, atol=1e-12)
    assert csi.sum() == pytest.approx(mon_points.sum(axis=1).mean() - dev_points.sum(axis=1).mean())
    # A Series of points by group gives the same table as the points_table rows.
    age = table.loc[["age"]]
    by_series = pd.Series(age["points"].to_numpy(), index=pd.IntervalIndex(age["bin"].tolist()))
    pd.testing.assert_frame_equal(
        csi_table(dev_groups["age"], mon_groups["age"], by_series),
        csi_table(dev_groups["age"], mon_groups["age"], age),
    )


def test_adding_a_constant_to_the_points_leaves_csi_unchanged():
    reference = ["a", "b", "b", "c", "c", "c"]
    current = ["a", "a", "b", "c", "c", "d"]
    points = pd.Series({"a": 12.0, "b": -3.0, "c": 7.5, "d": 1.0})
    base = csi_table(reference, current, points)["csi"].sum()
    shifted = csi_table(reference, current, points + 1000.0)["csi"].sum()
    assert shifted == pytest.approx(base, abs=1e-12)


def test_csi_rows_counts_and_shares_follow_psi_table():
    edges = [-np.inf, 1, 2, np.inf]
    reference = pd.cut(pd.Series([0.5, 1.5, 1.7, np.nan, 3.0]), edges)
    current = pd.cut(pd.Series([2.5, 2.7, np.nan, np.nan, 0.1, 1.2]), edges)
    points = pd.Series([3.0, 2.0, 1.0, 9.0], index=[*pd.IntervalIndex.from_breaks(edges), np.nan])
    table = csi_table(reference, current, points)
    stability = psi_table(reference, current)
    # Compare category codes: comparing interval indices hashes them, which warns on NumPy < 2.2.
    np.testing.assert_array_equal(table.index.codes, stability.index.codes)
    np.testing.assert_array_equal(table.index.codes, [0, 1, 2, -1])
    np.testing.assert_array_equal(
        table[["reference", "current"]].to_numpy(), stability[["reference", "current"]].to_numpy()
    )
    # Category order with missing values last, and the missing group gets its own points.
    assert table["points"].tolist() == [3.0, 2.0, 1.0, 9.0]
    np.testing.assert_allclose(table["reference_share"], [1 / 5, 2 / 5, 1 / 5, 1 / 5])
    np.testing.assert_allclose(table["current_share"], [1 / 6, 1 / 6, 2 / 6, 2 / 6])


def test_shares_are_raw_while_psi_table_smooths_them():
    table = csi_table(["a", "a", "b"], ["a", "a", "a"], {"a": 1.0, "b": 4.0})
    np.testing.assert_allclose(table["current_share"], [1.0, 0.0])
    assert psi_table(["a", "a", "b"], ["a", "a", "a"])["current_share"].iloc[1] > 0
    np.testing.assert_allclose(table["csi"], [1 / 3, -4 / 3])


def test_points_outside_the_samples_are_left_out():
    table = csi_table(["a", "b"], ["b", "b"], {"a": 1.0, "b": 2.0, "z": 50.0})
    assert table.index.tolist() == ["a", "b"]


def test_unseen_points_go_to_groups_without_points():
    message = r"1 group\(s\) have no points, such as 'new'; pass unseen_points"
    with pytest.raises(ValueError, match=message):
        csi_table(["a", "b"], ["a", "new"], {"a": 1.0, "b": 2.0})
    table = csi_table(["a", "b"], ["a", "new"], {"a": 1.0, "b": 2.0}, unseen_points=-4)
    assert table["points"].tolist() == [1.0, 2.0, -4.0]
    np.testing.assert_allclose(table["csi"], [0.0, -1.0, -2.0])
    # Missing values without a missing group in points are unseen too.
    with pytest.raises(ValueError, match=r"1 group\(s\) have no points, such as nan"):
        csi_table(["a", None], ["a", "a"], {"a": 1.0})
    table = csi_table(["a", None], ["a", "a"], {"a": 1.0, np.nan: 6.0})
    assert table["points"].tolist() == [1.0, 6.0]


def test_weights_are_sums_and_integer_weights_equal_repeated_rows():
    reference = pd.Series(["a", "b", "c"], index=[7, 8, 9])
    current = pd.Series(["a", "b", "b", "c"])
    points = {"a": 10.0, "b": 20.0, "c": 40.0}
    weighted = csi_table(
        reference,
        current,
        points,
        reference_weights=pd.Series([3, 1, 2], index=[7, 8, 9]),
        current_weights=[1, 2, 0, 4],
    )
    repeated = csi_table(
        ["a", "a", "a", "b", "c", "c"], ["a", "b", "b", "c", "c", "c", "c"], points
    )
    pd.testing.assert_frame_equal(weighted, repeated, check_dtype=False)
    assert weighted["reference"].dtype == float
    # Groups whose weights sum to zero in both samples are left out.
    table = csi_table(
        ["a", "b"], ["a", "b"], points, reference_weights=[1, 0], current_weights=[1, 0]
    )
    assert table.index.tolist() == ["a"]
    with pytest.raises(ValueError, match="reference_weights must have the same index as reference"):
        csi_table(reference, current, points, reference_weights=pd.Series([1, 1, 1]))


def test_series_with_other_indices_and_nullable_dtypes():
    reference = pd.Series([1, 2, 2, None], index=[30, 10, 20, 0], dtype="Int64")
    current = pd.Series([2, 2, 1], index=list("abc"), dtype="Int64")
    points = pd.Series([5.0, 8.0, 1.0], index=pd.Index([1, 2, None], dtype="Int64"))
    table = csi_table(reference, current, points)
    assert table["reference"].tolist() == [1, 2, 1] and table["current"].tolist() == [1, 2, 0]
    assert table["points"].tolist() == [5.0, 8.0, 1.0]
    np.testing.assert_allclose(table["csi"], [(1 / 3 - 1 / 4) * 5, (2 / 3 - 1 / 2) * 8, -1 / 4])
    nullable = pd.Series({"x": 1.5, "y": 2.5}, dtype="Float64")
    table = csi_table(["x", "y"], ["y", "y"], nullable)
    np.testing.assert_allclose(table["csi"], [-0.75, 1.25])


def test_points_table_rows_of_one_variable_only():
    tables = {"a": pd.Series({"lo": -0.5, "hi": 0.5}), "b": pd.Series({"lo": 0.1, "hi": -0.1})}
    table = points_table(tables, {"a": 1.0, "b": 1.0}, 0.0, **SCALE)
    with pytest.raises(ValueError, match="points must hold the groups of one variable, got 2"):
        csi_table(["lo"], ["hi"], table)
    result = csi_table(["lo", "hi"], ["hi", "hi"], table.loc[["a"]])
    assert result["points"].tolist() == table.loc[["a"], "points"].tolist()


@pytest.mark.parametrize(
    "points,message",
    [
        ([1.0, 2.0], "points must be a points_table result, a Series or a mapping"),
        (pd.DataFrame({"woe": [1.0]}), "points must have a points column, got columns"),
        ({"a": 1.0, "b": np.nan}, "points must be finite, got nan for group 'b'"),
        ({"a": 1.0, "b": np.inf}, "points must be finite, got inf for group 'b'"),
        (pd.Series({"a": 1.0, "b": None}, dtype="Float64"), "points must be finite, got <NA>"),
        ({"a": 1.0, "b": "high"}, "points must be numbers, got 'high' for group 'b'"),
        ({"a": 1.0, "b": True}, "points must be numbers, got True for group 'b'"),
        (pd.Series([1.0, 2.0], index=["a", "a"]), "points must give each group once, got 'a'"),
        (pd.Series([1.0, 2.0], index=[np.nan, None]), "points must give each group once, got nan"),
    ],
)
def test_invalid_points(points, message):
    with pytest.raises(ValueError, match=message):
        csi_table(["a", "b"], ["a", "b"], points)


@pytest.mark.parametrize("unseen_points", ["raise", np.nan, np.inf, True])
def test_invalid_unseen_points(unseen_points):
    with pytest.raises(ValueError, match="unseen_points must be a finite number, got"):
        csi_table(["a"], ["a"], {"a": 1.0}, unseen_points=unseen_points)


def test_csi_with_unseen_points_is_the_shift_of_the_average_points():
    # A new job category in the monitoring sample: woe_transform gives it WOE 0, so its
    # applicants receive the points of WOE 0, which must be passed as unseen_points.
    data = Dataset.bank_data()
    development = data.iloc[:3000]
    monitoring = data.iloc[3000:].copy()
    monitoring.iloc[:40, monitoring.columns.get_loc("job")] = "astronaut"
    table = woe_table(development["job"], development["target"])
    card = points_table({"job": table}, {"job": 0.9}, -2.0, **SCALE)

    def score(frame):
        woe = woe_transform(frame["job"], table, unseen=0.0)
        return woe_to_points(pd.DataFrame({"job": woe}), {"job": 0.9}, -2.0, **SCALE)["job"]

    shift = score(monitoring).mean() - score(development).mean()
    zero = woe_to_points(pd.DataFrame({"job": [0.0]}), {"job": 0.9}, -2.0, **SCALE)["job"][0]
    csi = csi_table(development["job"], monitoring["job"], card, unseen_points=zero)
    assert csi.loc["astronaut", "current"] == 40 and csi.loc["astronaut", "points"] == zero
    assert csi["csi"].sum() == pytest.approx(shift, rel=1e-10)
    # The WOE itself in place of its points gives a very different, wrong answer.
    wrong = csi_table(development["job"], monitoring["job"], card, unseen_points=0.0)
    assert abs(wrong["csi"].sum() - shift) > 10


@pytest.mark.parametrize(
    "reference,current,message",
    [
        ([], ["a"], "reference and current must both contain observations"),
        (["a"], np.array([["a"]]), "current must be one-dimensional, got 2 dimensions"),
    ],
)
def test_invalid_samples(reference, current, message):
    with pytest.raises(ValueError, match=message):
        csi_table(reference, current, {"a": 1.0})


def test_inputs_are_not_modified():
    reference = pd.Series(["a", "b", None])
    current = pd.Series(["b", "b", "a"])
    points = pd.Series({"a": 1.0, "b": 2.0, np.nan: 3.0})
    copies = reference.copy(), current.copy(), points.copy()
    csi_table(reference, current, points, reference_weights=[1, 2, 3])
    for original, copy in zip((reference, current, points), copies, strict=True):
        pd.testing.assert_series_equal(original, copy)


def test_tuple_labels_match_from_a_mapping():
    reference = pd.Series([("a", 1), ("b", 2)])
    current = pd.Series([("a", 1), ("a", 1)])
    table = csi_table(reference, current, {("a", 1): 10.0, ("b", 2): 20.0})
    assert table["points"].tolist() == [10.0, 20.0]
    np.testing.assert_allclose(table["csi"], [5.0, -10.0])


def _interval(left, right):
    return pd.Interval(float(left), float(right), closed="right")


EDGES = [-np.inf, 2, np.inf]
LOW, HIGH = _interval(-np.inf, 2), _interval(2, np.inf)


def _other_groups():
    """Samples that pd.concat would join by hashing interval categories with an infinite
    endpoint, which warns on NumPy < 2.2: both samples, the groups and their counts."""
    yield (
        "other edges",
        pd.cut(pd.Series([1.0, 2, 3, 4]), EDGES),
        pd.cut(pd.Series([1.0, 3, 4, 5]), [-np.inf, 3, np.inf]),
        {LOW: (2, 0), HIGH: (2, 0), _interval(-np.inf, 3): (0, 2), _interval(3, np.inf): (0, 2)},
    )
    yield (
        "unused categories removed",
        pd.cut(pd.Series([1.0, 3]), EDGES),
        pd.cut(pd.Series([1.0, 1]), EDGES).cat.remove_unused_categories(),
        {LOW: (1, 2), HIGH: (1, 0)},
    )
    yield (
        "object intervals",
        pd.cut(pd.Series([3.0, 1]), EDGES),
        pd.Series([LOW, LOW, HIGH], dtype=object),
        {HIGH: (1, 1), LOW: (1, 2)},
    )
    yield (
        "object intervals first",
        pd.Series([LOW, LOW, HIGH], dtype=object),
        pd.cut(pd.Series([3.0, 1]), EDGES),
        {LOW: (2, 1), HIGH: (1, 1)},
    )
    yield (
        "strings",
        pd.cut(pd.Series([1.0, 3, np.nan]), EDGES),
        pd.Series(["x", "x"]),
        {LOW: (1, 0), HIGH: (1, 0), np.nan: (1, 0), "x": (0, 2)},
    )
    yield (
        "strings first",
        pd.Series(["x", "x"]),
        pd.cut(pd.Series([3.0, np.nan]), EDGES),
        {"x": (2, 0), HIGH: (0, 1), np.nan: (0, 1)},
    )


@pytest.mark.parametrize(
    "reference,current,expected",
    [case[1:] for case in _other_groups()],
    ids=[case[0] for case in _other_groups()],
)
def test_interval_groups_of_other_categories_are_joined_without_hashing(
    reference, current, expected, monkeypatch
):
    def unhashable(self):
        raise AssertionError("CategoricalDtype was hashed")

    # Groups in first-observed order, as pd.concat gives them.
    labels = list(expected)
    points = {label: 10.0 * (number + 1) for number, label in enumerate(labels)}
    monkeypatch.setattr(pd.CategoricalDtype, "__hash__", unhashable)
    psi = psi_table(reference, current)
    csi = csi_table(reference, current, points)
    monkeypatch.undo()
    counts = np.array(list(expected.values()))
    for table in (psi, csi):
        # Compare the labels one by one: comparing indices hashes intervals on NumPy < 2.2.
        assert len(table) == len(labels)
        for label, expected_label in zip(table.index, labels, strict=True):
            assert (pd.isna(label) and pd.isna(expected_label)) or label == expected_label
        np.testing.assert_equal(table["reference"].to_numpy(), counts[:, 0])
        np.testing.assert_equal(table["current"].to_numpy(), counts[:, 1])
    shares = counts / counts.sum(axis=0)
    np.testing.assert_allclose(csi["csi"], (shares[:, 1] - shares[:, 0]) * list(points.values()))
    smoothed = np.where(counts == 0, 0.5, counts)
    smoothed = smoothed / smoothed.sum(axis=0)
    np.testing.assert_allclose(
        psi["psi"], (smoothed[:, 1] - smoothed[:, 0]) * np.log(smoothed[:, 1] / smoothed[:, 0])
    )


def test_interval_groups_on_other_edges_keep_the_interval_dtype():
    # As pd.concat gives them: intervals closed on the same side stay intervals.
    reference = pd.cut(pd.Series([1.0, 2, 3, 4]), EDGES)
    current = pd.cut(pd.Series([1.0, 3, 4, 5]), [-np.inf, 3, np.inf])
    assert isinstance(psi_table(reference, current).index, pd.IntervalIndex)
    assert psi_table(reference, current).index.dtype == pd.IntervalDtype("float64", "right")
    current = pd.cut(pd.Series([1.0, 3]), EDGES, right=False)
    assert psi_table(reference, current).index.dtype == object
