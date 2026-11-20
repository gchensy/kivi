import copy
import dataclasses
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import kivi
from kivi.binning import bin_edges, chimerge_edges, cut, monotonic_edges, tree_edges
from kivi.datasets import Dataset
from kivi.spec import BinningSpec, fit_spec
from kivi.woe import woe_table, woe_transform

INF = np.inf

# x has the special code -1 and two missing values; c has a missing label.
FRAME = pd.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -1.0, np.nan, np.nan],
        "c": ["a", "a", "b", "b", "a", "b", "c", "a", None, "b"],
    }
)
TARGET = pd.Series([0, 1, 0, 0, 1, 1, 1, 0, 1, 0], name="y")


def _fit(**kwargs):
    kwargs.setdefault("bins", 2)
    kwargs.setdefault("special", {"x": [-1]})
    return fit_spec(FRAME, TARGET, **kwargs)


@pytest.fixture(scope="module")
def bank():
    data = Dataset.bank_data()
    return data.iloc[:3000], data.iloc[3000:]


@pytest.fixture(scope="module")
def bank_spec(bank):
    development, _ = bank
    return fit_spec(
        development.drop(columns="target"),
        development["target"],
        method={
            "balance": "monotonic",
            "duration": "chimerge",
            "pdays": "tree",
            "campaign": "tree",
            "previous": "tree",
            "day": "uniform",
        },
        bins={"age": [-INF, 25, 35, 45, 55, 65, INF]},
        special={"pdays": [-1]},
        options={"duration": {"alpha": None}},
    )


def test_fit_matches_hand_calculation():
    spec = _fit()
    assert dict(spec.kinds) == {"x": "numeric", "c": "categorical"}
    # The median of the fitted values 1..6; -1 and the missing values are set aside.
    np.testing.assert_array_equal(spec.edges["x"], [-INF, 3.5, INF])
    assert spec.special["x"] == (-1,)
    assert list(spec.special) == ["x"]

    x = spec.tables["x"]
    assert x.index.tolist()[:3] == [-1, pd.Interval(-INF, 3.5), pd.Interval(3.5, INF)]
    assert pd.isna(x.index[3])
    assert x.index.name == "bin"
    assert x.columns.tolist() == [
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
    np.testing.assert_array_equal(x["non_event"], [1, 2, 1, 1])
    np.testing.assert_array_equal(x["event"], [1, 1, 2, 1])
    # Five events and five non-events, so the WOE is ln(event / non_event) per group.
    np.testing.assert_allclose(x["woe"], np.log([1, 0.5, 2, 1]))
    np.testing.assert_allclose(x["iv"].sum(), 2 * 0.2 * np.log(2))

    c = spec.tables["c"]
    assert c.index.tolist()[:3] == ["a", "b", "c"]
    assert pd.isna(c.index[3])
    np.testing.assert_array_equal(c["non_event"], [2, 3, 0, 0])
    np.testing.assert_array_equal(c["event"], [2, 1, 1, 1])
    # Zero non-event cells become 0.5: non-event shares 2/6, 3/6, 0.5/6, 0.5/6.
    np.testing.assert_allclose(c["woe"], np.log([0.4 * 3, 0.2 * 2, 0.2 * 12, 0.2 * 12]))
    assert spec.smoothing == 0.5
    assert spec.kivi_version == kivi.__version__
    assert spec.format_version == 1


def test_columns_follow_the_frame_or_the_argument():
    assert _fit().columns.tolist() == ["x", "c"]
    spec = _fit(columns=["c", "x"])
    assert spec.columns.tolist() == ["c", "x"]
    assert spec.cut(FRAME).columns.tolist() == ["c", "x"]
    assert _fit(columns=["c"], special=None).columns.tolist() == ["c"]


def test_cut_groups_like_binning_cut():
    spec = _fit()
    new = pd.DataFrame(
        {"x": [-1, 0.5, 3.5, 3.6, INF, None], "c": ["b", "z", None, "a", "a", "c"], "id": 1},
        index=[10, 11, 12, 13, 14, 15],
    )
    groups = spec.cut(new)
    assert groups.columns.tolist() == ["x", "c"]
    assert groups.index.tolist() == [10, 11, 12, 13, 14, 15]
    expected = cut(new["x"], [-INF, 3.5, INF], special=[-1])
    assert groups["x"].cat.categories.tolist() == expected.cat.categories.tolist()
    np.testing.assert_array_equal(groups["x"].cat.codes, [0, 1, 1, 2, 2, -1])
    # Categorical columns are returned as they are.
    pd.testing.assert_series_equal(groups["c"], new["c"])


def test_woe_transform_matches_hand_calculation():
    spec = _fit(unseen={"c": 0.0})
    new = pd.DataFrame(
        {"x": [-1, 0.5, 3.5, 3.6, INF, None], "c": ["b", "z", None, "a", "a", "c"]},
        index=[10, 11, 12, 13, 14, 15],
    )
    encoded = spec.woe_transform(new)
    assert encoded.index.tolist() == [10, 11, 12, 13, 14, 15]
    assert encoded.dtypes.tolist() == [np.float64, np.float64]
    # -1, the lower bin twice, the upper bin twice (inf included) and the missing group.
    np.testing.assert_allclose(encoded["x"], np.log([1, 0.5, 0.5, 2, 2, 1]))
    # "z" is unseen and gets 0; the missing label has its own group.
    np.testing.assert_allclose(encoded["c"], np.log([0.4, 1, 2.4, 1.2, 1.2, 2.4]))
    for column in ("x", "c"):
        groups = spec.cut(new)[column]
        np.testing.assert_array_equal(
            encoded[column], woe_transform(groups, spec.tables[column], unseen=0.0)
        )


@pytest.mark.parametrize(
    "fit,column,value,shown",
    [
        ({}, "c", "new", "'new'"),
        ({"special": {"x": [-1, -2]}}, "x", -2.0, "-2"),
        ({"bins": {"x": [-INF, 0, 3.5, INF]}}, "x", -5.0, r"Interval\(-inf, 0.0"),
    ],
)
def test_unseen_groups_raise_by_default(fit, column, value, shown):
    spec = _fit(**fit)
    with pytest.raises(
        ValueError, match=rf"column '{column}': 1 group\(s\) are not in table, such as {shown}"
    ):
        spec.woe_transform(FRAME.assign(**{column: value}).iloc[:1])


def test_missing_values_are_unseen_without_a_missing_group():
    observed = FRAME["c"].notna()
    spec = fit_spec(FRAME.loc[observed, ["c"]], TARGET[observed])
    assert not spec.tables["c"].index.hasnans
    with pytest.raises(ValueError, match="column 'c': 1 group.* such as nan"):
        spec.woe_transform(FRAME)


def test_unseen_policy_is_per_column():
    spec = _fit(special={"x": [-1, -2]}, unseen={"x": -0.25})
    assert dict(spec.unseen) == {"x": -0.25, "c": "raise"}
    new = pd.DataFrame({"x": [-2.0, 1.0], "c": ["a", "a"]})
    np.testing.assert_allclose(spec.woe_transform(new)["x"], [-0.25, np.log(0.5)])
    with pytest.raises(ValueError, match="column 'c': 1 group"):
        spec.woe_transform(new.assign(c=["a", "new"]))
    everywhere = _fit(unseen=1)
    assert dict(everywhere.unseen) == {"x": 1.0, "c": 1.0}
    assert isinstance(everywhere.unseen["x"], float)


def test_minus_infinity_is_unseen_rather_than_missing():
    new = pd.DataFrame({"x": [-INF, None], "c": ["a", "a"]})
    spec = _fit()
    assert spec.cut(new)["x"].isna().all()
    with pytest.raises(
        ValueError, match="column 'x': 1 value\\(s\\) fall outside the edges, such as -inf"
    ):
        spec.woe_transform(new)
    lenient = _fit(unseen={"x": 0.5})
    # The missing value keeps the WOE of the missing group, ln(1 / 1).
    np.testing.assert_allclose(lenient.woe_transform(new)["x"], [0.5, 0.0])


def test_minus_infinity_is_reported_without_a_missing_group():
    observed = FRAME["x"].notna()
    spec = fit_spec(FRAME.loc[observed, ["x"]], TARGET[observed], bins=2, special={"x": [-1]})
    with pytest.raises(ValueError, match="column 'x': 1 value\\(s\\) fall outside the edges"):
        spec.woe_transform(pd.DataFrame({"x": [-INF, 1.0]}))


def test_columns_accept_a_numpy_array():
    spec = _fit(columns=np.array(["c", "x"]))
    assert spec.columns.tolist() == ["c", "x"]
    assert all(type(name) is str for name in spec.kinds)
    assert repr(spec.select(np.array(["x"]))) == "BinningSpec(columns=['x'])"


def test_methods_match_the_binning_functions(bank, bank_spec):
    development, _ = bank
    target = development["target"]
    expected = {
        "balance": monotonic_edges(development["balance"], target, smoothing=0.5),
        "duration": chimerge_edges(development["duration"], target, alpha=None),
        "pdays": tree_edges(development["pdays"], target, special=[-1]),
        "previous": tree_edges(development["previous"], target),
        "day": bin_edges(development["day"], bins=5, method="uniform"),
        "age": np.array([-INF, 25, 35, 45, 55, 65, INF]),
    }
    for column, edges in expected.items():
        np.testing.assert_array_equal(bank_spec.edges[column], edges)
    np.testing.assert_array_equal(bank_spec.edges["pdays"], [-INF, 100, 177, 192, 370, INF])
    for column, edges in expected.items():
        special = [-1] if column == "pdays" else None
        table = woe_table(cut(development[column], edges, special=special), target)
        assert bank_spec.tables[column].index.tolist() == list(table.index)
        np.testing.assert_array_equal(bank_spec.tables[column].to_numpy(), table.to_numpy())
    job = woe_table(development["job"], target)
    assert bank_spec.tables["job"].index.tolist() == job.index.tolist()
    np.testing.assert_array_equal(bank_spec.tables["job"].to_numpy(), job.to_numpy())
    assert bank_spec.kinds["job"] == "categorical"
    assert list(bank_spec.special) == [
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous",
    ]


def test_options_reach_the_binning_function(bank):
    development, _ = bank
    target = development["target"]
    spec = fit_spec(
        development[["balance", "pdays"]],
        target,
        method="tree",
        bins={"balance": 3},
        special={"pdays": [-1]},
        options={"balance": {"min_bin_share": 0.2, "criterion": "gini"}},
    )
    np.testing.assert_array_equal(
        spec.edges["balance"],
        tree_edges(development["balance"], target, max_bins=3, min_bin_share=0.2, criterion="gini"),
    )
    np.testing.assert_array_equal(
        spec.edges["pdays"], tree_edges(development["pdays"], target, special=[-1])
    )


def test_monotonic_binning_uses_the_spec_smoothing(bank):
    development, _ = bank
    target = development["target"]
    for smoothing in (0.5, 5.0):
        spec = fit_spec(
            development[["duration"]],
            target,
            method="monotonic",
            bins={"duration": None},
            smoothing=smoothing,
        )
        edges = monotonic_edges(development["duration"], target, max_bins=None, smoothing=smoothing)
        np.testing.assert_array_equal(spec.edges["duration"], edges)
        woe = spec.tables["duration"]["woe"].to_numpy()
        assert (np.diff(woe) > 0).all() or (np.diff(woe) < 0).all()


def test_explicit_edges_are_used_as_given():
    spec = _fit(bins={"x": [-INF, 0, 3.5, INF]})
    np.testing.assert_array_equal(spec.edges["x"], [-INF, 0, 3.5, INF])
    # No fitted value lies in (-inf, 0], so the table leaves that interval out.
    assert spec.tables["x"].index.tolist()[:3] == [-1, pd.Interval(0, 3.5), pd.Interval(3.5, INF)]
    assert spec.cut(FRAME)["x"].cat.categories.tolist()[1] == pd.Interval(-INF, 0.0)


def test_tied_quantiles_warn_with_the_column_name(bank):
    development, _ = bank
    columns = ["age", "pdays"]
    with pytest.warns(
        UserWarning,
        match="column 'pdays': tied edges leave 1 of the 5 requested bins; the value -1 makes "
        "up 82%",
    ):
        spec = fit_spec(development[columns], development["target"])
    np.testing.assert_array_equal(spec.edges["pdays"], [-INF, INF])
    # With the code set aside, the remaining values have distinct quintiles.
    spec = fit_spec(development[columns], development["target"], special={"pdays": [-1]})
    np.testing.assert_array_equal(
        spec.edges["pdays"], bin_edges(development["pdays"], bins=5, special=[-1])
    )
    assert len(spec.edges["pdays"]) == 6


def test_column_without_values_to_fit_gets_one_interval():
    frame = pd.DataFrame(
        {
            "empty": np.full(10, np.nan),
            "coded": np.full(10, -1.0),
            "constant": np.full(10, 7.0),
            "labels": pd.Series([None] * 10, dtype=object),
        }
    )
    with pytest.warns(UserWarning) as caught:
        spec = fit_spec(frame, TARGET, method={"constant": "tree"}, special={"coded": [-1]})
    messages = [str(warning.message) for warning in caught]
    assert messages == [
        "column 'empty' has no values to fit apart from missing, special and zero-weight values; "
        "its edges are [-inf, inf]",
        "column 'coded' has no values to fit apart from missing, special and zero-weight values; "
        "its edges are [-inf, inf]",
    ]
    assert all(warning.filename == __file__ for warning in caught)
    for column in ("empty", "coded", "constant"):
        np.testing.assert_array_equal(spec.edges[column], [-INF, INF])
    # One group with every observation: shares 1 and 1, so the WOE is 0.
    for column, label in (("empty", np.nan), ("labels", np.nan), ("coded", -1)):
        table = spec.tables[column]
        assert len(table) == 1 and table["total"].iloc[0] == 10
        assert table.index[0] == label or (pd.isna(label) and pd.isna(table.index[0]))
        assert table["woe"].iloc[0] == 0.0
    assert spec.tables["constant"].index.tolist() == [pd.Interval(-INF, INF)]
    with pytest.raises(ValueError, match="column 'empty': 1 group"):
        spec.woe_transform(frame.assign(empty=3.0))


def test_integer_weights_match_repeated_rows():
    weights = np.array([1, 2, 1, 3, 1, 1, 2, 1, 1, 2])
    weighted = _fit(method="tree", weights=weights)
    repeated = fit_spec(
        FRAME.loc[FRAME.index.repeat(weights)],
        TARGET.loc[TARGET.index.repeat(weights)].rename(None),
        bins=2,
        special={"x": [-1]},
        method="tree",
    )
    np.testing.assert_array_equal(weighted.edges["x"], repeated.edges["x"])
    for column in ("x", "c"):
        assert weighted.tables[column].index.equals(repeated.tables[column].index)
        assert weighted.tables[column]["non_event"].dtype == np.float64
        np.testing.assert_array_equal(
            weighted.tables[column].to_numpy(), repeated.tables[column].to_numpy()
        )
    # c: a has weights 1, 2, 1, 1 with labels 0, 1, 1, 0; b has 1, 3, 1, 2 with 0, 0, 1, 0.
    np.testing.assert_array_equal(weighted.tables["c"]["non_event"], [2, 6, 0, 0])
    np.testing.assert_array_equal(weighted.tables["c"]["event"], [3, 1, 2, 1])
    loaded = BinningSpec.from_json(weighted.to_json())
    assert loaded == weighted
    assert loaded.tables["c"]["event"].dtype == np.float64


def test_zero_weights_and_weight_validation():
    weights = np.ones(10)
    weights[[0, 1, 2]] = 0
    spec = _fit(weights=weights, bins={"x": [-INF, 3.5, INF]})
    # The lower interval has weight zero, so it is unseen.
    assert pd.Interval(-INF, 3.5) not in spec.tables["x"].index.tolist()
    with pytest.raises(ValueError, match="weights must give both target classes"):
        _fit(weights=TARGET.to_numpy())
    with pytest.raises(ValueError, match="weights must have the same index as frame"):
        _fit(weights=pd.Series(1.0, index=range(1, 11)))
    with pytest.raises(ValueError, match="weights must have one weight per value of frame"):
        _fit(weights=[1.0, 2.0])


@pytest.mark.parametrize(
    "values,special",
    [
        (pd.array([1, 2, 3, 4, 5, 6, -1, -1, None, None], dtype="Int64"), [-1]),
        (pd.array([1.5, 2, 3, 4, 5, 6, -1, -1, None, None], dtype="Float64"), [-1]),
        (np.array([1, 2, 3, 4, 5, 6, -1, -1, np.nan, np.nan]), [-1.0]),
    ],
)
def test_nullable_numeric_dtypes(values, special):
    frame = FRAME.assign(x=values)
    spec = fit_spec(frame, TARGET, bins=2, special={"x": special})
    np.testing.assert_array_equal(spec.tables["x"]["total"], [2, 3, 3, 2])
    loaded = BinningSpec.from_json(spec.to_json())
    assert_frame_equal(loaded.woe_transform(frame), spec.woe_transform(frame))
    np.testing.assert_allclose(spec.woe_transform(frame)["x"].iloc[[0, 6, 8]], np.log([0.5, 1, 1]))


@pytest.mark.parametrize("dtype", ["string", "category", "boolean"])
def test_nullable_categorical_dtypes(dtype):
    labels = ["a", "b", None, "a"] if dtype != "boolean" else [True, False, None, True]
    frame = pd.DataFrame({"g": pd.array(labels, dtype=dtype)})
    spec = fit_spec(frame, [1, 0, 1, 0])
    assert spec.kinds["g"] == "categorical"
    first, second = labels[0], labels[1]
    assert spec.tables["g"].index.tolist()[:2] == [first, second]
    assert type(spec.tables["g"].index[0]) is type(first)
    loaded = BinningSpec.from_json(spec.to_json())
    assert loaded == spec
    encoded = loaded.woe_transform(frame)["g"]
    # Counts (non-event, event): a (1, 1), b (1, 0), missing (0, 1). With the zero cells
    # smoothed to 0.5, both classes have shares 0.4, 0.4, 0.2 in some order.
    np.testing.assert_allclose(encoded, np.log([1.0, 0.5, 2.0, 1.0]))


def test_bank_round_trip_reproduces_groups_and_woe(bank, bank_spec):
    _, monitoring = bank
    text = bank_spec.to_json()
    loaded = BinningSpec.from_json(text)
    assert loaded == bank_spec
    assert loaded.to_json() == text
    assert loaded.columns.equals(bank_spec.columns)
    assert dict(loaded.kinds) == dict(bank_spec.kinds)
    for column in bank_spec.columns:
        assert_frame_equal(loaded.tables[column], bank_spec.tables[column])
    for column in bank_spec.edges:
        np.testing.assert_array_equal(loaded.edges[column], bank_spec.edges[column])
    assert_frame_equal(loaded.cut(monitoring), bank_spec.cut(monitoring))
    encoded = loaded.woe_transform(monitoring)
    assert_frame_equal(encoded, bank_spec.woe_transform(monitoring))
    # The same values as the steps applied to one column alone.
    development, _ = bank
    edges = bank_spec.edges["pdays"]
    table = woe_table(cut(development["pdays"], edges, special=[-1]), development["target"])
    pdays = woe_transform(cut(monitoring["pdays"], edges, special=[-1]), table)
    np.testing.assert_array_equal(encoded["pdays"], pdays)
    assert encoded.index.equals(monitoring.index)
    assert encoded.columns.tolist() == bank_spec.columns.tolist()


def test_json_keeps_label_types_apart():
    frame = pd.DataFrame(
        {
            "flag": [True, False, True, False],
            "code": pd.Categorical([1, 2, 1, 2]),
            "rate": pd.Categorical([1.0, 2.5, 1.0, 2.5]),
            "text": ["1", "x", "1", "x"],
            "mixed": pd.Series([1, "1", 1, "1"], dtype=object),
        }
    )
    spec = fit_spec(frame, [1, 0, 0, 1])
    loaded = BinningSpec.from_json(spec.to_json())
    expected = {
        "flag": [True, False],
        "code": [1, 2],
        "rate": [1.0, 2.5],
        "text": ["1", "x"],
        "mixed": [1, "1"],
    }
    for column, labels in expected.items():
        for table in (spec.tables[column], loaded.tables[column]):
            assert table.index.tolist() == labels
            assert [type(label) for label in table.index] == [type(label) for label in labels]
    variables = {entry["name"]: entry for entry in json.loads(spec.to_json())["variables"]}
    assert [row["bin"] for row in variables["flag"]["table"]] == [True, False]
    assert '"bin": true' in spec.to_json() and '"bin": 1.0' in spec.to_json()
    assert '"bin": "1"' in spec.to_json() and '"bin": 1,' in spec.to_json()
    # True, 1 and 1.0 compare equal, so the JSON text decides equality.
    flipped = spec.to_dict()
    flipped["variables"][0]["table"][0]["bin"] = 1
    assert BinningSpec.from_dict(flipped) != spec


def test_json_encodes_infinities_and_the_missing_group():
    spec = _fit()
    data = json.loads(spec.to_json())
    x = data["variables"][0]
    assert x["edges"] == [{"float": "-inf"}, 3.5, {"float": "inf"}]
    assert x["special"] == [-1]
    assert [row["bin"] for row in x["table"]] == [
        -1,
        {"interval": [{"float": "-inf"}, 3.5], "closed": "right"},
        {"interval": [3.5, {"float": "inf"}], "closed": "right"},
        None,
    ]
    assert data["variables"][1]["table"][3]["bin"] is None
    assert "Infinity" not in spec.to_json() and "NaN" not in spec.to_json()
    loaded = BinningSpec.from_json(spec.to_json())
    assert pd.isna(loaded.tables["c"].index[3])
    np.testing.assert_array_equal(loaded.woe_transform(FRAME), spec.woe_transform(FRAME))


def test_json_is_deterministic():
    first, second = _fit(), _fit()
    assert first.to_json() == second.to_json()
    assert first == second and hash(first) == hash(second)
    data = first.to_dict()
    assert list(data) == ["format", "format_version", "kivi_version", "smoothing", "variables"]
    assert list(data["variables"][0]) == ["name", "kind", "edges", "special", "unseen", "table"]
    assert list(data["variables"][1]) == ["name", "kind", "unseen", "table"]
    assert list(data["variables"][0]["table"][0]) == ["bin"] + list(first.tables["x"].columns)
    assert first.to_json(indent=2) == json.dumps(data, indent=2, ensure_ascii=False)
    assert json.loads(first.to_json(indent=2)) == data
    assert first != _fit(smoothing=1.0)
    assert first.__eq__("spec") is NotImplemented


def test_non_ascii_labels_round_trip():
    frame = pd.DataFrame({"婚姻": ["已婚", "未婚", "已婚", "离异"], "x": [1, 2, 3, 4]})
    spec = fit_spec(frame, [1, 0, 0, 1], bins=2)
    text = spec.to_json()
    assert '"已婚"' in text and '"婚姻"' in text
    for source in (text, text.encode("utf-8")):
        loaded = BinningSpec.from_json(source)
        assert loaded == spec
        assert loaded.tables["婚姻"].index.tolist() == ["已婚", "未婚", "离异"]
    assert_frame_equal(BinningSpec.from_json(text).woe_transform(frame), spec.woe_transform(frame))


@pytest.mark.parametrize("closed", ["right", "left"])
def test_interval_labels_round_trip(closed):
    age = pd.Series([22, 37, 45, 61, 33, 58])
    bands = pd.cut(age, [0, 30, 60, 100], right=closed == "right")
    spec = fit_spec(pd.DataFrame({"band": bands}), [1, 0, 0, 1, 1, 0])
    assert spec.kinds["band"] == "categorical"
    intervals = [
        pd.Interval(0, 30, closed),
        pd.Interval(30, 60, closed),
        pd.Interval(60, 100, closed),
    ]
    loaded = BinningSpec.from_json(spec.to_json())
    assert loaded.tables["band"].index.tolist() == intervals
    assert all(isinstance(label.left, int) for label in loaded.tables["band"].index)
    row = json.loads(spec.to_json())["variables"][0]["table"][0]
    assert row["bin"] == {"interval": [0, 30], "closed": closed}
    new = pd.cut(pd.Series([25, 70]), [0, 30, 60, 100], right=closed == "right")
    encoded = loaded.woe_transform(pd.DataFrame({"band": new}))
    assert_frame_equal(encoded, spec.woe_transform(pd.DataFrame({"band": new})))
    # Counts (non-event, event): (0, 1), (3, 1), (0, 1). Non-event shares after smoothing
    # 0.5 / 4, 3 / 4, 0.5 / 4 and event shares 1 / 3 each, so the outer bands have ln(8 / 3).
    np.testing.assert_allclose(encoded["band"], np.log([8 / 3, 8 / 3]))
    np.testing.assert_allclose(loaded.tables["band"]["woe"], np.log([8 / 3, 4 / 9, 8 / 3]))


def test_newer_format_version_is_rejected():
    data = _fit().to_dict()
    data["format_version"] = 2
    with pytest.raises(ValueError, match="format_version 2 is newer than 1, the newest that kivi"):
        BinningSpec.from_dict(data)
    with pytest.raises(ValueError, match="upgrade kivi"):
        BinningSpec.from_json(json.dumps(data))


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_json_constants_are_rejected(token):
    text = _fit().to_json().replace("3.5", token, 1)
    with pytest.raises(ValueError, match=f"JSON must not contain {token}"):
        BinningSpec.from_json(text)


def _edit(path, value):
    """Return a function that sets a nested entry of the dict of a fitted spec."""

    def edit(data):
        target = data
        for key in path[:-1]:
            target = target[key]
        if value is _DELETE:
            del target[path[-1]]
        else:
            target[path[-1]] = value
        return data

    return edit


_DELETE = object()


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda data: [data], "data must be a mapping, got list"),
        (_edit(["format"], "other"), "format must be 'kivi.BinningSpec', got 'other'"),
        (_edit(["format_version"], 0), "format_version must be a positive integer, got 0"),
        (_edit(["format_version"], True), "format_version must be a positive integer, got True"),
        (_edit(["format_version"], "1"), "format_version must be a positive integer, got '1'"),
        (_edit(["extra"], 1), r"the specification has unknown keys \['extra'\]"),
        (_edit(["smoothing"], _DELETE), r"the specification lacks \['smoothing'\]"),
        (_edit(["smoothing"], 0), "smoothing must be finite and positive, got 0"),
        (_edit(["kivi_version"], 5), "kivi_version must be a string, got 5"),
        (_edit(["kivi_version"], None), "kivi_version must be a string, got None"),
        (_edit(["variables"], {}), "variables must be a list, got dict"),
        (_edit(["variables"], []), "kinds must be a non-empty mapping"),
        (_edit(["variables", 0], 1), "each variable must be a mapping, got int"),
        (_edit(["variables", 0, "name"], 1), "each variable must have a string name, got 1"),
        (_edit(["variables", 1, "name"], "x"), "variables must not repeat a name, got 'x' twice"),
        (_edit(["variables", 0, "kind"], "date"), "kind of column 'x' must be one of"),
        (_edit(["variables", 0, "edges"], _DELETE), r"column 'x' of the specification lacks"),
        (_edit(["variables", 1, "edges"], []), r"column 'c' of the specification has unknown"),
        (_edit(["variables", 0, "edges"], [3.5]), "edges of column 'x' must increase from -inf"),
        (_edit(["variables", 0, "edges"], "inf"), "edges of column 'x' must be a list, got str"),
        (_edit(["variables", 0, "edges", 0], {"float": "nan"}), "edges of column 'x' has an"),
        (_edit(["variables", 0, "special"], [None]), "special codes of column 'x' must be"),
        (_edit(["variables", 0, "unseen"], "zero"), "unseen of column 'x' must be 'raise' or"),
        (_edit(["variables", 0, "table"], []), "table of column 'x' must be a non-empty list"),
        (_edit(["variables", 0, "table", 0, "woe"], _DELETE), "must have the same keys"),
        (_edit(["variables", 0, "table", 0, "woe"], "0"), "must hold numbers, got woe='0'"),
        (_edit(["variables", 0, "table", 0, "woe"], True), "must hold numbers, got woe=True"),
        (_edit(["variables", 0, "table", 0, "bin"], {"x": 1}), "has an invalid value"),
        (_edit(["variables", 0, "table", 0, "bin"], -2), r"neither special codes nor .*\[-2\]"),
        (
            _edit(["variables", 0, "table", 1, "bin"], {"interval": [0, 3.5], "closed": "right"}),
            "neither special codes nor intervals of its edges",
        ),
        (_edit(["variables", 1, "table", 1, "bin"], "a"), "table of column 'c' must have one row"),
        (
            _edit(["variables", 1, "table"], [{"bin": "a", "event": 1}]),
            "table of column 'c' must have a woe column",
        ),
    ],
)
def test_from_dict_rejects_invalid_data(edit, message):
    with pytest.raises(ValueError, match=message):
        BinningSpec.from_dict(edit(_fit().to_dict()))


def test_spec_is_immutable():
    spec = _fit()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.smoothing = 1.0
    with pytest.raises(TypeError):
        spec.kinds["x"] = "categorical"
    with pytest.raises(TypeError):
        spec.unseen["x"] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        spec.edges["x"][1] = 0.0
    table = spec.tables["x"]
    table.loc[:, "woe"] = 99.0
    assert (spec.tables["x"]["woe"] != 99.0).all()
    assert spec.tables["x"] is not spec.tables["x"]
    assert len(spec.tables) == 2 and list(spec.tables) == ["x", "c"]
    # The constructor copies its inputs.
    tables = {"x": _fit().tables["x"], "c": _fit().tables["c"]}
    edges = {"x": np.array([-INF, 3.5, INF])}
    built = BinningSpec(
        kinds={"x": "numeric", "c": "categorical"},
        edges=edges,
        special={"x": [-1]},
        tables=tables,
        unseen={"x": "raise", "c": "raise"},
    )
    tables["x"].loc[:, "woe"] = 0.0
    edges["x"][1] = 0.0
    assert built == spec


def test_fit_does_not_mutate_its_inputs():
    frame, target = FRAME.copy(), TARGET.copy()
    special, bins = {"x": [-1]}, {"x": [-INF, 3.5, INF]}
    weights = np.ones(10)
    spec = fit_spec(frame, target, bins=bins, special=special, weights=weights)
    spec.woe_transform(frame)
    assert_frame_equal(frame, FRAME)
    pd.testing.assert_series_equal(target, TARGET)
    assert special == {"x": [-1]} and bins == {"x": [-INF, 3.5, INF]}
    np.testing.assert_array_equal(weights, np.ones(10))


def test_select_keeps_the_chosen_columns():
    spec = _fit(unseen={"c": 0.0})
    selected = spec.select(["c"])
    assert selected.columns.tolist() == ["c"]
    assert dict(selected.edges) == {} and dict(selected.unseen) == {"c": 0.0}
    assert_frame_equal(selected.tables["c"], spec.tables["c"])
    assert_frame_equal(selected.woe_transform(FRAME), spec.woe_transform(FRAME)[["c"]])
    assert spec.select(pd.Index(["c", "x"])).columns.tolist() == ["c", "x"]
    with pytest.raises(ValueError, match=r"columns must be in the specification, got \['z'\]"):
        spec.select(["z"])
    with pytest.raises(ValueError, match="columns must be a sequence of column names, got str"):
        spec.select("c")
    with pytest.raises(ValueError, match="columns must name at least one column"):
        spec.select([])
    with pytest.raises(ValueError, match=r"columns must not repeat a column, got \['c'\]"):
        spec.select(["c", "c"])


def test_replace_changes_the_unseen_policy():
    spec = _fit()
    lenient = dataclasses.replace(spec, unseen={**spec.unseen, "c": 0.0})
    assert lenient.unseen["c"] == 0.0 and spec.unseen["c"] == "raise"
    assert lenient.woe_transform(FRAME.assign(c="new"))["c"].tolist() == [0.0] * 10
    with pytest.raises(
        ValueError, match=r"unseen must have one entry per column, but has no entry"
    ):
        dataclasses.replace(spec, unseen={"c": 0.0})


def test_pickle_and_copy_round_trip():
    spec = _fit()
    for restored in (pickle.loads(pickle.dumps(spec)), copy.deepcopy(spec), copy.copy(spec)):
        assert restored == spec
        assert_frame_equal(restored.woe_transform(FRAME), spec.woe_transform(FRAME))


def test_repr_names_the_columns():
    assert repr(_fit()) == "BinningSpec(columns=['x', 'c'])"


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"frame": FRAME.to_dict()}, "frame must be a pandas DataFrame, got dict"),
        ({"frame": FRAME[[]]}, "frame must have at least one column to fit"),
        ({"frame": FRAME.set_axis([0, 1], axis=1)}, "column names must be strings, got 0"),
        ({"frame": FRAME.set_axis(["x", "x"], axis=1)}, r"more than one column named \['x'\]"),
        ({"columns": "x"}, "columns must be a sequence of column names, got str"),
        ({"columns": ["z"]}, r"columns must be columns of frame, got \['z'\]"),
        ({"columns": ["x", "x"]}, r"columns must not repeat a column, got \['x'\]"),
        ({"columns": [0]}, "column names must be strings, got 0"),
        ({"target": TARGET[:3]}, "frame and target must have identical indices"),
        ({"target": TARGET.to_numpy()[:3]}, "frame and target must have the same length, got 10"),
        ({"target": TARGET * 0}, "target must contain both 0 and 1"),
        ({"target": TARGET.where(TARGET > 0)}, "target must contain both 0 and 1"),
        ({"target": TARGET.rename("x")}, "columns include the target 'x'"),
        ({"method": "kmeans"}, "method must be one of 'quantile', 'uniform', 'chimerge', 'tree'"),
        ({"method": {"x": "kmeans"}}, "method of column 'x' must be one of"),
        ({"method": {"c": "tree"}}, r"method names categorical columns, .*: \['c'\]"),
        ({"method": {"z": "tree"}}, r"method names columns that are not fitted: \['z'\]"),
        ({"bins": 0}, "bins of column 'x' must be a positive integer or edges, got 0"),
        ({"bins": True}, "bins of column 'x' must be a positive integer or edges, got True"),
        ({"bins": None}, "bins of column 'x' must be a positive integer or edges, got None"),
        ({"bins": [-INF, INF]}, "bins must be an integer or a dict by column"),
        ({"bins": {"x": [0, 1, INF]}}, "edges of column 'x' must increase from -inf to inf"),
        ({"bins": {"x": [-INF, 3, 2, INF]}}, "edges of column 'x' must increase from -inf"),
        ({"bins": {"x": [-INF, INF, INF]}}, "edges of column 'x' must increase from -inf"),
        ({"bins": {"x": ["a", "b"]}}, "edges of column 'x' must be numbers"),
        (
            {"bins": {"x": [-INF, INF]}, "method": {"x": "tree"}},
            "method must not name column 'x', whose edges bins gives",
        ),
        (
            {"bins": {"x": [-INF, INF]}, "options": {"x": {"alpha": 1}}},
            "options of column 'x' must be empty when bins gives its edges",
        ),
        ({"bins": {"c": 3}}, r"bins names categorical columns"),
        ({"special": [-1]}, "special must be a dict by column, got list"),
        ({"special": {"c": ["a"]}}, r"special names categorical columns"),
        ({"special": {"x": [np.nan]}}, "special of column 'x' must be numbers or strings, not"),
        ({"special": {"x": -1}}, "special of column 'x' must be one-dimensional"),
        ({"special": {"x": "-1"}}, "special of column 'x' must be a list of codes"),
        (
            {"options": {"x": {"alpha": 1}}},
            r"bin_edges other than .* \(no options\), got \['alpha'",
        ),
        (
            {"method": "tree", "options": {"x": {"alpha": 1}}},
            r"tree_edges other than .* \(criterion, min_bin_share\), got \['alpha'\]",
        ),
        ({"method": "tree", "options": {"x": {"max_bins": 1}}}, r"got \['max_bins'\]"),
        ({"method": "monotonic", "options": {"x": {"smoothing": 1}}}, r"got \['smoothing'\]"),
        ({"method": "tree", "options": {"x": 1}}, "options of column 'x' must be a dict, got int"),
        ({"options": [1]}, "options must be a dict by column, got list"),
        ({"unseen": np.nan}, "unseen must be 'raise' or a finite number, got nan"),
        ({"unseen": "zero"}, "unseen must be 'raise' or a finite number, got 'zero'"),
        ({"unseen": {"c": True}}, "unseen of column 'c' must be 'raise' or a finite number"),
        ({"smoothing": 0}, "smoothing must be finite and positive, got 0"),
        ({"smoothing": True}, "smoothing must be finite and positive, got True"),
        ({"method": "tree", "options": {"x": {"criterion": "x"}}}, "column 'x': criterion must"),
        # Every row with an observed x is an event; only the missing rows are not.
        (
            {"method": "chimerge", "target": TARGET.where(FRAME["x"].isna(), 1)},
            "column 'x': target",
        ),
    ],
)
def test_invalid_fit_arguments(kwargs, message):
    arguments = {"frame": FRAME, "target": TARGET, "bins": 2, **kwargs}
    with pytest.raises(ValueError, match=message):
        fit_spec(arguments.pop("frame"), arguments.pop("target"), **arguments)


def test_columns_that_cannot_be_binned_name_the_column():
    with pytest.raises(ValueError, match="column 'x': reference values must contain finite"):
        fit_spec(FRAME.assign(x=INF), TARGET)
    dates = pd.DataFrame({"when": pd.to_datetime(["2024-01-01", "2024-02-01"] * 2)})
    with pytest.raises(
        ValueError,
        match="column 'when': labels must be strings, integers, floats, booleans or intervals, "
        "got Timestamp",
    ):
        fit_spec(dates, [0, 1, 1, 0])


def test_cut_and_transform_check_the_frame():
    spec = _fit()
    with pytest.raises(ValueError, match="frame must be a pandas DataFrame, got dict"):
        spec.cut({"x": [1]})
    with pytest.raises(ValueError, match=r"frame lacks columns of the specification: \['x'\]"):
        spec.woe_transform(FRAME[["c"]])
    with pytest.raises(ValueError, match=r"frame has more than one column named \['c'\]"):
        spec.cut(pd.concat([FRAME, FRAME[["c"]]], axis=1))
    with pytest.raises(ValueError, match="column 'x': "):
        spec.woe_transform(FRAME.assign(x="high"))


def test_duplicate_index_is_kept():
    frame = FRAME.set_axis([0] * 10)
    spec = fit_spec(frame, TARGET.to_numpy(), bins=2, special={"x": [-1]})
    encoded = spec.woe_transform(frame)
    assert encoded.index.tolist() == [0] * 10
    np.testing.assert_array_equal(encoded.to_numpy(), _fit().woe_transform(FRAME).to_numpy())


def test_kivi_version_is_recorded_and_kept(monkeypatch):
    spec = _fit()
    data = spec.to_dict()
    data["kivi_version"] = "0.0.1"
    assert BinningSpec.from_dict(data).kivi_version == "0.0.1"
    monkeypatch.setattr(kivi, "__version__", "9.9.9")
    assert _fit().kivi_version == "9.9.9"


def _fields(**changes):
    spec = _fit()
    fields = {
        "kinds": dict(spec.kinds),
        "edges": dict(spec.edges),
        "special": dict(spec.special),
        "tables": dict(spec.tables),
        "unseen": dict(spec.unseen),
    }
    return {**fields, **changes}


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"kinds": {}}, "kinds must be a non-empty mapping"),
        ({"kinds": {0: "numeric", "c": "categorical"}}, "column names must be strings, got 0"),
        ({"kinds": {"x": "date", "c": "categorical"}}, "kind of column 'x' must be one of"),
        ({"edges": [[-INF, INF]]}, "edges must be a mapping, got list"),
        (
            {"edges": {"x": [-INF, INF], "c": [-INF, INF]}},
            r"edges must have one entry per numeric column, but has an entry for \['c'\], which "
            "are not numeric columns",
        ),
        ({"special": {}}, r"special must have one entry per numeric column, but has no entry"),
        (
            {"tables": {"x": _fit().tables["x"], "c": [1]}},
            "table of column 'c' must be a DataFrame",
        ),
        (
            {
                "tables": {
                    "x": _fit().tables["x"],
                    "c": _fit().tables["c"].rename(columns={"iv": 1}),
                }
            },
            "table of column 'c' must have string column names other than 'bin'",
        ),
        (
            {"tables": {"x": _fit().tables["x"], "c": _fit().tables["c"].assign(woe=INF)}},
            "table of column 'c' must hold finite numbers, but column 'woe' does not",
        ),
        (
            {"tables": {"x": _fit().tables["x"], "c": _fit().tables["c"].assign(event="1")}},
            "table of column 'c' must hold finite numbers, but column 'event' does not",
        ),
        (
            {
                "tables": {
                    "x": _fit().tables["x"],
                    "c": pd.DataFrame(
                        {"woe": [0.0]},
                        index=[pd.Interval(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01"))],
                    ),
                }
            },
            "labels of the table of column 'c' must be strings, integers, floats, booleans or "
            "intervals, got Timestamp",
        ),
        ({"smoothing": -1.0}, "smoothing must be finite and positive, got -1.0"),
        ({"kivi_version": 5}, "kivi_version must be a string, got 5"),
    ],
)
def test_constructor_validates_its_fields(changes, message):
    with pytest.raises(ValueError, match=message):
        BinningSpec(**_fields(**changes))


def test_constructor_accepts_a_series_of_woe_values_as_table():
    tables = {"x": _fit().tables["x"], "c": _fit().tables["c"]["woe"].to_frame()}
    spec = BinningSpec(**_fields(tables=tables))
    assert spec.tables["c"].columns.tolist() == ["woe"]
    np.testing.assert_allclose(spec.woe_transform(FRAME)["c"], _fit().woe_transform(FRAME)["c"])
    assert repr(spec.tables) == "<tables of ['x', 'c']>"
    data = spec.to_dict()
    data["variables"][1]["table"] = [1]
    with pytest.raises(ValueError, match="must have a mapping with a bin for each group"):
        BinningSpec.from_dict(data)
