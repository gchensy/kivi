"""Grouped categorical columns in fit_spec and BinningSpec."""

import copy
import dataclasses
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kivi.binning import category_groups, chimerge_groups, group_categories
from kivi.datasets import Dataset
from kivi.spec import BinningSpec, fit_spec
from kivi.woe import woe_table

INF = np.inf

# The frame of tests/test_spec.py: x has the special code -1, c a missing label.
# c counts (non-event, event): a (2, 2), b (3, 1), c (0, 1) and missing (0, 1).
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


def _chimerge(**kwargs):
    return _fit(method={"c": "chimerge"}, options={"c": {"alpha": None}}, **kwargs)


def _rare(**kwargs):
    return _fit(method={"c": "rare"}, options={"c": {"min_share": 0.2}}, **kwargs)


def _labels(index):
    return ["<NA>" if pd.isna(label) else label for label in index]


@pytest.fixture(scope="module")
def bank():
    data = Dataset.bank_data()
    return data.iloc[:3000], data.iloc[3000:]


def test_chimerge_grouping_matches_hand_calculation():
    spec = _chimerge()
    # Sorted by WOE: b (3, 1), a (2, 2), c (0, 1). The pair (b, a) has the smaller chi-square,
    # 8 * 4**2 / (4 * 4 * 5 * 3) = 0.53 against 5 * 2**2 / (4 * 1 * 2 * 3) = 0.83.
    assert spec.groups["c"] == {"b": "b, a", "a": "b, a", "c": "c"}
    table = spec.tables["c"]
    assert _labels(table.index) == ["b, a", "c", "<NA>"]
    np.testing.assert_array_equal(table["non_event"], [5, 0, 0])
    np.testing.assert_array_equal(table["event"], [3, 1, 1])
    # Zero non-event cells become 0.5: non-event shares 5/6, 0.5/6, 0.5/6.
    np.testing.assert_allclose(table["woe"], np.log([18 / 25, 12 / 5, 12 / 5]))
    assert spec.format_version == 2


def test_the_grouping_and_table_are_those_of_the_public_functions():
    spec = _chimerge()
    groups = chimerge_groups(FRAME["c"], TARGET, max_groups=2, alpha=None)
    assert list(spec.groups["c"].items()) == list(groups.items())
    expected = woe_table(group_categories(FRAME["c"], groups), TARGET)
    np.testing.assert_allclose(spec.tables["c"].to_numpy(), expected.to_numpy())
    assert _labels(spec.tables["c"].index) == _labels(expected.index)
    rare = _rare()
    assert rare.groups["c"] == category_groups(FRAME["c"], min_share=0.2)
    assert rare.groups["c"] == {"a": "a", "b": "b", "c": "other"}
    assert _labels(rare.tables["c"].index) == ["a", "b", "other", "<NA>"]
    np.testing.assert_array_equal(rare.tables["c"]["event"], [2, 1, 1, 1])


def test_columns_without_a_grouping_are_fitted_as_before():
    plain = _fit()
    for spec in (_chimerge(), _rare()):
        assert_frame_equal(spec.tables["x"], plain.tables["x"])
        np.testing.assert_array_equal(spec.edges["x"], plain.edges["x"])
        assert list(spec.groups) == ["c"]
    assert dict(plain.groups) == {} and plain.format_version == 1
    data = plain.to_dict()
    assert data["format_version"] == 1
    assert all("groups" not in entry for entry in data["variables"])
    # A string method still applies to the numeric columns only.
    assert dict(_fit(method="chimerge").groups) == {}


def test_bins_sets_the_number_of_groups_and_options_reach_the_grouping():
    assert set(_chimerge(bins=1).groups["c"].values()) == {"b, a, c"}
    assert _chimerge(bins={"x": 2, "c": 3}).groups["c"] == {"b": "b", "a": "a", "c": "c"}
    assert np.array_equal(_chimerge(bins={"c": 3}).edges["x"], _fit(bins=5).edges["x"])
    # With alpha=0.05, every merge is insignificant and one group remains.
    spec = _fit(method={"c": "chimerge"}, options={"c": {"min_group_share": 0}})
    assert set(spec.groups["c"].values()) == {"b, a, c"}
    spec = _fit(method={"c": "rare"}, options={"c": {"min_count": 2, "rare_label": "few"}})
    assert spec.groups["c"] == {"a": "a", "b": "b", "c": "few"}
    # The default min_share of 0.05 pools nothing here.
    assert _fit(method={"c": "rare"}).groups["c"] == {"a": "a", "b": "b", "c": "c"}


def test_special_codes_of_a_grouped_column_keep_their_own_group():
    spec = _rare(special={"x": [-1], "c": ["c", "z"]})
    assert list(spec.groups["c"].items()) == [("c", "c"), ("z", "z"), ("a", "a"), ("b", "b")]
    assert _labels(spec.tables["c"].index) == ["c", "a", "b", "<NA>"]
    spec = _chimerge(special={"x": [-1], "c": ["c"]}, bins={"x": 2, "c": 1})
    assert spec.groups["c"] == {"c": "c", "b": "b, a", "a": "b, a"}


def test_cut_maps_grouped_columns_and_keeps_unmapped_categories():
    spec = _chimerge()
    sample = pd.DataFrame(
        {"x": [1.0, 7.0, 2.0, 3.0], "c": ["a", "new", None, "c"]}, index=[4, 3, 2, 1]
    )
    groups = spec.cut(sample)
    assert groups.index.equals(sample.index)
    assert groups["c"].cat.categories.tolist() == ["b, a", "c", "new"]
    assert groups["c"].cat.codes.tolist() == [0, 2, -1, 1]
    # Without unmapped categories the categories are the groups alone.
    assert spec.cut(FRAME)["c"].cat.categories.tolist() == ["b, a", "c"]
    # A category that is also the label of a group would make its group ambiguous.
    with pytest.raises(
        ValueError,
        match="column 'c': the category 'b, a' is not in its grouping but is the label of one",
    ):
        spec.cut(pd.DataFrame({"x": [1.0], "c": ["b, a"]}))
    with pytest.raises(ValueError, match="the category 'other' is not in its grouping"):
        _rare().cut(pd.DataFrame({"x": [1.0], "c": ["other"]}))


def test_woe_transform_gives_unmapped_categories_the_unseen_policy():
    spec = _chimerge()
    sample = pd.DataFrame({"x": [1.0, 2.0, 3.0], "c": ["new", "a", "b, a"]})
    with pytest.raises(
        ValueError,
        match=r"column 'c': 2 value\(s\) have categories that its grouping does not map, such "
        "as 'new'; give the column an unseen WOE",
    ):
        spec.woe_transform(sample)
    lenient = dataclasses.replace(spec, unseen={"x": "raise", "c": -9.0})
    encoded = lenient.woe_transform(sample)["c"]
    # 'b, a' is the label of a group but not a category, so it is unseen as well.
    np.testing.assert_allclose(encoded, [-9.0, np.log(18 / 25), -9.0])
    missing = lenient.woe_transform(pd.DataFrame({"x": [1.0], "c": [None]}))["c"]
    np.testing.assert_allclose(missing, [np.log(12 / 5)])
    np.testing.assert_allclose(
        spec.woe_transform(FRAME)["c"], spec.tables["c"]["woe"].iloc[[0, 0, 0, 0, 0, 0, 1, 0, 2, 0]]
    )


def test_grouped_specifications_round_trip_through_json():
    spec = _chimerge(special={"x": [-1], "c": [7, "z"]}, bins={"x": 2, "c": 1})
    data = spec.to_dict()
    assert data["format_version"] == 2
    entry = data["variables"][1]
    assert list(entry) == ["name", "kind", "groups", "unseen", "table"]
    assert entry["groups"] == [
        {"group": 7, "categories": [7]},
        {"group": "z", "categories": ["z"]},
        {"group": "b, a, c", "categories": ["b", "a", "c"]},
    ]
    text = spec.to_json(indent=2)
    loaded = BinningSpec.from_json(text)
    assert loaded == spec and loaded.to_json(indent=2) == text
    assert list(loaded.groups["c"].items()) == list(spec.groups["c"].items())
    assert [type(category) for category in loaded.groups["c"]][:2] == [int, str]
    assert_frame_equal(loaded.woe_transform(FRAME), spec.woe_transform(FRAME))
    assert pickle.loads(pickle.dumps(spec)) == spec
    assert copy.deepcopy(spec) == spec and hash(loaded) == hash(spec)


def test_version_1_files_are_read_unchanged_and_cannot_hold_groups():
    text = _fit().to_json()
    assert json.loads(text)["format_version"] == 1
    assert BinningSpec.from_json(text) == _fit()
    data = _chimerge().to_dict()
    data["format_version"] = 1
    with pytest.raises(ValueError, match=r"column 'c' of the specification has unknown keys"):
        BinningSpec.from_dict(data)
    # A version 2 file without groups describes an ungrouped specification.
    data = _fit().to_dict()
    data["format_version"] = 2
    assert BinningSpec.from_dict(data) == _fit()


def _edit_groups(value):
    def edit(data):
        data["variables"][1]["groups"] = value
        return data

    return edit


@pytest.mark.parametrize(
    "groups,message",
    [
        ({"b": "g"}, r"groups of column 'c' must be a list, got dict"),
        ([["b", "g"]], "must have a mapping of a group and its categories per group"),
        ([{"group": "g"}], "must have a mapping of a group and its categories per group"),
        ([{"group": "g", "categories": []}], "must list at least one category for each group"),
        ([{"group": "g", "categories": "b"}], "must list at least one category for each group"),
        (
            [{"group": "g", "categories": ["a"]}, {"group": "g", "categories": ["b"]}],
            "groups of column 'c' must not repeat a group, got 'g' twice",
        ),
        (
            [{"group": "g", "categories": ["a", "b"]}, {"group": "h", "categories": ["a"]}],
            "groups of column 'c' must not repeat a category",
        ),
        ([{"group": None, "categories": ["a"]}], "must not have missing categories or groups"),
        ([{"group": "g", "categories": [None]}], "must not have missing categories or groups"),
        ([{"group": {"x": 1}, "categories": ["a"]}], "groups of column 'c' has an invalid value"),
        (
            [{"group": "g", "categories": ["a", "b", "c"]}],
            r"table of column 'c' has groups that its grouping lacks: \['b, a', 'c'\]",
        ),
    ],
)
def test_damaged_groups_are_rejected(groups, message):
    data = _edit_groups(groups)(_chimerge().to_dict())
    with pytest.raises(ValueError, match=message):
        BinningSpec.from_dict(data)


def _fields(**changes):
    spec = _chimerge()
    fields = {
        "kinds": dict(spec.kinds),
        "edges": dict(spec.edges),
        "special": dict(spec.special),
        "tables": dict(spec.tables),
        "unseen": dict(spec.unseen),
        "groups": dict(spec.groups),
    }
    return {**fields, **changes}


@pytest.mark.parametrize(
    "groups,message",
    [
        ([1], "groups must be a mapping, got list"),
        ({"x": {1.0: 1.0}}, r"groups must have entries for categorical columns only, .*\['x'\]"),
        ({"z": {}}, r"groups must have entries for categorical columns only, .*\['z'\]"),
        ({"c": ["a"]}, "grouping of column 'c' must be a mapping from category to group, got list"),
        ({"c": {None: "b, a"}}, "grouping of column 'c' must not have missing categories"),
        ({"c": {"a": np.nan}}, "grouping of column 'c' must not have missing categories"),
        (
            {"c": {pd.Timestamp("2024-01-01"): "b, a"}},
            "categories of the grouping of column 'c' must be strings, integers, floats",
        ),
        ({"c": {"a": "g"}}, r"table of column 'c' has groups that its grouping lacks"),
    ],
)
def test_constructor_validates_the_groupings(groups, message):
    with pytest.raises(ValueError, match=message):
        BinningSpec(**_fields(groups=groups))


def test_groupings_are_canonical_copies():
    spec = _chimerge()
    edited = {"c": "c", "b": "b, a", "a": "b, a"}
    replaced = dataclasses.replace(spec, groups={"c": edited})
    # The categories of each group are kept together, in the order of the groups.
    assert list(replaced.groups["c"].items()) == [("c", "c"), ("b", "b, a"), ("a", "b, a")]
    assert BinningSpec(**_fields(groups={"c": {"a": "b, a", "c": "c", "b": "b, a"}})).groups[
        "c"
    ] == {"a": "b, a", "b": "b, a", "c": "c"}
    assert list(
        BinningSpec(**_fields(groups={"c": {"a": "b, a", "c": "c", "b": "b, a"}})).groups["c"]
    ) == ["a", "b", "c"]
    copy_ = spec.groups["c"]
    copy_["a"] = "c"
    assert spec.groups["c"]["a"] == "b, a"
    with pytest.raises(TypeError):
        spec.groups["c"] = {}
    numpy_labels = {np.str_("a"): np.str_("b, a"), "b": "b, a", "c": "c"}
    category, group = next(
        iter(BinningSpec(**_fields(groups={"c": numpy_labels})).groups["c"].items())
    )
    assert type(category) is str and type(group) is str


def test_select_keeps_the_groupings():
    spec = _chimerge()
    assert spec.select(["c"]).groups["c"] == spec.groups["c"]
    assert spec.select(["c"]).format_version == 2
    plain = spec.select(["x"])
    assert dict(plain.groups) == {} and plain.format_version == 1


def test_integer_weights_repeat_rows():
    weights = pd.Series([1, 2, 0, 3, 1, 1, 2, 1, 1, 2])
    rows = FRAME.index.repeat(weights)
    repeated_frame = FRAME.loc[rows].reset_index(drop=True)
    repeated_target = TARGET.loc[rows].reset_index(drop=True)
    for method, options in (("chimerge", {"alpha": None}), ("rare", {"min_share": 0.15})):
        kwargs = {
            "bins": 2,
            "special": {"x": [-1]},
            "method": {"c": method},
            "options": {"c": options},
        }
        weighted = fit_spec(FRAME, TARGET, weights=weights, **kwargs)
        plain = fit_spec(repeated_frame, repeated_target, **kwargs)
        assert list(weighted.groups["c"].items()) == list(plain.groups["c"].items())
        np.testing.assert_allclose(weighted.tables["c"].to_numpy(), plain.tables["c"].to_numpy())


def test_categories_with_zero_weight_are_left_out_of_the_grouping():
    weights = np.where(FRAME["c"] == "c", 0, 1)
    spec = _rare(weights=weights)
    assert spec.groups["c"] == {"a": "a", "b": "b"}
    assert _labels(spec.tables["c"].index) == ["a", "b", "<NA>"]
    with pytest.raises(ValueError, match="have categories that its grouping does not map"):
        spec.woe_transform(FRAME)
    assert spec.cut(FRAME)["c"].cat.categories.tolist() == ["a", "b", "c"]


def test_a_grouped_column_without_values_to_group_warns():
    frame = FRAME.assign(c=["s"] * 4 + [None] * 6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = fit_spec(
            frame, TARGET, method={"c": "rare"}, special={"c": ["s"]}, bins=2, columns=["c"]
        )
    assert [str(warning.message) for warning in caught] == [
        "column 'c' has no values to group apart from missing, special and zero-weight values; "
        "its groups are its special codes"
    ]
    assert spec.groups["c"] == {"s": "s"}
    assert BinningSpec.from_json(spec.to_json()) == spec


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"method": {"c": 1}}, r"method names categorical columns, which can only be grouped"),
        ({"method": {"c": "quantile"}}, r"with 'chimerge' or 'rare': \['c'\]"),
        ({"method": "rare"}, "method must be one of 'quantile', 'uniform', 'chimerge', 'tree'"),
        ({"method": {"x": "rare"}}, "method of column 'x' must be one of 'quantile'"),
        ({"bins": {"z": 2}}, r"bins names columns that are not fitted: \['z'\]"),
        (
            {"method": {"c": "rare"}, "bins": {"c": 2}},
            r"bins names columns grouped with 'rare', which take min_share or min_count in "
            r"options instead: \['c'\]",
        ),
        (
            {"method": {"c": "chimerge"}, "bins": {"c": [-INF, INF]}},
            "bins of column 'c' must be a positive integer with 'chimerge', got",
        ),
        (
            {"method": {"c": "chimerge"}, "bins": {"c": None}},
            "bins of column 'c' must be a positive integer with 'chimerge', got None",
        ),
        (
            {"method": {"c": "chimerge"}, "options": {"c": {"max_groups": 2}}},
            r"options of column 'c' must be arguments of chimerge_groups other than those "
            r"fit_spec sets \(alpha, min_group_share\), got \['max_groups'\]",
        ),
        (
            {"method": {"c": "rare"}, "options": {"c": {"special": ["a"]}}},
            r"category_groups other than .* \(min_count, min_share, rare_label\)",
        ),
        ({"method": {"c": "rare"}, "options": {"c": 1}}, "options of column 'c' must be a dict"),
        ({"options": {"c": {}}}, r"options names categorical columns, which keep their labels"),
        (
            {"special": {"c": ["a"]}},
            r"special names categorical columns, which keep their labels unless method groups "
            r"them: \['c'\]",
        ),
        (
            {"method": {"c": "rare"}, "special": {"c": [True]}},
            "special of column 'c' must be non-missing labels other than booleans, got True",
        ),
        (
            {"method": {"c": "rare"}, "special": {"c": [None]}},
            "special of column 'c' must be non-missing labels other than booleans, got None",
        ),
        (
            {"method": {"c": "rare"}, "special": {"c": "a"}},
            "special of column 'c' must be a list of codes, got 'a'",
        ),
        (
            {"method": {"c": "rare"}, "special": {"c": [pd.Timestamp("2024-01-01")]}},
            "special of column 'c' must be strings, integers, floats, booleans or intervals",
        ),
        (
            {"method": {"c": "rare"}, "options": {"c": {"min_share": 2}}},
            "column 'c': min_share must be between 0 and 1, got 2",
        ),
        (
            {"method": {"c": "rare"}, "options": {"c": {"rare_label": "a", "min_share": 0.2}}},
            "column 'c': rare_label 'a' is the label of a category that is not rare",
        ),
    ],
)
def test_invalid_grouping_arguments(kwargs, message):
    kwargs = {"bins": 2, **kwargs}
    with pytest.raises(ValueError, match=message):
        fit_spec(FRAME, TARGET, **kwargs)


def test_chimerge_needs_both_classes_in_the_grouped_rows():
    # Only the row with a missing label has an event.
    target = TARGET.mask(FRAME["c"].notna(), 0)
    with pytest.raises(
        ValueError, match="column 'c': target must contain both 0 and 1 where values are observed$"
    ):
        fit_spec(FRAME, target, method={"c": "chimerge"}, bins=2)


def test_bank_categories_group_and_encode_the_later_sample(bank):
    development, monitoring = bank
    spec = fit_spec(
        development,
        development["target"],
        columns=["balance", "job", "education", "month", "housing"],
        method={"job": "chimerge", "month": "chimerge", "education": "rare"},
        special={"job": ["unknown"]},
        options={"education": {"min_share": 0.1}},
    )
    assert list(spec.groups) == ["job", "education", "month"]
    assert spec.groups["education"]["unknown"] == "other"
    for name in spec.groups:
        groups = spec.groups[name]
        assert set(groups) == set(development[name])
        assert spec.tables[name]["total"].sum() == len(development)
    assert spec.tables["job"].index[0] == "unknown"
    encoded = spec.woe_transform(monitoring)
    assert np.isfinite(encoded.to_numpy()).all()
    assert BinningSpec.from_json(spec.to_json()) == spec
    assert_frame_equal(
        spec.select(["balance", "housing"]).tables["housing"], spec.tables["housing"]
    )


def test_the_newest_format_version_is_a_class_constant():
    assert BinningSpec.FORMAT_VERSION == 2
    assert _fit().format_version == 1 and _chimerge().format_version == 2
    data = _chimerge().to_dict()
    data["format_version"] = BinningSpec.FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="format_version 3 is newer than 2"):
        BinningSpec.from_dict(data)


def test_an_unseen_group_receives_new_categories():
    spec = _rare(unseen={"x": "raise", "c": {"group": "other"}})
    assert spec.unseen["c"] == {"group": "other"}
    sample = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "c": ["new", "a", "other", None]})
    groups = spec.cut(sample)["c"]
    # Unmapped categories, including one named like the group, join the group; no new category.
    assert groups.cat.categories.tolist() == ["a", "b", "other"]
    assert groups.tolist()[:3] == ["other", "a", "other"] and pd.isna(groups.iloc[3])
    table = spec.tables["c"]["woe"]
    encoded = spec.woe_transform(sample)["c"]
    np.testing.assert_allclose(
        encoded, [table.iloc[2], table.iloc[0], table.iloc[2], table.iloc[3]]
    )
    # Without a missing group in development, missing values get the group's WOE too.
    dropped = _rare(
        unseen={"x": "raise", "c": {"group": "other"}}, weights=FRAME["c"].notna().astype(int)
    )
    assert not pd.isna(dropped.tables["c"].index).any()
    missing = dropped.woe_transform(pd.DataFrame({"x": [1.0], "c": [None]}))["c"]
    np.testing.assert_allclose(missing, [dropped.tables["c"]["woe"].iloc[2]])


def test_the_unseen_group_follows_a_refit_and_round_trips():
    spec = _rare(unseen={"x": "raise", "c": {"group": "other"}})
    entry = spec.to_dict()["variables"][1]
    assert entry["unseen"] == {"group": "other"}
    loaded = BinningSpec.from_json(spec.to_json())
    assert loaded == spec and loaded.unseen["c"] == {"group": "other"}
    assert pickle.loads(pickle.dumps(spec)) == spec
    copy_ = spec.unseen["c"]
    copy_["group"] = "a"
    assert spec.unseen["c"] == {"group": "other"}
    # An interval group is encoded like any label.
    interval = pd.Interval(0.0, 1.0)
    groups = {"a": interval, "b": "b", "c": "b"}
    table = woe_table(group_categories(FRAME["c"], groups), TARGET)
    replaced = dataclasses.replace(
        spec,
        groups={"c": groups},
        tables={**spec.tables, "c": table},
        unseen={"x": "raise", "c": {"group": interval}},
    )
    assert BinningSpec.from_json(replaced.to_json()).unseen["c"] == {"group": interval}


@pytest.mark.parametrize(
    "unseen,message",
    [
        ({"c": {"group": "zzz"}}, "unseen of column 'c' names the group 'zzz', which its table"),
        ({"x": {"group": 1.0}}, "unseen of column 'x' names a group, which only grouped"),
        ({"c": {"group": None}}, "unseen of column 'c' must name a non-missing group"),
        ({"c": {"name": "other"}}, "unseen of column 'c' must be 'raise' or a finite number"),
        ({"c": {"group": [1]}}, "the group of unseen of column 'c' must be strings"),
    ],
)
def test_invalid_unseen_groups(unseen, message):
    with pytest.raises(ValueError, match=message):
        _rare(unseen=unseen)


def test_an_unseen_group_needs_a_grouped_column_in_the_file():
    data = _rare(unseen={"x": "raise", "c": {"group": "other"}}).to_dict()
    del data["variables"][1]["groups"]
    data["variables"][1]["table"] = _fit().to_dict()["variables"][1]["table"]
    with pytest.raises(ValueError, match="names a group, which only grouped categorical columns"):
        BinningSpec.from_dict(data)
    data["variables"][1]["unseen"] = {"group": {"bad": 1}}
    with pytest.raises(ValueError, match="unseen of column 'c' has an invalid value"):
        BinningSpec.from_dict(data)
