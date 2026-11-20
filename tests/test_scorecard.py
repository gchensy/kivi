import math

import numpy as np
import pandas as pd
import pytest

from kivi.datasets import Dataset
from kivi.scorecard import (
    pd_to_score,
    points_table,
    reason_codes,
    score_scaling,
    score_to_pd,
    woe_to_points,
)
from kivi.woe import woe_table, woe_transform

SCALE = {"pdo": 20, "base_score": 600, "base_odds": 50}
# factor = 20 / ln 2 and offset = 600 - factor * ln 50, worked out by hand.
FACTOR = 28.85390081777927
OFFSET = 487.1228762045055

TABLES = {
    "housing": pd.Series({"no": -0.4, "yes": 0.3}),
    "contact": pd.Series({"cellular": -0.2, "unknown": 0.6}),
}
COEFFICIENTS = {"housing": 0.8, "contact": 0.5}
INTERCEPT = -2.0


def test_scaling_constants():
    factor, offset = score_scaling(**SCALE)
    assert factor == pytest.approx(FACTOR, rel=1e-15)
    assert offset == pytest.approx(OFFSET, rel=1e-15)


def test_scores_follow_base_odds_and_pdo():
    # Odds of 50, 100 and 25 to 1 give the base score and the base score plus or minus pdo.
    scores = pd_to_score(np.array([1 / 51, 1 / 101, 1 / 26]), **SCALE)
    np.testing.assert_allclose(scores, [600, 620, 580], rtol=1e-13)
    # Odds of 9 to 1.
    assert pd_to_score(0.1, **SCALE) == pytest.approx(OFFSET + FACTOR * math.log(9), rel=1e-15)


def test_score_and_pd_are_inverse():
    prob = np.array([1e-9, 0.001, 0.02, 0.5, 0.97, 1 - 1e-9])
    np.testing.assert_allclose(score_to_pd(pd_to_score(prob, **SCALE), **SCALE), prob, rtol=1e-6)
    np.testing.assert_allclose(score_to_pd([580, 600, 620], **SCALE), [1 / 26, 1 / 51, 1 / 101])


def test_extreme_scores_do_not_overflow():
    np.testing.assert_equal(score_to_pd([-1e6, 1e6, -np.inf, np.inf], **SCALE), [1, 0, 1, 0])


def test_conversions_keep_series_shapes_and_missing_values():
    prob = pd.Series([0.02, np.nan], index=[7, 3], name="pd")
    scores = pd_to_score(prob, **SCALE)
    assert isinstance(scores, pd.Series)
    assert scores.name == "pd" and scores.index.tolist() == [7, 3]
    assert np.isnan(scores[3])
    back = score_to_pd(scores, **SCALE)
    assert back.index.tolist() == [7, 3] and back[7] == pytest.approx(0.02)
    assert isinstance(pd_to_score(0.02, **SCALE), float)
    assert isinstance(score_to_pd(600, **SCALE), float)
    assert pd_to_score(np.full((2, 3), 0.5), **SCALE).shape == (2, 3)


@pytest.mark.parametrize("prob,message", [(0, "got 0.0"), (1, "got 1.0"), ([0.5, 1.5], "got 1.5")])
def test_invalid_probabilities(prob, message):
    with pytest.raises(ValueError, match=f"prob must be strictly between 0 and 1, {message}"):
        pd_to_score(prob, **SCALE)


@pytest.mark.parametrize(
    "scale,message",
    [
        ({"pdo": 0}, "pdo must be positive, got 0"),
        ({"pdo": -20}, "pdo must be positive, got -20"),
        ({"base_odds": 0}, "base_odds must be positive, got 0"),
        ({"base_score": np.nan}, "base_score must be a finite number, got nan"),
        ({"pdo": True}, "pdo must be a finite number, got True"),
        ({"base_odds": "50"}, "base_odds must be a finite number, got '50'"),
    ],
)
def test_invalid_scaling(scale, message):
    with pytest.raises(ValueError, match=message):
        score_scaling(**{**SCALE, **scale})
    with pytest.raises(ValueError, match=message):
        score_to_pd(600, **{**SCALE, **scale})


def test_points_match_hand_calculation():
    table = points_table(TABLES, COEFFICIENTS, INTERCEPT, **SCALE)
    assert table.index.names == ["variable", "bin"]
    assert table.columns.tolist() == ["woe", "points"]
    assert table.index.tolist() == [
        ("housing", "no"),
        ("housing", "yes"),
        ("contact", "cellular"),
        ("contact", "unknown"),
    ]
    # Each variable carries half of offset - factor * intercept.
    base = (OFFSET + 2 * FACTOR) / 2
    expected = [
        base + FACTOR * 0.8 * 0.4,
        base - FACTOR * 0.8 * 0.3,
        base + FACTOR * 0.5 * 0.2,
        base - FACTOR * 0.5 * 0.6,
    ]
    np.testing.assert_allclose(table["points"], expected, rtol=1e-14)
    np.testing.assert_allclose(table["points"], [281.6486, 265.4905, 275.3008, 263.7592], atol=1e-4)
    np.testing.assert_equal(table["woe"].to_numpy(), [-0.4, 0.3, -0.2, 0.6])


def test_points_add_up_to_the_model_score():
    table = points_table(TABLES, COEFFICIENTS, INTERCEPT, **SCALE)
    total = table.loc[("housing", "yes"), "points"] + table.loc[("contact", "unknown"), "points"]
    log_odds = INTERCEPT + 0.8 * 0.3 + 0.5 * 0.6
    prob = 1 / (1 + math.exp(-log_odds))
    assert total == pytest.approx(pd_to_score(prob, **SCALE), rel=1e-14)


def test_coefficients_set_the_order_and_extra_tables_are_ignored():
    tables = {**TABLES, "unused": pd.Series({"x": 1.0})}
    coefficients = pd.Series({"contact": 0.5, "housing": 0.8})
    table = points_table(tables, coefficients, INTERCEPT, **SCALE)
    assert table.index.get_level_values("variable").unique().tolist() == ["contact", "housing"]
    np.testing.assert_allclose(
        table.loc["housing", "points"],
        points_table(TABLES, COEFFICIENTS, INTERCEPT, **SCALE).loc["housing", "points"],
    )


def _bank_model():
    data = Dataset.bank_data()
    development = data.iloc[:3000]
    edges = np.array([-np.inf, 25, 35, 45, 55, 65, np.inf])

    def to_groups(frame):
        return pd.DataFrame(
            {
                "age": pd.cut(frame["age"], edges),
                "housing": frame["housing"],
                "loan": frame["loan"],
                "poutcome": frame["poutcome"],
            }
        )

    groups = to_groups(development)
    tables = {column: woe_table(groups[column], development["target"]) for column in groups}
    coefficients = {"age": 0.7, "housing": 0.7, "loan": 0.8, "poutcome": 0.9}
    return data, to_groups, tables, coefficients


def test_points_table_on_interval_and_shared_labels():
    _, _, tables, coefficients = _bank_model()
    table = points_table(tables, coefficients, -2.0, **SCALE)
    # housing and loan share the labels "no" and "yes"; age has interval labels.
    assert len(table) == 6 + 2 + 2 + 4
    assert table.index.levels[1].dtype == object
    assert table.loc["age"].index[0] == pd.Interval(-np.inf, 25.0)
    loan_yes = 0.8 * tables["loan"].loc["yes", "woe"]
    housing_yes = 0.7 * tables["housing"].loc["yes", "woe"]
    base = (OFFSET + 2 * FACTOR) / 4
    assert table.loc[("loan", "yes"), "points"] == pytest.approx(base - FACTOR * loan_yes)
    assert table.loc[("housing", "yes"), "points"] == pytest.approx(base - FACTOR * housing_yes)


def test_missing_group_gets_points():
    groups = pd.Series(["a", None, "b", "a", None, "b"])
    target = pd.Series([1, 0, 0, 0, 1, 0])
    table = points_table({"x": woe_table(groups, target)}, {"x": 1.0}, 0.0, **SCALE)
    labels = table.index.get_level_values("bin")
    assert labels[0] == "a" and pd.isna(labels[1]) and labels[2] == "b"
    # Event shares 0.4, 0.4, 0.2 (b's zero count smoothed to 0.5); non-event shares 1/4, 1/4, 1/2.
    np.testing.assert_allclose(table["woe"], np.log([1.6, 1.6, 0.4]))
    np.testing.assert_allclose(table["points"], OFFSET - FACTOR * np.log([1.6, 1.6, 0.4]))


def test_applicant_points_match_the_points_table():
    data, to_groups, tables, coefficients = _bank_model()
    current = data.iloc[3000:]
    groups = to_groups(current)
    woe = pd.DataFrame({column: woe_transform(groups[column], tables[column]) for column in groups})
    points = woe_to_points(woe, coefficients, -2.0, **SCALE)
    assert points.index.equals(current.index)
    table = points_table(tables, coefficients, -2.0, **SCALE)
    # Interval groups are compared by their codes, which follow the table rows.
    np.testing.assert_array_equal(
        points["age"].to_numpy(), table.loc["age", "points"].to_numpy()[groups["age"].cat.codes]
    )
    for column in ["housing", "loan", "poutcome"]:
        lookup = table.loc[column, "points"]
        np.testing.assert_array_equal(points[column].to_numpy(), lookup[groups[column]].to_numpy())
    log_odds = -2.0 + woe.to_numpy() @ np.array([0.7, 0.7, 0.8, 0.9])
    expected = pd_to_score(1 / (1 + np.exp(-log_odds)), **SCALE)
    np.testing.assert_allclose(points.sum(axis=1), expected, rtol=1e-12)


def test_woe_to_points_accepts_a_mapping_and_ignores_extra_columns():
    woe = {"contact": [0.6, -0.2], "housing": [-0.4, 0.3], "other": [9.0, 9.0]}
    points = woe_to_points(woe, COEFFICIENTS, INTERCEPT, **SCALE)
    assert points.columns.tolist() == ["housing", "contact"]
    np.testing.assert_allclose(
        points.to_numpy(), [[281.6486, 263.7592], [265.4905, 275.3008]], atol=1e-4
    )


def test_reason_codes_rank_points_lost():
    table = points_table(TABLES, COEFFICIENTS, INTERCEPT, **SCALE)
    woe = pd.DataFrame(
        {"housing": [-0.4, 0.3, 0.3, -0.4], "contact": [-0.2, 0.6, -0.2, 0.6]}, index=list("abcd")
    )
    points = woe_to_points(woe, COEFFICIENTS, INTERCEPT, **SCALE)
    reasons = reason_codes(points, table, top=2)
    assert reasons.index.tolist() == list("abcd")
    assert reasons.columns.tolist() == ["reason_1", "reason_2"]
    # housing costs 0.8 * 0.7 * factor = 16.2 points, contact 0.5 * 0.8 * factor = 11.5.
    assert reasons.to_numpy().tolist() == [
        [None, None],
        ["housing", "contact"],
        ["housing", None],
        ["contact", None],
    ]


def test_reason_codes_ties_and_padding():
    table = points_table(
        {"a": pd.Series({"lo": 0.0, "hi": 1.0}), "b": pd.Series({"lo": 0.0, "hi": 1.0})},
        {"a": 1.0, "b": 1.0},
        0.0,
        **SCALE,
    )
    points = woe_to_points(
        pd.DataFrame({"a": [1.0], "b": [1.0]}), {"b": 1.0, "a": 1.0}, 0.0, **SCALE
    )
    reasons = reason_codes(points, table, top=3)
    assert reasons.to_numpy().tolist() == [["b", "a", None]]


@pytest.mark.parametrize(
    "points,table,top,message",
    [
        (np.zeros((1, 2)), None, 3, "points must be a DataFrame, got ndarray"),
        (pd.DataFrame({"housing": [1.0]}), None, 0, "top must be a positive integer, got 0"),
        (pd.DataFrame({"housing": [1.0]}), None, 2.0, "top must be a positive integer, got 2.0"),
        (pd.DataFrame({"housing": [1.0]}), pd.DataFrame({"woe": [1.0]}), 3, "points_table result"),
        (pd.DataFrame({"other": [1.0]}), "table", 3, "table has no entry for 1 variable"),
        (pd.DataFrame({"housing": [np.nan]}), "table", 3, "points must be finite"),
    ],
)
def test_invalid_reason_code_inputs(points, table, top, message):
    if isinstance(table, str):
        table = points_table(TABLES, COEFFICIENTS, INTERCEPT, **SCALE)
    with pytest.raises(ValueError, match=message):
        reason_codes(points, table, top=top)


@pytest.mark.parametrize(
    "tables,coefficients,intercept,message",
    [
        ([1], COEFFICIENTS, 0.0, "tables must be a mapping of variable to table, got list"),
        (TABLES, [0.8, 0.5], 0.0, "coefficients must be a mapping or Series by variable, got list"),
        (TABLES, {}, 0.0, "coefficients must name at least one variable"),
        (TABLES, pd.Series([1.0, 2.0], index=["housing"] * 2), 0.0, "name each variable once"),
        (TABLES, {"housing": np.nan}, 0.0, "finite numbers, got nan for 'housing'"),
        (TABLES, {"housing": "high"}, 0.0, "finite numbers, got 'high' for 'housing'"),
        (TABLES, {"housing": 0.8, "age": 1.0}, 0.0, "tables has no entry for 1 variable.*'age'"),
        (TABLES, COEFFICIENTS, np.inf, "intercept must be a finite number, got inf"),
        (TABLES, COEFFICIENTS, True, "intercept must be a finite number, got True"),
        ({"housing": pd.DataFrame({"iv": [0.1]})}, {"housing": 1.0}, 0.0, "must have a woe column"),
        ({"housing": [0.1]}, {"housing": 1.0}, 0.0, "woe_table result or a Series, got list"),
        ({"housing": pd.Series([np.nan])}, {"housing": 1.0}, 0.0, "must have finite WOE values"),
    ],
)
def test_invalid_points_table_inputs(tables, coefficients, intercept, message):
    with pytest.raises(ValueError, match=message):
        points_table(tables, coefficients, intercept, **SCALE)


@pytest.mark.parametrize(
    "woe,message",
    [
        (np.zeros((2, 2)), "woe must be a DataFrame or a mapping, got ndarray"),
        (pd.DataFrame({"housing": [0.1]}), "woe has no entry for 1 variable.*'contact'"),
        (pd.DataFrame({"housing": [0.1], "contact": [np.nan]}), "infinite values in 'contact'"),
    ],
)
def test_invalid_woe_inputs(woe, message):
    with pytest.raises(ValueError, match=message):
        woe_to_points(woe, COEFFICIENTS, INTERCEPT, **SCALE)


def test_inputs_are_not_modified():
    tables = {name: table.copy() for name, table in TABLES.items()}
    coefficients = pd.Series(COEFFICIENTS)
    woe = pd.DataFrame({"housing": [-0.4, 0.3], "contact": [0.6, -0.2]})
    before = (coefficients.copy(), woe.copy())
    table = points_table(tables, coefficients, INTERCEPT, **SCALE)
    reason_codes(woe_to_points(woe, coefficients, INTERCEPT, **SCALE), table)
    for name, original in TABLES.items():
        pd.testing.assert_series_equal(tables[name], original)
    pd.testing.assert_series_equal(coefficients, before[0])
    pd.testing.assert_frame_equal(woe, before[1])
