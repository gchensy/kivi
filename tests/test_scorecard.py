"""Tests of kivi.scorecard against hand calculations, the functions it builds on and saved files.

The golden files in ``tests/data`` were written by this module run as a script,
``PYTHONPATH=src python tests/test_scorecard.py``, with the code that introduced
version 1 of the format of `ScorecardSpec` (kivi 1.10.0):
``scorecard_spec_1.10.0.json`` with ``to_json(indent=2)`` of `golden_spec`, and
``scorecard_spec_1.10.0.scores.json`` with the rows of `golden_frame` and the
points, scores, probabilities, grades and reason codes that the loaded
specification gives them. ``scorecard_spec_1.10.0.calibrated.json`` holds the
``to_dict`` of the two `calibrated_specs`, with a shift and a Platt
calibration and integer grade labels, and the scores, probabilities and
grades that each gives the rows of `calibrated_frame`. Later releases must
read the files and score the rows as saved, so the files must not change; a
change that breaks them needs a new format version. Scores, grades and the
lookups of an isotonic calibration are compared exactly, and the
probabilities of a shift or Platt calibration, which go through ``exp`` and
``log``, to a relative ``RTOL``, because processors round those differently.
"""

import copy
import dataclasses
import functools
import json
import math
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

import kivi
from kivi.calibration import (
    assign_grades,
    central_tendency_shift,
    isotonic_fit,
    isotonic_predict,
    master_scale,
    platt_fit,
    platt_predict,
    quasi_moment_fit,
)
from kivi.datasets import load_bank
from kivi.scorecard import (
    ScorecardSpec,
    pd_to_score,
    points_table,
    reason_codes,
    score_scaling,
    score_to_pd,
    woe_to_points,
)
from kivi.spec import fit_spec
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


def _points_of(table, variable, label):
    """Look up the points of one group of a points_table result."""
    rows = table.loc[[variable]]
    return rows["points"].to_numpy()[rows["bin"].to_numpy() == label][0]


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
    probability = np.array([1e-9, 0.001, 0.02, 0.5, 0.97, 1 - 1e-9])
    np.testing.assert_allclose(
        score_to_pd(pd_to_score(probability, **SCALE), **SCALE), probability, rtol=1e-6
    )
    np.testing.assert_allclose(score_to_pd([580, 600, 620], **SCALE), [1 / 26, 1 / 51, 1 / 101])


def test_extreme_scores_do_not_overflow():
    np.testing.assert_equal(score_to_pd([-1e6, 1e6, -np.inf, np.inf], **SCALE), [1, 0, 1, 0])


def test_conversions_keep_series_shapes_and_missing_values():
    probability = pd.Series([0.02, np.nan], index=[7, 3], name="pd")
    scores = pd_to_score(probability, **SCALE)
    assert isinstance(scores, pd.Series)
    assert scores.name == "pd" and scores.index.tolist() == [7, 3]
    assert np.isnan(scores[3])
    back = score_to_pd(scores, **SCALE)
    assert back.index.tolist() == [7, 3] and back[7] == pytest.approx(0.02)
    assert isinstance(pd_to_score(0.02, **SCALE), float)
    assert isinstance(score_to_pd(600, **SCALE), float)
    assert pd_to_score(np.full((2, 3), 0.5), **SCALE).shape == (2, 3)


@pytest.mark.parametrize(
    "probability,message", [(0, "got 0.0"), (1, "got 1.0"), ([0.5, 1.5], "got 1.5")]
)
def test_invalid_probabilities(probability, message):
    with pytest.raises(
        ValueError, match=f"probability must be strictly between 0 and 1, {message}"
    ):
        pd_to_score(probability, **SCALE)


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


def test_scaling_returns_python_floats():
    # A float32 pdo used to give float32 results, and pd_to_score of the base odds 599.99999.
    factor, offset = score_scaling(pdo=np.float32(20), base_score=np.int64(600), base_odds=50)
    assert type(factor) is float and type(offset) is float
    assert (factor, offset) == score_scaling(**SCALE)
    score = pd_to_score(1 / 51, pdo=np.float32(20), base_score=600, base_odds=np.float32(50))
    assert score == pytest.approx(600, abs=1e-12)


STRING_DTYPE = "(object|str)"  # pandas 3 reads a list of strings as dtype str


@pytest.mark.parametrize(
    "convert, values, message",
    [
        # score_to_pd of a date used to give 0.0, and pd_to_score of strings 550.52 and 527.12.
        (score_to_pd, pd.Series(pd.to_datetime(["2020-01-01"])), "score.*got dtype datetime64"),
        (score_to_pd, ["600"], f"score.*got dtype {STRING_DTYPE} with the value '600'$"),
        (
            pd_to_score,
            ["0.1", "0.2"],
            f"probability.*got dtype {STRING_DTYPE} with the value '0.1'$",
        ),
        (pd_to_score, np.array([0.1 + 0j]), "probability.*got dtype complex128$"),
        (score_to_pd, np.array([600 + 1j]), "score.*got dtype complex128$"),
        (
            pd_to_score,
            np.array([0.1, "a"], dtype=object),
            "probability.*got dtype object with the value 'a'$",
        ),
    ],
)
def test_conversions_reject_values_that_are_not_real_numbers(convert, values, message):
    name, rest = message.split(".*")
    with pytest.raises(ValueError, match=f"^{name} must be real and numeric, {rest}"):
        convert(values, **SCALE)


def test_conversions_read_booleans_and_pd_na():
    np.testing.assert_array_equal(
        score_to_pd([True, False], **SCALE), score_to_pd([1.0, 0.0], **SCALE)
    )
    scores = pd_to_score([0.02, pd.NA], **SCALE)
    assert scores[0] == pytest.approx(pd_to_score(0.02, **SCALE)) and np.isnan(scores[1])
    probability = score_to_pd(pd.Series([600, pd.NA], dtype=object), **SCALE)
    assert probability[0] == pytest.approx(1 / 51) and np.isnan(probability[1])


def test_points_match_hand_calculation():
    table = points_table(TABLES, COEFFICIENTS, INTERCEPT, **SCALE)
    assert table.columns.tolist() == ["bin", "woe", "points"]
    assert table.index.name == "variable"
    assert table.index.tolist() == ["housing", "housing", "contact", "contact"]
    assert table["bin"].tolist() == ["no", "yes", "cellular", "unknown"]
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
    total = _points_of(table, "housing", "yes") + _points_of(table, "contact", "unknown")
    log_odds = INTERCEPT + 0.8 * 0.3 + 0.5 * 0.6
    probability = 1 / (1 + math.exp(-log_odds))
    assert total == pytest.approx(pd_to_score(probability, **SCALE), rel=1e-14)


def test_coefficients_set_the_order_and_extra_tables_are_ignored():
    tables = {**TABLES, "unused": pd.Series({"x": 1.0})}
    coefficients = pd.Series({"contact": 0.5, "housing": 0.8})
    table = points_table(tables, coefficients, INTERCEPT, **SCALE)
    assert table.index.unique().tolist() == ["contact", "housing"]
    np.testing.assert_allclose(
        table.loc["housing", "points"],
        points_table(TABLES, COEFFICIENTS, INTERCEPT, **SCALE).loc["housing", "points"],
    )


def _bank_model():
    data = load_bank()
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


def test_points_table_names_a_woe_value_that_is_not_a_number():
    tables = {**TABLES, "housing": pd.Series({"no": "low", "yes": 0.3})}
    with pytest.raises(ValueError, match="^the table of 'housing' must hold numbers, got 'low'$"):
        points_table(tables, COEFFICIENTS, INTERCEPT, **SCALE)


def test_points_table_on_interval_and_shared_labels():
    _, _, tables, coefficients = _bank_model()
    table = points_table(tables, coefficients, -2.0, **SCALE)
    # housing and loan share the labels "no" and "yes"; age has interval labels.
    assert len(table) == 6 + 2 + 2 + 4
    assert table["bin"].dtype == object
    assert table.loc[["age"], "bin"].iloc[0] == pd.Interval(-np.inf, 25.0)
    loan_yes = 0.8 * tables["loan"].loc["yes", "woe"]
    housing_yes = 0.7 * tables["housing"].loc["yes", "woe"]
    base = (OFFSET + 2 * FACTOR) / 4
    assert _points_of(table, "loan", "yes") == pytest.approx(base - FACTOR * loan_yes)
    assert _points_of(table, "housing", "yes") == pytest.approx(base - FACTOR * housing_yes)


def test_points_table_keeps_labels_that_compare_equal():
    tables = {
        "flag": pd.Series([0.2, -0.1], index=pd.Index([True, False], dtype=object)),
        "count": pd.Series([0.3, -0.3], index=[1, 0]),
        "rate": pd.Series([0.5], index=[1.0]),
    }
    table = points_table(tables, {"flag": 1.0, "count": 1.0, "rate": 1.0}, 0.0, **SCALE)
    labels = table["bin"].tolist()
    assert labels == [True, False, 1, 0, 1.0]
    assert [type(label) for label in labels] == [bool, bool, int, int, float]
    np.testing.assert_allclose(table["woe"], [0.2, -0.1, 0.3, -0.3, 0.5])


def test_missing_group_gets_points():
    groups = pd.Series(["a", None, "b", "a", None, "b"])
    target = pd.Series([1, 0, 0, 0, 1, 0])
    table = points_table({"x": woe_table(groups, target)}, {"x": 1.0}, 0.0, **SCALE)
    labels = table["bin"].tolist()
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
        rows = table.loc[[column]]
        lookup = pd.Series(rows["points"].to_numpy(), index=rows["bin"].tolist())
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


def test_woe_to_points_does_not_align_a_mapping_of_series():
    # The contact WOE at x used to be taken from the label x, -0.2, not from position 0.
    housing = pd.Series([-0.4, 0.3], index=["x", "y"])
    for index in (["y", "x"], ["y", "z"]):
        contact = pd.Series([0.6, -0.2], index=index)
        with pytest.raises(
            ValueError,
            match=r"woe\['housing'\] and woe\['contact'\] must have identical indices",
        ):
            woe_to_points(
                {"housing": housing, "contact": contact}, COEFFICIENTS, INTERCEPT, **SCALE
            )
    contact = pd.Series([0.6, -0.2], index=["x", "y"])
    points = woe_to_points(
        {"housing": housing, "contact": contact}, COEFFICIENTS, INTERCEPT, **SCALE
    )
    assert points.index.tolist() == ["x", "y"]
    expected = woe_to_points(
        pd.DataFrame({"housing": [-0.4, 0.3], "contact": [0.6, -0.2]}),
        COEFFICIENTS,
        INTERCEPT,
        **SCALE,
    )
    np.testing.assert_array_equal(points.to_numpy(), expected.to_numpy())
    # Values that are not Series are matched by position, and so are the Series' rows.
    mixed = woe_to_points(
        {"housing": housing, "contact": [0.6, -0.2]}, COEFFICIENTS, INTERCEPT, **SCALE
    )
    pd.testing.assert_frame_equal(mixed, points)


@pytest.mark.parametrize("function", ["points_table", "woe_to_points"])
def test_coefficients_with_the_intercept_get_a_clear_message(function):
    # The coef column of logistic_regression starts with the intercept; this used to say
    # "tables has no entry for 1 variable(s): 'intercept'".
    coefficients = pd.Series({"intercept": INTERCEPT, **COEFFICIENTS})
    first = (
        TABLES if function == "points_table" else pd.DataFrame({"housing": [0.3], "contact": [0.6]})
    )
    with pytest.raises(
        ValueError,
        match="^coefficients has an 'intercept' entry; drop it and pass it as intercept$",
    ):
        globals()[function](first, coefficients, INTERCEPT, **SCALE)


def test_reason_codes_reject_a_repeated_column():
    # The repeated column used to give reasons such as housing, housing, contact.
    table = points_table(TABLES, COEFFICIENTS, INTERCEPT, **SCALE)
    woe = pd.DataFrame({"housing": [0.3], "contact": [0.6]})
    points = woe_to_points(woe, COEFFICIENTS, INTERCEPT, **SCALE)
    with pytest.raises(
        ValueError, match="^points must have one column per variable, got 'housing' twice$"
    ):
        reason_codes(pd.concat([points, points[["housing"]]], axis=1), table)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_reason_codes_reject_non_finite_points_in_the_table(value):
    # A NaN in a hand-edited table used to give None, housing, None.
    table = points_table(TABLES, COEFFICIENTS, INTERCEPT, **SCALE)
    table.iloc[1, table.columns.get_loc("points")] = value
    points = woe_to_points(
        pd.DataFrame({"housing": [0.3], "contact": [0.6]}), COEFFICIENTS, INTERCEPT, **SCALE
    )
    with pytest.raises(
        ValueError, match=f"^table must have finite points, got {value!r} for 'housing'$"
    ):
        reason_codes(points, table)
    # Only the variables of points are checked.
    reason_codes(points[["contact"]], table)


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


def test_reason_codes_ignore_rounding_noise():
    data, to_groups, tables, coefficients = _bank_model()
    groups = to_groups(data.iloc[3000:])
    woe = pd.DataFrame({column: woe_transform(groups[column], tables[column]) for column in groups})
    table = points_table(tables, coefficients, -2.0, **SCALE)
    exact = reason_codes(woe_to_points(woe, coefficients, -2.0, **SCALE), table)
    assert exact["reason_1"].isna().sum() > 0
    # WOE stored with six decimals or in single precision gives the same groups and reasons.
    for noisy in (woe.round(6), woe.astype(np.float32)):
        reasons = reason_codes(woe_to_points(noisy, coefficients, -2.0, **SCALE), table)
        pd.testing.assert_frame_equal(reasons, exact)
    # Points one atol away from their group are noise; beyond it they match no group.
    points = woe_to_points(woe, coefficients, -2.0, **SCALE)
    pd.testing.assert_frame_equal(reason_codes(points + 0.9e-4, table), exact)
    with pytest.raises(ValueError, match="points of 'age' must match a group of table within"):
        reason_codes(points + 2e-4, table)


def test_reason_codes_reject_points_from_another_model():
    table = points_table(TABLES, COEFFICIENTS, INTERCEPT, **SCALE)
    woe = pd.DataFrame({"housing": [-0.4], "contact": [0.6]})
    points = woe_to_points(woe, COEFFICIENTS, INTERCEPT - 1.0, **SCALE)
    # Each variable gains factor / 2 = 14.43 points from the lower intercept.
    with pytest.raises(ValueError, match=r"got 296\.07.*nearest group 281\.64"):
        reason_codes(points, table)


def test_reason_from_a_missing_group_and_a_negative_coefficient():
    tables = {"x": pd.Series({"a": -0.3, np.nan: 0.8}), "y": pd.Series({"p": 0.5, "q": -0.2})}
    woe = pd.DataFrame({"x": [0.8, -0.3, 0.8], "y": [-0.2, -0.2, 0.5]})
    table = points_table(tables, {"x": 1.0, "y": 1.0}, -2.0, **SCALE)
    points = woe_to_points(woe, {"x": 1.0, "y": 1.0}, -2.0, **SCALE)
    # The missing group of x costs 1.1 * factor = 31.7 points, group p of y 0.7 * factor = 20.2.
    assert reason_codes(points, table).to_numpy().tolist() == [
        ["x", None, None],
        [None, None, None],
        ["x", "y", None],
    ]
    # With a negative coefficient the missing group of x becomes its best group.
    coefficients = {"x": -0.5, "y": 1.0}
    table = points_table(tables, coefficients, -2.0, **SCALE)
    points = woe_to_points(woe, coefficients, -2.0, **SCALE)
    assert pd.isna(table.loc[["x"]].sort_values("points")["bin"].iloc[-1])
    assert reason_codes(points, table).to_numpy().tolist() == [
        [None, None, None],
        ["x", None, None],
        ["y", None, None],
    ]


def test_reason_codes_accept_rounded_points_and_table():
    table = points_table(TABLES, COEFFICIENTS, INTERCEPT, **SCALE)
    woe = pd.DataFrame({"housing": [0.3], "contact": [-0.2]})
    points = woe_to_points(woe, COEFFICIENTS, INTERCEPT, **SCALE)
    assert reason_codes(points.round(), table.round()).to_numpy().tolist() == [
        ["housing", None, None]
    ]
    with pytest.raises(ValueError, match="points of 'housing' must match a group"):
        reason_codes(points.round(), table)


@pytest.mark.parametrize(
    "points,table,top,message",
    [
        (np.zeros((1, 2)), None, 3, "points must be a DataFrame, got ndarray"),
        (pd.DataFrame({"housing": [1.0]}), None, 0, "top must be a positive integer, got 0"),
        (pd.DataFrame({"housing": [1.0]}), None, 2.0, "top must be a positive integer, got 2.0"),
        (pd.DataFrame({"housing": [1.0]}), pd.DataFrame({"woe": [1.0]}), 3, "points_table result"),
        (pd.DataFrame({"housing": [1.0]}), "no points", 3, "must have a points column"),
        (pd.DataFrame({"other": [1.0]}), "table", 3, "table has no entry for 1 variable"),
        (pd.DataFrame({"housing": [np.nan]}), "table", 3, "points must be finite, got nan in"),
    ],
)
def test_invalid_reason_code_inputs(points, table, top, message):
    if isinstance(table, str):
        built = points_table(TABLES, COEFFICIENTS, INTERCEPT, **SCALE)
        table = built.drop(columns="points") if table == "no points" else built
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


def _unseen_model():
    table = points_table(TABLES, COEFFICIENTS, INTERCEPT, **SCALE)
    housing = pd.Series(["no", "rented", "yes"], index=[7, 8, 9])
    contact = pd.Series(["unknown", "unknown", "cellular"], index=[7, 8, 9])
    woe = pd.DataFrame(
        {
            "housing": woe_transform(housing, TABLES["housing"], unseen=0.0),
            "contact": woe_transform(contact, TABLES["contact"]),
        }
    )
    return table, woe_to_points(woe, COEFFICIENTS, INTERCEPT, **SCALE)


def test_reason_codes_exempt_the_points_of_unseen_groups():
    table, points = _unseen_model()
    # A WOE of 0 gives the base points, (offset - factor * intercept) / 2 for both variables.
    base = (OFFSET + FACTOR * 2.0) / 2
    zero = woe_to_points(
        pd.DataFrame({"housing": [0.0], "contact": [0.0]}), COEFFICIENTS, INTERCEPT, **SCALE
    )
    np.testing.assert_allclose(zero.iloc[0], [base, base], rtol=1e-14)
    assert points.loc[8, "housing"] == pytest.approx(base, rel=1e-14)
    message = r"points of 'housing' must match a group of table within atol=0.0001, got 272\.4"
    with pytest.raises(ValueError, match=message):
        reason_codes(points, table)
    expected = [["contact", None, None], ["contact", None, None], ["housing", None, None]]
    for unseen_points in (base, zero.iloc[0], {"housing": base}, pd.Series({"housing": base})):
        reasons = reason_codes(points, table, unseen_points=unseen_points)
        assert reasons.index.tolist() == [7, 8, 9]
        assert reasons.to_numpy().tolist() == expected
    # Without unseen points, unseen_points changes nothing.
    matched = points.drop(index=8)
    pd.testing.assert_frame_equal(
        reason_codes(matched, table, unseen_points=base), reason_codes(matched, table)
    )


def test_unseen_points_by_variable_exempt_only_their_variable():
    table, points = _unseen_model()
    base = (OFFSET + FACTOR * 2.0) / 2
    # Points of contact equal to the base points match no contact group: they are exempt
    # only when unseen_points names contact.
    points.loc[9, "contact"] = base
    with pytest.raises(ValueError, match=r"points of 'contact' must match a group of table"):
        reason_codes(points, table, unseen_points={"housing": base})
    reasons = reason_codes(points, table, unseen_points={"housing": base, "contact": base})
    assert reasons.to_numpy().tolist() == [
        ["contact", None, None],
        ["contact", None, None],
        ["housing", None, None],
    ]


def test_mismatches_other_than_unseen_points_still_raise():
    table, points = _unseen_model()
    base = (OFFSET + FACTOR * 2.0) / 2
    # WOE stored with one decimal too few moves points far from every group: atol, not
    # unseen_points, is the remedy, so the error names both.
    coarse = woe_to_points(
        pd.DataFrame({"housing": [-0.43], "contact": [0.6]}), COEFFICIENTS, INTERCEPT, **SCALE
    )
    message = (
        r"points of 'housing' must match a group of table within atol=0.0001 or "
        r"unseen_points 272\.4.*check atol and that points and table come from the same model"
    )
    with pytest.raises(ValueError, match=message):
        reason_codes(coarse, table, unseen_points=base)
    assert reason_codes(coarse, table, atol=1.0).to_numpy().tolist() == [["contact", None, None]]
    other = woe_to_points(
        pd.DataFrame({"housing": [0.0], "contact": [0.6]}), COEFFICIENTS, INTERCEPT - 1.0, **SCALE
    )
    with pytest.raises(ValueError, match="same model and scaling"):
        reason_codes(other, table, unseen_points=base)


def test_unseen_points_within_atol_of_a_group_match_that_group():
    table = points_table(TABLES, COEFFICIENTS, INTERCEPT, **SCALE)
    woe = pd.DataFrame({"housing": [0.3], "contact": [-0.2]})
    points = woe_to_points(woe, COEFFICIENTS, INTERCEPT, **SCALE)
    yes = _points_of(table, "housing", "yes")
    # unseen_points equal to group yes cannot be told apart from it, so yes costs points.
    assert reason_codes(points, table, unseen_points=yes).to_numpy().tolist() == [
        ["housing", None, None]
    ]


@pytest.mark.parametrize(
    "unseen_points,message",
    [
        ("ignore", "unseen_points must be finite numbers, got 'ignore' for 'housing'"),
        (np.nan, "unseen_points must be finite numbers, got nan for 'housing'"),
        (True, "unseen_points must be finite numbers, got True for 'housing'"),
        ({"contact": np.inf}, "unseen_points must be finite numbers, got inf for 'contact'"),
        (pd.Series([1.0, 2.0], index=["housing", "housing"]), "name each variable once"),
    ],
)
def test_invalid_unseen_points(unseen_points, message):
    table = points_table(TABLES, COEFFICIENTS, INTERCEPT, **SCALE)
    points = woe_to_points(
        pd.DataFrame({"housing": [0.3], "contact": [-0.2]}), COEFFICIENTS, INTERCEPT, **SCALE
    )
    with pytest.raises(ValueError, match=message):
        reason_codes(points, table, unseen_points=unseen_points)


# ScorecardSpec

COLUMNS = ["age", "balance", "pdays", "housing", "contact", "job", "education"]
# Not in the order of the binning, which from_model follows.
BANK_MODEL = (
    {
        "pdays": 0.9,
        "age": 0.7,
        "balance": 0.6,
        "housing": 0.8,
        "contact": 0.65,
        "job": 0.55,
        "education": 0.5,
    },
    -2.1,
)


@functools.cache
def _bank_fit():
    """A binning of every kind of column, fitted on the first 3,000 bank rows.

    ``balance`` has missing values, ``pdays`` the special code -1, ``job`` and
    ``education`` are grouped, and the columns have each unseen policy: a WOE
    for ``balance`` and ``contact``, the group ``"other"`` for ``job`` and
    ``"raise"`` for the others. ``duration`` is not in the model.
    """
    data = load_bank().iloc[:3000].copy()
    data.loc[data.index[::40], "balance"] = np.nan
    binning = fit_spec(
        data[[*COLUMNS, "duration"]],
        data["target"],
        method={"age": "chimerge", "pdays": "tree", "job": "rare", "education": "chimerge"},
        bins={"balance": 4},
        special={"pdays": [-1]},
        options={"job": {"min_share": 0.05}},
        unseen={"balance": 0.1, "contact": 0.0, "job": {"group": "other"}},
    )
    return binning, data


def bank_binning():
    return _bank_fit()[0]


def development():
    return _bank_fit()[1].copy()


def scoring_frame():
    """Later rows with an unseen contact, an unseen job, missing and infinite balances."""
    frame = load_bank().iloc[3000:3300][[*COLUMNS, "duration"]].copy()
    frame["balance"] = frame["balance"].astype(float)
    frame.iloc[0, frame.columns.get_loc("contact")] = "satellite"
    frame.iloc[1, frame.columns.get_loc("job")] = "astronaut"
    frame.iloc[2, frame.columns.get_loc("balance")] = np.nan
    frame.iloc[3, frame.columns.get_loc("balance")] = np.inf
    frame.iloc[4, frame.columns.get_loc("balance")] = -np.inf
    return frame


def bank_card(**options):
    return ScorecardSpec.from_model(bank_binning(), *BANK_MODEL, **SCALE, **options)


def model_woe(frame):
    """The WOE of the model's columns, in the order of its coefficients."""
    return bank_binning().woe_transform(frame)[list(BANK_MODEL[0])]


def model_log_odds(frame):
    """The log odds of the model, summed column by column."""
    woe = model_woe(frame)
    log_odds = np.full(len(frame), BANK_MODEL[1])
    for name, coefficient in BANK_MODEL[0].items():
        log_odds = log_odds + coefficient * woe[name].to_numpy()
    return log_odds


def test_unrounded_card_matches_points_table_and_woe_to_points():
    card = bank_card()
    assert list(card.binning.columns) == list(BANK_MODEL[0])
    assert card.binning == bank_binning().select(list(BANK_MODEL[0]))
    expected = points_table(card.binning.tables, *BANK_MODEL, **SCALE)
    pd.testing.assert_frame_equal(card.table, expected, check_index_type=False)
    frame = scoring_frame()
    points = card.points(frame)
    reference = woe_to_points(model_woe(frame), *BANK_MODEL, **SCALE)
    assert list(points.columns) == list(BANK_MODEL[0])
    assert points.index.equals(frame.index)
    np.testing.assert_allclose(points.to_numpy(), reference.to_numpy(), rtol=1e-12, atol=0)
    log_odds = model_log_odds(frame)
    probability = 1 / (1 + np.exp(-log_odds))
    score = card.score(frame)
    assert score.name == "score" and score.index.equals(frame.index)
    np.testing.assert_allclose(score, pd_to_score(probability, **SCALE), rtol=1e-12, atol=0)
    result = card.probability(frame)
    assert result.name == "probability" and result.index.equals(frame.index)
    np.testing.assert_allclose(result, probability, rtol=1e-12, atol=0)


def test_unseen_points_of_from_model_are_those_of_woe_to_points():
    card = bank_card()
    assert list(card.unseen_points) == ["balance", "contact"]
    row = pd.DataFrame({name: [0.0] for name in BANK_MODEL[0]}).assign(balance=0.1)
    expected = woe_to_points(row, *BANK_MODEL, **SCALE).iloc[0]
    assert card.unseen_points["balance"] == expected["balance"]
    assert card.unseen_points["contact"] == expected["contact"]
    points = card.points(scoring_frame())
    assert points["contact"].iloc[0] == expected["contact"]
    assert points["balance"].iloc[3] == points["balance"].iloc[4] == expected["balance"]


def test_uncalibrated_probability_is_score_to_pd_of_the_score():
    for card in (bank_card(), bank_card(decimals=0)):
        frame = scoring_frame()
        score = card.score(frame)
        assert card.probability(frame).equals(score_to_pd(score, **SCALE).rename("probability"))


def test_identical_rows_get_bit_identical_scores():
    card = bank_card()
    frame = scoring_frame()
    # Each row many times, interleaved with the others and in another order.
    repeated = pd.concat([frame] * 20).sample(frac=1.0, random_state=0)
    score = card.score(repeated)
    alone = card.score(frame)
    for label in frame.index:
        values = score.loc[[label]].to_numpy()
        assert (values == alone[label]).all(), label
    probability = card.probability(repeated)
    assert probability.groupby(level=0).nunique().eq(1).all()


@pytest.mark.parametrize("decimals", [0, 1, -1])
def test_rounded_scores_are_column_sums_of_the_rounded_table(decimals):
    card = bank_card(decimals=decimals)
    unrounded = bank_card()
    np.testing.assert_array_equal(
        card.table["points"], unrounded.table.round({"points": decimals})["points"]
    )
    assert (
        dict(card.unseen_points)
        == pd.Series(dict(unrounded.unseen_points)).round(decimals).to_dict()
    )
    frame = scoring_frame()
    points = card.points(frame)
    expected = woe_to_points(model_woe(frame), *BANK_MODEL, **SCALE).round(decimals)
    np.testing.assert_array_equal(points.to_numpy(), expected.to_numpy())
    total = np.zeros(len(frame))
    for name in card.binning.columns:
        total = total + expected[name].to_numpy()
    np.testing.assert_array_equal(card.score(frame).to_numpy(), total)
    np.testing.assert_array_equal(card.probability(frame).to_numpy(), score_to_pd(total, **SCALE))


# -0.5 rounds to -0.0, which the specification stores as 0.0.
@pytest.mark.parametrize("base_score,rounded", [(600.5, 600.0), (601.5, 602.0), (-0.5, 0.0)])
def test_rounding_halves_to_even_as_pandas_does(base_score, rounded):
    # With base odds of 1 and no intercept, a WOE of 0 has exactly the base score as points.
    binning = fit_spec(
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}), [0, 1, 1, 0], bins=2, unseen={"x": 0.0}
    )
    scaling = {"pdo": 20, "base_score": base_score, "base_odds": 1}
    card = ScorecardSpec.from_model(binning, {"x": 1.0}, 0.0, **scaling, decimals=0)
    unrounded = ScorecardSpec.from_model(binning, {"x": 1.0}, 0.0, **scaling)
    assert unrounded.unseen_points["x"] == base_score
    assert card.unseen_points["x"] == rounded
    assert math.copysign(1.0, card.unseen_points["x"]) == 1.0
    assert card.table["points"].tolist() == unrounded.table["points"].round().tolist()


def test_negative_zeros_are_stored_as_zeros():
    # Points in (-0.5, 0) round to -0.0, which scores exactly as 0.0.
    binning = fit_spec(
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}), [0, 1, 1, 0], bins=2, unseen={"x": 0.0}
    )
    scaling = {"pdo": 20, "base_score": -0.25, "base_odds": 1}
    card = ScorecardSpec.from_model(binning, {"x": 1.0}, 0.0, **scaling, decimals=0)
    assert pd.Series(dict(card.unseen_points)).round(0).iloc[0] == 0.0
    assert math.copysign(1.0, card.unseen_points["x"]) == 1.0
    table = card.table
    negative = table.assign(points=[-0.0, -0.0])
    positive = table.assign(points=[0.0, 0.0])
    one = ScorecardSpec(binning, negative, **scaling, unseen_points={"x": -0.0})
    other = ScorecardSpec(binning, positive, **scaling, unseen_points={"x": 0.0})
    assert one == other and hash(one) == hash(other) and one.to_json() == other.to_json()
    assert "-0.0" not in one.to_json()
    assert all(math.copysign(1.0, value) == 1.0 for value in one.table["points"])
    # So does every other number of the specification.
    calibrated = dataclasses.replace(
        one,
        base_score=-0.0,
        calibration={"kind": "platt", "slope": 1.0, "intercept": -0.0},
        scale=master_scale([0.05, 0.5]).assign(lower=[-0.0, np.sqrt(0.05 * 0.5)]),
    )
    assert "-0.0" not in calibrated.to_json()
    isotonic = dataclasses.replace(
        one,
        calibration={
            "kind": "isotonic",
            "edges": [-np.inf, -0.0, np.inf],
            "probability": [0.2, -0.0],
        },
    )
    assert "-0.0" not in isotonic.to_json()
    text = other.to_json().replace('"x": 0.0', '"x": -0.0')
    assert "-0.0" in text
    assert ScorecardSpec.from_json(text) == other


def test_from_model_keeps_the_columns_of_the_coefficients():
    model = ({"housing": 0.8, "age": 0.7}, -2.0)
    card = ScorecardSpec.from_model(bank_binning(), *model, **SCALE)
    assert list(card.binning.columns) == ["housing", "age"]
    assert card.unseen_points is None
    series = ScorecardSpec.from_model(bank_binning(), pd.Series(model[0]), model[1], **SCALE)
    assert series == card


# Calibration


def development_log_odds(card, frame):
    factor, offset = score_scaling(**SCALE)
    return (offset - card.score(frame)) / factor


def test_shift_calibration_moves_the_mean_probability_to_the_central_tendency():
    card = bank_card(decimals=0)
    sample = development()
    shift = central_tendency_shift(card.probability(sample), central_tendency=0.2)
    shifted = dataclasses.replace(card, calibration={"kind": "shift", "shift": shift})
    probability = shifted.probability(sample)
    assert probability.mean() == pytest.approx(0.2, rel=1e-12, abs=0)
    uncalibrated = card.probability(sample)
    expected = uncalibrated / (uncalibrated + (1 - uncalibrated) * np.exp(-shift))
    np.testing.assert_allclose(probability, expected, rtol=1e-12, atol=0)
    frame = scoring_frame()
    np.testing.assert_array_equal(
        shifted.probability(frame),
        platt_predict(development_log_odds(card, frame).to_numpy(), 1.0, shift),
    )


def test_platt_calibration_is_platt_predict_of_the_log_odds():
    card = bank_card(decimals=0)
    sample = development()
    log_odds = development_log_odds(card, sample)
    slope, intercept = platt_fit(log_odds, sample["target"])
    fitted = dataclasses.replace(
        card, calibration={"kind": "platt", "slope": slope, "intercept": intercept}
    )
    frame = scoring_frame()
    np.testing.assert_array_equal(
        fitted.probability(frame),
        platt_predict(development_log_odds(card, frame), slope, intercept),
    )
    slope, intercept = quasi_moment_fit(log_odds, central_tendency=0.15, accuracy_ratio=0.3)
    matched = dataclasses.replace(
        card, calibration={"kind": "platt", "slope": slope, "intercept": intercept}
    )
    assert matched.probability(sample).mean() == pytest.approx(0.15, rel=1e-9)


def test_isotonic_calibration_is_isotonic_predict_of_the_score():
    card = bank_card(decimals=0)
    sample = development()
    edges, probability = isotonic_fit(card.score(sample), sample["target"], increasing=False)
    fitted = dataclasses.replace(
        card, calibration={"kind": "isotonic", "edges": edges, "probability": probability}
    )
    frame = scoring_frame()
    np.testing.assert_array_equal(
        fitted.probability(frame), isotonic_predict(card.score(frame), edges, probability)
    )
    stored = fitted.calibration
    np.testing.assert_array_equal(stored["edges"], edges)
    assert not stored["edges"].flags.writeable and not stored["probability"].flags.writeable


def test_grades_are_assign_grades_of_the_calibrated_probability():
    scale = master_scale(np.geomspace(0.02, 0.32, 5), labels=list("ABCDE"))
    card = bank_card(decimals=0, calibration={"kind": "shift", "shift": -0.3}, scale=scale)
    frame = scoring_frame()
    grades = card.grade(frame)
    assert grades.name == "grade" and grades.index.equals(frame.index)
    expected = assign_grades(card.probability(frame), scale)
    assert grades.astype(object).tolist() == expected.astype(object).tolist()
    assert grades.dtype == expected.dtype
    pd.testing.assert_frame_equal(card.scale, scale)


def test_grade_needs_a_scale():
    with pytest.raises(ValueError, match="^grade needs a master scale; build the specification"):
        bank_card().grade(scoring_frame())


# Unseen groups and the binning


def test_points_follow_the_groups_of_binning_woe_transform():
    card = bank_card(decimals=0)
    frame = scoring_frame()
    points = card.points(frame)
    woe = card.binning.woe_transform(frame)
    table = card.table
    for name in card.binning.columns:
        rows = table.loc[[name]]
        for value, woe_value in zip(points[name], woe[name], strict=True):
            matches = rows["points"][rows["woe"] == woe_value]
            if name in card.unseen_points and woe_value == card.binning.unseen[name]:
                assert value == card.unseen_points[name] or value in matches.tolist()
            else:
                assert value in matches.tolist(), name


def test_a_raise_column_raises_as_binning_woe_transform_does():
    card = bank_card()
    frame = scoring_frame()
    for column, value in (("housing", "rented"), ("age", np.inf), ("education", "doctorate")):
        changed = frame.copy()
        if column == "age":
            changed[column] = changed[column].astype(float)
        changed.iloc[5, changed.columns.get_loc(column)] = value
        with pytest.raises(ValueError) as expected:
            card.binning.woe_transform(changed)
        with pytest.raises(ValueError, match=f"^column '{column}': ") as raised:
            card.points(changed)
        assert str(raised.value) == str(expected.value)


def test_a_missing_value_without_a_missing_group_raises():
    frame = scoring_frame()
    frame["age"] = frame["age"].astype(float)
    frame.iloc[0, frame.columns.get_loc("age")] = np.nan
    with pytest.raises(ValueError, match="^column 'age': 1 group.*not in table, such as nan"):
        bank_card().score(frame)


def test_an_unseen_group_column_gives_the_points_of_its_group():
    card = bank_card(decimals=0)
    frame = scoring_frame()
    other = _points_of(card.table, "job", "other")
    assert card.points(frame)["job"].iloc[1] == other
    assert (
        card.binning.woe_transform(frame)["job"].iloc[1]
        == card.binning.tables["job"].loc[["other"], "woe"].iloc[0]
    )


def test_missing_values_get_the_points_of_the_missing_group():
    card = bank_card()
    missing = card.table.loc[["balance"]]
    missing = missing["points"][missing["bin"].isna()].iloc[0]
    assert card.points(scoring_frame())["balance"].iloc[2] == missing


@pytest.mark.parametrize("decimals", [None, 0, 1])
@pytest.mark.parametrize("top,atol", [(3, 1e-4), (2, 0.5), (7, 3.0)])
def test_reason_codes_match_the_module_function(decimals, top, atol):
    card = bank_card(decimals=decimals)
    frame = scoring_frame()
    points = card.points(frame)
    expected = reason_codes(
        points, card.table, top=top, atol=atol, unseen_points=dict(card.unseen_points)
    )
    reasons = card.reason_codes(frame, top=top, atol=atol)
    # Rows 0, 3 and 4 have unseen values. Their unseen points lie more than 1 point from every
    # group, so reason_codes matches them to no group as long as atol is smaller.
    rows = frame.index if atol < 1 else frame.index.delete([0, 3, 4])
    assert reasons.loc[rows].equals(expected.loc[rows])
    assert "balance" not in reasons.iloc[[3, 4]].to_numpy().ravel().tolist()
    # The unseen contact costs nothing, so it is not a reason of the first row.
    assert "contact" not in card.reason_codes(frame, top=7).iloc[0].tolist()
    with pytest.raises(ValueError, match="points of 'balance' must match a group of table"):
        reason_codes(points, card.table)


@pytest.mark.parametrize("decimals", [None, 0])
@pytest.mark.parametrize(
    "column,group,new",
    [
        # The worst housing group, and a contact group between the best and the worst.
        ("housing", "no", "rented"),
        ("contact", "telephone", "satellite"),
    ],
)
def test_unseen_values_with_the_points_of_a_group_are_no_reason(decimals, column, group, new):
    bank = load_bank()
    first = fit_spec(bank, bank["target"], columns=["age", column], bins=3)
    woe = first.tables[column].loc[[group], "woe"].iloc[0]
    binning = fit_spec(bank, bank["target"], columns=["age", column], bins=3, unseen={column: woe})
    card = ScorecardSpec.from_model(
        binning, {"age": 0.7, column: 0.8}, -2.0, **SCALE, decimals=decimals
    )
    table = card.table
    assert card.unseen_points[column] == _points_of(table, column, group)
    assert _points_of(table, column, group) < table.loc[[column], "points"].max()
    best = table.loc[[column]].sort_values("points")["bin"].iloc[-1]
    frame = pd.DataFrame({"age": [70, 70, 30], column: [new, group, new]})
    reasons = card.reason_codes(frame, top=2)
    # An unseen value costs what the best group would.
    as_best = card.reason_codes(frame.assign(**{column: [best, group, best]}), top=2)
    assert reasons.iloc[[0, 2]].equals(as_best.iloc[[0, 2]])
    assert column not in reasons.iloc[[0, 2]].to_numpy().ravel().tolist()
    assert column in reasons.iloc[1].tolist()
    # Seen groups cost what reason_codes gives them; it cannot tell the unseen value apart.
    points = card.points(frame)
    module = reason_codes(points, table, top=2, unseen_points=dict(card.unseen_points))
    assert reasons.iloc[[1]].equals(module.iloc[[1]])
    assert column in module.iloc[0].tolist()


@pytest.mark.parametrize(
    "options,message",
    [
        ({"top": 0}, "^top must be a positive integer, got 0$"),
        ({"top": True}, "^top must be a positive integer, got True$"),
        ({"atol": 0.0}, "^atol must be a finite positive number, got 0.0$"),
        ({"atol": np.inf}, "^atol must be a finite positive number, got inf$"),
    ],
)
def test_invalid_reason_options_of_the_card(options, message):
    with pytest.raises(ValueError, match=message):
        bank_card().reason_codes(scoring_frame(), **options)


# Immutability, equality and the JSON format


def test_fields_are_read_only_copies():
    scale = master_scale([0.05, 0.2, 0.5])
    card = bank_card(scale=scale)
    table = card.table
    table.loc[:, "points"] = 0.0
    assert (card.table["points"] != 0).all()
    copied = card.scale
    copied.loc[:, "probability"] = 0.3
    pd.testing.assert_frame_equal(card.scale, scale)
    with pytest.raises(TypeError):
        card.unseen_points["contact"] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        card.table = table
    with pytest.raises(dataclasses.FrozenInstanceError):
        card.pdo = 30.0


def test_replace_validates_again():
    card = bank_card()
    assert dataclasses.replace(card, pdo=40).pdo == 40.0
    with pytest.raises(ValueError, match="^pdo must be positive"):
        dataclasses.replace(card, pdo=-1)
    with pytest.raises(ValueError, match="unseen_points lacks the points of \\['contact'\\]"):
        dataclasses.replace(card, unseen_points={"balance": 1.0})


def test_scaling_is_stored_as_floats():
    card = bank_card()
    assert (card.pdo, card.base_score, card.base_odds) == (20.0, 600.0, 50.0)
    assert all(type(value) is float for value in (card.pdo, card.base_score, card.base_odds))
    assert card == ScorecardSpec.from_model(
        bank_binning(), *BANK_MODEL, pdo=20.0, base_score=np.int64(600), base_odds=50.0
    )


def test_equality_hash_pickle_and_repr():
    card = bank_card(decimals=0)
    same = bank_card(decimals=0)
    assert card == same and hash(card) == hash(same) and len({card, same}) == 1
    assert card != bank_card(decimals=1)
    assert card != "card"
    assert pickle.loads(pickle.dumps(card)) == card
    assert copy.deepcopy(card) == card
    assert repr(card) == (
        "ScorecardSpec(columns=['pdays', 'age', 'balance', 'housing', 'contact', 'job', "
        "'education'])"
    )


def full_card():
    """A card with every optional field set."""
    card = bank_card(decimals=0)
    edges, probability = isotonic_fit(
        card.score(development()), development()["target"], increasing=False
    )
    return dataclasses.replace(
        card,
        calibration={"kind": "isotonic", "edges": edges, "probability": probability},
        scale=master_scale(np.geomspace(0.02, 0.32, 5), labels=list("ABCDE")),
    )


@pytest.mark.parametrize(
    "calibration",
    [
        None,
        {"kind": "shift", "shift": -0.25},
        {"kind": "platt", "slope": 0.9, "intercept": 0.1},
        "isotonic",
    ],
)
@pytest.mark.parametrize("labels", [list("ABCDE"), None, [1.5, 2.5, 3.5, 4.5, 5.5]])
def test_json_round_trip(calibration, labels):
    card = full_card() if calibration == "isotonic" else bank_card(calibration=calibration)
    card = dataclasses.replace(card, scale=master_scale(np.geomspace(0.02, 0.32, 5), labels=labels))
    text = card.to_json()
    loaded = ScorecardSpec.from_json(text)
    assert loaded == card and loaded.to_json() == text
    assert ScorecardSpec.from_json(text.encode()) == card
    assert ScorecardSpec.from_dict(json.loads(card.to_json(indent=2))) == card
    pd.testing.assert_frame_equal(loaded.table, card.table)
    pd.testing.assert_frame_equal(loaded.scale, card.scale)
    frame = scoring_frame()
    assert loaded.score(frame).equals(card.score(frame))
    assert loaded.probability(frame).equals(card.probability(frame))
    assert loaded.grade(frame).equals(card.grade(frame))


def test_to_dict_has_every_key_with_null_for_unset_fields():
    model = ({"housing": 0.8}, -2.0)
    data = ScorecardSpec.from_model(bank_binning(), *model, **SCALE).to_dict()
    assert list(data) == [
        "format",
        "format_version",
        "kivi_version",
        "scaling",
        "binning",
        "points",
        "unseen_points",
        "calibration",
        "scale",
    ]
    assert data["format"] == "kivi.ScorecardSpec" and data["format_version"] == 1
    assert data["kivi_version"] == kivi.__version__
    assert data["scaling"] == {"pdo": 20.0, "base_score": 600.0, "base_odds": 50.0}
    assert data["binning"] == bank_binning().select(["housing"]).to_dict()
    assert list(data["points"]) == ["housing"] and len(data["points"]["housing"]) == 2
    assert data["unseen_points"] is None and data["calibration"] is None
    assert data["scale"] is None
    card = full_card()
    full = card.to_dict()
    assert full["unseen_points"] == dict(card.unseen_points)
    assert list(full["unseen_points"]) == ["balance", "contact"]
    edges = card.calibration["edges"]
    assert full["calibration"] == {
        "kind": "isotonic",
        "edges": [{"float": "-inf"}, *edges[1:-1].tolist(), {"float": "inf"}],
        "probability": card.calibration["probability"].tolist(),
    }
    scale = card.scale
    assert full["scale"][0] == {
        "grade": "A",
        "lower": 0.0,
        "probability": scale["probability"].iloc[0],
        "upper": scale["upper"].iloc[0],
    }
    assert [row["grade"] for row in full["scale"]] == list("ABCDE")


def test_kivi_version_is_kept():
    card = ScorecardSpec(
        bank_binning().select(["housing"]),
        bank_card().table.loc[["housing"]],
        **SCALE,
        kivi_version="1.10.0",
    )
    assert card.kivi_version == "1.10.0"
    assert ScorecardSpec.from_json(card.to_json()).kivi_version == "1.10.0"


# Errors


def small_inputs():
    """The binning and the points table of a card on housing and age."""
    binning = bank_binning().select(["housing", "age"])
    table = points_table(binning.tables, {"housing": 0.8, "age": 0.7}, -2.0, **SCALE)
    return binning, table


def test_binning_must_be_a_binning_spec():
    binning, table = small_inputs()
    message = "^binning must be a BinningSpec, such as a fit_spec result, got dict$"
    with pytest.raises(ValueError, match=message):
        ScorecardSpec({}, table, **SCALE)
    with pytest.raises(ValueError, match=message):
        ScorecardSpec.from_model({}, {"housing": 0.8}, -2.0, **SCALE)


def test_invalid_scaling_of_the_card():
    binning, table = small_inputs()
    with pytest.raises(ValueError, match="^base_odds must be positive, got 0$"):
        ScorecardSpec(binning, table, **{**SCALE, "base_odds": 0})


def reordered(table):
    return pd.concat([table.loc[["age"]], table.loc[["housing"]]])


def interleaved(table):
    return table.iloc[[0, 2, 1, 3, 4]]


@pytest.mark.parametrize(
    "change,message",
    [
        (
            lambda table: table.reset_index(),
            "^table must be a points_table result, got a DataFrame",
        ),
        (lambda table: table["points"], "^table must be a points_table result, got Series$"),
        (
            lambda table: table.assign(iv=0.0),
            r"^table must have the columns 'bin', 'woe' and 'points' of a points_table result, "
            r"got \['bin', 'woe', 'points', 'iv'\]$",
        ),
        (
            lambda table: table.drop(columns="woe"),
            r"^table must have the columns .*, got \['bin', 'points'\]$",
        ),
        (
            lambda table: pd.concat([table, table.loc[["age"]].rename(index={"age": "pdays"})]),
            r"^table has rows for variables that binning lacks: \['pdays'\]$",
        ),
        (
            lambda table: table.loc[["housing"]],
            r"^table has no rows for the columns \['age'\] of binning; keep only the columns of "
            r"the model with BinningSpec.select$",
        ),
        (
            lambda table: table.iloc[:-1],
            r"^table has 2 row\(s\) for 'age', whose table in binning has 3$",
        ),
        (
            reordered,
            r"^table must list the groups of each variable together, in the order of the columns "
            r"of binning \['housing', 'age'\]$",
        ),
        (interleaved, "^table must list the groups of each variable together"),
        (
            lambda table: table.assign(woe=table["woe"].astype(str)),
            "^woe of table must be real and numeric, got dtype",
        ),
        (
            lambda table: table.assign(points=table["points"] > 270),
            "^points of table must be real and numeric, got dtype bool$",
        ),
        (
            lambda table: table.assign(bin=["yes", "no", *table["bin"].iloc[2:]]),
            r"^bin of table must list the groups of binning in their order, got 'yes' where the "
            r"table of 'housing' in binning has 'no'$",
        ),
        (
            lambda table: table.assign(bin=[np.nan, *table["bin"].iloc[1:]]),
            "^bin of table must list the groups of binning in their order, got nan where",
        ),
        (
            lambda table: table.assign(bin=[np.array([1, 2]), *table["bin"].iloc[1:]]),
            r"^bin of table must list the groups of binning in their order, got array\(\[1, 2\]\)",
        ),
        (
            lambda table: table.assign(woe=table["woe"].round(2)),
            r"^woe of table must equal the WOE of binning, got 0\.33 where binning has "
            r"0\.33.* for the group 'no' of 'housing'; round only the points column$",
        ),
        (
            lambda table: table.assign(points=[1.0, 2.0, np.inf, 3.0, 4.0]),
            r"^points of table must be finite, got inf for the group Interval\(-inf, 29\.0, "
            r"closed='right'\) of 'age'$",
        ),
        (
            lambda table: table.assign(points=[np.nan, 2.0, 1.0, 3.0, 4.0]),
            "^points of table must be finite, got nan for the group 'no' of 'housing'$",
        ),
    ],
)
def test_invalid_tables(change, message):
    binning, table = small_inputs()
    with pytest.raises(ValueError, match=message):
        ScorecardSpec(binning, change(table), **SCALE)


def test_labels_must_keep_booleans_apart_from_numbers():
    frame = pd.DataFrame({"flag": [True, False, True, False, True, False]})
    binning = fit_spec(frame, [1, 0, 0, 0, 1, 1])
    table = points_table(binning.tables, {"flag": 0.5}, -1.0, **SCALE)
    card = ScorecardSpec(binning, table, **SCALE)
    assert card.table["bin"].tolist() == [True, False]
    with pytest.raises(ValueError, match="got 1 where the table of 'flag' in binning has True$"):
        ScorecardSpec(binning, table.assign(bin=[1, 0]), **SCALE)


def test_a_table_with_equal_labels_gets_the_labels_of_binning():
    binning, table = small_inputs()
    labels = table["bin"].tolist()
    labels[2] = pd.Interval(-np.inf, 29, closed="right")
    card = ScorecardSpec(binning, table.assign(bin=labels).iloc[:, [2, 0, 1]], **SCALE)
    assert card.table.columns.tolist() == ["bin", "woe", "points"]
    assert card.table["bin"].iloc[2] == binning.tables["age"].index[0]
    assert ScorecardSpec(binning, table, **SCALE) == card


@pytest.mark.parametrize(
    "unseen_points,message",
    [
        (
            None,
            r"^unseen_points must give the points of unseen groups of the columns whose unseen "
            r"policy in binning is a WOE, got None for \['balance', 'contact'\]$",
        ),
        (
            pd.Series([1.0, 2.0, 3.0], index=["balance", "contact", "contact"]),
            "^unseen_points must name each column once, got 'contact' twice$",
        ),
        ([1.0, 2.0], "^unseen_points must be a mapping or Series by column, got list$"),
        (
            {"balance": 1.0, "contact": 2.0, "job": 3.0},
            r"^unseen_points has points for columns whose unseen policy in binning is not a WOE: "
            r"\['job'\]$",
        ),
        (
            {"balance": 1.0},
            r"^unseen_points lacks the points of \['contact'\], whose unseen policy in binning "
            "is a WOE$",
        ),
        (
            {"balance": 1.0, "contact": np.nan},
            "^unseen_points must be finite numbers, got nan for 'contact'$",
        ),
        (
            {"balance": True, "contact": 2.0},
            "^unseen_points must be finite numbers, got True for 'balance'$",
        ),
        (
            {"balance": "1", "contact": 2.0},
            "^unseen_points must be finite numbers, got '1' for 'balance'$",
        ),
    ],
)
def test_invalid_unseen_points_of_the_card(unseen_points, message):
    card = bank_card()
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(card, unseen_points=unseen_points)


def test_unseen_points_as_a_series_or_empty_mapping():
    card = bank_card()
    series = pd.Series(dict(card.unseen_points)).iloc[::-1]
    replaced = dataclasses.replace(card, unseen_points=series)
    assert replaced == card and list(replaced.unseen_points) == ["balance", "contact"]
    binning, table = small_inputs()
    assert ScorecardSpec(binning, table, **SCALE, unseen_points={}).unseen_points is None


@pytest.mark.parametrize(
    "calibration,message",
    [
        ([0.1], "^calibration must be a mapping with a kind, got list$"),
        (
            {"kind": "beta"},
            r"^the kind of calibration must be one of \('shift', 'platt', 'isotonic'\), got "
            "'beta'$",
        ),
        ({"shift": 0.1}, "^the kind of calibration must be one of .*, got None$"),
        (
            {"kind": "shift", "delta": 0.1},
            r"^a shift calibration must have the keys \['kind', 'shift'\], got \['kind', "
            r"'delta'\]$",
        ),
        (
            {"kind": "platt", "slope": 1.0},
            r"^a platt calibration must have the keys \['kind', 'slope', 'intercept'\]",
        ),
        (
            {"kind": "shift", "shift": np.nan},
            "^the shift of a shift calibration must be a finite number, got nan$",
        ),
        (
            {"kind": "platt", "slope": 1.0, "intercept": True},
            "^the intercept of a platt calibration must be a finite number, got True$",
        ),
        (
            {"kind": "platt", "slope": "1", "intercept": 0.0},
            "^the slope of a platt calibration must be a finite number, got '1'$",
        ),
        (
            {"kind": "platt", "slope": -0.5, "intercept": 0.0},
            "^the slope of a platt calibration must not be negative, which would give higher "
            r"scores a higher probability, got -0\.5$",
        ),
        (
            {"kind": "isotonic", "edges": [0.0, 500.0, np.inf], "probability": [0.2, 0.1]},
            r"^isotonic calibration: edges must increase from -inf to inf, got \[0\.0, 500\.0, "
            r"inf\]$",
        ),
        (
            {"kind": "isotonic", "edges": [-np.inf, 500.0, np.inf], "probability": [0.2]},
            "^isotonic calibration: probability must give one value for each of the 2 band",
        ),
        (
            {"kind": "isotonic", "edges": [-np.inf, 500.0, np.inf], "probability": [1.2, 0.1]},
            "^isotonic calibration: probability must be",
        ),
        (
            {"kind": "isotonic", "edges": [-np.inf, 500.0, np.inf], "probability": [0.1, 0.2]},
            r"^isotonic calibration: probability must not rise with the score, as isotonic_fit "
            r"gives it with increasing=False, got 0\.1 followed by 0\.2$",
        ),
    ],
)
def test_invalid_calibrations(calibration, message):
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(bank_card(), calibration=calibration)


def test_calibrations_are_stored_with_floats():
    card = dataclasses.replace(
        bank_card(), calibration={"kind": "platt", "slope": np.float32(0.5), "intercept": 1}
    )
    assert dict(card.calibration) == {"kind": "platt", "slope": 0.5, "intercept": 1.0}
    assert type(card.calibration["intercept"]) is float
    flat = dataclasses.replace(
        bank_card(), calibration={"kind": "platt", "slope": 0, "intercept": -1.0}
    )
    assert flat.probability(scoring_frame()).nunique() == 1
    single = dataclasses.replace(
        bank_card(),
        calibration={"kind": "isotonic", "edges": (-np.inf, np.inf), "probability": [0.1]},
    )
    assert (single.probability(scoring_frame()) == 0.1).all()
    with pytest.raises(TypeError):
        card.calibration["slope"] = 1.0


def test_scales_are_checked():
    scale = master_scale([0.05, 0.2, 0.5], labels=["A", "B", "C"])
    card = bank_card()
    for invalid, message in [
        ([0.1], "^scale must be a DataFrame from master_scale, got list$"),
        (
            scale[["lower", "upper"]],
            r"^scale must have the columns 'lower', 'probability' and 'upper' of a master_scale "
            r"result, got \['lower', 'upper'\]$",
        ),
        (scale.assign(upper=[0.1, 0.3, 0.9]), "^scale must cover"),
        (
            scale.assign(probability=[0.05, 0.2, 1.0]),
            r"^probability of scale must be strictly between 0 and 1, got 1\.0$",
        ),
        (
            scale.set_axis(pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])),
            "^scale must be labelled by strings, integers, finite floats or booleans, got "
            "Timestamp",
        ),
        (
            scale.set_axis([1.0, 2.0, np.inf]),
            "^scale must be labelled by strings, integers, finite floats or booleans, got inf "
            "of type float$",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(card, scale=invalid)


def test_scales_keep_their_labels():
    card = bank_card()
    for labels in (
        [True, False],
        np.array([3, 1], dtype=np.int32),
        ["low", "high"],
        [0.5, 1.5],
    ):
        scale = master_scale([0.05, 0.5], labels=labels)
        stored = dataclasses.replace(card, scale=scale).scale
        assert stored.index.name == "grade"
        assert stored.index.tolist() == list(labels)
        assert ScorecardSpec.from_json(dataclasses.replace(card, scale=scale).to_json()) == (
            dataclasses.replace(card, scale=scale)
        )


def test_kivi_version_must_be_a_string():
    binning, table = small_inputs()
    with pytest.raises(ValueError, match="^kivi_version must be a string, got 1.1$"):
        ScorecardSpec(binning, table, **SCALE, kivi_version=1.1)


@pytest.mark.parametrize(
    "decimals,message",
    [
        (True, "^decimals must be an integer, got True$"),
        (1.5, "^decimals must be an integer, got 1.5$"),
        ("0", "^decimals must be an integer, got '0'$"),
    ],
)
def test_invalid_decimals(decimals, message):
    with pytest.raises(ValueError, match=message):
        bank_card(decimals=decimals)


def test_from_model_needs_every_column_of_the_coefficients():
    with pytest.raises(ValueError, match="^binning has no entry for 1 variable\\(s\\): 'loan'$"):
        ScorecardSpec.from_model(bank_binning(), {"age": 0.5, "loan": 0.3}, -2.0, **SCALE)
    with pytest.raises(ValueError, match="^coefficients has an 'intercept' entry; drop it"):
        ScorecardSpec.from_model(bank_binning(), {"age": 0.5, "intercept": -2.0}, -2.0, **SCALE)
    with pytest.raises(ValueError, match="^intercept must be a finite number, got nan$"):
        ScorecardSpec.from_model(bank_binning(), {"age": 0.5}, np.nan, **SCALE)


def card_dict():
    return json.loads(full_card().to_json())


def changed(path, value):
    """The dict of `full_card` with the value at ``path`` replaced, or removed for KeyError."""
    data = card_dict()
    target = data
    for key in path[:-1]:
        target = target[key]
    if value is KeyError:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return data


@pytest.mark.parametrize(
    "data,message",
    [
        ([1], "^data must be a mapping, got list$"),
        (
            changed(["format"], "kivi.BinningSpec"),
            "^data is not a kivi scorecard specification: format must be 'kivi.ScorecardSpec', "
            "got 'kivi.BinningSpec'$",
        ),
        (
            changed(["format_version"], True),
            "^format_version must be a positive integer, got True$",
        ),
        (changed(["format_version"], 0), "^format_version must be a positive integer, got 0$"),
        (changed(["format_version"], "1"), "^format_version must be a positive integer, got '1'$"),
        (
            changed(["format_version"], 2),
            "^format_version 2 is newer than 1, the newest that kivi .* reads; upgrade kivi to "
            "load this specification$",
        ),
        (changed(["scale"], KeyError), r"^the specification lacks \['scale'\]$"),
        (
            {**card_dict(), "notes": "", "extra": 1},
            r"^the specification has unknown keys \['extra', 'notes'\]$",
        ),
        (changed(["kivi_version"], None), "^kivi_version must be a string, got None$"),
        (changed(["scaling"], [20, 600, 50]), "^scaling must be a mapping, got a list$"),
        (changed(["scaling", "pdo"], KeyError), r"^scaling lacks \['pdo'\]$"),
        (changed(["scaling", "pdo"], "20"), "^pdo of scaling must be numbers, got '20'$"),
        (changed(["scaling", "pdo"], -20), "^pdo must be positive, got -20$"),
        (
            changed(["binning", "format_version"], 3),
            "^binning: format_version 3 is newer than 2",
        ),
        (changed(["binning"], None), "^binning: data must be a mapping, got NoneType$"),
        (changed(["points"], None), "^points must be a mapping by column, got null$"),
        (changed(["points", "age"], KeyError), r"^points lacks \['age'\]$"),
        (changed(["points", "duration"], [1.0]), r"^points has unknown keys \['duration'\]$"),
        (
            changed(["points", "age"], {"0": 1.0}),
            "^points of 'age' must be a list, got an object$",
        ),
        (
            changed(["points", "age"], [1.0]),
            "^points of 'age' must have one value per group of its table in binning, 3, got 1$",
        ),
        (
            changed(["points", "housing"], [1.0, True]),
            "^points of 'housing' must be numbers, got True$",
        ),
        (
            changed(["points", "housing"], [1.0, {"float": "nan"}]),
            r"^points of 'housing' must be numbers, got \{'float': 'nan'\}$",
        ),
        (
            changed(["points", "housing"], [1.0, {"float": "inf"}]),
            "^points of table must be finite, got inf for the group 'yes' of 'housing'$",
        ),
        (
            changed(["unseen_points"], [1.0]),
            "^unseen_points must be a mapping or null, got a list$",
        ),
        (
            changed(["unseen_points", "contact"], "1"),
            "^unseen_points must be numbers, got '1'$",
        ),
        (
            changed(["unseen_points", "contact"], KeyError),
            r"^unseen_points lacks the points of \['contact'\]",
        ),
        (
            changed(["calibration"], "shift"),
            "^calibration must be a mapping or null, got a string$",
        ),
        (
            changed(["calibration", "edges"], {"0": 1}),
            "^edges of the isotonic calibration must be a list, got an object$",
        ),
        (
            changed(["calibration", "probability"], ["0.1"]),
            "^probability of the isotonic calibration must be numbers, got '0.1'$",
        ),
        (
            changed(["calibration", "edges"], KeyError),
            r"^an? isotonic calibration must have the keys \['kind', 'edges', 'probability'\]",
        ),
        (
            changed(["calibration"], {"kind": "shift", "shift": "0.1"}),
            "^the shift of a shift calibration must be a finite number, got '0.1'$",
        ),
        (changed(["scale"], []), "^scale must be a non-empty list of grades or null, got a list$"),
        (
            changed(["scale"], {}),
            "^scale must be a non-empty list of grades or null, got an object$",
        ),
        (changed(["scale", 1], "B"), r"^scale\[1\] must be a mapping, got a string$"),
        (changed(["scale", 1, "upper"], KeyError), r"^scale\[1\] lacks \['upper'\]$"),
        (
            changed(["scale", 1, "grade"], None),
            r"^the grade of scale\[1\] must be a string, number or boolean, got null$",
        ),
        (
            changed(["scale", 1, "grade"], ["B"]),
            r"^the grade of scale\[1\] must be a string, number or boolean, got a list$",
        ),
        (
            changed(["scale", 1, "lower"], None),
            r"^lower of scale\[1\] must be numbers, got None$",
        ),
        (changed(["scale", 1, "grade"], "A"), "^scale must be indexed by unique"),
    ],
)
def test_invalid_dicts(data, message):
    with pytest.raises(ValueError, match=message):
        ScorecardSpec.from_dict(data)


def test_isotonic_edges_are_written_and_read_as_tagged_infinities():
    data = card_dict()
    assert data["calibration"]["edges"][0] == {"float": "-inf"}
    data["calibration"]["edges"][0] = {"float": "inf"}
    with pytest.raises(ValueError, match="^isotonic calibration: edges must increase"):
        ScorecardSpec.from_dict(data)
    data["calibration"]["edges"][0] = 4
    with pytest.raises(ValueError, match=r"^isotonic calibration: edges must increase .*\[4\.0, "):
        ScorecardSpec.from_dict(data)


@pytest.mark.parametrize(
    "text,message",
    [
        (None, "^text must be JSON text as a str or bytes, got NoneType; read a file first"),
        ("{", "^Expecting property name"),
        (
            '{"format": "kivi.ScorecardSpec", "format": "kivi.ScorecardSpec"}',
            "^JSON objects must not repeat a key, got 'format' twice$",
        ),
        (
            '{"format": NaN}',
            "^JSON must not contain NaN; kivi writes missing labels as null and infinite numbers "
            'as {"float": "inf"}$',
        ),
        ('{"format": -Infinity}', "^JSON must not contain -Infinity"),
    ],
)
def test_invalid_json(text, message):
    with pytest.raises(ValueError, match=message):
        ScorecardSpec.from_json(text)


def test_nested_binnings_reject_repeated_keys_too():
    text = full_card().to_json()
    repeated = text.replace('"smoothing": 0.5', '"smoothing": 0.5, "smoothing": 0.25', 1)
    with pytest.raises(ValueError, match="^JSON objects must not repeat a key, got 'smoothing'"):
        ScorecardSpec.from_json(repeated)


# An integer that JSON and Python hold exactly but that no float can.
HUGE = 10**400
TOO_LARGE = "must be numbers that fit a float, got an integer too large for one$"


def with_huge(path):
    """The dict of `full_card` with HUGE at ``path``, the last key indexing a list if numeric."""
    data = card_dict()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = HUGE
    return data


@pytest.mark.parametrize(
    "path,message",
    [
        (["scaling", "pdo"], f"^pdo of scaling {TOO_LARGE}"),
        (["points", "age", 0], f"^points of 'age' {TOO_LARGE}"),
        (["unseen_points", "contact"], f"^unseen_points {TOO_LARGE}"),
        (["calibration", "edges", 1], f"^edges of the isotonic calibration {TOO_LARGE}"),
        (
            ["calibration", "probability", 0],
            f"^probability of the isotonic calibration {TOO_LARGE}",
        ),
        (["scale", 1, "lower"], rf"^lower of scale\[1\] {TOO_LARGE}"),
    ],
)
def test_json_integers_too_large_for_a_float(path, message):
    data = with_huge(path)
    with pytest.raises(ValueError, match=message):
        ScorecardSpec.from_dict(data)
    with pytest.raises(ValueError, match=message):
        ScorecardSpec.from_json(json.dumps(data))


@pytest.mark.parametrize(
    "calibration,message",
    [
        ({"kind": "shift", "shift": HUGE}, "shift"),
        ({"kind": "platt", "slope": 1.0, "intercept": HUGE}, "intercept"),
    ],
)
def test_json_calibration_parameters_too_large_for_a_float(calibration, message):
    data = card_dict()
    data["calibration"] = calibration
    with pytest.raises(ValueError, match=f"^the {message} of the calibration {TOO_LARGE}"):
        ScorecardSpec.from_json(json.dumps(data))


def huge_scale():
    """A master scale whose first PD is HUGE; pandas 2.0 cannot infer floats from it."""
    scale = master_scale([0.05, 0.2]).astype(object)
    scale["probability"] = pd.Series([HUGE, 0.2], index=scale.index, dtype=object)
    return scale


def test_arguments_too_large_for_a_float():
    card = bank_card()
    for options, message in [
        ({"pdo": HUGE}, "^pdo must be a finite number, got 1000"),
        (
            {"unseen_points": {"balance": HUGE, "contact": 1.0}},
            "^unseen_points must be finite numbers, got 1000",
        ),
        (
            {"calibration": {"kind": "platt", "slope": HUGE, "intercept": 0.0}},
            "^the slope of a platt calibration must be a finite number, got 1000",
        ),
        (
            {
                "calibration": {
                    "kind": "isotonic",
                    "edges": [-np.inf, HUGE, np.inf],
                    "probability": [0.2, 0.1],
                }
            },
            "^calibration must hold numbers that fit a float$",
        ),
        (
            {"scale": huge_scale()},
            "^scale must hold numbers that fit a float$",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(card, **options)
    table = card.table.astype({"points": object})
    table.iloc[0, 2] = HUGE
    with pytest.raises(ValueError, match="^table must hold numbers that fit a float$"):
        dataclasses.replace(card, table=table)
    with pytest.raises(ValueError, match="^base_score must be a finite number, got 1000"):
        score_scaling(**{**SCALE, "base_score": HUGE})
    with pytest.raises(ValueError, match="^intercept must be a finite number, got 1000"):
        points_table(TABLES, COEFFICIENTS, HUGE, **SCALE)
    with pytest.raises(ValueError, match="^coefficients must be finite numbers, got 1000"):
        woe_to_points(pd.DataFrame({"housing": [0.3]}), {"housing": HUGE}, -2.0, **SCALE)
    table = points_table(TABLES, COEFFICIENTS, INTERCEPT, **SCALE)
    points = woe_to_points(
        pd.DataFrame({"housing": [0.3], "contact": [-0.2]}), COEFFICIENTS, INTERCEPT, **SCALE
    )
    with pytest.raises(ValueError, match="^unseen_points must be finite numbers, got 1000"):
        reason_codes(points, table, unseen_points=HUGE)


# Golden files

DATA = pathlib.Path(__file__).parent / "data"
GOLDEN = DATA / "scorecard_spec_1.10.0.json"
GOLDEN_SCORES = DATA / "scorecard_spec_1.10.0.scores.json"
GOLDEN_CALIBRATED = DATA / "scorecard_spec_1.10.0.calibrated.json"
# The last digits of exp and log may differ from the file's on another CPU.
RTOL = 1e-12


def golden_spec():
    """The specification of the golden file.

    Its binning has numeric columns with a special code and with missing
    values, grouped categorical columns and every unseen policy; the card
    has rounded points, unseen points, an isotonic calibration, whose edges
    are infinite at both ends, and a master scale.
    """
    binning = dataclasses.replace(bank_binning(), kivi_version="1.10.0")
    card = ScorecardSpec.from_model(binning, *BANK_MODEL, **SCALE, decimals=0)
    sample = development()
    edges, probability = isotonic_fit(card.score(sample), sample["target"], increasing=False)
    return dataclasses.replace(
        card,
        calibration={"kind": "isotonic", "edges": edges, "probability": probability},
        scale=master_scale(np.geomspace(0.02, 0.32, 5), labels=list("ABCDE")),
        kivi_version="1.10.0",
    )


def golden_frame():
    """The rows that the golden scores belong to, with every kind of value."""
    return scoring_frame().iloc[:40]


def encoded(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, float) and math.isinf(value):
        return {"float": "inf" if value > 0 else "-inf"}
    return value


def decoded(value):
    if value is None:
        return np.nan
    if isinstance(value, dict):
        return float(value["float"])
    return value


def saved_frame(saved, binning):
    """The rows of the scores file, with float numeric columns and object categorical ones."""
    columns = {}
    for name, values in saved["columns"].items():
        values = [decoded(value) for value in values]
        numeric = name not in binning.kinds or binning.kinds[name] == "numeric"
        columns[name] = np.array(values, dtype=float if numeric else object)
    return pd.DataFrame(columns, index=saved["index"])


def test_the_golden_file_loads_and_scores_as_saved():
    text = GOLDEN.read_text(encoding="utf-8")
    card = ScorecardSpec.from_json(text)
    assert card.to_json(indent=2) + "\n" == text
    assert card.kivi_version == "1.10.0" and card.binning.kivi_version == "1.10.0"
    # The file has what it is meant to cover.
    binning = card.binning
    assert list(binning.columns) == list(BANK_MODEL[0])
    assert binning.special["pdays"] == (-1,)
    assert binning.tables["balance"].index.isna().any()
    assert set(binning.groups) == {"job", "education"}
    assert {name: type(binning.unseen[name]).__name__ for name in binning.columns} == {
        "pdays": "str",
        "age": "str",
        "balance": "float",
        "housing": "str",
        "contact": "float",
        "job": "dict",
        "education": "str",
    }
    assert list(card.unseen_points) == ["balance", "contact"]
    assert card.calibration["kind"] == "isotonic"
    assert card.scale.index.tolist() == list("ABCDE")
    saved = json.loads(GOLDEN_SCORES.read_text(encoding="utf-8"))
    frame = saved_frame(saved["frame"], binning)
    points = card.points(frame)
    assert list(points.columns) == list(saved["points"])
    for name, values in saved["points"].items():
        assert points[name].tolist() == values, name
    assert card.score(frame).tolist() == saved["score"]
    # An isotonic calibration looks the probability up, so it is exact on every processor.
    assert card.probability(frame).tolist() == saved["probability"]
    assert card.grade(frame).astype(object).tolist() == saved["grade"]
    assert card.reason_codes(frame).to_numpy().tolist() == saved["reasons"]


def calibrated_specs():
    """Small specifications with a shift and a Platt calibration and integer grades."""
    sample = development()
    binning = dataclasses.replace(
        fit_spec(sample[["age", "housing"]], sample["target"], bins=3), kivi_version="1.10.0"
    )
    card = ScorecardSpec.from_model(
        binning,
        {"age": 0.7, "housing": 0.8},
        -2.0,
        **SCALE,
        decimals=0,
        scale=master_scale(np.geomspace(0.05, 0.4, 4)),
    )
    card = dataclasses.replace(card, kivi_version="1.10.0")
    shift = central_tendency_shift(card.probability(sample), central_tendency=0.15)
    slope, intercept = platt_fit(development_log_odds(card, sample), sample["target"])
    return {
        "shift": dataclasses.replace(card, calibration={"kind": "shift", "shift": shift}),
        "platt": dataclasses.replace(
            card, calibration={"kind": "platt", "slope": slope, "intercept": intercept}
        ),
    }


def calibrated_frame():
    return load_bank().iloc[3000:3012][["age", "housing"]]


def test_the_calibrated_golden_file_loads_and_scores_as_saved():
    saved = json.loads(GOLDEN_CALIBRATED.read_text(encoding="utf-8"))
    assert list(saved["cards"]) == ["shift", "platt"]
    for kind, content in saved["cards"].items():
        card = ScorecardSpec.from_dict(content["spec"])
        # The same keys, in the same order, with the same numbers.
        assert json.dumps(card.to_dict()) == json.dumps(content["spec"])
        assert ScorecardSpec.from_json(json.dumps(content["spec"])) == card
        assert card.kivi_version == "1.10.0" and card.calibration["kind"] == kind
        assert list(content["spec"]["calibration"]) == [
            "kind",
            *{"shift": ["shift"], "platt": ["slope", "intercept"]}[kind],
        ]
        assert [row["grade"] for row in content["spec"]["scale"]] == [1, 2, 3, 4]
        assert card.scale.index.dtype == np.int64
        frame = saved_frame(saved["frame"], card.binning)
        assert card.score(frame).tolist() == content["score"]
        np.testing.assert_allclose(
            card.probability(frame), content["probability"], rtol=RTOL, atol=0
        )
        grades = card.grade(frame).astype(object).tolist()
        assert grades == content["grade"] and all(type(grade) is int for grade in grades)


def write_golden_files():
    """Write the golden files from `golden_spec`, `calibrated_specs` and their frames."""
    GOLDEN.write_text(golden_spec().to_json(indent=2) + "\n", encoding="utf-8")
    card = ScorecardSpec.from_json(GOLDEN.read_text(encoding="utf-8"))
    frame = golden_frame()
    points = card.points(frame)
    saved = {
        "frame": {
            "index": frame.index.tolist(),
            "columns": {
                name: [encoded(value) for value in frame[name].tolist()] for name in frame.columns
            },
        },
        "points": {name: points[name].tolist() for name in points.columns},
        "score": card.score(frame).tolist(),
        "probability": card.probability(frame).tolist(),
        "grade": card.grade(frame).astype(object).tolist(),
        "reasons": card.reason_codes(frame).to_numpy().tolist(),
    }
    GOLDEN_SCORES.write_text(json.dumps(saved, indent=1) + "\n", encoding="utf-8")
    frame = calibrated_frame()
    cards = {}
    for kind, card in calibrated_specs().items():
        card = ScorecardSpec.from_dict(card.to_dict())
        cards[kind] = {
            "spec": card.to_dict(),
            "score": card.score(frame).tolist(),
            "probability": card.probability(frame).tolist(),
            "grade": card.grade(frame).astype(object).tolist(),
        }
    saved = {
        "frame": {
            "index": frame.index.tolist(),
            "columns": {name: frame[name].tolist() for name in frame.columns},
        },
        "cards": cards,
    }
    GOLDEN_CALIBRATED.write_text(json.dumps(saved, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_golden_files()
    sys.exit(0)
