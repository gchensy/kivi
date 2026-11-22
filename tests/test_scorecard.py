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
