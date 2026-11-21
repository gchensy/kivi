import math

import numpy as np
import pandas as pd
import pytest

from kivi.binning import bin_edges
from kivi.datasets import load_bank, make_applications
from kivi.metrics import gini, ks, roc_auc
from kivi.monitoring import monitoring_table, stability_table
from kivi.stability import psi, psi_table

COLUMNS = ["total", "known", "event", "event_rate", "auc", "gini", "ks", "psi"]

# Three months of four scores. With bins=2 the January median, 0.25, splits the bands:
# January has 2 and 2 scores in them, February 1 and 3, March 0 and 4.
SCORE = [0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.6, 0.7, 0.5, 0.6, 0.8, 0.9]
TARGET = [0, 0, 1, 1, 0, 1, 0, 1, 0, np.nan, 1, np.nan]
PERIOD = ["2024-01"] * 4 + ["2024-02"] * 4 + ["2024-03"] * 4


def hand_psi(reference, current, smoothing=0.5):
    """PSI of two count lists, written out from the definition."""
    reference = [smoothing if count == 0 else count for count in reference]
    current = [smoothing if count == 0 else count for count in current]
    total = 0.0
    for a, b in zip(reference, current, strict=True):
        q, p = a / sum(reference), b / sum(current)
        total += (p - q) * math.log(p / q)
    return total


@pytest.fixture(scope="module")
def applications():
    """Applications over six months; the outcomes of the last month are partly unknown."""
    frame = make_applications(n_applicants=3000, rng=7)
    rng = np.random.default_rng(3)
    months = pd.period_range("2024-01", periods=6, freq="M")
    frame["month"] = months[np.sort(rng.integers(0, 6, len(frame)))]
    target = frame["target"].astype(float)
    immature = (frame["month"] == months[-1]).to_numpy() & (rng.random(len(frame)) < 0.6)
    target[immature] = np.nan
    frame["outcome"] = target
    return frame


def period_rows(frame, period, edges, reference_score):
    """Recompute each row of monitoring_table from one period at a time."""
    rows = {}
    for label, group in frame.groupby(period, sort=True):
        known = group[group["outcome"].notna()]
        both = known["outcome"].nunique() == 2
        score, target = known["incumbent_pd"], known["outcome"].astype(int)
        current = pd.cut(group["incumbent_pd"], edges)
        rows[label] = {
            "total": len(group),
            "known": len(known),
            "event": int(known["outcome"].sum()),
            "event_rate": known["outcome"].mean() if len(known) else np.nan,
            "auc": roc_auc(score, target) if both else np.nan,
            "gini": gini(score, target) if both else np.nan,
            "ks": ks(score, target) if both else np.nan,
            "psi": psi_table(pd.cut(reference_score, edges), current)["psi"].sum(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


# monitoring_table


def test_monitoring_table_matches_hand_calculation():
    table = monitoring_table(SCORE, PERIOD, TARGET, reference="2024-01", bins=2)
    assert table.columns.tolist() == COLUMNS
    assert table.index.tolist() == ["2024-01", "2024-02", "2024-03"]
    assert table.index.name == "period"
    assert table["total"].tolist() == [4, 4, 4]
    assert table["known"].tolist() == [4, 4, 2]
    assert table["event"].tolist() == [2, 2, 1]
    assert table["event_rate"].tolist() == [0.5, 0.5, 0.5]
    # February: events 0.3 and 0.7 beat non-events 0.2 and 0.6 in 3 of 4 pairs.
    assert table["auc"].tolist() == [1.0, 0.75, 1.0]
    assert table["gini"].tolist() == [1.0, 0.5, 1.0]
    assert table["ks"].tolist() == [1.0, 0.5, 1.0]
    assert table["psi"].iloc[0] == 0.0
    assert table["psi"].iloc[1] == pytest.approx(hand_psi([2, 2], [1, 3]))
    assert table["psi"].iloc[2] == pytest.approx(hand_psi([2, 2], [0, 4]))
    assert table["psi"].iloc[2] == pytest.approx(0.8087, abs=1e-4)


def test_monitoring_table_rows_match_metrics_of_each_period(applications):
    reference = pd.Period("2024-01", "M")
    table = monitoring_table(
        applications["incumbent_pd"],
        applications["month"],
        applications["outcome"],
        reference=reference,
    )
    first = applications.loc[applications["month"] == reference, "incumbent_pd"]
    edges = bin_edges(first, bins=10)
    expected = period_rows(applications, "month", edges, first)
    assert isinstance(table.index, pd.PeriodIndex)
    assert table.index.tolist() == expected.index.tolist()
    assert table["total"].sum() == 3000
    pd.testing.assert_frame_equal(
        table, expected[COLUMNS], check_dtype=False, check_names=False, check_index_type=False
    )
    # The last month has unknown outcomes; they count in total only.
    last = table.iloc[-1]
    assert last["known"] < last["total"]


def test_monitoring_table_reference_sample(applications):
    development = make_applications(n_applicants=2000, rng=8)["incumbent_pd"]
    table = monitoring_table(
        applications["incumbent_pd"],
        applications["month"],
        applications["outcome"],
        reference=development,
        bins=5,
    )
    edges = bin_edges(development, bins=5)
    expected = period_rows(applications, "month", edges, development)
    np.testing.assert_allclose(table["psi"], expected["psi"], rtol=1e-12)
    np.testing.assert_allclose(table["auc"], expected["auc"], rtol=1e-12)
    # Every month comes from the same population as the development sample.
    assert (table["psi"] < 0.05).all()
    # A list works as well as a Series.
    listed = monitoring_table(
        applications["incumbent_pd"].tolist(),
        applications["month"].tolist(),
        applications["outcome"].tolist(),
        reference=development.tolist(),
        bins=5,
    )
    np.testing.assert_allclose(listed["psi"], table["psi"], rtol=1e-12)


def test_monitoring_table_integer_bins_are_bin_edges_of_the_reference(applications):
    score, target, month = (applications[c] for c in ("incumbent_pd", "outcome", "month"))
    reference = pd.Period("2024-02", "M")
    edges = bin_edges(score[month == reference], bins=4)
    by_count = monitoring_table(score, month, target, reference=reference, bins=4)
    by_edges = monitoring_table(score, month, target, reference=reference, bins=edges)
    pd.testing.assert_frame_equal(by_count, by_edges)
    assert by_count.loc[reference, "psi"] == 0.0


def test_monitoring_table_missing_target_counts_in_total_only():
    target = [np.nan, 0, 1, None, 1, np.nan]
    table = monitoring_table(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ["a"] * 6, pd.Series(target, dtype=object), reference="a"
    )
    assert table["total"].tolist() == [6]
    assert table["known"].tolist() == [3]
    assert table["event"].tolist() == [2]
    assert table["event_rate"].tolist() == [pytest.approx(2 / 3)]
    # Known rows: 0.2 (0), 0.3 (1), 0.5 (1).
    assert table["auc"].tolist() == [1.0]


def test_monitoring_table_nullable_target():
    target = pd.array([0, 1, pd.NA, 1], dtype="Int64")
    table = monitoring_table([0.1, 0.4, 0.3, 0.2], [1, 1, 1, 1], target, reference=1)
    assert table["known"].tolist() == [3]
    assert table["event"].tolist() == [2]
    # Events 0.4 and 0.2 against the non-event 0.1.
    assert table["auc"].tolist() == [1.0]


def test_monitoring_table_period_without_both_classes_has_nan_discrimination():
    score = [0.1, 0.5, 0.2, 0.6, 0.3, 0.7]
    target = [0, 1, 0, 0, np.nan, 1]
    table = monitoring_table(score, ["a", "a", "b", "b", "c", "c"], target, reference="a")
    assert table["known"].tolist() == [2, 2, 1]
    assert table["event"].tolist() == [1, 0, 1]
    assert table["event_rate"].tolist() == [0.5, 0.0, 1.0]
    assert table["auc"].iloc[0] == 1.0
    for column in ("auc", "gini", "ks"):
        assert table[column].iloc[1:].isna().all()
    assert table["psi"].notna().all()


def test_monitoring_table_empty_categorical_period_keeps_its_row():
    categories = ["2024-03", "2024-01", "2024-02"]
    period = pd.Categorical(["2024-03"] * 3 + ["2024-02"] * 3, categories=categories, ordered=True)
    score = [0.1, 0.2, 0.3, 0.1, 0.2, 0.9]
    target = [0, 1, 1, 0, 1, 0]
    table = monitoring_table(
        score, period, target, reference="2024-03", bins=[-np.inf, 0.15, np.inf]
    )
    assert isinstance(table.index, pd.CategoricalIndex)
    assert table.index.tolist() == categories
    assert table.index.dtype == period.dtype
    empty = table.loc["2024-01"]
    assert empty[["total", "known", "event"]].tolist() == [0, 0, 0]
    assert empty[["event_rate", "auc", "gini", "ks", "psi"]].isna().all()
    assert table.loc["2024-03", "psi"] == 0.0
    assert table.loc["2024-02", "psi"] == pytest.approx(hand_psi([1, 2], [1, 2]))


def test_monitoring_table_unordered_categorical_follows_category_order():
    period = pd.Categorical(["b", "a", "b", "a"], categories=["b", "a", "c"])
    table = monitoring_table([0.1, 0.2, 0.3, 0.4], period, [0, 1, 1, 0], reference="b")
    assert table.index.tolist() == ["b", "a", "c"]
    assert table["total"].tolist() == [2, 2, 0]


def test_monitoring_table_sorts_other_periods():
    table = monitoring_table([0.1, 0.2, 0.3, 0.4], [3, 1, 2, 1], [0, 1, 1, 0], reference=1)
    assert table.index.tolist() == [1, 2, 3]
    assert table["total"].tolist() == [2, 1, 1]
    dates = pd.to_datetime(["2024-02-01", "2024-01-01", "2024-02-01", "2024-01-01"])
    by_date = monitoring_table([0.1, 0.2, 0.3, 0.4], dates, [0, 1, 1, 0], reference=dates[1])
    assert isinstance(by_date.index, pd.DatetimeIndex)
    assert by_date.index.tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]


def test_monitoring_table_without_target():
    table = monitoring_table(SCORE, PERIOD, reference="2024-01", bins=2)
    assert table.columns.tolist() == ["total", "psi"]
    with_target = monitoring_table(SCORE, PERIOD, TARGET, reference="2024-01", bins=2)
    pd.testing.assert_frame_equal(table, with_target[["total", "psi"]])


def test_monitoring_table_smoothing():
    table = monitoring_table(SCORE, PERIOD, TARGET, reference="2024-01", bins=2, smoothing=0.1)
    assert table["psi"].iloc[2] == pytest.approx(hand_psi([2, 2], [0, 4], smoothing=0.1))
    assert table["psi"].iloc[2] == pytest.approx(psi([2, 2], [0, 4], smoothing=0.1))


def test_monitoring_table_leaves_out_bands_empty_in_both_samples():
    # Interpolated quartiles of [0, 10] leave the two middle bands empty in the reference;
    # psi_table never sees a group that neither sample has.
    score = [0.0, 10.0, 0.0, 1.0, 9.0, 10.0]
    period = ["ref", "ref", "now", "now", "now", "now"]
    table = monitoring_table(score, period, reference="ref", bins=4)
    edges = bin_edges([0.0, 10.0], bins=4)
    assert len(edges) == 5
    expected = psi_table(pd.cut([0.0, 10.0], edges), pd.cut([0.0, 1.0, 9.0, 10.0], edges))
    assert len(expected) == 2
    assert table.loc["now", "psi"] == pytest.approx(expected["psi"].sum())
    assert table.loc["now", "psi"] == pytest.approx(hand_psi([1, 1], [2, 2]))


def test_monitoring_table_integer_weights_equal_repeated_rows(applications):
    rng = np.random.default_rng(4)
    weights = rng.integers(0, 4, len(applications))
    repeat = np.repeat(np.arange(len(applications)), weights)
    score, target, month = (
        applications[column].to_numpy() for column in ("incumbent_pd", "outcome", "month")
    )
    edges = [-np.inf, 0.04, 0.07, 0.1, 0.15, 0.25, np.inf]
    reference = pd.Period("2024-03", "M")
    weighted = monitoring_table(
        score, month, target, reference=reference, bins=edges, weights=weights
    )
    repeated = monitoring_table(
        score[repeat], month[repeat], target[repeat], reference=reference, bins=edges
    )
    pd.testing.assert_frame_equal(weighted, repeated, check_dtype=False, rtol=1e-12)
    assert weighted["total"].sum() == weights.sum()

    development = make_applications(n_applicants=500, rng=9)["incumbent_pd"].to_numpy()
    development_weights = rng.integers(0, 3, len(development))
    sample = np.repeat(np.arange(len(development)), development_weights)
    weighted = monitoring_table(
        score,
        month,
        target,
        reference=development,
        bins=edges,
        weights=weights,
        reference_weights=development_weights,
    )
    repeated = monitoring_table(
        score[repeat], month[repeat], target[repeat], reference=development[sample], bins=edges
    )
    pd.testing.assert_frame_equal(weighted, repeated, check_dtype=False, rtol=1e-12)


def test_monitoring_table_weighted_rows_match_weighted_metrics(applications):
    rng = np.random.default_rng(6)
    weights = pd.Series(rng.uniform(0.5, 3.0, len(applications)), index=applications.index)
    score, target, month = (applications[c] for c in ("incumbent_pd", "outcome", "month"))
    reference = pd.Period("2024-01", "M")
    table = monitoring_table(score, month, target, reference=reference, bins=5, weights=weights)
    first = month == reference
    edges = bin_edges(score[first], bins=5, weights=weights[first])
    for label in table.index:
        rows = (month == label) & target.notna()
        w = weights[rows]
        assert table.loc[label, "total"] == pytest.approx(weights[month == label].sum())
        assert table.loc[label, "known"] == pytest.approx(w.sum())
        assert table.loc[label, "event"] == pytest.approx(w[target[rows] == 1].sum())
        assert table.loc[label, "auc"] == pytest.approx(
            roc_auc(score[rows], target[rows], weights=w)
        )
        assert table.loc[label, "ks"] == pytest.approx(ks(score[rows], target[rows], weights=w))
        expected = psi_table(
            pd.cut(score[first], edges),
            pd.cut(score[month == label], edges),
            reference_weights=weights[first],
            current_weights=weights[month == label],
        )
        assert table.loc[label, "psi"] == pytest.approx(expected["psi"].sum())


def test_monitoring_table_zero_weight_periods():
    score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    target = [0, 1, 0, 1, 0, 1]
    weights = [1, 1, 0, 0, 1, 1]
    edges = [-np.inf, 0.15, np.inf]
    table = monitoring_table(
        score, ["a", "a", "b", "b", "c", "c"], target, reference="a", bins=edges, weights=weights
    )
    # Repeated rows would have no period b.
    assert table.index.tolist() == ["a", "c"]
    categorical = pd.Categorical(["a", "a", "b", "b", "c", "c"])
    table = monitoring_table(score, categorical, target, reference="a", bins=edges, weights=weights)
    assert table.index.tolist() == ["a", "b", "c"]
    assert table.loc["b", "total"] == 0
    assert table.loc[["b"], ["event_rate", "auc", "psi"]].isna().all(axis=None)
    # A period whose events have no weight has no discrimination.
    table = monitoring_table(
        score, ["a"] * 6, target, reference="a", bins=edges, weights=[1, 0, 1, 0, 1, 0]
    )
    assert table["event"].tolist() == [0.0]
    assert np.isnan(table["auc"].iloc[0])


def test_monitoring_table_warns_when_ties_leave_fewer_bands():
    score = [0.1] * 6 + [0.2, 0.3]
    message = "tied edges leave 2 of the 4 requested bins; .*pass fewer bins or explicit edges"
    with pytest.warns(UserWarning, match=message) as caught:
        table = monitoring_table(score, ["a"] * 8, reference="a", bins=4)
    assert "special" not in str(caught[0].message)
    assert caught[0].filename == __file__
    assert table["psi"].tolist() == [0.0]


def test_monitoring_table_tuple_periods():
    period = pd.Series([(2024, 1)] * 3 + [(2024, 2)] * 3)
    score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    table = monitoring_table(score, period, reference=(2024, 1), bins=2)
    assert table.index.tolist() == [(2024, 1), (2024, 2)]
    # The tuple is a period label, not a sample of the scores 2024 and 1.
    assert table["psi"].iloc[0] == 0.0
    assert table["psi"].iloc[1] == pytest.approx(hand_psi([2, 1], [0, 3]))
    groups = pd.DataFrame({"housing": ["own", "rent", "own", "rent", "rent", "rent"]})
    stability = stability_table(groups, period, reference=(2024, 1))
    assert stability["housing"].iloc[0] == 0.0
    with pytest.raises(ValueError, match=r"got \(2024, 3\), which is not one of the periods"):
        monitoring_table(score, period, reference=(2024, 3))


def test_monitoring_table_date_string_selects_one_period():
    dates = pd.to_datetime(["2024-01-01"] * 2 + ["2024-02-01"] * 2)
    score = [0.1, 0.2, 0.3, 0.4]
    by_string = monitoring_table(score, dates, reference="2024-01", bins=2)
    by_date = monitoring_table(score, dates, reference=dates[0], bins=2)
    pd.testing.assert_frame_equal(by_string, by_date)
    with pytest.raises(
        ValueError,
        match="reference must name one period of period, got '2024', which matches 2 periods",
    ):
        monitoring_table(score, dates, reference="2024")


def test_monitoring_table_zero_dimensional_array_is_a_label():
    by_array = monitoring_table(SCORE, PERIOD, TARGET, reference=np.array("2024-01"), bins=2)
    by_label = monitoring_table(SCORE, PERIOD, TARGET, reference="2024-01", bins=2)
    pd.testing.assert_frame_equal(by_array, by_label)


def test_monitoring_table_lists_three_periods_in_the_error():
    message = "got 'e', which is not one of the periods 'a', 'b', 'c', ...$"
    with pytest.raises(ValueError, match=message):
        monitoring_table([0.1, 0.2, 0.3, 0.4], ["a", "b", "c", "d"], reference="e")


def test_monitoring_table_weighted_reference_sample_with_integer_bins(applications):
    rng = np.random.default_rng(10)
    development = make_applications(n_applicants=800, rng=11)["incumbent_pd"]
    development_weights = rng.uniform(0.2, 2.0, len(development))
    score, target, month = (applications[c] for c in ("incumbent_pd", "outcome", "month"))
    table = monitoring_table(
        score,
        month,
        target,
        reference=development,
        bins=6,
        reference_weights=development_weights,
    )
    edges = bin_edges(development, bins=6, weights=development_weights)
    pd.testing.assert_frame_equal(
        table,
        monitoring_table(
            score,
            month,
            target,
            reference=development,
            bins=edges,
            reference_weights=development_weights,
        ),
    )
    for label in table.index:
        expected = psi_table(
            pd.cut(development, edges),
            pd.cut(score[month == label], edges),
            reference_weights=development_weights,
        )
        assert table.loc[label, "psi"] == pytest.approx(expected["psi"].sum())


def test_monitoring_table_does_not_mutate_inputs(applications):
    frame = applications.copy()
    reference = make_applications(n_applicants=300, rng=2)["incumbent_pd"]
    reference_copy = reference.copy()
    weights = pd.Series(np.ones(len(frame)), index=frame.index)
    monitoring_table(
        frame["incumbent_pd"],
        frame["month"],
        frame["outcome"],
        reference=reference,
        weights=weights,
        reference_weights=np.ones(len(reference)),
    )
    pd.testing.assert_frame_equal(frame, applications)
    pd.testing.assert_series_equal(reference, reference_copy)
    assert (weights == 1).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"reference": "2024-04"},
            "reference must be a period of period or a reference sample, "
            "got '2024-04', which is not one of the periods '2024-01', '2024-02', '2024-03'",
        ),
        ({"reference": []}, "reference must not be empty"),
        ({"reference": [0.1, np.nan]}, r"reference must be finite, got 1 missing .* nan"),
        ({"reference": ["a", "b"]}, "reference must be real and numeric"),
        ({"reference": np.ones((2, 2))}, "reference must be one-dimensional"),
        (
            {"reference": "2024-01", "reference_weights": [1, 1, 1, 1]},
            "reference_weights must not be given with a reference period",
        ),
        (
            {"reference": [0.1, 0.2], "reference_weights": [1]},
            "reference_weights must have one weight per value of reference, got 1 for 2",
        ),
        ({"reference": "2024-01", "bins": 0}, "bins must be a positive integer or at least two"),
        ({"reference": "2024-01", "bins": [0.5, 0.2]}, "bins must be a positive integer"),
        (
            {"reference": "2024-01", "bins": [0.15, 0.5, 1.0]},
            r"score has 1 value\(s\) outside the edges in bins, such as 0.1",
        ),
        (
            {"reference": [0.0, 0.5], "bins": [0.05, 0.5, 1.0]},
            r"reference has 1 value\(s\) outside the edges in bins, such as 0.0",
        ),
        (
            {"reference": ["2024-01"]},
            "reference must be real and numeric scores or a period label, got dtype .*; pass "
            "a period label as a scalar, not in a list",
        ),
        ({"reference": np.array("2024-04")}, "got '2024-04', which is not one of the periods"),
        ({"reference": {"2024-01"}}, "reference must be a period of period"),
        (
            {
                "reference": pd.Series([0.1, 0.2], index=[1, 2]),
                "reference_weights": pd.Series([1, 1]),
            },
            "reference_weights must have the same index as reference",
        ),
        (
            {"reference": "2024-01", "bins": [0.05, 0.5, 0.85]},
            r"score has 1 value\(s\) outside the edges in bins, such as 0.9",
        ),
        ({"reference": "2024-01", "smoothing": 0}, "smoothing must be finite and positive"),
        ({"reference": "2024-01", "weights": [1] * 11}, "weights must have one weight per value"),
        (
            {"reference": "2024-01", "weights": [-1] + [1] * 11},
            "weights must be finite and non-negative",
        ),
    ],
)
def test_monitoring_table_rejects_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        monitoring_table(SCORE, PERIOD, TARGET, **kwargs)


def test_monitoring_table_rejects_invalid_data():
    with pytest.raises(ValueError, match="score and period must have the same length, got 3 and 2"):
        monitoring_table([0.1, 0.2, 0.3], ["a", "b"], reference="a")
    with pytest.raises(ValueError, match="score and period must not be empty"):
        monitoring_table([], [], reference="a")
    with pytest.raises(ValueError, match="target must have one label per value of score, got 2"):
        monitoring_table([0.1, 0.2, 0.3], ["a"] * 3, [0, 1], reference="a")
    with pytest.raises(ValueError, match="target must contain only 0, 1 and missing values, got 2"):
        monitoring_table([0.1, 0.2, 0.3], ["a"] * 3, [0, 1, 2], reference="a")
    with pytest.raises(ValueError, match=r"score must be finite, got 1 missing or infinite .* inf"):
        monitoring_table([0.1, np.inf, 0.3], ["a"] * 3, reference="a")
    with pytest.raises(ValueError, match="score must be real and numeric, got dtype"):
        monitoring_table(["x", "y"], ["a"] * 2, reference="a")
    with pytest.raises(ValueError, match="period must not have missing values, got 1"):
        monitoring_table([0.1, 0.2, 0.3], ["a", None, "a"], reference="a")
    categorical = pd.Categorical(["a", "a"], categories=["a", "b"])
    with pytest.raises(ValueError, match="reference period 'b' has no observations"):
        monitoring_table([0.1, 0.2], categorical, reference="b")
    with pytest.raises(ValueError, match="reference period 'b' has no observations with positive"):
        monitoring_table([0.1, 0.2], ["a", "b"], reference="b", weights=[1, 0])
    dates = pd.to_datetime(["2024-01-01", "2024-01-15"])
    # A partial date matches both periods.
    with pytest.raises(
        ValueError,
        match="reference must name one period of period, got '2024-01', which matches 2 periods",
    ):
        monitoring_table([0.1, 0.2], dates, reference="2024-01")
    score = pd.Series([0.1, 0.2], index=[1, 2])
    with pytest.raises(ValueError, match="score and period must have identical indices"):
        monitoring_table(score, pd.Series(["a", "a"]), reference="a")
    with pytest.raises(ValueError, match="score and target must have identical indices"):
        monitoring_table(score, ["a", "a"], pd.Series([0, 1]), reference="a")
    with pytest.raises(ValueError, match="score and weights must have identical indices"):
        monitoring_table(score, ["a", "a"], reference="a", weights=pd.Series([1, 1]))


# stability_table


@pytest.fixture(scope="module")
def bank():
    data = load_bank()
    rng = np.random.default_rng(12)
    quarters = pd.period_range("2024Q1", periods=4, freq="Q")
    data["quarter"] = quarters[rng.integers(0, 4, len(data))]
    edges = bin_edges(data["age"].iloc[:2000], bins=5)
    data["age_band"] = pd.cut(data["age"], edges)
    return data


VARIABLES = ["age_band", "job", "marital", "poutcome"]


def test_stability_table_matches_hand_calculation():
    groups = pd.DataFrame(
        {
            "housing": ["own", "own", "rent", "rent", "own", "rent", "rent", "rent"],
            "region": ["north", "south", "north", "south", "north", "north", "south", None],
        }
    )
    period = ["2024-01"] * 4 + ["2024-02"] * 4
    table = stability_table(groups, period, reference="2024-01")
    assert table.index.tolist() == ["2024-01", "2024-02"]
    assert table.index.name == "period"
    assert table.columns.tolist() == ["housing", "region"]
    assert table.columns.name == "variable"
    assert table.iloc[0].tolist() == [0.0, 0.0]
    assert table.loc["2024-02", "housing"] == pytest.approx(hand_psi([2, 2], [1, 3]))
    # north, south and missing: January has no missing region.
    assert table.loc["2024-02", "region"] == pytest.approx(hand_psi([2, 2, 0], [2, 1, 1]))


def test_stability_table_matches_psi_table_of_each_period(bank):
    reference = pd.Period("2024Q1", "Q")
    table = stability_table(bank[VARIABLES], bank["quarter"], reference=reference)
    assert isinstance(table.index, pd.PeriodIndex)
    assert table.columns.tolist() == VARIABLES
    first = bank[bank["quarter"] == reference]
    for label, group in bank.groupby("quarter"):
        for column in VARIABLES:
            expected = psi_table(first[column], group[column])["psi"].sum()
            assert table.loc[label, column] == pytest.approx(expected, rel=1e-12, abs=1e-15)
    assert (table.loc[reference] == 0).all()


def test_stability_table_reference_sample(bank):
    development, monitoring = bank.iloc[:2000], bank.iloc[2000:]
    table = stability_table(monitoring[VARIABLES], monitoring["quarter"], reference=development)
    for label, group in monitoring.groupby("quarter"):
        for column in VARIABLES:
            expected = psi_table(development[column], group[column])["psi"].sum()
            assert table.loc[label, column] == pytest.approx(expected, rel=1e-12)
    # Extra columns of the reference are ignored.
    wider = stability_table(monitoring[VARIABLES[1:]], monitoring["quarter"], reference=development)
    pd.testing.assert_frame_equal(wider, table[VARIABLES[1:]])


def test_stability_table_reference_with_other_dtype():
    groups = pd.DataFrame({"grade": pd.Categorical(["A", "B", "C", "C"])})
    reference = pd.DataFrame({"grade": ["A", "A", "B", "D"]})
    table = stability_table(groups, ["x", "x", "y", "y"], reference=reference)
    # A, B and D in the reference against A and B in x, without C, which neither has,
    # and against C and C in y.
    assert table.loc["x", "grade"] == pytest.approx(hand_psi([2, 1, 1], [1, 1, 0]))
    assert table.loc["y", "grade"] == pytest.approx(hand_psi([2, 1, 1, 0], [0, 0, 0, 2]))
    expected = psi_table(reference["grade"], groups["grade"].iloc[2:])["psi"].sum()
    assert table.loc["y", "grade"] == pytest.approx(expected)


def test_stability_table_integer_weights_equal_repeated_rows(bank):
    rng = np.random.default_rng(13)
    weights = rng.integers(0, 4, len(bank))
    repeat = np.repeat(np.arange(len(bank)), weights)
    reference = pd.Period("2024Q2", "Q")
    weighted = stability_table(
        bank[VARIABLES], bank["quarter"], reference=reference, weights=weights
    )
    repeated = stability_table(
        bank[VARIABLES].iloc[repeat], bank["quarter"].iloc[repeat], reference=reference
    )
    pd.testing.assert_frame_equal(weighted, repeated, rtol=1e-12)

    development = bank.iloc[:500]
    development_weights = rng.integers(0, 3, len(development))
    sample = np.repeat(np.arange(len(development)), development_weights)
    weighted = stability_table(
        bank[VARIABLES],
        bank["quarter"],
        reference=development,
        weights=weights,
        reference_weights=development_weights,
    )
    repeated = stability_table(
        bank[VARIABLES].iloc[repeat],
        bank["quarter"].iloc[repeat],
        reference=development.iloc[sample],
    )
    pd.testing.assert_frame_equal(weighted, repeated, rtol=1e-12)


def test_stability_table_weighted_rows_match_weighted_psi_table(bank):
    rng = np.random.default_rng(14)
    weights = pd.Series(rng.uniform(0.1, 3.0, len(bank)), index=bank.index)
    development = bank.iloc[:1500]
    development_weights = rng.uniform(0.5, 1.5, len(development))
    for reference, reference_weights in [
        (pd.Period("2024Q3", "Q"), None),
        (development, development_weights),
    ]:
        table = stability_table(
            bank[VARIABLES],
            bank["quarter"],
            reference=reference,
            weights=weights,
            reference_weights=reference_weights,
        )
        if reference_weights is None:
            rows = bank["quarter"] == reference
            reference, reference_weights = bank[rows], weights[rows]
        for label, group in bank.groupby("quarter"):
            for column in VARIABLES:
                expected = psi_table(
                    reference[column],
                    group[column],
                    reference_weights=reference_weights,
                    current_weights=weights[group.index],
                )
                assert table.loc[label, column] == pytest.approx(expected["psi"].sum(), rel=1e-12)


def test_stability_table_reference_with_other_interval_categories():
    # Bands cut on other edges: pd.concat would hash the interval categories, which warns
    # on NumPy < 2.2. Equal intervals still match.
    current = pd.DataFrame({"age": pd.cut([25, 35, 45, 35], [-np.inf, 30, 40, np.inf])})
    reference = pd.DataFrame({"age": pd.cut([25, 35, 25], [-np.inf, 30, np.inf])})
    table = stability_table(current, ["a", "a", "b", "b"], reference=reference)
    # (-inf, 30] twice and (30, inf] once in the reference; a has (-inf, 30] and (30, 40].
    assert table.loc["a", "age"] == pytest.approx(hand_psi([2, 1, 0], [1, 0, 1]))
    # b has (40, inf] and (30, 40].
    assert table.loc["b", "age"] == pytest.approx(hand_psi([2, 1, 0, 0], [0, 0, 1, 1]))
    reference = pd.DataFrame({"age": current["age"].iloc[:2].cat.remove_unused_categories()})
    table = stability_table(current, ["a", "a", "b", "b"], reference=reference)
    assert table.loc["a", "age"] == 0.0


def test_stability_table_empty_categorical_period():
    groups = pd.DataFrame({"housing": ["own", "rent", "own", "own"]})
    period = pd.Categorical(["q1", "q1", "q3", "q3"], categories=["q1", "q2", "q3"], ordered=True)
    table = stability_table(groups, period, reference="q1")
    assert table.index.tolist() == ["q1", "q2", "q3"]
    assert np.isnan(table.loc["q2", "housing"])
    assert table.loc["q3", "housing"] == pytest.approx(hand_psi([1, 1], [2, 0]))


def test_stability_table_does_not_mutate_inputs(bank):
    frame = bank.copy()
    stability_table(frame[VARIABLES], frame["quarter"], reference=frame.iloc[:100])
    pd.testing.assert_frame_equal(frame, bank)


def test_stability_table_rejects_invalid_arguments():
    groups = pd.DataFrame({"housing": ["own", "rent"], "region": ["n", "s"]})
    period = ["a", "b"]
    with pytest.raises(ValueError, match="groups must be a DataFrame, got Series"):
        stability_table(groups["housing"], period, reference="a")
    with pytest.raises(ValueError, match="groups must have at least one column"):
        stability_table(groups[[]], period, reference="a")
    nested = pd.DataFrame({("x", "housing"): ["own", "rent"]})
    with pytest.raises(ValueError, match="groups must have one level of column names, got 2"):
        stability_table(nested, period, reference="a")
    with pytest.raises(ValueError, match="groups must have unique column names, got 'housing'"):
        stability_table(groups[["housing", "housing"]], period, reference="a")
    with pytest.raises(ValueError, match="groups must have at least one row"):
        stability_table(groups.iloc[:0], [], reference="a")
    with pytest.raises(ValueError, match="period must have one value per row of groups, got 1"):
        stability_table(groups, ["a"], reference="a")
    with pytest.raises(ValueError, match="groups and period must have identical indices"):
        stability_table(groups, pd.Series(period, index=[5, 6]), reference="a")
    with pytest.raises(ValueError, match="reference must be a period of period or a DataFrame"):
        stability_table(groups, period, reference=["own", "rent"])
    with pytest.raises(ValueError, match="reference must be a period of period or a reference"):
        stability_table(groups, period, reference="c")
    with pytest.raises(ValueError, match="reference must have every column of groups, got no"):
        stability_table(groups, period, reference=groups[["housing"]])
    with pytest.raises(ValueError, match="reference must have at least one row"):
        stability_table(groups, period, reference=groups.iloc[:0])
    with pytest.raises(ValueError, match="reference must have unique column names"):
        stability_table(groups, period, reference=groups[["housing", "region", "housing"]])
    with pytest.raises(ValueError, match="reference_weights must not be given with a reference"):
        stability_table(groups, period, reference="a", reference_weights=[1, 1])
    with pytest.raises(ValueError, match="reference_weights must have the same index as reference"):
        stability_table(
            groups, period, reference=groups, reference_weights=pd.Series([1, 1], index=[3, 4])
        )
    with pytest.raises(ValueError, match="reference_weights must have one weight per value of"):
        stability_table(groups, period, reference=groups, reference_weights=[1])
    with pytest.raises(ValueError, match="weights must have one weight per value of groups"):
        stability_table(groups, period, reference="a", weights=[1])
    with pytest.raises(ValueError, match="reference period 'b' has no observations with positive"):
        stability_table(groups, period, reference="b", weights=[1, 0])
    with pytest.raises(ValueError, match="period must not have missing values"):
        stability_table(groups, ["a", None], reference="a")
    with pytest.raises(ValueError, match="smoothing must be finite and positive"):
        stability_table(groups, period, reference="a", smoothing=-1)
