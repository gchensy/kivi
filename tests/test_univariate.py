import math
import statistics
import warnings

import numpy as np
import pandas as pd
import pytest

from kivi.binning import bin_edges, cut
from kivi.datasets import load_bank
from kivi.univariate import profile_table
from kivi.woe import woe_table

BASE = [
    "kind",
    "count",
    "missing",
    "missing_share",
    "distinct",
    "top",
    "top_share",
    "special",
    "special_share",
    "mean",
    "std",
    "min",
    "q01",
    "q50",
    "q99",
    "max",
]
TARGETED = [
    "event_rate",
    "missing_event_rate",
    "non_missing_event_rate",
    "missing_lift",
    "non_missing_lift",
    "iv",
    "tied_bins",
    "gini",
]

FRAME = pd.DataFrame(
    {
        "x": [1, 2, 2, 3, 4, np.nan, -1, 5],
        "c": ["a", "b", "a", None, "b", "a", "c", "a"],
    }
)
TARGET = pd.Series([0, 0, 1, 0, 1, 1, 0, 1])


def test_profile_by_hand():
    profile = profile_table(FRAME, TARGET, bins=2, special={"x": [-1]})
    assert list(profile.columns) == BASE + TARGETED
    assert profile.index.name == "variable"
    assert profile.index.tolist() == ["x", "c"]
    x, c = profile.loc["x"], profile.loc["c"]

    assert x["kind"] == "numeric"
    assert (x["count"], x["missing"], x["distinct"], x["special"]) == (7, 1, 6, 1)
    assert x["missing_share"] == x["special_share"] == 1 / 8
    assert x["top"] == 2 and x["top_share"] == pytest.approx(2 / 7)
    # Summary of the regular values 1, 2, 2, 3, 4, 5, without the missing value and -1.
    assert x["mean"] == pytest.approx(17 / 6)
    assert x["std"] == pytest.approx(statistics.stdev([1, 2, 2, 3, 4, 5]))
    assert (x["min"], x["q50"], x["max"]) == (1, 2.5, 5)
    # Linear interpolation at positions 0.05 and 4.95 of the six sorted values.
    assert x["q01"] == pytest.approx(1.05) and x["q99"] == pytest.approx(4.95)
    assert x["event_rate"] == 0.5
    assert x["missing_event_rate"] == 1 and x["missing_lift"] == 2
    assert x["non_missing_event_rate"] == pytest.approx(3 / 7)
    assert x["non_missing_lift"] == pytest.approx(6 / 7)
    # Groups -1, (-inf, 2.5], (2.5, inf] and missing hold (non-event, event) counts (1, 0),
    # (2, 1), (1, 2) and (0, 1); with zeros smoothed to 0.5 every class share is a multiple of
    # 1 / 4.5 and every WOE is +-ln 2.
    assert x["iv"] == pytest.approx((0.5 + 1 + 1 + 0.5) / 4.5 * math.log(2))
    assert not x["tied_bins"]
    # Events 2, 4, 5 against non-events 1, 2, 3: 7.5 of 9 pairs, so AUC = 5/6.
    assert x["gini"] == pytest.approx(2 * 7.5 / 9 - 1)

    assert c["kind"] == "categorical"
    assert (c["count"], c["missing"], c["distinct"], c["special"]) == (7, 1, 3, 0)
    assert c["top"] == "a" and c["top_share"] == pytest.approx(4 / 7)
    assert all(np.isnan(c[name]) for name in ["mean", "std", "min", "q01", "q50", "q99", "max"])
    assert c["missing_event_rate"] == 0 and c["missing_lift"] == 0
    assert c["non_missing_event_rate"] == pytest.approx(4 / 7)
    # Groups a, b, c and missing: non-event shares 1/4 each, event shares 3, 1, 0.5, 0.5 of 5.
    expected = sum((e - 0.25) * math.log(e / 0.25) for e in [3 / 5, 1 / 5, 0.5 / 5, 0.5 / 5])
    assert c["iv"] == pytest.approx(expected)
    assert not c["tied_bins"]
    assert np.isnan(c["gini"])


def test_profile_iv_equals_the_quick_binning_with_the_public_functions():
    profile = profile_table(FRAME, TARGET, bins=2, special={"x": [-1]})
    edges = bin_edges(FRAME["x"], bins=2, special=[-1])
    groups = cut(FRAME["x"], edges, special=[-1])
    assert profile.loc["x", "iv"] == woe_table(groups, TARGET)["iv"].sum()
    assert profile.loc["c", "iv"] == woe_table(FRAME["c"], TARGET)["iv"].sum()


def test_profile_without_target_describes_only():
    profile = profile_table(FRAME)
    assert list(profile.columns) == BASE
    assert profile["count"].dtype == np.int64
    assert profile["distinct"].dtype == np.int64
    # Without special codes, -1 is a regular value.
    assert profile.loc["x", "min"] == -1
    assert profile.loc["x", "special"] == 0


def test_profile_reports_tied_quantiles_instead_of_warning():
    frame = pd.DataFrame({"flag": [0] * 8 + [1] * 2, "spread": np.arange(10.0)})
    target = [0, 1, 0, 0, 1, 0, 0, 0, 1, 1]
    profile = profile_table(frame, target)  # no warning: pytest turns warnings into errors
    assert profile["tied_bins"].tolist() == [True, False]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        edges = bin_edges(frame["flag"], bins=5)
    assert "tied edges" in str(caught[0].message)
    assert profile.loc["flag", "iv"] == woe_table(pd.cut(frame["flag"], edges), target)["iv"].sum()


def test_special_codes_are_values_but_not_numbers():
    frame = pd.DataFrame({"pdays": [-1, -1, -1, 10, 20, 30, 40, -1]})
    target = [0, 1, 0, 1, 0, 1, 1, 0]
    plain = profile_table(frame, target, bins=2).loc["pdays"]
    coded = profile_table(frame, target, bins=2, special={"pdays": [-1]}).loc["pdays"]
    # The code is the most frequent value either way.
    assert plain["top"] == coded["top"] == -1
    assert plain["top_share"] == coded["top_share"] == 0.5
    assert coded["special"] == 4 and coded["special_share"] == 0.5
    assert (coded["min"], coded["mean"], coded["max"]) == (10, 25, 40)
    assert plain["min"] == -1
    # Gini of 10, 20, 30, 40 with labels 1, 0, 1, 1: 2 of 3 pairs ranked right.
    assert coded["gini"] == pytest.approx(2 * 2 / 3 - 1)
    edges = bin_edges(frame["pdays"], bins=2, special=[-1])
    groups = cut(frame["pdays"], edges, special=[-1])
    assert coded["iv"] == woe_table(groups, target)["iv"].sum()


def test_infinite_special_code_and_all_missing_column():
    frame = pd.DataFrame(
        {"x": [np.inf, 1.0, 2.0, np.inf, 3.0, 4.0], "empty": [np.nan] * 6},
    )
    target = [1, 0, 1, 1, 0, 0]
    profile = profile_table(frame, target, special={"x": [np.inf]})
    assert profile.loc["x", "max"] == 4 and profile.loc["x", "special"] == 2
    empty = profile.loc["empty"]
    assert (empty["count"], empty["missing"], empty["distinct"]) == (0, 6, 0)
    assert empty["missing_share"] == 1 and np.isnan(empty["top"]) and np.isnan(empty["top_share"])
    assert np.isnan(empty["mean"]) and np.isnan(empty["gini"])
    assert np.isnan(empty["non_missing_event_rate"]) and empty["missing_lift"] == 1
    # One group, the missing values, carries no information.
    assert empty["iv"] == 0 and empty["tied_bins"]


def test_kinds_follow_fit_spec():
    frame = pd.DataFrame(
        {
            "flag": [True, False, True, True],
            "level": pd.Categorical(["b", "a", "b", "a"], categories=["a", "b"]),
            "codes": pd.Series([1, 2, 1, 2], dtype=object),
            "nullable": pd.Series([1, None, 3, 3], dtype="Int64"),
            "real": [0.5, 1.5, 2.5, 3.5],
        }
    )
    profile = profile_table(frame, [0, 1, 0, 1])  # an object column of numbers does not warn
    assert profile["kind"].tolist() == [
        "categorical",
        "categorical",
        "categorical",
        "numeric",
        "numeric",
    ]
    # Equally frequent values: the first category wins, else the first value seen.
    assert profile.loc["level", "top"] == "a"
    assert profile.loc["codes", "top"] == 1
    assert profile.loc["flag", "top"] is True
    assert profile.loc["nullable", "top"] == 3 and profile.loc["nullable", "missing"] == 1
    assert profile.loc["nullable", "mean"] == pytest.approx(7 / 3)


def test_profile_does_not_mutate_inputs():
    frame, target = FRAME.copy(), TARGET.copy()
    special = {"x": [-1]}
    profile_table(frame, target, special=special, weights=np.arange(1, 9))
    pd.testing.assert_frame_equal(frame, FRAME)
    pd.testing.assert_series_equal(target, TARGET)
    assert special == {"x": [-1]}


def _weighted_sample():
    rng = np.random.default_rng(11)
    n = 60
    frame = pd.DataFrame(
        {
            "x": np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n).round(2)),
            "code": np.where(rng.random(n) < 0.3, -1, rng.integers(0, 50, n)),
            "label": rng.choice(["a", "b", "c", None], n),
            "flag": rng.integers(0, 2, n),
        }
    )
    target = pd.Series(rng.integers(0, 2, n))
    weights = rng.integers(0, 4, n)
    return frame, target, weights


def test_integer_weights_equal_repeated_rows():
    frame, target, weights = _weighted_sample()
    special = {"code": [-1]}
    weighted = profile_table(frame, target, special=special, weights=weights)
    positions = np.repeat(np.arange(len(frame)), weights)
    rows = frame.iloc[positions].reset_index(drop=True)
    labels = target.iloc[positions].reset_index(drop=True)
    # With unit weights every statistic matches, including the weighted quantiles.
    unit = profile_table(rows, labels, special=special, weights=np.ones(len(rows)))
    pd.testing.assert_frame_equal(weighted, unit)
    # Without weights the quantiles interpolate, so only the other columns match.
    repeated = profile_table(rows, labels, special=special)
    exact = [name for name in BASE + TARGETED if name not in ("q01", "q50", "q99", "iv")]
    pd.testing.assert_frame_equal(weighted[exact], repeated[exact], check_dtype=False)
    # The weighted quantiles are those of NumPy's inverted CDF on the repeated rows.
    values = rows["x"].dropna().to_numpy()
    expected = [np.sort(values)[math.ceil(p * len(values)) - 1] for p in (0.01, 0.5, 0.99)]
    assert weighted.loc["x", ["q01", "q50", "q99"]].tolist() == expected


def test_zero_weights_drop_rows():
    frame, target, weights = _weighted_sample()
    weights = np.where(np.arange(len(frame)) % 3 == 0, 0, 1)
    weighted = profile_table(frame, target, weights=weights)
    kept = weights > 0
    dropped = profile_table(
        frame[kept].reset_index(drop=True),
        target[kept].reset_index(drop=True),
        weights=np.ones(kept.sum()),
    )
    pd.testing.assert_frame_equal(weighted, dropped)


def test_bank_profile_matches_the_iv_screening_of_the_risk_workflow():
    data = load_bank()
    development = data.iloc[:3000]
    profile = profile_table(development.drop(columns="target"), development["target"])
    iv = profile["iv"].sort_values(ascending=False).round(3)
    assert iv.head(4).to_dict() == {
        "duration": 1.594,
        "poutcome": 0.423,
        "month": 0.334,
        "contact": 0.215,
    }
    assert profile.index[profile["tied_bins"]].tolist() == ["campaign", "pdays", "previous"]
    assert profile.loc["pdays", "top"] == -1
    assert profile.loc["pdays", "top_share"] == 0.825
    coded = profile_table(development[["pdays"]], development["target"], special={"pdays": [-1]})
    assert not coded.loc["pdays", "tied_bins"]
    assert coded.loc["pdays", "iv"] > 0.1


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"frame": FRAME["x"]}, "frame must be a pandas DataFrame, got Series"),
        ({"frame": FRAME.iloc[:0]}, "at least one row"),
        ({"columns": ["x", "y"]}, "columns must be columns of frame, got \\['y'\\]"),
        ({"columns": ["x", "x"]}, "must not repeat a column"),
        ({"bins": 0}, "bins must be a positive integer, got 0"),
        ({"target": FRAME["x"].rename("c")}, "columns include the target 'c'"),
        ({"target": [0, 1, 2, 0, 1, 0, 1, 0]}, "target must contain both 0 and 1"),
        ({"target": [0] * 8}, "target must contain both 0 and 1"),
        ({"target": TARGET.set_axis(range(1, 9))}, "identical indices"),
        ({"target": [0, 1]}, "same length, got 8 and 2"),
        ({"weights": [-1] + [1] * 7}, "finite and non-negative"),
        ({"target": TARGET, "weights": 1 - TARGET}, "both target classes a positive total"),
        ({"special": [-1]}, "special must be a dict by column, got list"),
        ({"special": {"y": [-1]}}, "not profiled: \\['y'\\]"),
        ({"special": {"c": ["a"]}}, "categorical columns, which keep their labels: \\['c'\\]"),
        ({"special": {"x": ["-1"]}}, "special of column 'x' must be non-missing numbers"),
    ],
)
def test_invalid_profile_arguments(arguments, message):
    arguments = {"frame": FRAME, **arguments}
    with pytest.raises(ValueError, match=message):
        profile_table(**arguments)


def test_profile_rejects_infinite_values_that_are_not_special():
    frame = pd.DataFrame({"x": [1.0, np.inf, 2.0]})
    with pytest.raises(ValueError, match="column 'x' has infinite values"):
        profile_table(frame)
