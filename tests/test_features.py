"""Tests of kivi.features against a brute-force reference, pandas and saved files.

The reference computes every window of every observation with plain pandas,
one observation at a time, from the definitions in the docstring of
`window_features`. The golden files in ``tests/data`` were written by this
module run as a script, ``PYTHONPATH=src python tests/test_features.py``,
with the code that introduced the format of `FeatureSpec` (kivi 1.3.0):
``feature_spec_1.3.0.json`` with ``to_json(indent=2)`` and
``feature_spec_1.3.0.features.json`` with the features that the loaded
specification computes on `golden_tables`. Once 1.3.0 is released they must
not change: later releases read the file and compute the same features from
it, and a change that breaks that needs a new format version.
"""

import copy
import dataclasses
import json
import math
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

import kivi
from kivi import features
from kivi.features import FeatureSpec, window_features

DATA = pathlib.Path(__file__).parent / "data"
GOLDEN = DATA / "feature_spec_1.3.0.json"
GOLDEN_FEATURES = DATA / "feature_spec_1.3.0.features.json"
ALL = ["count", "sum", "mean", "min", "max", "std", "nunique", "first", "last", "days_since_last"]
INTEGERS = ("count", "nunique")
WINDOWS = ["1M", "3M", "12M", "30D", ("3M", "1M"), ("60D", "30D")]
AGGREGATIONS = {
    "amount": ALL,
    "kind": ["count", "nunique", "days_since_last"],
    "flag": ["sum", "mean", "first"],
}
KEYS = {"id": "customer", "time": "time", "observation_time": "date"}


def shift(t, bound):
    """The time ``bound``, such as "3M" or "90D", before ``t``, on the local calendar."""
    count, unit = int(bound[:-1]), bound[-1]
    offset = pd.DateOffset(months=count) if unit == "M" else pd.Timedelta(days=count)
    if t.tz is None:
        return t - offset
    local = t.tz_localize(None) - offset
    return local.tz_localize(t.tz, ambiguous=True, nonexistent="shift_forward")


def label(window):
    return "_".join(bound.lower() for bound in ((window,) if isinstance(window, str) else window))


def aggregate(name, present, times, t):
    """One aggregation of the non-missing values of a window, in the order of time."""
    if name == "count":
        return len(present)
    if name == "nunique":
        return present.nunique()
    if name == "days_since_last":
        return (t - times.iloc[-1]) / pd.Timedelta(days=1) if len(present) else math.nan
    numbers = present.astype(float)
    if not len(numbers):
        return 0.0 if name == "sum" else math.nan
    if name == "first":
        return numbers.iloc[0]
    if name == "last":
        return numbers.iloc[-1]
    return getattr(numbers, name)()


def brute_force(
    events,
    observations,
    *,
    id,
    time,
    observation_time,
    windows=("3M", "6M", "12M"),
    aggregations=None,
    ratios=None,
    closed="left",
    prefix="",
):
    """Compute the features of each observation with a loop and plain pandas."""
    aggregations = aggregations or {}
    names, stems = [], {}
    for window in windows:
        stems[f"{prefix}events_{label(window)}"] = (f"{prefix}events", label(window))
        for column, aggregation_names in aggregations.items():
            for name in aggregation_names:
                stems[f"{prefix}{column}_{name}_{label(window)}"] = (
                    f"{prefix}{column}_{name}",
                    label(window),
                )
    names = list(stems)
    rows = []
    for _, observation in observations.iterrows():
        t = observation[observation_time]
        mine = events[events[id] == observation[id]]
        # A stable sort keeps the order of the rows among events of one time.
        mine = mine.iloc[np.argsort(mine[time].to_numpy(), kind="stable")]
        row = {}
        for window in windows:
            outer, inner = (window, None) if isinstance(window, str) else window
            low, high = shift(t, outer), t if inner is None else shift(t, inner)
            if closed == "left":
                inside = mine[(mine[time] >= low) & (mine[time] < high)]
            else:
                inside = mine[(mine[time] > low) & (mine[time] <= high)]
            row[f"{prefix}events_{label(window)}"] = len(inside)
            for column, aggregation_names in aggregations.items():
                kept = inside[inside[column].notna()]
                for name in aggregation_names:
                    row[f"{prefix}{column}_{name}_{label(window)}"] = aggregate(
                        name, kept[column], kept[time], t
                    )
        rows.append(row)
    data = {}
    for name in names:
        integer = name.startswith(f"{prefix}events_") or stems[name][0].endswith(INTEGERS)
        data[name] = np.array([row[name] for row in rows], dtype=np.int64 if integer else float)
    for numerator, denominator in ratios or []:
        top, bottom = stems[numerator], stems[denominator]
        name = f"{numerator}_over_{bottom[1] if top[0] == bottom[0] else denominator}"
        with np.errstate(divide="ignore", invalid="ignore"):
            values = data[numerator] / data[denominator].astype(float)
        data[name] = np.where(data[denominator] == 0, np.nan, values)
    return pd.DataFrame(data, index=observations.index)


def assert_same(result, expected):
    """Equal columns, index and dtypes; counts exactly, floats to rounding."""
    assert list(result.columns) == list(expected.columns)
    assert result.index.equals(expected.index)
    for name in expected:
        assert result[name].dtype == expected[name].dtype, name
        if expected[name].dtype == np.int64:
            np.testing.assert_array_equal(result[name], expected[name], err_msg=name)
        else:
            np.testing.assert_allclose(
                result[name], expected[name], rtol=1e-10, atol=1e-9, err_msg=name
            )


def localized(times, zone):
    """Naive times as wall times of ``zone``, skipped ones shifted, repeated ones late."""
    if zone is None:
        return times
    return times.dt.tz_localize(
        zone, ambiguous=np.zeros(len(times), dtype=bool), nonexistent="shift_forward"
    )


def random_tables(seed, zone=None):
    """Events and observations with ties, month ends, events on window bounds and gaps."""
    rng = np.random.default_rng(seed)
    n = 300
    # A coarse grid of days and hours makes several events of one time frequent.
    days = pd.to_timedelta(rng.integers(0, 760, n), unit="D")
    hours = pd.to_timedelta(rng.choice([0, 0, 0, 2, 12], n), unit="h")
    events = pd.DataFrame(
        {
            "customer": rng.integers(0, 6, n),
            "time": localized(pd.Series(pd.Timestamp("2023-01-31") + days + hours), zone),
            "amount": rng.normal(50, 20, n).round(1),
            "kind": rng.choice(["cash", "shop", "fee"], n).astype(object),
            "flag": rng.random(n) < 0.3,
        }
    )
    events.loc[rng.random(n) < 0.1, "amount"] = np.nan
    events.loc[rng.random(n) < 0.1, "kind"] = None
    dates = pd.Series(
        pd.to_datetime(
            [
                "2024-02-29",
                "2024-03-31",
                "2024-05-31",
                "2023-03-31",
                "2024-01-31 12:00",
                "2024-12-31",
                "2024-03-01",
                "2024-10-27 02:30",
                "2024-04-30 02:30",
            ],
            format="ISO8601",
        )
    )
    observations = pd.DataFrame(
        {
            # Customer 9 has no events.
            "customer": rng.choice([0, 1, 2, 3, 4, 5, 9], 30),
            "date": localized(dates, zone).sample(30, replace=True, random_state=seed).to_numpy(),
        }
    )
    # Events exactly on the bounds of the windows of some observations, and at their times.
    bounds = []
    for _, observation in observations.iloc[:10].iterrows():
        t = observation["date"]
        for bound in [None, "1M", "3M", "12M", "30D", "60D"]:
            bounds.append(
                {
                    "customer": observation["customer"],
                    "time": t if bound is None else shift(t, bound),
                    "amount": float(rng.integers(1, 9)),
                    "kind": "bound",
                    "flag": True,
                }
            )
    events = pd.concat([events, pd.DataFrame(bounds)], ignore_index=True)
    # Several events of one time with other values.
    repeated = events.sample(40, random_state=seed).assign(amount=lambda x: x["amount"] + 1)
    events = pd.concat([events, repeated], ignore_index=True)
    events = events.sample(frac=1, random_state=seed)
    events.index = pd.Index(rng.permutation(len(events)) + 1000)
    observations.index = pd.Index([f"o{i}" for i in range(len(observations))])
    # A repeated observation, as when features are backfilled twice.
    observations = pd.concat([observations, observations.iloc[[0]]])
    return events, observations


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("closed", ["left", "right"])
@pytest.mark.parametrize("zone", [None, "Asia/Shanghai", "Europe/Berlin"])
def test_window_features_match_the_brute_force(seed, closed, zone):
    events, observations = random_tables(seed, zone)
    options = {"windows": WINDOWS, "aggregations": AGGREGATIONS, "closed": closed}
    ratios = [("amount_sum_1m", "amount_sum_3m_1m"), ("events_1m", "amount_count_12m")]
    result = window_features(events, observations, **KEYS, **options, ratios=ratios)
    assert_same(result, brute_force(events, observations, **KEYS, **options, ratios=ratios))


def test_the_brute_force_sees_the_cases_it_is_meant_to_cover():
    events, observations = random_tables(0)
    assert events.duplicated(["customer", "time"]).sum() > 20
    assert events["amount"].isna().any() and events["kind"].isna().any()
    merged = observations.merge(events, on="customer")
    assert (merged["time"] == merged["date"]).any()
    assert (merged["time"] == merged["date"] - pd.DateOffset(months=3)).any()
    assert (observations["date"] == pd.Timestamp("2024-02-29")).any()
    result = window_features(events, observations, **KEYS, windows=["1M"])
    assert (result["events_1m"] == 0).any()


def test_events_at_or_after_the_observation_time_never_count():
    events, observations = random_tables(2)
    observations = observations.drop_duplicates("customer")
    options = {"windows": WINDOWS, "aggregations": AGGREGATIONS}
    before = window_features(events, observations, **KEYS, **options)
    dates = events["customer"].map(observations.set_index("customer")["date"])
    later = (events["time"] >= dates).to_numpy()
    assert later.sum() > 10
    changed = events.copy()
    changed.loc[later, "amount"] = 1e12
    changed.loc[later, "kind"] = "leak"
    changed.loc[later, "flag"] = True
    # New events at and after the observation times.
    new = observations.assign(time=observations["date"], amount=-1e12, kind="new", flag=True)
    late = new.assign(time=new["time"] + pd.Timedelta(hours=1))
    columns = ["customer", "time", "amount", "kind", "flag"]
    changed = pd.concat([changed, new[columns], late[columns]], ignore_index=True)
    pd.testing.assert_frame_equal(window_features(changed, observations, **KEYS, **options), before)
    # With closed="right" the events at the observation time count, but not the later ones.
    right = {**options, "closed": "right"}
    after = changed.loc[
        (changed["time"] > changed["customer"].map(observations.set_index("customer")["date"]))
    ]
    kept = changed.drop(after.index)
    moved = changed.copy()
    moved.loc[after.index, "amount"] = 7e11
    pd.testing.assert_frame_equal(
        window_features(moved, observations, **KEYS, **right),
        window_features(kept, observations, **KEYS, **right),
    )


@pytest.mark.parametrize("closed", ["left", "right"])
def test_day_windows_match_pandas_rolling_at_the_times_of_the_events(closed):
    events, _ = random_tables(3)
    # pandas 2.0 ends every window at the current row, so the windows of events of one time
    # differ from those of an observation at that time; the brute force covers such ties.
    events = events.drop_duplicates(["customer", "time"])
    events = events.sort_values(["customer", "time"], kind="stable")
    observations = events[["customer", "time"]].rename(columns={"time": "date"})
    names = ["count", "sum", "mean", "min", "max", "std"]
    result = window_features(
        events, observations, **KEYS, windows=["90D"], aggregations={"amount": names}, closed=closed
    )
    rolling = events.groupby("customer").rolling("90D", on="time", closed=closed)["amount"]
    for name in names:
        # The groups come in the order of the customers and the rows in the order of time, as
        # the events are sorted.
        expected = getattr(rolling, name)().to_numpy()
        if name in ("count", "sum"):
            expected = np.nan_to_num(expected)
        np.testing.assert_allclose(
            result[f"amount_{name}_90d"], expected, rtol=1e-10, atol=1e-9, err_msg=name
        )


def test_a_worked_example_by_hand():
    events = pd.DataFrame(
        {
            "customer": ["a", "a", "a", "a", "b"],
            "time": pd.to_datetime(
                [
                    "2024-02-28 23:00",
                    "2024-02-29",
                    "2024-03-15 12:00",
                    "2024-03-31",
                    "2024-01-31",
                ],
                format="ISO8601",
            ),
            "amount": [10.0, 20.0, np.nan, 40.0, 5.0],
        }
    )
    observations = pd.DataFrame(
        {"customer": ["a", "b", "c"], "date": pd.to_datetime(["2024-03-31"] * 3)}
    )
    result = window_features(
        events,
        observations,
        **KEYS,
        windows=["1M"],
        aggregations={"amount": ["count", "sum", "first", "last", "days_since_last"]},
    )
    expected = pd.DataFrame(
        {
            # The month before 31 March starts on 29 February; 31 March is not before itself.
            "events_1m": [2, 0, 0],
            "amount_count_1m": [1, 0, 0],
            "amount_sum_1m": [20.0, 0.0, 0.0],
            "amount_first_1m": [20.0, np.nan, np.nan],
            "amount_last_1m": [20.0, np.nan, np.nan],
            "amount_days_since_last_1m": [31.0, np.nan, np.nan],
        }
    )
    pd.testing.assert_frame_equal(result, expected)
    # With closed="right" the event at the observation time counts but not the one on the
    # lower bound.
    right = window_features(events, observations, **KEYS, windows=["1M"], closed="right")
    assert right["events_1m"].tolist() == [2, 0, 0]


def test_days_since_last_counts_fractions_of_a_day_from_the_observation_time():
    events = pd.DataFrame(
        {"customer": [1, 1], "time": pd.to_datetime(["2024-01-01 06:00", "2024-03-30 12:00"])}
    )
    events["x"] = 1.0
    observations = pd.DataFrame({"customer": [1], "date": pd.to_datetime(["2024-03-31"])})
    result = window_features(
        events,
        observations,
        **KEYS,
        windows=["3M", ("3M", "1M")],
        aggregations={"x": ["days_since_last"]},
    )
    assert result["x_days_since_last_3m"].tolist() == [0.5]
    assert result["x_days_since_last_3m_1m"].tolist() == [89.75]


def test_bounds_in_the_hours_that_clocks_skip_or_repeat():
    zone = "Europe/Berlin"
    events = pd.DataFrame(
        {
            "customer": [1, 1, 2, 2, 3],
            "time": pd.to_datetime(
                [
                    # 03:00 CEST, the end of the hour skipped on 31 March, and a second before.
                    "2024-03-31 01:00:00",
                    "2024-03-31 00:59:59",
                    # 02:15 CEST before the repeated hour of 27 October, and 02:45 CET in it.
                    "2024-10-27 00:15",
                    "2024-10-27 01:45",
                    # 02:45 CEST, in the first occurrence of the repeated hour.
                    "2024-10-27 00:45",
                ],
                format="ISO8601",
                utc=True,
            ).tz_convert(zone),
        }
    )
    observations = pd.DataFrame(
        {
            "customer": [1, 2, 3],
            "date": pd.to_datetime(
                ["2024-04-30 02:30", "2024-11-27 02:30"] + ["2024-11-27 02:30"]
            ).tz_localize(zone),
        }
    )
    result = window_features(events, observations, **KEYS, windows=["30D", "1M"])
    # 30 days before 30 April 02:30 is 31 March 02:30, which does not exist: 03:00 instead.
    # 30 days before 27 November is 28 October, after both events of customer 2.
    assert result["events_30d"].tolist() == [1, 0, 0]
    # A month before 27 November 02:30 is 02:30 on 27 October, taken at its first occurrence,
    # in summer time: the event at 02:45 in winter time counts and the one at 02:15 in summer
    # time does not. The event of customer 3 at 02:45 in summer time lies between the two
    # occurrences of 02:30, so it counts only because the first one is taken.
    assert result["events_1m"].tolist() == [2, 1, 1]


def test_sums_keep_the_precision_of_each_entity():
    events = pd.DataFrame(
        {
            "customer": [1, 1, 2, 2],
            "time": pd.to_datetime(["2024-01-02", "2024-01-03"] * 2),
            "amount": [1e17, 3.0, 0.1, 0.2],
        }
    )
    observations = pd.DataFrame({"customer": [1, 2], "date": pd.to_datetime(["2024-02-01"] * 2)})
    result = window_features(
        events, observations, **KEYS, windows=["1M"], aggregations={"amount": ["sum"]}
    )
    assert result["amount_sum_1m"].tolist() == [1e17 + 3.0, 0.1 + 0.2]


def test_sums_keep_their_precision_after_large_values_of_the_same_entity():
    events = pd.DataFrame(
        {
            "customer": [1, 1, 1],
            "time": pd.to_datetime(["2020-01-01", "2024-03-10", "2024-03-20"]),
            "amount": [1e17, 0.1, 0.2],
        }
    )
    observations = pd.DataFrame({"customer": [1], "date": pd.to_datetime(["2024-03-31"])})
    result = window_features(
        events, observations, **KEYS, windows=["1M"], aggregations={"amount": ["sum", "mean"]}
    )
    assert result.iloc[0].tolist() == [2, 0.1 + 0.2, (0.1 + 0.2) / 2]
    # Large values at random places before small windows of one entity.
    rng = np.random.default_rng(11)
    n = 2000
    events = pd.DataFrame(
        {
            "customer": 1,
            "time": pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n) * 12, unit="h"),
            "amount": np.where(rng.random(n) < 0.05, 1e15, rng.random(n)),
        }
    )
    observations = pd.DataFrame(
        {"customer": 1, "date": events["time"].sample(200, random_state=0).to_numpy()}
    )
    result = window_features(
        events, observations, **KEYS, windows=["7D", "90D"], aggregations={"amount": ["sum"]}
    )
    for _, row in observations.iterrows():
        for days in (7, 90):
            inside = events.loc[
                (events["time"] >= row["date"] - pd.Timedelta(days=days))
                & (events["time"] < row["date"]),
                "amount",
            ]
            expected = math.fsum(inside)
            got = result.loc[_, f"amount_sum_{days}d"]
            assert got == pytest.approx(expected, rel=1e-13, abs=1e-13), (row["date"], days)


def test_windows_are_expanded_in_chunks(monkeypatch):
    events, observations = random_tables(12)
    options = {"windows": WINDOWS, "aggregations": AGGREGATIONS}
    whole = window_features(events, observations, **KEYS, **options)
    # Chunks smaller than some windows, which then form chunks of their own.
    monkeypatch.setattr(features, "_CHUNK", 5)
    assert [(c.start, c.stop) for c in features._chunks(np.array([2, 2, 9, 0, 1, 4]))] == [
        (0, 2),
        (2, 3),
        (3, 6),
    ]
    pd.testing.assert_frame_equal(window_features(events, observations, **KEYS, **options), whole)


def test_std_is_accurate_for_large_values_with_a_small_spread():
    values = 1e9 + np.array([1.0, 2.0, 3.0, 4.0])
    events = pd.DataFrame(
        {"customer": 1, "time": pd.date_range("2024-01-01", periods=4), "amount": values}
    )
    observations = pd.DataFrame({"customer": [1], "date": pd.to_datetime(["2024-02-01"])})
    result = window_features(
        events, observations, **KEYS, windows=["1M"], aggregations={"amount": ["std"]}
    )
    assert result["amount_std_1m"].iloc[0] == pytest.approx(pd.Series([1.0, 2, 3, 4]).std())


def test_text_and_boolean_columns():
    events = pd.DataFrame(
        {
            "customer": [1, 1, 1, 1],
            "time": pd.date_range("2024-01-01", periods=4),
            "kind": pd.Series(["cash", None, "shop", "cash"], dtype="string"),
            "flag": pd.array([True, pd.NA, False, True], dtype="boolean"),
            "count": pd.array([1, 2, pd.NA, 4], dtype="Int64"),
        }
    )
    observations = pd.DataFrame({"customer": [1], "date": pd.to_datetime(["2024-02-01"])})
    result = window_features(
        events,
        observations,
        **KEYS,
        windows=["1M"],
        aggregations={
            "kind": ["count", "nunique"],
            "flag": ["sum", "mean"],
            "count": ["sum", "max"],
        },
    )
    assert result.iloc[0].tolist() == [4, 3, 2, 2.0, 2 / 3, 7.0, 4.0]


def test_ratios_name_their_columns_and_divide_by_zero_into_nan():
    events, observations = random_tables(4)
    # Customer 8 has no events.
    observations = pd.concat([observations, observations.iloc[:1].assign(customer=8)])
    result = window_features(
        events,
        observations,
        **KEYS,
        windows=["1M", "3M"],
        aggregations={"amount": ["sum"]},
        ratios=[("p_amount_sum_1m", "p_amount_sum_3m"), ("p_amount_sum_1m", "p_events_3m")],
        prefix="p_",
    )
    assert list(result.columns[-2:]) == [
        "p_amount_sum_1m_over_3m",
        "p_amount_sum_1m_over_p_events_3m",
    ]
    zero = result["p_amount_sum_3m"] == 0
    assert zero.any() and result.loc[zero, "p_amount_sum_1m_over_3m"].isna().all()
    expected = result["p_amount_sum_1m"] / result["p_events_3m"].replace(0, np.nan)
    pd.testing.assert_series_equal(
        result["p_amount_sum_1m_over_p_events_3m"], expected, check_names=False
    )


def test_column_names_and_the_prefix_keep_their_case():
    events = pd.DataFrame(
        {
            "customer": [1, 1],
            "time": pd.to_datetime(["2024-01-05", "2024-01-06"]),
            "Amount": [1.0, 2.0],
            "amount": [10.0, 20.0],
        }
    )
    observations = pd.DataFrame({"customer": [1], "date": pd.to_datetime(["2024-02-01"])})
    result = window_features(
        events,
        observations,
        **KEYS,
        windows=["1M", ("2M", "1M")],
        aggregations={"Amount": ["sum"], "amount": ["sum"]},
        ratios=[("Cash_Amount_sum_1m", "Cash_amount_sum_1m")],
        prefix="Cash_",
    )
    assert list(result.columns) == [
        "Cash_events_1m",
        "Cash_Amount_sum_1m",
        "Cash_amount_sum_1m",
        "Cash_events_2m_1m",
        "Cash_Amount_sum_2m_1m",
        "Cash_amount_sum_2m_1m",
        "Cash_Amount_sum_1m_over_Cash_amount_sum_1m",
    ]
    assert result.iloc[0].tolist() == [2, 3.0, 30.0, 0, 0.0, 0.0, 0.1]


def test_a_pair_is_one_window_only_inside_the_list():
    events, observations = random_tables(13)
    pair = window_features(events, observations, **KEYS, windows=[("6M", "3M")])
    two = window_features(events, observations, **KEYS, windows=("6M", "3M"))
    assert list(pair.columns) == ["events_6m_3m"]
    assert list(two.columns) == ["events_6m", "events_3m"]
    pd.testing.assert_series_equal(
        pair["events_6m_3m"], two["events_6m"] - two["events_3m"], check_names=False
    )


def test_columns_come_window_by_window_with_a_prefix():
    events, observations = random_tables(5)
    result = window_features(
        events,
        observations,
        **KEYS,
        windows=["1M", ("2M", "1M")],
        aggregations={"amount": ["sum", "max"], "kind": ["nunique"]},
        prefix="all_",
    )
    assert list(result.columns) == [
        "all_events_1m",
        "all_amount_sum_1m",
        "all_amount_max_1m",
        "all_kind_nunique_1m",
        "all_events_2m_1m",
        "all_amount_sum_2m_1m",
        "all_amount_max_2m_1m",
        "all_kind_nunique_2m_1m",
    ]
    default = window_features(events, observations, **KEYS)
    assert list(default.columns) == ["events_3m", "events_6m", "events_12m"]
    assert (default.dtypes == np.int64).all()


def test_the_result_follows_the_rows_of_the_observations():
    events, observations = random_tables(6)
    result = window_features(events, observations, **KEYS, aggregations={"amount": ["sum"]})
    assert result.index.equals(observations.index)
    shuffled = observations.iloc[::-1]
    pd.testing.assert_frame_equal(
        window_features(events, shuffled, **KEYS, aggregations={"amount": ["sum"]}),
        result.iloc[::-1],
    )
    assert result.iloc[0].equals(result.iloc[-1])


def test_empty_tables():
    events, observations = random_tables(7)
    options = {
        "windows": ["1M"],
        "aggregations": AGGREGATIONS,
        "ratios": [("events_1m", "amount_sum_1m")],
    }
    empty = window_features(events, observations.iloc[:0], **KEYS, **options)
    assert empty.shape == (0, 18) and empty.index.equals(observations.index[:0])
    full = window_features(events, observations, **KEYS, **options)
    pd.testing.assert_series_equal(empty.dtypes, full.dtypes)
    nothing = window_features(events.iloc[:0], observations, **KEYS, **options)
    assert_same(nothing, brute_force(events.iloc[:0], observations, **KEYS, **options))
    assert (nothing.select_dtypes("int64") == 0).all().all()
    assert nothing["amount_sum_1m"].eq(0).all() and nothing["amount_mean_1m"].isna().all()


def test_ids_of_other_dtypes_and_absent_entities():
    events = pd.DataFrame(
        {
            "customer": pd.Categorical(["x", "y", "x"]),
            "time": pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07"]),
        }
    )
    observations = pd.DataFrame(
        {"customer": ["x", "z", "y"], "date": pd.to_datetime(["2024-02-01"] * 3)}
    )
    result = window_features(events, observations, **KEYS, windows=["1M"])
    assert result["events_1m"].tolist() == [2, 0, 1]


def test_times_of_other_resolutions():
    events = pd.DataFrame(
        {"customer": [1, 1], "time": pd.to_datetime(["2024-01-05", "2024-01-31"]).as_unit("s")}
    )
    observations = pd.DataFrame(
        {"customer": [1], "date": pd.to_datetime(["2024-02-29"]).as_unit("ms")}
    )
    result = window_features(events, observations, **KEYS, windows=["1M"])
    assert result["events_1m"].tolist() == [1]


def test_inputs_are_left_unchanged():
    events, observations = random_tables(8)
    events = events.assign(time_copy=events["time"])
    before = (events.copy(deep=True), observations.copy(deep=True))
    window_features(
        events,
        observations,
        **KEYS,
        windows=WINDOWS,
        aggregations=AGGREGATIONS,
        ratios=[("events_1m", "events_3m")],
    )
    spec = FeatureSpec(
        id="customer",
        observation_time="date",
        events=[{"time": "time", "aggregations": AGGREGATIONS}],
    )
    spec.compute(observations, events={"events": events})
    pd.testing.assert_frame_equal(events, before[0])
    pd.testing.assert_frame_equal(observations, before[1])


@pytest.mark.parametrize(
    "change,message",
    [
        ({"events": [1]}, "^events must be a DataFrame, got list$"),
        ({"observations": None}, "^observations must be a DataFrame, got NoneType$"),
        ({"id": 1}, "^id must be a string, got 1$"),
        ({"time": ["time"]}, r"^time must be a string, got \['time'\]$"),
        ({"observation_time": 1.5}, "^observation_time must be a string, got 1.5$"),
        ({"id": "x"}, "^id must name a column of observations, got 'x'$"),
        ({"time": "x"}, "^time must name a column of events, got 'x'$"),
        (
            {"observation_time": "x"},
            "^observation_time must name a column of observations, got 'x'$",
        ),
        (
            {"time": "amount"},
            "^time column 'amount' of events must hold datetimes, got dtype float64; convert it "
            "with pandas.to_datetime$",
        ),
        (
            {"windows": "3M"},
            r"^windows must be a list of windows such as '3M', '90D' or \('6M', '3M'\), got '3M'; "
            r"pass \['3M'\] for one window$",
        ),
        (
            {"windows": pd.Series([1.0, 2.0])},
            r"^windows must be a list of windows .*, got \[1.0, 2.0\]$",
        ),
        ({"windows": []}, "^windows must have at least one window$"),
        (
            {"windows": ["3m"]},
            "^windows must count months or days such as '3M' or '90D', got '3m'$",
        ),
        ({"windows": ["0M"]}, "^windows must count months or days .*, got '0M'$"),
        ({"windows": ["1Y"]}, "^windows must count months or days .*, got '1Y'$"),
        (
            {"windows": [3]},
            r"^each window of windows must be a string such as '3M' or a pair such as \('6M', "
            r"'3M'\), got 3$",
        ),
        (
            {"windows": [("6M", "3M", "1M")]},
            r"^each window of windows must be .*, got \('6M', '3M', '1M'\)$",
        ),
        (
            {"windows": [("3M", "6M")]},
            r"^a pair of windows must reach further back with its first bound, in the unit of its "
            r"second, such as \('6M', '3M'\), got \('3M', '6M'\)$",
        ),
        ({"windows": [["3M", "60D"]]}, r"^a pair of windows must .*, got \('3M', '60D'\)$"),
        ({"windows": [("3M", "3M")]}, r"^a pair of windows must .*, got \('3M', '3M'\)$"),
        ({"windows": ["3M", "6M", "3M"]}, "^windows must not repeat a window, got '3M'$"),
        (
            {"aggregations": ["sum"]},
            r"^aggregations must be a dict from column names to lists of aggregations, got "
            r"\['sum'\]$",
        ),
        (
            {"aggregations": pd.Series([1.0, 2.0])},
            r"^aggregations must be a dict .*, got \[1.0, 2.0\]$",
        ),
        ({"aggregations": {1: ["sum"]}}, "^column names must be strings, got 1$"),
        (
            {"aggregations": {"amount": "sum"}},
            r"^aggregations of column 'amount' must be a list such as \['sum', 'max'\], got "
            r"'sum'; pass \['sum'\] for one aggregation$",
        ),
        (
            {"aggregations": {"amount": []}},
            "^aggregations of column 'amount' must name at least one aggregation$",
        ),
        (
            {"aggregations": {"amount": ["median"]}},
            r"^aggregations must be among \['count', 'sum', 'mean', 'min', 'max', 'std', "
            r"'nunique', 'first', 'last', 'days_since_last'\], got 'median'$",
        ),
        ({"aggregations": {"amount": [None]}}, "^aggregations must be among .*, got None$"),
        (
            {"aggregations": {"amount": ["sum", "max", "sum"]}},
            "^aggregations of column 'amount' must not repeat an aggregation, got 'sum'$",
        ),
        ({"aggregations": {"x": ["sum"]}}, "^aggregations must name a column of events, got 'x'$"),
        (
            {"aggregations": {"time": ["max"]}},
            "^column 'time' of events must be real and numeric, got dtype datetime64.*; only "
            "count, nunique and days_since_last take values other than numbers$",
        ),
        (
            {"aggregations": {"kind": ["sum"]}},
            "^column 'kind' of events must be real and numeric, got dtype object with the value "
            "'a'; only count, nunique and days_since_last take values other than numbers$",
        ),
        (
            {"ratios": ("events_3m", "events_6m")},
            r"^each ratio of ratios must be a pair of column names such as \('amount_sum_3m', "
            r"'amount_sum_12m'\), got 'events_3m'$",
        ),
        (
            {"ratios": "events_3m"},
            "^ratios must be a list of pairs of column names, got 'events_3m'$",
        ),
        ({"ratios": [("events_3m",)]}, r"^each ratio of ratios must be .*, got \('events_3m',\)$"),
        (
            {"ratios": [("events_3m", 1)]},
            r"^each ratio of ratios must be .*, got \('events_3m', 1\)$",
        ),
        (
            {"ratios": [("events_3m", "events_1m")]},
            "^ratios must name features of the windows and aggregations, such as 'events_3m', got "
            "'events_1m'$",
        ),
        (
            {"ratios": [("events_3m", "events_3m")]},
            r"^ratios must divide two different features, got \('events_3m', 'events_3m'\)$",
        ),
        (
            {"ratios": [("events_3m", "events_6m"), ["events_3m", "events_6m"]]},
            "^the result would have two columns named 'events_3m_over_6m'; remove the repeated "
            "ratio$",
        ),
        ({"closed": "both"}, "^closed must be 'left' or 'right', got 'both'$"),
        (
            {"closed": pd.Series([1.0, 2.0])},
            r"^closed must be 'left' or 'right', got \[1.0, 2.0\]$",
        ),
        ({"prefix": 1}, "^prefix must be a string, got 1$"),
    ],
)
def test_invalid_arguments_raise(change, message):
    events = pd.DataFrame(
        {
            "customer": [1, 2],
            "time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "amount": [1.0, 2.0],
            "kind": ["a", "b"],
        }
    ).astype({"kind": object})
    observations = pd.DataFrame({"customer": [1], "date": pd.to_datetime(["2024-02-01"])})
    arguments = {"events": events, "observations": observations, **KEYS, **change}
    with pytest.raises(ValueError, match=message):
        window_features(**arguments)


@pytest.mark.parametrize(
    "change,message",
    [
        (
            lambda e, o: (e.assign(customer=[1, None]), o),
            "^id column 'customer' of events must not contain missing values, got 1$",
        ),
        (
            lambda e, o: (e, o.assign(customer=[np.nan])),
            "^id column 'customer' of observations must not contain missing values, got 1$",
        ),
        (
            lambda e, o: (e.assign(time=[pd.NaT, pd.Timestamp("2024-01-01")]), o),
            "^time column 'time' of events must not contain missing values, got 1$",
        ),
        (
            lambda e, o: (e, o.assign(date=pd.to_datetime(["2024-01-01"]).tz_localize("UTC"))),
            "^the times of events and of observations must both be tz-naive or share one time "
            "zone, got tz-naive and 'UTC'$",
        ),
        (
            lambda e, o: (
                e.assign(time=e["time"].dt.tz_localize("Asia/Shanghai")),
                o.assign(date=o["date"].dt.tz_localize("UTC")),
            ),
            "^the times of events and of observations must both be tz-naive or share one time "
            "zone, got 'Asia/Shanghai' and 'UTC'$",
        ),
        (
            lambda e, o: (e.assign(amount=[1.0, np.inf]), o),
            "^column 'amount' of events must hold finite numbers or missing values, got inf$",
        ),
        (
            lambda e, o: (e.set_axis(["customer", "time", "customer"], axis=1), o),
            "^events has more than one column named 'customer'$",
        ),
        (
            lambda e, o: (e, o.assign(date=["2024-02-01"]).astype({"date": object})),
            "^observation_time column 'date' of observations must hold datetimes, got dtype "
            "object; convert it with pandas.to_datetime$",
        ),
    ],
)
def test_invalid_tables_raise(change, message):
    events = pd.DataFrame(
        {
            "customer": [1, 2],
            "time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "amount": [1.0, 2.0],
        }
    )
    observations = pd.DataFrame({"customer": [1], "date": pd.to_datetime(["2024-02-01"])})
    events, observations = change(events, observations)
    with pytest.raises(ValueError, match=message):
        window_features(events, observations, **KEYS, aggregations={"amount": ["sum"]})


# FeatureSpec


def two_groups(**changes):
    """A specification of all events and of the cash withdrawals, with ratios across them."""
    fields = {
        "id": "customer",
        "observation_time": "date",
        "events": [
            {"time": "time", "windows": WINDOWS, "aggregations": AGGREGATIONS},
            {
                "table": "cash",
                "time": "time",
                "windows": ["3M", ("6M", "3M")],
                "aggregations": {"amount": ["sum", "max"]},
                "closed": "right",
                "prefix": "cash_",
            },
        ],
        "ratios": [("cash_amount_sum_3m", "amount_sum_3m_1m"), ("events_1m", "events_3m")],
    }
    return FeatureSpec(**{**fields, **changes})


def test_compute_gives_the_result_of_window_features():
    events, observations = random_tables(9)
    cash = events[events["kind"] == "cash"]
    spec = two_groups()
    result = spec.compute(observations, events={"events": events, "cash": cash})
    first = window_features(
        events, observations, **KEYS, windows=WINDOWS, aggregations=AGGREGATIONS
    )
    second = window_features(
        cash,
        observations,
        **KEYS,
        windows=["3M", ("6M", "3M")],
        aggregations={"amount": ["sum", "max"]},
        closed="right",
        prefix="cash_",
    )
    expected = pd.concat([first, second], axis=1)
    expected["cash_amount_sum_3m_over_amount_sum_3m_1m"] = expected[
        "cash_amount_sum_3m"
    ] / expected["amount_sum_3m_1m"].replace(0, np.nan)
    expected["events_1m_over_3m"] = expected["events_1m"] / expected["events_3m"].replace(0, np.nan)
    pd.testing.assert_frame_equal(result, expected)
    assert list(spec.columns) == list(result.columns)
    one = FeatureSpec(
        id="customer",
        observation_time="date",
        events=[{"time": "time", "windows": ["1M"], "aggregations": {"amount": ["sum"]}}],
        ratios=[("amount_sum_1m", "events_1m")],
    )
    pd.testing.assert_frame_equal(
        one.compute(observations, events=events),
        window_features(
            events,
            observations,
            **KEYS,
            windows=["1M"],
            aggregations={"amount": ["sum"]},
            ratios=[("amount_sum_1m", "events_1m")],
        ),
    )


def test_a_specification_normalizes_its_fields_and_is_read_only():
    spec = FeatureSpec(id="customer", observation_time="date", events=[{"time": "time"}])
    group = spec.events[0]
    assert dict(group) == {
        "table": "events",
        "time": "time",
        "windows": ("3M", "6M", "12M"),
        "aggregations": {},
        "closed": "left",
        "prefix": "",
    }
    assert spec.ratios == () and spec.panel is None and spec.kivi_version == kivi.__version__
    with pytest.raises(TypeError):
        group["time"] = "other"  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.id = "other"  # type: ignore[misc]
    spec = two_groups()
    assert spec.events[1]["windows"] == ("3M", ("6M", "3M"))
    assert spec.events[0]["aggregations"]["kind"] == ("count", "nunique", "days_since_last")
    assert spec.ratios == (
        ("cash_amount_sum_3m", "amount_sum_3m_1m"),
        ("events_1m", "events_3m"),
    )
    assert repr(spec).startswith("FeatureSpec(columns=['events_1m', 'amount_count_1m', ")
    assert spec.columns[-1] == "events_1m_over_3m" and len(spec.columns) == 6 * 17 + 2 * 3 + 2


def test_round_trips_equality_and_replace():
    spec = two_groups()
    assert FeatureSpec.from_json(spec.to_json()) == spec
    assert FeatureSpec.from_json(spec.to_json(indent=2).encode()) == spec
    assert FeatureSpec.from_dict(json.loads(spec.to_json())) == spec
    assert pickle.loads(pickle.dumps(spec)) == spec
    assert copy.deepcopy(spec) == spec
    assert hash(two_groups()) == hash(spec) and len({spec, two_groups()}) == 1
    assert spec != two_groups(kivi_version="0.0.0")
    assert spec != two_groups(ratios=())
    assert spec.__eq__("spec") is NotImplemented
    replaced = dataclasses.replace(spec, ratios=[("events_1m", "events_12m")])
    assert replaced.columns[-1] == "events_1m_over_12m"
    assert dataclasses.replace(spec) == spec
    with pytest.raises(ValueError, match="^ratios must name features"):
        dataclasses.replace(spec, ratios=[("events_1m", "events_2m")])


def test_compute_takes_one_table_or_a_dict_of_tables():
    events, observations = random_tables(10)
    spec = FeatureSpec(
        id="customer",
        observation_time="date",
        events=[
            {"table": "tx", "time": "time", "windows": ["1M"]},
            {"table": "tx", "time": "time", "windows": ["1M"], "prefix": "again_"},
        ],
    )
    pd.testing.assert_frame_equal(
        spec.compute(observations, events=events), spec.compute(observations, events={"tx": events})
    )
    with pytest.raises(ValueError, match=r"^events\['tx'\] must be a DataFrame, got list$"):
        spec.compute(observations, events={"tx": []})
    with pytest.raises(
        ValueError, match=r"^time must name a column of events\['tx'\], got 'time'$"
    ):
        spec.compute(observations, events={"tx": events.drop(columns="time")})
    spec = two_groups()
    with pytest.raises(
        ValueError,
        match=r"^events must be a dict from table names to DataFrames, because the groups read "
        r"the tables \['events', 'cash'\]$",
    ):
        spec.compute(observations, events=events)
    with pytest.raises(
        ValueError, match="^events must be a DataFrame or a dict of DataFrames, got list$"
    ):
        spec.compute(observations, events=[events])
    with pytest.raises(
        ValueError, match="^events must have a table for every group, but lacks 'cash'$"
    ):
        spec.compute(observations, events={"events": events})
    with pytest.raises(ValueError, match=r"^events has tables that no group reads: \['other'\]$"):
        spec.compute(observations, events={"events": events, "cash": events, "other": events})


@pytest.mark.parametrize(
    "change,message",
    [
        ({"id": 1}, "^id must be a string, got 1$"),
        ({"observation_time": None}, "^observation_time must be a string, got None$"),
        (
            {"events": []},
            r"^events must be a non-empty list of dicts, one per group of features, got \[\]$",
        ),
        (
            {"events": {"time": "time"}},
            r"^events must be a non-empty list of dicts, .*, got \{'time': 'time'\}$",
        ),
        ({"events": ["time"]}, r"^events\[0\] must be a dict, got 'time'$"),
        (
            {"events": [{"time": "time", "window": ["3M"]}]},
            r"^events\[0\] has unknown keys \['window'\]; the keys are \['table', 'time', "
            r"'windows', 'aggregations', 'closed', 'prefix'\]$",
        ),
        (
            {"events": [{"windows": ["3M"]}]},
            r"^events\[0\] lacks 'time', the column with the time of each event$",
        ),
        (
            {"events": [{"time": "time", "table": 1}]},
            r"^table of events\[0\] must be a string, got 1$",
        ),
        ({"events": [{"time": 1}]}, r"^time of events\[0\] must be a string, got 1$"),
        (
            {"events": [{"time": "time", "windows": ["3M", "3M"]}]},
            r"^windows of events\[0\] must not repeat a window, got '3M'$",
        ),
        (
            {"events": [{"time": "time"}, {"time": "time", "windows": ["1Y"]}]},
            r"^windows of events\[1\] must count months or days such as '3M' or '90D', got '1Y'$",
        ),
        (
            {"events": [{"time": "time", "aggregations": {"x": ["median"]}}]},
            r"^aggregations of events\[0\] must be among .*, got 'median'$",
        ),
        (
            {"events": [{"time": "time", "closed": "both"}]},
            r"^closed of events\[0\] must be 'left' or 'right', got 'both'$",
        ),
        (
            {"events": [{"time": "time", "prefix": None}]},
            r"^prefix of events\[0\] must be a string, got None$",
        ),
        (
            {"events": [{"time": "time"}, {"time": "time", "windows": ["1M", "3M"]}]},
            "^the groups of events would have two columns named 'events_3m'; give the groups "
            "different prefixes or remove the repeated ratio$",
        ),
        (
            {"ratios": [("events_3m", "x")]},
            "^ratios must name features of the windows and aggregations, such as 'events_3m', got "
            "'x'$",
        ),
        (
            {"panel": {"status": "status"}},
            r"^panel must be None, got \{'status': 'status'\}; panel features are not available "
            r"yet$",
        ),
        ({"kivi_version": 1.3}, "^kivi_version must be a string, got 1.3$"),
    ],
)
def test_invalid_specifications_raise(change, message):
    fields = {"id": "customer", "observation_time": "date", "events": [{"time": "time"}]}
    with pytest.raises(ValueError, match=message):
        FeatureSpec(**{**fields, **change})


def damaged(change):
    data = two_groups().to_dict()
    change(data)
    return data


@pytest.mark.parametrize(
    "data,message",
    [
        ([], "^data must be a mapping, got list$"),
        (
            damaged(lambda d: d.update(format="kivi.BinningSpec")),
            "^data is not a kivi feature specification: format must be 'kivi.FeatureSpec', got "
            "'kivi.BinningSpec'$",
        ),
        (
            damaged(lambda d: d.update(format_version=True)),
            "^format_version must be a positive integer, got True$",
        ),
        (
            damaged(lambda d: d.update(format_version=0)),
            "^format_version must be a positive integer, got 0$",
        ),
        (
            damaged(lambda d: d.update(format_version=2)),
            "^format_version 2 is newer than 1, the newest that kivi .* reads; upgrade kivi to "
            "load this specification$",
        ),
        (damaged(lambda d: d.pop("panel")), r"^the specification lacks \['panel'\]$"),
        (
            damaged(lambda d: d.update(groups=[])),
            r"^the specification has unknown keys \['groups'\]$",
        ),
        (damaged(lambda d: d.update(events={})), "^events must be a list, got dict$"),
        (
            damaged(lambda d: d.update(events=[])),
            r"^events must be a non-empty list of dicts, .*, got \[\]$",
        ),
        (damaged(lambda d: d["events"].append([])), r"^events\[2\] must be a mapping, got list$"),
        (damaged(lambda d: d["events"][0].pop("closed")), r"^events\[0\] lacks \['closed'\]$"),
        (
            damaged(lambda d: d["events"][1].update(where="x")),
            r"^events\[1\] has unknown keys \['where'\]$",
        ),
        (
            # The layout of format version 1 before the review, a list of columns.
            damaged(
                lambda d: d["events"][0].update(
                    aggregations=[{"column": "amount", "aggregations": ["sum"]}]
                )
            ),
            r"^aggregations of events\[0\] must be a dict from column names to lists of "
            r"aggregations, got \[\{'column': 'amount', 'aggregations': \['sum'\]\}\]$",
        ),
        (
            damaged(lambda d: d["events"][0]["aggregations"].update({1: ["sum"]})),
            "^column names must be strings, got 1$",
        ),
        (
            damaged(lambda d: d["events"][0]["aggregations"].update(amount="sum")),
            r"^aggregations of events\[0\] of column 'amount' must be a list such as",
        ),
        (
            damaged(lambda d: d["events"][0]["windows"].append(["6M", 3])),
            r"^each window of windows of events\[0\] must be .*, got \['6M', 3\]$",
        ),
        (
            damaged(lambda d: d.update(ratios=[["events_1m"]])),
            r"^each ratio of ratios must be .*, got \['events_1m'\]$",
        ),
        (damaged(lambda d: d.update(panel=[])), r"^panel must be None, got \[\]"),
        (
            damaged(lambda d: d.update(kivi_version=None)),
            "^kivi_version must be a string, got None$",
        ),
    ],
)
def test_invalid_dicts_raise(data, message):
    with pytest.raises(ValueError, match=message):
        FeatureSpec.from_dict(data)


def test_invalid_json_raises():
    with pytest.raises(ValueError, match="^text must be JSON text as a str or bytes, got dict; "):
        FeatureSpec.from_json({})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        FeatureSpec.from_json("{")
    # json.loads would keep the last of two values of one key.
    text = two_groups().to_json().replace('"amount": [', '"amount": ["sum"], "amount": [', 1)
    with pytest.raises(
        ValueError, match="^JSON objects must not repeat a key, got 'amount' twice$"
    ):
        FeatureSpec.from_json(text)


def test_the_dict_holds_the_arguments_of_the_constructor():
    spec = two_groups()
    data = spec.to_dict()
    assert list(data) == [
        "format",
        "format_version",
        "kivi_version",
        "id",
        "observation_time",
        "events",
        "ratios",
        "panel",
    ]
    assert data["events"][1] == {
        "table": "cash",
        "time": "time",
        "windows": ["3M", ["6M", "3M"]],
        "aggregations": {"amount": ["sum", "max"]},
        "closed": "right",
        "prefix": "cash_",
    }
    assert list(data["events"][0]["aggregations"]) == ["amount", "kind", "flag"]
    arguments = ["id", "observation_time", "events", "ratios", "panel", "kivi_version"]
    assert FeatureSpec(**{name: data[name] for name in arguments}) == spec


def test_a_missing_kivi_version_is_rejected_even_though_the_constructor_fills_it_in():
    data = two_groups().to_dict()
    data["kivi_version"] = None
    with pytest.raises(ValueError, match="^kivi_version must be a string, got None$"):
        FeatureSpec.from_dict(data)


# Golden files


def golden_spec():
    """The specification of the golden file, with every key of format version 1."""
    return FeatureSpec(
        id="customer",
        observation_time="date",
        events=[
            {
                "time": "time",
                "windows": ["1M", "3M", ("6M", "3M"), "90D"],
                "aggregations": {"amount": ALL[:6] + ALL[7:], "merchant": ["count", "nunique"]},
            },
            {
                "table": "cash",
                "time": "time",
                "windows": ["3M"],
                "aggregations": {"amount": ["sum"]},
                "prefix": "cash_",
            },
            {
                "table": "balances",
                "time": "day",
                "windows": ["30D", ("60D", "30D")],
                "aggregations": {"balance": ["mean", "last"]},
                "closed": "right",
                "prefix": "daily_",
            },
        ],
        ratios=[
            ("amount_sum_3m", "amount_sum_6m_3m"),
            ("cash_amount_sum_3m", "amount_sum_3m"),
            ("daily_balance_mean_30d", "daily_balance_mean_60d_30d"),
        ],
        kivi_version="1.3.0",
    )


def golden_tables():
    """The tables on which the golden features were computed."""
    rng = np.random.default_rng(13)
    n = 120
    events = pd.DataFrame(
        {
            "customer": rng.integers(0, 4, n),
            "time": pd.Timestamp("2023-06-30")
            + pd.to_timedelta(rng.integers(0, 400, n), unit="D")
            + pd.to_timedelta(rng.integers(0, 24, n), unit="h"),
            "amount": rng.normal(80, 30, n).round(2),
            "merchant": rng.choice(["food", "fuel", "travel", "atm"], n).astype(object),
        }
    )
    events.loc[rng.random(n) < 0.1, "amount"] = np.nan
    days = pd.date_range("2024-03-01", "2024-06-30", freq="D")
    balances = pd.DataFrame(
        {
            "customer": np.repeat([0, 1, 2], len(days)),
            "day": np.tile(days, 3),
            "balance": rng.normal(1000, 200, 3 * len(days)).round(2),
        }
    )
    observations = pd.DataFrame(
        {
            "customer": [0, 1, 2, 3, 4, 0],
            "date": pd.to_datetime(
                ["2024-05-31", "2024-06-30", "2024-04-30", "2024-05-15", "2024-05-31", "2024-06-30"]
            ),
        }
    )
    tables = {
        "events": events,
        "cash": events[events["merchant"] == "atm"],
        "balances": balances,
    }
    return tables, observations


def test_the_golden_file_loads_and_computes_the_golden_features():
    text = GOLDEN.read_text(encoding="utf-8")
    loaded = FeatureSpec.from_json(text)
    assert loaded == golden_spec()
    assert loaded.to_json(indent=2) + "\n" == text
    tables, observations = golden_tables()
    result = loaded.compute(observations, events=tables)
    saved = json.loads(GOLDEN_FEATURES.read_text(encoding="utf-8"))
    assert list(result.columns) == saved["columns"]
    assert result.index.tolist() == saved["index"]
    for name in saved["columns"]:
        expected = [np.nan if value is None else value for value in saved["data"][name]]
        if result[name].dtype == np.int64:
            assert result[name].tolist() == expected, name
        else:
            assert result[name].dtype == np.float64
            # The last digit of a sum may differ on another processor.
            np.testing.assert_allclose(result[name], expected, rtol=1e-12, atol=0, err_msg=name)


def write_golden_files():
    """Write the golden files from `golden_spec` and `golden_tables`."""
    spec = golden_spec()
    GOLDEN.write_text(spec.to_json(indent=2) + "\n", encoding="utf-8")
    tables, observations = golden_tables()
    result = FeatureSpec.from_json(GOLDEN.read_text(encoding="utf-8")).compute(
        observations, events=tables
    )
    data = {
        name: [None if pd.isna(value) else value for value in result[name].tolist()]
        for name in result.columns
    }
    saved = {"columns": list(result.columns), "index": result.index.tolist(), "data": data}
    GOLDEN_FEATURES.write_text(json.dumps(saved, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_golden_files()
    sys.exit(0)
