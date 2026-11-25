"""Workout LGDs, conversion factors, LGD tables and the fractional logit of kivi.lgd."""

import functools
import math
import re

import numpy as np
import pandas as pd
import pytest

from kivi import _logit
from kivi.lgd import conversion_factor, fractional_regression, lgd_table, workout_lgd
from kivi.selection import SeparationError, logistic_regression


def two_sided(z):
    return math.erfc(abs(z) / math.sqrt(2))


# --- workout_lgd --------------------------------------------------------------------------


def hand_example(**options):
    return workout_lgd(
        ["A", "B", "C"],
        ["2023-01-01"] * 3,
        [100.0, 200.0, 50.0],
        ["A", "A", "B", "B"],
        ["2024-01-01", "2024-01-01", "2022-01-01", "2023-01-01"],
        [55.0, -11.0, -5.0, 200.0],
        **{"rate": 0.1, **options},
    )


def test_workout_lgd_matches_a_hand_calculation():
    result = hand_example()
    assert result.index.tolist() == ["A", "B", "C"] and result.index.name == "loan"
    assert result.columns.tolist() == ["ead", "recovery", "cost", "lgd"]
    np.testing.assert_array_equal(result["ead"], [100.0, 200.0, 50.0])
    # A year after default, 55 and 11 are worth 50 and 10; a year before, 5 is worth 5.5.
    np.testing.assert_allclose(result["recovery"], [50.0, 200.0, 0.0], rtol=1e-14)
    np.testing.assert_allclose(result["cost"], [10.0, 5.5, 0.0], rtol=1e-14)
    np.testing.assert_allclose(result["lgd"], [0.6, 5.5 / 200, 1.0], rtol=1e-13)
    # A flow on the day of default is not discounted, and a loan without flows lost everything.
    assert result.loc["B", "recovery"] == 200.0
    assert result.loc["C"].tolist() == [50.0, 0.0, 0.0, 1.0]


def test_rate_per_loan_and_act_365():
    # 2024 is a leap year: 366 days from 1 January 2024 to 1 January 2025.
    result = workout_lgd(
        ["A", "B"],
        ["2024-01-01", "2024-03-01"],
        [100.0, 100.0],
        ["A", "B", "B"],
        ["2025-01-01", "2024-03-31", "2023-03-01"],
        [80.0, 50.0, -10.0],
        rate=[0.05, 0.12],
    )
    np.testing.assert_allclose(result.loc["A", "recovery"], 80.0 / 1.05 ** (366 / 365), rtol=1e-14)
    np.testing.assert_allclose(result.loc["B", "recovery"], 50.0 / 1.12 ** (30 / 365), rtol=1e-14)
    np.testing.assert_allclose(result.loc["B", "cost"], 10.0 * 1.12 ** (366 / 365), rtol=1e-14)
    lgd = (100 + result["cost"] - result["recovery"]) / 100
    np.testing.assert_allclose(result["lgd"], lgd, rtol=1e-14)


def test_rate_zero_gives_undiscounted_sums():
    result = workout_lgd(
        [1, 2], ["2020-01-01", "2020-06-30"], [10.0, 4.0], [2, 1, 1], ["2030-01-01"] * 3,
        [5.0, 3.0, -1.0], rate=0.0,
    )  # fmt: skip
    np.testing.assert_array_equal(result["recovery"], [3.0, 5.0])
    np.testing.assert_array_equal(result["cost"], [1.0, 0.0])
    np.testing.assert_array_equal(result["lgd"], [0.8, -0.25])


def test_lgd_is_not_clipped():
    result = workout_lgd(
        ["A", "B"], ["2024-01-01"] * 2, [100.0, 100.0], ["A", "B", "B"],
        ["2024-02-01", "2024-02-01", "2024-02-01"], [120.0, 10.0, -30.0], rate=0.0,
    )  # fmt: skip
    np.testing.assert_allclose(result["lgd"], [-0.2, 1.2], rtol=1e-14)


def brute_force(loans, default_dates, ead, flow_loans, flow_dates, amounts, rates):
    """The LGD of each loan by a loop over its flows."""
    rows = {}
    for loan, date, exposure, rate in zip(loans, default_dates, ead, rates, strict=True):
        recovery = cost = 0.0
        for other, day, amount in zip(flow_loans, flow_dates, amounts, strict=True):
            if other == loan:
                value = amount * (1 + rate) ** (-(day - date).days / 365)
                if amount > 0:
                    recovery += value
                else:
                    cost -= value
        rows[loan] = (exposure, recovery, cost, (exposure + cost - recovery) / exposure)
    return pd.DataFrame.from_dict(rows, orient="index", columns=["ead", "recovery", "cost", "lgd"])


@pytest.mark.parametrize("seed", range(5))
def test_workout_lgd_matches_a_loop_over_flows(seed):
    rng = np.random.default_rng(seed)
    n_loans, n_flows = 40, 300
    loans = [f"L{i}" for i in rng.permutation(n_loans)]
    default_dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(
        rng.integers(0, 1500, n_loans), unit="D"
    )
    ead = rng.uniform(1e3, 1e5, n_loans)
    rates = rng.uniform(0.0, 0.15, n_loans)
    which = rng.integers(0, n_loans, n_flows)
    flow_loans = [loans[i] for i in which]
    # Recoveries up to four years after default; costs from a year before.
    after = rng.integers(0, 4 * 365, n_flows)
    before = rng.integers(-365, 4 * 365, n_flows)
    recovery = rng.random(n_flows) < 0.7
    offsets = np.where(recovery, after, before)
    flow_dates = default_dates[which] + pd.to_timedelta(offsets, unit="D")
    amounts = np.where(recovery, rng.uniform(0, 3e4, n_flows), -rng.uniform(0, 3e3, n_flows))
    result = workout_lgd(
        pd.Series(loans, index=np.arange(n_loans) * 3),
        pd.Series(default_dates, index=np.arange(n_loans) * 3),
        pd.Series(ead, index=np.arange(n_loans) * 3),
        flow_loans,
        flow_dates,
        amounts,
        rate=pd.Series(rates, index=np.arange(n_loans) * 3),
    )
    expected = brute_force(loans, default_dates, ead, flow_loans, flow_dates, amounts, rates)
    assert result.index.tolist() == loans
    np.testing.assert_allclose(result.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=0)


def test_dates_are_read_like_delinquency_panel():
    # A time of day is ignored, and tz-aware dates keep their local date.
    default = pd.Series(pd.to_datetime(["2023-01-01 23:00"]).tz_localize("Asia/Shanghai"))
    result = workout_lgd(
        ["A"], default, [100.0], ["A"], [pd.Timestamp("2024-01-01 01:00")], [110.0], rate=0.1
    )
    np.testing.assert_allclose(result["recovery"], [100.0], rtol=1e-14)


def test_flows_may_be_empty_and_have_any_index():
    result = workout_lgd(["A", "B"], ["2023-01-01"] * 2, [1.0, 2.0], [], [], [], rate=0.05)
    assert result["lgd"].tolist() == [1.0, 1.0] and result["recovery"].tolist() == [0.0, 0.0]
    assert (result.dtypes == np.float64).all()
    flows = pd.DataFrame(
        {"loan": ["B", "A"], "date": ["2023-01-01"] * 2, "amount": [1.0, 0.5]}, index=[10, 20]
    )
    result = workout_lgd(
        pd.Series(["A", "B"]), ["2023-01-01"] * 2, [1.0, 2.0],
        flows["loan"], flows["date"], flows["amount"], rate=0.05,
    )  # fmt: skip
    assert result["lgd"].tolist() == [0.5, 0.5]


def test_index_keeps_the_dtype_of_the_loans():
    result = workout_lgd(
        [3, 1, 2], ["2023-01-01"] * 3, [1.0] * 3, [1], ["2023-01-01"], [1.0], rate=0
    )
    assert result.index.tolist() == [3, 1, 2] and result.index.dtype == np.int64
    assert result["lgd"].tolist() == [1.0, 0.0, 1.0]


def test_drawings_after_default_count_in_the_loss_or_in_the_exposure():
    loans, default = ["A"], ["2023-01-01"]
    drawn = 20 / 1.1 ** (182 / 365)
    # A drawing is a negative flow: it counts in the loss (EBA/GL/2017/16, paragraph 142).
    flows = (["A", "A"], ["2023-07-02", "2024-01-01"], [-20.0, 99.0])
    result = workout_lgd(loans, default, [100.0], *flows, rate=0.1)
    np.testing.assert_allclose(result["lgd"], [(100 + drawn - 90) / 100], rtol=1e-14)
    # Where the CCF includes it, it moves to the exposure, which counts in the loss as well
    # (paragraph 140): once in the loss and once in the exposure.
    drawings = workout_lgd(loans, default, [100.0], ["A"], ["2023-07-02"], [-20.0], rate=0.1)
    np.testing.assert_allclose(drawings["cost"], [drawn], rtol=1e-14)
    ead = 100.0 + drawings["cost"].to_numpy()
    result = workout_lgd(loans, default, ead, ["A"], ["2024-01-01"], [99.0], rate=0.1)
    np.testing.assert_allclose(result["ead"], [100 + drawn], rtol=1e-14)
    np.testing.assert_allclose(result["lgd"], [(100 + drawn - 90) / (100 + drawn)], rtol=1e-14)


def test_workout_lgd_does_not_mutate_its_inputs():
    loans = pd.Series(["A", "B"])
    dates = pd.Series(["2023-01-01", "2023-01-02"])
    ead = pd.Series([1.0, 2.0])
    flows = pd.DataFrame({"loan": ["A"], "date": ["2023-02-01"], "amount": [0.5]})
    copies = [loans.copy(), dates.copy(), ead.copy(), flows.copy()]
    workout_lgd(loans, dates, ead, flows["loan"], flows["date"], flows["amount"], rate=0.1)
    for value, copy in zip([loans, dates, ead, flows], copies, strict=True):
        assert value.equals(copy)


LOANS = {"loan": ["A", "B"], "default_date": ["2023-01-01"] * 2, "ead": [100.0, 50.0]}
FLOWS = {"flow_loan": ["A"], "flow_date": ["2023-06-01"], "flow_amount": [10.0]}


def call(rate=0.1, **changes):
    arguments = {**LOANS, **FLOWS, **changes}
    return workout_lgd(*arguments.values(), rate=rate)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"loan": ["A", "A"]}, "loan must contain each loan once, got 'A' more than once"),
        ({"loan": ["A", None]}, "loan must not contain missing values, got 1"),
        ({"loan": []}, "loan, default_date and ead must have the same length, got 0, 2 and 2"),
        (
            {"loan": [], "default_date": [], "ead": []},
            "loan, default_date and ead must not be empty",
        ),
        ({"flow_loan": ["C"]}, "flow_loan must contain only loans of loan, got 'C'"),
        ({"flow_loan": [None]}, "flow_loan must not contain missing values, got 1"),
        (
            {"flow_amount": [1.0, 2.0]},
            "flow_loan, flow_date and flow_amount must have the same length, got 1, 1 and 2",
        ),
        ({"ead": [100.0, 0.0]}, "ead must be positive, got 0.0"),
        ({"ead": [-1.0, 50.0]}, "ead must be positive, got -1.0"),
        ({"ead": [np.nan, 50.0]}, "ead must not contain missing values, got 1"),
        ({"ead": [np.inf, 50.0]}, "ead must be finite, got inf"),
        ({"ead": [True, True]}, "ead must be real and numeric, got dtype bool"),
        ({"ead": ["a", "b"]}, "ead must be real and numeric, got dtype"),
        ({"flow_amount": [np.nan]}, "flow_amount must not contain missing values, got 1"),
        ({"flow_amount": [True]}, "flow_amount must be real and numeric, got dtype bool"),
        ({"flow_amount": [-np.inf]}, "flow_amount must be finite, got -inf"),
        ({"default_date": [1, 2]}, "default_date must hold dates, got dtype int64"),
        ({"default_date": ["2023-01-01", None]}, "default_date must not contain missing values"),
        ({"default_date": ["2023-01-01", "soon"]}, "default_date must hold dates, got 'soon'"),
        (
            {"default_date": pd.period_range("2023-01", periods=2, freq="M")},
            "default_date must hold dates, got dtype period[M]",
        ),
        ({"flow_date": [True]}, "flow_date must hold dates, got dtype bool"),
        (
            {"flow_date": ["2022-12-31"]},
            "flow_amount must not be positive before the date of default: loan 'A' recovers "
            "10.0 on 2022-12-31, before its default on 2023-01-01",
        ),
    ],
)
def test_workout_lgd_errors(changes, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        call(**changes)


@pytest.mark.parametrize(
    "rate,message",
    [
        (-1.0, "rate must be finite and greater than -1, got -1.0"),
        ([0.1, np.nan], "rate must be finite and greater than -1, got nan"),
        (np.inf, "rate must be finite and greater than -1, got inf"),
        ([0.1, 0.1, 0.1], "rate must be one rate or one per loan, got shape (3,) for 2 loans"),
        ([[0.1, 0.1]], "rate must be one rate or one per loan, got shape (1, 2) for 2 loans"),
        (True, "rate must be real and numeric, got dtype bool"),
        ("0.1", "rate must be real and numeric, got dtype <U3"),
    ],
)
def test_rate_errors(rate, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        call(rate=rate)


def test_indices_of_the_loans_and_of_the_flows_must_match():
    with pytest.raises(ValueError, match="loan and ead must have identical indices"):
        call(loan=pd.Series(["A", "B"]), ead=pd.Series([1.0, 2.0], index=[1, 2]))
    with pytest.raises(ValueError, match="loan and rate must have identical indices"):
        call(loan=pd.Series(["A", "B"]), rate=pd.Series([0.1, 0.1], index=[1, 2]))
    with pytest.raises(ValueError, match="flow_loan and flow_amount must have identical indices"):
        call(flow_loan=pd.Series(["A"]), flow_amount=pd.Series([1.0], index=[5]))
    # A cost before default is allowed; the flows' index need not match the loans'.
    result = call(flow_date=["2022-01-01"], flow_amount=pd.Series([-1.0], index=[7]))
    np.testing.assert_allclose(result.loc["A", "cost"], 1.1 ** (365 / 365), rtol=1e-14)


# --- conversion_factor --------------------------------------------------------------------


def test_conversion_factor_closed_forms():
    ccf = conversion_factor(
        [600.0, 800.0, 300.0, 900.0, 1500.0, 1200.0, np.nan],
        [400.0, 400.0, 500.0, 1000.0, 1000.0, 1100.0, 0.0],
        1000.0,
    )
    expected = [200 / 600, 400 / 600, -200 / 500, np.nan, np.nan, np.nan, np.nan]
    np.testing.assert_allclose(ccf, expected, rtol=1e-15)
    # Drawn above the limit gives NaN, not the ratio of two negative numbers.
    assert np.isnan(conversion_factor(1200.0, 1100.0, 1000.0))
    assert conversion_factor(1300.0, 200.0, 1000.0) == pytest.approx(1100 / 800, rel=1e-15)
    assert conversion_factor(200.0, 200.0, 1000.0) == 0.0
    assert np.isnan(conversion_factor([200.0], [pd.NA], 1000.0)).all()
    np.testing.assert_allclose(np.maximum(ccf, 0), [1 / 3, 2 / 3, 0, *[np.nan] * 4], rtol=1e-15)


def test_conversion_factor_shapes():
    assert isinstance(conversion_factor(600.0, 400.0, 1000.0), float)
    assert isinstance(conversion_factor([600.0], 400.0, 1000.0), np.ndarray)
    drawn = pd.Series([400.0, 500.0], index=["x", "y"], name="drawn")
    result = conversion_factor(600.0, drawn, 1000.0)
    assert isinstance(result, pd.Series) and result.name == "ccf"
    assert result.index.tolist() == ["x", "y"]
    np.testing.assert_allclose(result, [1 / 3, 0.2], rtol=1e-15)
    result = conversion_factor(pd.Series([600.0, 700.0]), [400.0, 500.0], pd.Series([1e3, 1e3]))
    np.testing.assert_allclose(result, [1 / 3, 0.4], rtol=1e-15)
    grid = conversion_factor([[600.0], [700.0]], [400.0, 500.0], 1000.0)
    assert grid.shape == (2, 2)


@pytest.mark.parametrize(
    "arguments,message",
    [
        (([True], [0.0], [1.0]), "ead must be real and numeric, got dtype bool"),
        (([1.0], ["a"], [1.0]), "drawn must be real and numeric, got dtype"),
        (([1.0], [0.0], [np.inf]), "limit must be finite or missing, got inf"),
        (
            ([1.0, 2.0], [0.0, 0.0, 0.0], 1.0),
            "ead, drawn, limit cannot be broadcast together: ead (2,), drawn (3,), limit ()",
        ),
        (
            (pd.Series([1.0]), pd.Series([0.0], index=[3]), 1.0),
            "ead and drawn must have identical indices",
        ),
        (
            (pd.Series([1.0]), [0.0, 0.0], 1.0),
            "must broadcast to the length 1 of their Series index, got shape (2,)",
        ),
    ],
)
def test_conversion_factor_errors(arguments, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        conversion_factor(*arguments)


# --- lgd_table ----------------------------------------------------------------------------


def random_lgds(seed, n=600):
    rng = np.random.default_rng(seed)
    groups = pd.Series(rng.choice(["a", "b", "c", None], n, p=[0.4, 0.3, 0.25, 0.05]))
    # Bimodal LGDs with a few negative values and values above 1.
    lgd = np.where(rng.random(n) < 0.5, rng.beta(0.5, 4, n), rng.beta(4, 0.5, n))
    lgd = lgd + np.where(rng.random(n) < 0.1, rng.uniform(-0.2, 0.2, n), 0.0)
    lgd[rng.random(n) < 0.1] = 1.0
    lgd[rng.random(n) < 0.1] = 0.0
    return groups, pd.Series(lgd)


def grouped(groups, lgd):
    """The columns of lgd_table by a pandas groupby, in the order in which the groups appear."""
    # Group by the codes of first appearance: pandas 2.0 puts the missing group last.
    codes, _ = pd.factorize(groups, use_na_sentinel=False)
    by = pd.Series(lgd.to_numpy()).groupby(codes)
    return pd.DataFrame(
        {
            "total": by.size(),
            "mean": by.mean(),
            "std": by.std(),
            "min": by.min(),
            "max": by.max(),
            "zero_share": by.apply(lambda values: (values <= 0).mean()),
            "full_share": by.apply(lambda values: (values >= 1).mean()),
        }
    )


@pytest.mark.parametrize("floor", [0.0, None, 0.05])
@pytest.mark.parametrize("seed", range(3))
def test_lgd_table_matches_a_groupby_after_flooring(seed, floor):
    groups, lgd = random_lgds(seed)
    table = lgd_table(groups, lgd, floor=floor)
    floored = lgd if floor is None else lgd.clip(lower=floor)
    expected = grouped(groups, floored)
    assert table.index.name == "bin"
    labels = groups.drop_duplicates()
    assert table.index.equals(pd.Index(labels)) and table.index.isna().sum() == 1
    assert table.columns.tolist() == [
        "total", "total_share", "mean", "std", "min", "max", "zero_share", "full_share"
    ]  # fmt: skip
    np.testing.assert_array_equal(table["total"], expected["total"])
    np.testing.assert_allclose(table["total_share"], expected["total"] / len(lgd), rtol=1e-15)
    for column in ("mean", "std", "zero_share", "full_share"):
        np.testing.assert_allclose(table[column], expected[column], rtol=1e-12, atol=0)
    for column in ("min", "max"):
        np.testing.assert_array_equal(table[column], expected[column])
    if floor is None:
        assert (table["min"] < 0).any()
    else:
        assert (table["min"] >= floor).all()


def test_lgd_table_integer_weights_equal_repeated_rows():
    groups, lgd = random_lgds(7, n=200)
    weights = np.random.default_rng(1).integers(0, 4, len(lgd))
    weighted = lgd_table(groups, lgd, weights=weights)
    repeated = lgd_table(groups.repeat(weights), lgd.repeat(weights))
    # Zero weights do not count in min and max, nor in the order of the rows here.
    pd.testing.assert_index_equal(weighted.index, repeated.index)
    np.testing.assert_allclose(weighted.to_numpy(), repeated.to_numpy(), rtol=1e-12, atol=0)


def test_lgd_table_weights_leave_out_rows_and_groups_without_weight():
    table = lgd_table(["a", "a", "b", "a"], [0.2, 0.9, 0.5, 0.4], weights=[1.0, 0.0, 0.0, 3.0])
    assert table.index.tolist() == ["a"]
    assert table.loc["a", "max"] == 0.4 and table.loc["a", "total"] == 4.0
    np.testing.assert_allclose(table.loc["a", "mean"], (0.2 + 3 * 0.4) / 4, rtol=1e-15)
    np.testing.assert_allclose(table.loc["a", "std"], math.sqrt((0.15**2 + 3 * 0.05**2) / 3))
    # A total weight of 1 or less has no sample variance.
    table = lgd_table(["a", "a"], [0.2, 0.4], weights=[0.5, 0.5])
    assert np.isnan(table.loc["a", "std"])


def test_lgd_table_constant_groups_and_categorical_order():
    groups = pd.Categorical(["low", "high", "high", None, "low"], categories=["low", "mid", "high"])
    table = lgd_table(groups, [0.3, 1.0, 1.0, 0.7, 0.3])
    assert table.index.tolist()[:2] == ["low", "high"] and pd.isna(table.index[2])
    assert table["mean"].tolist() == [0.3, 1.0, 0.7]
    assert table["std"].tolist()[:2] == [0.0, 0.0] and np.isnan(table["std"].iloc[2])
    assert table["full_share"].tolist() == [0.0, 1.0, 0.0]
    assert table["total"].dtype == np.int64


@pytest.mark.parametrize(
    "arguments,options,message",
    [
        ((["a"], [np.nan]), {}, "lgd must not contain missing values, got 1"),
        ((["a"], [np.inf]), {}, "lgd must be finite, got inf"),
        ((["a"], [True]), {}, "lgd must be real and numeric, got dtype bool"),
        ((["a", "b"], [0.1]), {}, "groups and lgd must have the same length, got 2 and 1"),
        (([], []), {}, "groups and lgd must contain observations"),
        (
            (pd.Series(["a"]), pd.Series([0.1], index=[4])),
            {},
            "groups and lgd must have identical indices",
        ),
        ((["a"], [0.1]), {"floor": True}, "floor must be None or a finite number, got True"),
        ((["a"], [0.1]), {"floor": -np.inf}, "floor must be None or a finite number, got -inf"),
        ((["a"], [0.1]), {"floor": "0"}, "floor must be None or a finite number, got '0'"),
        ((["a"], [0.1]), {"weights": [-1.0]}, "weights must be finite and non-negative"),
        ((["a"], [0.1]), {"weights": [0.0]}, "weights must have a finite, positive total"),
        (
            (["a"], pd.Series([0.1])),
            {"weights": pd.Series([1.0], index=[2])},
            "lgd and weights must have identical indices",
        ),
    ],
)
def test_lgd_table_errors(arguments, options, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        lgd_table(*arguments, **options)


# --- fractional_regression ----------------------------------------------------------------


def fractional_data(seed, n=400):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "ltv": rng.uniform(0.2, 1.4, n),
            "unsecured": rng.integers(0, 2, n),
            "age": rng.normal(size=n),
        }
    )
    mean = 1 / (1 + np.exp(-(-2.0 + 1.5 * X["ltv"] + 1.2 * X["unsecured"] + 0.2 * X["age"])))
    response = rng.beta(2 * mean, 2 * (1 - mean))
    # Exact 0 and 1, as cures and total losses give them.
    response[rng.random(n) < 0.1] = 0.0
    response[rng.random(n) < 0.1] = 1.0
    return X, pd.Series(response)


def quasi_log_likelihood(response, mean, weights):
    return float(np.sum(weights * (response * np.log(mean) + (1 - response) * np.log1p(-mean))))


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_fractional_regression_matches_statsmodels(seed, weighted):
    sm = pytest.importorskip("statsmodels.api")
    X, response = fractional_data(seed)
    weights = np.random.default_rng(seed + 10).integers(1, 4, len(X)) if weighted else None
    result = fractional_regression(X, response, weights=weights)
    model = sm.GLM(
        response,
        sm.add_constant(X.astype(float)),
        family=sm.families.Binomial(),
        freq_weights=weights,
    ).fit(cov_type="HC0", tol=1e-14)
    np.testing.assert_allclose(result["coef"], model.params, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result["se"], model.bse, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result["z"], model.tvalues, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result["p_value"], model.pvalues, rtol=1e-9, atol=0)
    mass = np.ones(len(X)) if weights is None else weights
    llf = quasi_log_likelihood(response.to_numpy(), model.mu, mass)
    np.testing.assert_allclose(result.attrs["log_likelihood"], llf, rtol=1e-12)
    assert result.attrs["converged"] is True
    assert result.index.tolist() == ["intercept", "ltv", "unsecured", "age"]
    assert result.index.name == "variable"
    assert result.columns.tolist() == ["coef", "se", "z", "p_value"]


def test_fractional_regression_hand_example():
    X = pd.DataFrame({"unsecured": [0, 0, 0, 0, 1, 1, 1, 1]})
    response = [0.0, 0.1, 0.5, 0.2, 0.3, 0.9, 0.4, 1.0]
    result = fractional_regression(X, response)
    logit = [math.log(0.2 / 0.8), math.log(0.65 / 0.35)]
    np.testing.assert_allclose(result["coef"], [logit[0], logit[1] - logit[0]], rtol=1e-12)
    # The sandwich of a saturated model: sum of squared residuals over the squared information.
    first, second = 0.14 / (4 * 0.2 * 0.8) ** 2, 0.37 / (4 * 0.65 * 0.35) ** 2
    np.testing.assert_allclose(result["se"], np.sqrt([first, first + second]), rtol=1e-10)
    np.testing.assert_allclose(result["p_value"], [two_sided(z) for z in result["z"]])


def test_binary_response_gives_the_coefficients_of_logistic_regression():
    X, response = fractional_data(4)
    binary = (response > 0.5).astype(float)
    robust = fractional_regression(X, binary)
    model = logistic_regression(X, binary.astype(int))
    np.testing.assert_array_equal(robust["coef"], model["coef"])
    assert robust.attrs["log_likelihood"] == model.attrs["log_likelihood"]
    # The standard errors are robust instead of model-based.
    assert not np.allclose(robust["se"], model["se"], rtol=1e-3)


def test_fractional_regression_integer_weights_equal_repeated_rows():
    X, response = fractional_data(5, n=120)
    weights = np.random.default_rng(2).integers(0, 4, len(X))
    weighted = fractional_regression(X, response, weights=weights)
    repeated = fractional_regression(X.loc[X.index.repeat(weights)], response.repeat(weights))
    np.testing.assert_allclose(weighted.to_numpy(), repeated.to_numpy(), rtol=1e-10, atol=0)
    np.testing.assert_allclose(
        weighted.attrs["log_likelihood"], repeated.attrs["log_likelihood"], rtol=1e-12
    )


@pytest.mark.parametrize(
    "x,response",
    [
        ([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0]),  # binary responses, complete separation
        ([0, 0, 1, 1], [0.3, 0.5, 1.0, 1.0]),  # every unsecured loan lost everything
        ([0, 0, 1, 1, 1], [0.0, 0.0, 0.9, 1.0, 1.0]),  # quasi-complete: x = 0 lost nothing
        ([0, 0, 1, 1, 2, 2], [0.0, 0.0, 0.4, 0.6, 1.0, 1.0]),  # fractions on the boundary
        ([1, 2, 3, 4, 5], [0.0, 0.0, 0.5, 1.0, 1.0]),
    ],
)
def test_responses_of_0_and_1_can_be_separated(x, response):
    with pytest.raises(SeparationError, match="X separates the responses of 0 from those of 1"):
        fractional_regression(pd.DataFrame({"x": x}), response)


@pytest.mark.parametrize(
    "x,response",
    [
        ([1, 2, 3, 4], [0.1, 0.2, 0.8, 0.9]),  # ordered fractions are never separated
        ([0, 0, 1, 1, 1], [0.0, 0.2, 0.9, 1.0, 1.0]),  # a fraction in each group bounds both
        ([1, 2, 3, 4, 5], [0.0, 0.0, 0.5, 1.0, 0.999999]),
        ([1, 2, 3, 4], [1e-12, 1e-12, 1.0 - 1e-12, 1.0 - 1e-12]),
    ],
)
def test_fractions_between_0_and_1_prevent_separation(x, response):
    result = fractional_regression(pd.DataFrame({"x": x}), response)
    assert result.attrs["converged"] and np.isfinite(result["coef"]).all()


def test_fit_with_exact_residuals_of_zero():
    # A saturated model of equal responses fits every row exactly, so the robust se is 0.
    fitted = _logit.fit(
        np.column_stack([np.ones(4), [0.0, 0.0, 1.0, 1.0]]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        robust=True,
    )
    assert fitted.coef.tolist() == [0.0, 0.0] and fitted.se.tolist() == [0.0, 0.0]
    assert np.isnan(fitted.z).all() and np.isnan(fitted.p_value).all()


def test_robust_covariance_of_binary_logit_is_the_sandwich():
    design = np.column_stack([np.ones(8), [0.0, 0, 0, 0, 1, 1, 1, 1]])
    target = np.array([0.0, 0, 0, 1, 0, 1, 1, 1])
    model = _logit.fit(design, target)
    robust = _logit.fit(design, target, robust=True)
    np.testing.assert_array_equal(robust.coef, model.coef)
    p = 1 / (1 + np.exp(-design @ model.coef))
    information = (design * (p * (1 - p))[:, None]).T @ design
    meat = (design * ((target - p) ** 2)[:, None]).T @ design
    inverse = np.linalg.inv(information)
    np.testing.assert_allclose(robust.cov, inverse @ meat @ inverse, rtol=1e-12)


def test_fractional_regression_accepts_arrays():
    X, response = fractional_data(6, n=60)
    result = fractional_regression(X.to_numpy(), response.to_list())
    assert result.index.tolist() == ["intercept", 0, 1, 2]
    np.testing.assert_allclose(result["coef"], fractional_regression(X, response)["coef"])


def test_fractional_regression_warns_when_it_does_not_converge(monkeypatch):
    monkeypatch.setattr(_logit, "fit", functools.partial(_logit.fit, max_iter=1))
    X, response = fractional_data(0, n=50)
    with pytest.warns(RuntimeWarning, match="fractional regression did not converge after 1") as r:
        result = fractional_regression(X, response)
    assert r[0].filename == __file__ and result.attrs["converged"] is False


def singular_at(column, *args, **kwargs):
    raise _logit.SingularError(column)


def test_fractional_regression_names_a_singular_column(monkeypatch):
    monkeypatch.setattr(_logit, "fit", functools.partial(singular_at, 2))
    X, response = fractional_data(0, n=50)
    with pytest.raises(ValueError, match="X column 'unsecured' is so close to a linear"):
        fractional_regression(X, response)
    monkeypatch.setattr(_logit, "fit", functools.partial(singular_at, 0))
    with pytest.raises(ValueError, match="X column 'intercept' is so close"):
        fractional_regression(X, response)


X_SMALL = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})


@pytest.mark.parametrize(
    "X,response,options,message",
    [
        (X_SMALL, [0.1, 0.2, 1.2, 0.3], {}, "response must be between 0 and 1, got 1.2"),
        (X_SMALL, [0.1, -0.1, 0.2, 0.3], {}, "response must be between 0 and 1, got -0.1"),
        (X_SMALL, [0.1, np.nan, 0.2, 0.3], {}, "response must not contain missing values, got 1"),
        (
            X_SMALL,
            [True, False, True, False],
            {},
            "response must be real and numeric, got dtype bool",
        ),
        (X_SMALL, [0.0] * 4, {}, "response must not be 0 for every row"),
        (X_SMALL, [1.0] * 4, {}, "response must not be 1 for every row"),
        (
            X_SMALL,
            [0.0, 0.0, 0.5, 0.0],
            {"weights": [1, 1, 0, 1]},
            "response must not be 0 for every row on the rows with positive weight",
        ),
        (
            X_SMALL,
            [0.1, 0.2, 0.3],
            {},
            "X and response must have the same number of rows, got 4 and 3",
        ),
        (
            X_SMALL,
            pd.Series([0.1, 0.2, 0.3, 0.4], index=[1, 2, 3, 4]),
            {},
            "X and response must have identical indices",
        ),
        (
            X_SMALL,
            [0.1, 0.2, 0.3, 0.4],
            {"weights": pd.Series([1.0] * 4, index=[1, 2, 3, 4])},
            "X and weights must have identical indices",
        ),
        (
            X_SMALL,
            [0.1, 0.2, 0.3, 0.4],
            {"weights": [1.0, 1.0]},
            "weights must have one weight per row of X, got 2 for 4",
        ),
        (X_SMALL, [0.1, 0.2, 0.3, 0.4], {"weights": [1, -1, 1, 1]}, "weights must be finite"),
        (
            pd.DataFrame({"x": list("abcd")}),
            [0.1] * 4,
            {},
            "X column 'x' must be numeric, got dtype",
        ),
        (pd.DataFrame({"x": [1.0] * 4}), [0.1, 0.2, 0.3, 0.4], {}, "X column 'x' is constant;"),
        (
            pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]}),
            [0.1, 0.2, 0.3, 0.4],
            {},
            "X column 'y' is a linear combination of the intercept and the columns before it",
        ),
        (
            pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 1.0, 2.0, 2.0]}),
            [0.1, 0.2, 0.3, 0.4],
            {"weights": [1, 1, 0, 0]},
            "X column 'y' is constant on the rows with positive weight",
        ),
        (pd.DataFrame({"intercept": [1.0, 2.0]}), [0.1, 0.2], {}, "labelled 'intercept'"),
        (pd.DataFrame({"x": [1.0]}), [0.1], {}, "X must have at least 2 rows, got 1"),
        (pd.DataFrame(index=range(3)), [0.1] * 3, {}, "X must have at least one column"),
        (
            pd.DataFrame([[1.0, 2.0], [3.0, 5.0]], columns=["a", "a"]),
            [0.1, 0.2],
            {},
            "X must have unique column labels, got 'a' more than once",
        ),
        (
            pd.DataFrame({"x": [1.0, np.nan, 2.0]}),
            [0.1, 0.2, 0.3],
            {},
            "X column 'x' must not contain missing or infinite values",
        ),
        ([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], {}, "X must be two-dimensional, got 1 dimensions"),
    ],
)
def test_fractional_regression_errors(X, response, options, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        fractional_regression(X, response, **options)
