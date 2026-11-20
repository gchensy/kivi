import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from kivi._stats import betainc, chi2_sf, gammaincc
from kivi.calibration import (
    binomial_test,
    calibration_table,
    corrected_intercept,
    herfindahl_index,
    hosmer_lemeshow,
    jeffreys_test,
    prior_correction,
)
from kivi.datasets import Dataset

# Upper 5% and 1% points of the chi-square distribution, as tabulated.
CHI2_QUANTILES = [
    (1, 3.841458821, 6.634896601),
    (2, 5.991464547, 9.210340372),
    (3, 7.814727903, 11.34486673),
    (5, 11.07049769, 15.08627247),
    (10, 18.30703805, 23.20925116),
    (20, 31.41043284, 37.56623479),
    (100, 124.3421134, 135.8067232),
]


@pytest.mark.parametrize("dof,q95,q99", CHI2_QUANTILES)
def test_chi2_tail_at_tabulated_quantiles(dof, q95, q99):
    assert chi2_sf(q95, dof) == pytest.approx(0.05, rel=1e-8)
    assert chi2_sf(q99, dof) == pytest.approx(0.01, rel=1e-8)


@pytest.mark.parametrize("x", [0.01, 0.5, 1.0, 2.5, 7.0, 30.0, 200.0])
def test_chi2_and_gamma_closed_forms(x):
    assert chi2_sf(x, 2) == pytest.approx(math.exp(-x / 2), rel=1e-12)
    assert chi2_sf(x, 1) == pytest.approx(math.erfc(math.sqrt(x / 2)), rel=1e-11)
    tail3 = math.erfc(math.sqrt(x / 2)) + math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
    assert chi2_sf(x, 3) == pytest.approx(tail3, rel=1e-11)
    poisson = math.exp(-x) * sum(x**k / math.factorial(k) for k in range(4))
    assert gammaincc(4, x) == pytest.approx(poisson, rel=1e-11)
    assert chi2_sf(0, 3) == 1.0


def test_incomplete_beta_closed_forms():
    x = np.linspace(0, 1, 21)
    np.testing.assert_allclose(betainc(1, 1, x), x, atol=1e-15)
    np.testing.assert_allclose(betainc(3.5, 1, x), x**3.5, atol=1e-14)
    np.testing.assert_allclose(betainc(1, 4, x), 1 - (1 - x) ** 4, atol=1e-14)
    # Beta(1/2, 1/2) is the arcsine distribution and Beta(1/2, 3/2) has a similar closed form.
    np.testing.assert_allclose(betainc(0.5, 0.5, x), 2 / np.pi * np.arcsin(np.sqrt(x)), atol=1e-14)
    closed = 2 / np.pi * (np.arcsin(np.sqrt(x)) + np.sqrt(x * (1 - x)))
    np.testing.assert_allclose(betainc(0.5, 1.5, x), closed, atol=1e-14)
    np.testing.assert_allclose(betainc([2, 30, 700], [2, 30, 700], 0.5), 0.5, atol=1e-13)
    np.testing.assert_allclose(betainc(2.5, 7, x) + betainc(7, 2.5, 1 - x), 1, atol=1e-14)
    np.testing.assert_equal(betainc(2, 3, [-0.5, 0, 1, 1.5]), [0, 0, 1, 1])


@pytest.mark.parametrize("n,p", [(50, Fraction(3, 10)), (500, Fraction(13, 1000))])
def test_incomplete_beta_matches_exact_binomial_tails(n, p):
    for k in range(1, n + 1, max(1, n // 25)):
        exact = sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1))
        assert betainc(k, n - k + 1, float(p)) == pytest.approx(float(exact), rel=1e-11, abs=1e-300)


# Exact binomial tails summed with mpmath at 40 digits, and the Jeffreys value from
# numerical integration of the Beta(d + 1/2, N - d + 1/2) density at the same precision.
@pytest.mark.parametrize(
    "test,event,total,expected",
    [
        (binomial_test, 50_300, 10**7, 0.089769730480718044),
        (binomial_test, 5_007_000, 10**9, 0.00085205602053193747),
        (jeffreys_test, 50_300, 10**7, 0.089407659340101566),
    ],
)
def test_large_grades_match_high_precision_values(test, event, total, expected):
    assert test(event, total, 0.005) == pytest.approx(expected, rel=1e-11)


def test_incomplete_beta_for_one_large_parameter():
    # A grade of a million without events: I_p(1/2, N + 1/2) tends to erf(sqrt(N p)).
    total, p = 10**6, 2e-6
    assert jeffreys_test(0, total, p) == pytest.approx(math.erf(math.sqrt(total * p)), rel=1e-5)


def test_chi2_tail_at_infinity_and_missing_values():
    assert chi2_sf(math.inf, 3) == 0.0
    with pytest.raises(ValueError, match="must not be NaN"):
        chi2_sf(math.nan, 3)
    # A grade with a vanishing PD gives an infinite statistic and a p-value of zero.
    statistic, pvalue = hosmer_lemeshow([1, 1, 1], [10] * 3, [1e-320, 0.5, 0.5], ddof=0)
    assert statistic == math.inf and pvalue == 0.0


def test_prior_correction_matches_hand_calculation():
    # omega = (0.02 / 0.98) / (0.5 / 0.5) = 1 / 49.
    corrected = prior_correction([0.0, 0.05, 0.2, 0.5, 1.0], sample_rate=0.5, population_rate=0.02)
    expected = [0, 0.05 / (0.05 + 49 * 0.95), 0.2 / (0.2 + 49 * 0.8), 0.02, 1]
    np.testing.assert_allclose(corrected, expected, rtol=1e-14)
    assert corrected_intercept(0.0, sample_rate=0.5, population_rate=0.02) == pytest.approx(
        -math.log(49), rel=1e-15
    )


def test_prior_correction_agrees_with_the_intercept():
    linear = np.array([-3.0, -1.2, 0.0, 0.7, 2.5])
    intercept = -1.5
    prob = 1 / (1 + np.exp(-(intercept + linear)))
    rates = {"sample_rate": 0.3, "population_rate": 0.04}
    shifted = corrected_intercept(intercept, **rates)
    np.testing.assert_allclose(
        prior_correction(prob, **rates), 1 / (1 + np.exp(-(shifted + linear))), rtol=1e-13
    )
    np.testing.assert_allclose(prior_correction(prob, sample_rate=0.3, population_rate=0.3), prob)


def test_prior_correction_keeps_shapes_and_missing_values():
    prob = pd.Series([0.5, np.nan], index=[4, 2], name="pd")
    corrected = prior_correction(prob, sample_rate=0.5, population_rate=0.02)
    assert corrected.index.tolist() == [4, 2] and corrected.name == "pd"
    assert corrected[4] == pytest.approx(0.02) and np.isnan(corrected[2])
    assert isinstance(prior_correction(0.5, sample_rate=0.5, population_rate=0.02), float)


@pytest.mark.parametrize(
    "prob,rates,message",
    [
        (1.2, {}, "prob must be between 0 and 1, got 1.2"),
        ([0.5, -0.1], {}, "prob must be between 0 and 1, got -0.1"),
        (0.5, {"sample_rate": 0}, "sample_rate must be strictly between 0 and 1, got 0"),
        (
            0.5,
            {"population_rate": 1.0},
            "population_rate must be strictly between 0 and 1, got 1.0",
        ),
        (0.5, {"sample_rate": True}, "sample_rate must be strictly between 0 and 1, got True"),
    ],
)
def test_invalid_prior_correction(prob, rates, message):
    with pytest.raises(ValueError, match=message):
        prior_correction(prob, **{"sample_rate": 0.5, "population_rate": 0.02, **rates})


def test_invalid_intercept_correction():
    with pytest.raises(ValueError, match="intercept must be a finite number, got nan"):
        corrected_intercept(np.nan, sample_rate=0.5, population_rate=0.02)


def test_calibration_table_counts_and_rates():
    grades = pd.Series(["B", "A", "B", "A", "B", "C"])
    target = pd.Series([1, 0, 0, 0, 1, 0])
    prob = pd.Series([0.4, 0.1, 0.2, 0.3, 0.6, 0.05])
    table = calibration_table(grades, target, prob)
    assert table.index.name == "bin"
    assert table.index.tolist() == ["B", "A", "C"]
    assert table.columns.tolist() == ["total", "event", "event_rate", "mean_prob"]
    assert table["total"].tolist() == [3, 2, 1]
    assert table["event"].tolist() == [2, 0, 0]
    np.testing.assert_allclose(table["event_rate"], [2 / 3, 0, 0])
    np.testing.assert_allclose(table["mean_prob"], [0.4, 0.2, 0.05])
    assert table["total"].dtype == np.int64 and table["event"].dtype == np.int64


def test_calibration_table_follows_category_order_with_missing_last():
    grades = pd.Categorical(["low", None, "high", "low"], categories=["high", "mid", "low"])
    table = calibration_table(grades, [0, 1, 1, 0], [0.1, 0.5, 0.9, 0.2])
    assert table.index[:2].tolist() == ["high", "low"]
    assert pd.isna(table.index[2])
    assert table["total"].tolist() == [1, 2, 1]
    np.testing.assert_allclose(table["mean_prob"], [0.9, 0.15, 0.5])


def test_calibration_table_puts_missing_labels_in_their_own_group():
    grades = pd.Series(["B", None, "A", "B", None])
    table = calibration_table(grades, [1, 0, 0, 1, 1], [0.4, 0.1, 0.2, 0.6, 0.3])
    assert table.index[[0, 2]].tolist() == ["B", "A"] and pd.isna(table.index[1])
    assert table["total"].tolist() == [2, 2, 1]
    assert table["event"].tolist() == [2, 1, 0]
    np.testing.assert_allclose(table["mean_prob"], [0.5, 0.2, 0.2])


def test_master_scale_on_bank_data():
    data = Dataset.bank_data()
    # A stand-in PD: the balance rank, mapped to [0.01, 0.3].
    prob = 0.01 + 0.29 * data["balance"].rank(pct=True)
    edges = [0, 0.05, 0.1, 0.2, 1]
    grades = pd.cut(prob, edges, labels=[1, 2, 3, 4])
    table = calibration_table(grades, data["target"], prob)
    assert table.index.tolist() == [1, 2, 3, 4]
    assert table["total"].sum() == len(data)
    assert table["event"].sum() == data["target"].sum()
    codes = grades.cat.codes.to_numpy()
    np.testing.assert_array_equal(table["total"], np.bincount(codes))
    np.testing.assert_array_equal(table["event"], np.bincount(codes, weights=data["target"]))
    np.testing.assert_allclose(table["mean_prob"], prob.groupby(codes).mean())
    # Interval labels from pandas.cut keep their order too.
    intervals = calibration_table(pd.cut(prob, edges), data["target"], prob)
    np.testing.assert_array_equal(intervals["total"], table["total"])


@pytest.mark.parametrize(
    "groups,target,prob,message",
    [
        (pd.Series([1, 2]), pd.Series([0, 1], index=[1, 2]), [0.1, 0.2], "identical indices"),
        ([1, 2], [0, 1, 1], [0.1, 0.2], "same length, got 2, 3 and 2"),
        ([], [], [], "must contain observations"),
        ([1, 2], [0, 2], [0.1, 0.2], "target must contain only 0 and 1, got 2"),
        ([1, 2], [0, None], [0.1, 0.2], "target must contain only 0 and 1, got nan"),
        ([1, 2], [0, 1], [0.1, np.nan], "prob must not have missing values"),
        ([1, 2], [0, 1], [0.1, 1.5], "prob must be between 0 and 1, got 1.5"),
        (np.ones((2, 2)), [0, 1], [0.1, 0.2], "groups must be one-dimensional, got 2 dimensions"),
    ],
)
def test_invalid_calibration_table_inputs(groups, target, prob, message):
    with pytest.raises(ValueError, match=message):
        calibration_table(groups, target, prob)


def test_binomial_test_matches_hand_calculation():
    # P(D >= 3) for D ~ Bin(10, 0.1) is 1 - 0.9^10 - 10 * 0.1 * 0.9^9 - 45 * 0.01 * 0.9^8.
    tail = 1 - 0.9**10 - 0.9**9 - 0.45 * 0.9**8
    pvalues = binomial_test([0, 3, 10], 10, 0.1)
    np.testing.assert_allclose(pvalues, [1, tail, 1e-10], rtol=1e-12)
    assert binomial_test(3, 10, 0.1) == pytest.approx(0.0701908264, rel=1e-9)
    assert isinstance(binomial_test(3, 10, 0.1), float)
    np.testing.assert_equal(binomial_test([0, 2, 2], [5, 5, 5], [0, 0, 1]), [1, 0, 1])


def test_jeffreys_test_matches_closed_forms():
    p = np.array([0.01, 0.2, 0.7])
    # One observation: Beta(1/2, 3/2) without the event and Beta(3/2, 1/2) with it.
    without = 2 / np.pi * (np.arcsin(np.sqrt(p)) + np.sqrt(p * (1 - p)))
    with_event = 1 - 2 / np.pi * (np.arcsin(np.sqrt(1 - p)) + np.sqrt(p * (1 - p)))
    np.testing.assert_allclose(jeffreys_test(0, 1, p), without, rtol=1e-13)
    # The closed form loses digits to cancellation near 0.
    np.testing.assert_allclose(jeffreys_test(1, 1, p), with_event, rtol=1e-11)


def test_tests_agree_on_the_direction():
    # More events than predicted gives small p-values; fewer gives large ones.
    event, count, prob = np.array([1, 4, 12]), 200, 0.02
    for pvalues in (binomial_test(event, count, prob), jeffreys_test(event, count, prob)):
        assert pvalues[0] > 0.9 and pvalues[2] < 0.01
        assert np.all(np.diff(pvalues) < 0)


@pytest.mark.parametrize("test", [binomial_test, jeffreys_test])
@pytest.mark.parametrize(
    "event,count,prob,message",
    [
        (1, 0, 0.1, "total must be a positive whole number, got 0.0"),
        (1, 2.5, 0.1, "total must be a positive whole number, got 2.5"),
        (1, np.inf, 0.1, "total must be a positive whole number, got inf"),
        (-1, 5, 0.1, "event must be a non-negative whole number, got -1.0"),
        (np.nan, 5, 0.1, "event must be a non-negative whole number, got nan"),
        (6, 5, 0.1, "event must not exceed total, got 6.0 events among 5.0"),
        (1, 5, 1.1, "prob must be between 0 and 1, got 1.1"),
        (1, 5, np.nan, "prob must not have missing values"),
        ([1, 2], [5, 5, 5], 0.1, "cannot be broadcast together"),
    ],
)
def test_invalid_grade_test_inputs(test, event, count, prob, message):
    with pytest.raises(ValueError, match=message):
        test(event, count, prob)


@pytest.mark.parametrize("test", [binomial_test, jeffreys_test])
def test_grade_tests_keep_the_series_index(test):
    event = pd.Series([1, 5], index=["A", "B"])
    total = pd.Series([100, 10], index=["A", "B"])
    pvalues = test(event, total, 0.05)
    assert isinstance(pvalues, pd.Series)
    assert pvalues.index.tolist() == ["A", "B"]
    np.testing.assert_array_equal(pvalues.to_numpy(), test([1, 5], [100, 10], 0.05))
    with pytest.raises(ValueError, match="identical indices when they are Series"):
        test(event, total.iloc[::-1], 0.05)
    with pytest.raises(ValueError, match="broadcast to the length 2 of their Series index"):
        test(event, np.full((3, 1), 100), 0.05)


def test_hosmer_lemeshow_matches_hand_calculation():
    event, count, prob = [2, 6, 15], [100, 100, 100], [0.03, 0.06, 0.12]
    # (2 - 3)^2 / (3 * 0.97) + 0 + (15 - 12)^2 / (12 * 0.88)
    expected = 1 / 2.91 + 9 / 10.56
    statistic, pvalue = hosmer_lemeshow(event, count, prob)
    assert statistic == pytest.approx(expected, rel=1e-14)
    assert pvalue == pytest.approx(math.erfc(math.sqrt(expected / 2)), rel=1e-12)
    statistic, pvalue = hosmer_lemeshow(event, count, prob, ddof=0)
    tail3 = math.erfc(math.sqrt(expected / 2)) + math.sqrt(2 * expected / math.pi) * math.exp(
        -expected / 2
    )
    assert pvalue == pytest.approx(tail3, rel=1e-12)


def test_hosmer_lemeshow_from_a_calibration_table():
    prob = np.repeat([0.1, 0.2, 0.3, 0.4], 10)
    target = np.zeros(40, dtype=int)
    target[[0, 10, 11, 20, 21, 22, 30, 31, 32, 33]] = 1
    table = calibration_table(prob, target, prob)
    statistic, _ = hosmer_lemeshow(table["event"], table["total"], table["mean_prob"])
    assert statistic == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize(
    "event,count,prob,ddof,message",
    [
        ([1, 2], [10, 10], [0.1, 0.2], 2, "2 group\\(s\\) with ddof=2 leave 0 degrees of freedom"),
        ([1, 2, 3], [10, 10, 10], [0.1, 0, 0.3], 2, "strictly between 0 and 1, got 0.0"),
        ([1, 2, 3], [10, 10, 10], [0.1, 0.2, 0.3], -1, "non-negative integer, got -1"),
        ([1, 2, 3], [10, 10, 10], [0.1, 0.2, 0.3], 1.0, "non-negative integer, got 1.0"),
        (1, 10, 0.1, 0, "one-dimensional, got 0 dimensions"),
    ],
)
def test_invalid_hosmer_lemeshow_inputs(event, count, prob, ddof, message):
    with pytest.raises(ValueError, match=message):
        hosmer_lemeshow(event, count, prob, ddof=ddof)


def test_hosmer_lemeshow_rejects_misaligned_series():
    event = pd.Series([1, 2, 3], index=list("abc"))
    total = pd.Series([10, 10, 10], index=list("cba"))
    with pytest.raises(ValueError, match="identical indices when they are Series"):
        hosmer_lemeshow(event, total, [0.1, 0.2, 0.3], ddof=0)


def test_herfindahl_index_matches_hand_calculation():
    assert herfindahl_index([50, 30, 20]) == pytest.approx(0.38, rel=1e-15)
    assert herfindahl_index([50, 30, 20], normalize=True) == pytest.approx(
        (0.38 - 1 / 3) / (2 / 3), rel=1e-14
    )
    assert herfindahl_index([5, 5, 5, 5]) == pytest.approx(0.25)
    assert herfindahl_index([5, 5, 5, 5], normalize=True) == pytest.approx(0, abs=1e-15)
    # Empty grades count towards the number of grades.
    assert herfindahl_index([7, 0, 0]) == 1.0
    assert herfindahl_index(pd.Series([7, 0, 0]), normalize=True) == 1.0


@pytest.mark.parametrize(
    "counts,normalize,message",
    [
        ([[1, 2]], False, "one-dimensional, got 2 dimensions"),
        ([1, -1], False, "finite and non-negative, got -1.0"),
        ([1, np.nan], False, "finite and non-negative, got nan"),
        ([0, 0], False, "positive sum, got 0.0"),
        ([3], True, "at least two grades, got 1"),
        ([1, 2], "no", "normalize must be True or False, got 'no'"),
    ],
)
def test_invalid_herfindahl_inputs(counts, normalize, message):
    with pytest.raises(ValueError, match=message):
        herfindahl_index(counts, normalize=normalize)


def test_inputs_are_not_modified():
    grades = pd.Series(["a", "b", "a"])
    target = pd.Series([0, 1, 1])
    prob = pd.Series([0.2, 0.4, 0.6])
    copies = grades.copy(), target.copy(), prob.copy()
    table = calibration_table(grades, target, prob)
    prior_correction(prob, sample_rate=0.5, population_rate=0.1)
    event, count = table["event"].to_numpy(), table["total"].to_numpy()
    binomial_test(event, count, 0.3)
    jeffreys_test(event, count, 0.3)
    for original, copy in zip((grades, target, prob), copies, strict=True):
        pd.testing.assert_series_equal(original, copy)
    np.testing.assert_array_equal(table["event"], [1, 1])
