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
    isotonic_fit,
    isotonic_predict,
    jeffreys_test,
    platt_fit,
    platt_predict,
    prior_correction,
)
from kivi.datasets import Dataset
from kivi.selection import SeparationError

INF = np.inf

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
    probability = 1 / (1 + np.exp(-(intercept + linear)))
    rates = {"sample_rate": 0.3, "population_rate": 0.04}
    shifted = corrected_intercept(intercept, **rates)
    np.testing.assert_allclose(
        prior_correction(probability, **rates), 1 / (1 + np.exp(-(shifted + linear))), rtol=1e-13
    )
    np.testing.assert_allclose(
        prior_correction(probability, sample_rate=0.3, population_rate=0.3), probability
    )


def test_prior_correction_keeps_shapes_and_missing_values():
    probability = pd.Series([0.5, np.nan], index=[4, 2], name="pd")
    corrected = prior_correction(probability, sample_rate=0.5, population_rate=0.02)
    assert corrected.index.tolist() == [4, 2] and corrected.name == "pd"
    assert corrected[4] == pytest.approx(0.02) and np.isnan(corrected[2])
    assert isinstance(prior_correction(0.5, sample_rate=0.5, population_rate=0.02), float)


@pytest.mark.parametrize(
    "probability,rates,message",
    [
        (1.2, {}, "probability must be between 0 and 1, got 1.2"),
        ([0.5, -0.1], {}, "probability must be between 0 and 1, got -0.1"),
        (0.5, {"sample_rate": 0}, "sample_rate must be strictly between 0 and 1, got 0"),
        (
            0.5,
            {"population_rate": 1.0},
            "population_rate must be strictly between 0 and 1, got 1.0",
        ),
        (0.5, {"sample_rate": True}, "sample_rate must be strictly between 0 and 1, got True"),
    ],
)
def test_invalid_prior_correction(probability, rates, message):
    with pytest.raises(ValueError, match=message):
        prior_correction(probability, **{"sample_rate": 0.5, "population_rate": 0.02, **rates})


def test_invalid_intercept_correction():
    with pytest.raises(ValueError, match="intercept must be a finite number, got nan"):
        corrected_intercept(np.nan, sample_rate=0.5, population_rate=0.02)


def test_calibration_table_counts_and_rates():
    grades = pd.Series(["B", "A", "B", "A", "B", "C"])
    target = pd.Series([1, 0, 0, 0, 1, 0])
    probability = pd.Series([0.4, 0.1, 0.2, 0.3, 0.6, 0.05])
    table = calibration_table(grades, target, probability)
    assert table.index.name == "bin"
    assert table.index.tolist() == ["B", "A", "C"]
    assert table.columns.tolist() == ["total", "event", "event_rate", "mean_probability"]
    assert table["total"].tolist() == [3, 2, 1]
    assert table["event"].tolist() == [2, 0, 0]
    np.testing.assert_allclose(table["event_rate"], [2 / 3, 0, 0])
    np.testing.assert_allclose(table["mean_probability"], [0.4, 0.2, 0.05])
    assert table["total"].dtype == np.int64 and table["event"].dtype == np.int64


def test_calibration_table_follows_category_order_with_missing_last():
    grades = pd.Categorical(["low", None, "high", "low"], categories=["high", "mid", "low"])
    table = calibration_table(grades, [0, 1, 1, 0], [0.1, 0.5, 0.9, 0.2])
    assert table.index[:2].tolist() == ["high", "low"]
    assert pd.isna(table.index[2])
    assert table["total"].tolist() == [1, 2, 1]
    np.testing.assert_allclose(table["mean_probability"], [0.9, 0.15, 0.5])


def test_calibration_table_puts_missing_labels_in_their_own_group():
    grades = pd.Series(["B", None, "A", "B", None])
    table = calibration_table(grades, [1, 0, 0, 1, 1], [0.4, 0.1, 0.2, 0.6, 0.3])
    assert table.index[[0, 2]].tolist() == ["B", "A"] and pd.isna(table.index[1])
    assert table["total"].tolist() == [2, 2, 1]
    assert table["event"].tolist() == [2, 1, 0]
    np.testing.assert_allclose(table["mean_probability"], [0.5, 0.2, 0.2])


def test_master_scale_on_bank_data():
    data = Dataset.bank_data()
    # A stand-in PD: the balance rank, mapped to [0.01, 0.3].
    probability = 0.01 + 0.29 * data["balance"].rank(pct=True)
    edges = [0, 0.05, 0.1, 0.2, 1]
    grades = pd.cut(probability, edges, labels=[1, 2, 3, 4])
    table = calibration_table(grades, data["target"], probability)
    assert table.index.tolist() == [1, 2, 3, 4]
    assert table["total"].sum() == len(data)
    assert table["event"].sum() == data["target"].sum()
    codes = grades.cat.codes.to_numpy()
    np.testing.assert_array_equal(table["total"], np.bincount(codes))
    np.testing.assert_array_equal(table["event"], np.bincount(codes, weights=data["target"]))
    np.testing.assert_allclose(table["mean_probability"], probability.groupby(codes).mean())
    # Interval labels from pandas.cut keep their order too.
    intervals = calibration_table(pd.cut(probability, edges), data["target"], probability)
    np.testing.assert_array_equal(intervals["total"], table["total"])


@pytest.mark.parametrize(
    "groups,target,probability,message",
    [
        (pd.Series([1, 2]), pd.Series([0, 1], index=[1, 2]), [0.1, 0.2], "identical indices"),
        ([1, 2], [0, 1, 1], [0.1, 0.2], "same length, got 2, 3 and 2"),
        ([], [], [], "must contain observations"),
        ([1, 2], [0, 2], [0.1, 0.2], "target must contain only 0 and 1, got 2"),
        ([1, 2], [0, None], [0.1, 0.2], "target must contain only 0 and 1, got nan"),
        ([1, 2], [0, 1], [0.1, np.nan], "probability must not have missing values"),
        ([1, 2], [0, 1], [0.1, 1.5], "probability must be between 0 and 1, got 1.5"),
        (np.ones((2, 2)), [0, 1], [0.1, 0.2], "groups must be one-dimensional, got 2 dimensions"),
    ],
)
def test_invalid_calibration_table_inputs(groups, target, probability, message):
    with pytest.raises(ValueError, match=message):
        calibration_table(groups, target, probability)


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
    event, count, probability = np.array([1, 4, 12]), 200, 0.02
    for pvalues in (
        binomial_test(event, count, probability),
        jeffreys_test(event, count, probability),
    ):
        assert pvalues[0] > 0.9 and pvalues[2] < 0.01
        assert np.all(np.diff(pvalues) < 0)


@pytest.mark.parametrize("test", [binomial_test, jeffreys_test])
@pytest.mark.parametrize(
    "event,count,probability,message",
    [
        (1, 0, 0.1, "total must be a positive whole number, got 0.0"),
        (1, 2.5, 0.1, "total must be a positive whole number, got 2.5"),
        (1, np.inf, 0.1, "total must be a positive whole number, got inf"),
        (-1, 5, 0.1, "event must be a non-negative whole number, got -1.0"),
        (np.nan, 5, 0.1, "event must be a non-negative whole number, got nan"),
        (6, 5, 0.1, "event must not exceed total, got 6.0 events among 5.0"),
        (1, 5, 1.1, "probability must be between 0 and 1, got 1.1"),
        (1, 5, np.nan, "probability must not have missing values"),
        ([1, 2], [5, 5, 5], 0.1, "cannot be broadcast together"),
    ],
)
def test_invalid_grade_test_inputs(test, event, count, probability, message):
    with pytest.raises(ValueError, match=message):
        test(event, count, probability)


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
    event, count, probability = [2, 6, 15], [100, 100, 100], [0.03, 0.06, 0.12]
    # (2 - 3)^2 / (3 * 0.97) + 0 + (15 - 12)^2 / (12 * 0.88)
    expected = 1 / 2.91 + 9 / 10.56
    statistic, pvalue = hosmer_lemeshow(event, count, probability)
    assert statistic == pytest.approx(expected, rel=1e-14)
    assert pvalue == pytest.approx(math.erfc(math.sqrt(expected / 2)), rel=1e-12)
    statistic, pvalue = hosmer_lemeshow(event, count, probability, ddof=0)
    tail3 = math.erfc(math.sqrt(expected / 2)) + math.sqrt(2 * expected / math.pi) * math.exp(
        -expected / 2
    )
    assert pvalue == pytest.approx(tail3, rel=1e-12)


def test_hosmer_lemeshow_from_a_calibration_table():
    probability = np.repeat([0.1, 0.2, 0.3, 0.4], 10)
    target = np.zeros(40, dtype=int)
    target[[0, 10, 11, 20, 21, 22, 30, 31, 32, 33]] = 1
    table = calibration_table(probability, target, probability)
    statistic, _ = hosmer_lemeshow(table["event"], table["total"], table["mean_probability"])
    assert statistic == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize(
    "event,count,probability,ddof,message",
    [
        ([1, 2], [10, 10], [0.1, 0.2], 2, "2 group\\(s\\) with ddof=2 leave 0 degrees of freedom"),
        ([1, 2, 3], [10, 10, 10], [0.1, 0, 0.3], 2, "strictly between 0 and 1, got 0.0"),
        ([1, 2, 3], [10, 10, 10], [0.1, 0.2, 0.3], -1, "non-negative integer, got -1"),
        ([1, 2, 3], [10, 10, 10], [0.1, 0.2, 0.3], 1.0, "non-negative integer, got 1.0"),
        (1, 10, 0.1, 0, "one-dimensional, got 0 dimensions"),
    ],
)
def test_invalid_hosmer_lemeshow_inputs(event, count, probability, ddof, message):
    with pytest.raises(ValueError, match=message):
        hosmer_lemeshow(event, count, probability, ddof=ddof)


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
    probability = pd.Series([0.2, 0.4, 0.6])
    copies = grades.copy(), target.copy(), probability.copy()
    table = calibration_table(grades, target, probability)
    prior_correction(probability, sample_rate=0.5, population_rate=0.1)
    event, count = table["event"].to_numpy(), table["total"].to_numpy()
    binomial_test(event, count, 0.3)
    jeffreys_test(event, count, 0.3)
    for original, copy in zip((grades, target, probability), copies, strict=True):
        pd.testing.assert_series_equal(original, copy)
    np.testing.assert_array_equal(table["event"], [1, 1])


def test_platt_fit_on_a_binary_score_reproduces_the_two_event_rates():
    # With two distinct scores the logistic curve passes through both event rates:
    # 1 event of 4 at score 2 and 3 of 4 at score 5, so logit(1/4) = 2a + b and
    # logit(3/4) = 5a + b.
    score = [2, 2, 2, 2, 5, 5, 5, 5]
    target = [1, 0, 0, 0, 1, 1, 0, 1]
    slope, intercept = platt_fit(score, target)
    assert slope == pytest.approx(2 * math.log(3) / 3, rel=1e-12)
    assert intercept == pytest.approx(-math.log(3) - 2 * slope, rel=1e-12)
    assert isinstance(slope, float) and isinstance(intercept, float)
    np.testing.assert_allclose(platt_predict([2, 5], slope, intercept), [0.25, 0.75], rtol=1e-12)


def test_platt_fit_matches_statsmodels_on_bank_data():
    data = Dataset.bank_data()
    # statsmodels Logit(target, add_constant(duration)).fit(tol=1e-14).
    slope, intercept = platt_fit(data["duration"], data["target"])
    assert slope == pytest.approx(0.003549552890267578, rel=1e-10)
    assert intercept == pytest.approx(-3.255934555378699, rel=1e-10)
    slope, intercept = platt_fit(data["age"], data["target"])
    assert slope == pytest.approx(0.012903913401860033, rel=1e-10)
    assert intercept == pytest.approx(-2.576818417293876, rel=1e-10)


def test_platt_fit_matches_the_event_rate_and_follows_the_scale():
    rng = np.random.default_rng(5)
    score = rng.normal(size=500)
    target = (rng.random(500) < 1 / (1 + np.exp(-(0.4 + 1.3 * score)))).astype(int)
    slope, intercept = platt_fit(score, target)
    probability = platt_predict(score, slope, intercept)
    # The score equations: fitted events match observed events overall and weighted by score.
    assert probability.sum() == pytest.approx(target.sum(), rel=1e-10)
    assert probability @ score == pytest.approx(target @ score, rel=1e-9)
    # A linear change of the score changes the parameters, not the probabilities.
    rescaled = platt_fit(2 * score + 3, target)
    assert rescaled[0] == pytest.approx(slope / 2, rel=1e-10)
    np.testing.assert_allclose(platt_predict(2 * score + 3, *rescaled), probability, rtol=1e-10)
    # Points that fall as the risk rises get a negative slope and the same probabilities.
    points = platt_fit(600 - 50 * score, target)
    assert points[0] == pytest.approx(-slope / 50, rel=1e-10)
    np.testing.assert_allclose(platt_predict(600 - 50 * score, *points), probability, rtol=1e-9)


def test_platt_integer_weights_equal_repeated_rows():
    score = np.array([0.3, -1.2, 0.8, 2.0, -0.5, 1.1, 0.0])
    target = np.array([0, 0, 1, 1, 1, 0, 0])
    weights = np.array([2, 1, 0, 3, 1, 2, 4])
    repeated = np.repeat(np.arange(7), weights)
    weighted = platt_fit(score, target, weights=weights)
    np.testing.assert_allclose(weighted, platt_fit(score[repeated], target[repeated]), rtol=1e-12)
    np.testing.assert_allclose(
        platt_fit(pd.Series(score, index=range(7, 14)), target, weights=weights / 10),
        weighted,
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "score,target",
    [
        ([1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1]),
        ([4.0, 3.0, 2.0, 1.0], [0, 0, 1, 1]),
        # Quasi-complete: the classes overlap only at the tied score 2.
        ([1.0, 2.0, 2.0, 3.0, 3.0], [0, 0, 1, 1, 1]),
    ],
)
def test_platt_fit_raises_when_the_score_separates_the_classes(score, target):
    with pytest.raises(SeparationError, match="score separates the classes of target"):
        platt_fit(score, target)
    # The isotonic calibration of a separating score is 0 below the split and 1 above it, with
    # the event rate of the tied score in between.
    edges, probabilities = isotonic_fit(score, target, increasing=score[0] < score[-1])
    assert probabilities.min() == 0 and probabilities.max() == 1


def test_platt_fit_rejects_a_constant_score():
    with pytest.raises(ValueError, match="score must vary to fit a slope, got only 3.0"):
        platt_fit([3, 3, 3], [0, 1, 0])
    with pytest.raises(ValueError, match="score must vary to fit a slope, got only 1.0"):
        platt_fit([1, 1, 2], [0, 1, 1], weights=[1, 1, 0])


def test_platt_on_series_with_other_indices_and_nullable_dtypes():
    score = pd.Series([2, 2, 2, 2, 5, 5, 5, 5], index=list("hgfedcba"), dtype="Float64")
    target = pd.Series([1, 0, 0, 0, 1, 1, 0, 1], index=list("hgfedcba"), dtype="Int64")
    slope, intercept = platt_fit(score, target)
    assert slope == pytest.approx(2 * math.log(3) / 3, rel=1e-12)
    with pytest.raises(ValueError, match="score and target must have identical indices"):
        platt_fit(score, target.sort_index())
    probability = platt_predict(
        pd.Series([2.0, None, 5.0], index=[9, 8, 7], dtype="Float64"), slope, intercept
    )
    assert probability.index.tolist() == [9, 8, 7]
    np.testing.assert_allclose(probability, [0.25, np.nan, 0.75], rtol=1e-12)


def test_platt_predict_shapes_and_extreme_scores():
    # 1 / (1 + exp(-(1.5 * s - 0.5))) at s = -2, 0 and 2.
    expected = 1 / (1 + np.exp([3.5, 0.5, -2.5]))
    np.testing.assert_allclose(platt_predict([-2.0, 0.0, 2.0], 1.5, -0.5), expected, rtol=1e-14)
    assert isinstance(platt_predict(0.0, 1.5, -0.5), float)
    series = platt_predict(pd.Series([1.0, np.nan], index=[3, 1], name="score"), 1.5, -0.5)
    assert series.index.tolist() == [3, 1] and series.name == "score" and np.isnan(series[1])
    np.testing.assert_array_equal(
        platt_predict([-1e308, 1e308, -np.inf, np.inf], 2.0, 0.0), [0, 1, 0, 1]
    )
    assert platt_predict(-800.0, 1.0, 0.0) == pytest.approx(math.exp(-800), rel=1e-12)


@pytest.mark.parametrize(
    "score,slope,intercept,message",
    [
        ([0.1], np.nan, 0.0, "slope must be a finite number, got nan"),
        ([0.1], 1.0, True, "intercept must be a finite number, got True"),
        (["a"], 1.0, 0.0, "score must be real and numeric, got dtype <U1"),
        (pd.Series(["a"]), 1.0, 0.0, "score must be real and numeric, got dtype"),
        (np.array([1 + 1j]), 1.0, 0.0, "score must be real and numeric, got dtype complex128"),
    ],
)
def test_invalid_platt_predict(score, slope, intercept, message):
    with pytest.raises(ValueError, match=message):
        platt_predict(score, slope, intercept)


def test_isotonic_fit_matches_hand_calculation():
    # Rates by score: 0, 1, 0, 0, 1, 0, 1, 1. Pooling gives 0 | 1/3 for 0.2-0.4 | 1/2 for
    # 0.5-0.6 | 1 for 0.7-0.8, where the equal rates of 0.7 and 0.8 are pooled as well.
    score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    target = [0, 1, 0, 0, 1, 0, 1, 1]
    edges, probabilities = isotonic_fit(score, target)
    np.testing.assert_array_equal(edges, [-INF, 0.1, 0.4, 0.6, INF])
    np.testing.assert_allclose(probabilities, [0, 1 / 3, 1 / 2, 1], rtol=1e-15)
    np.testing.assert_allclose(
        isotonic_predict(score, edges, probabilities), [0, 1 / 3, 1 / 3, 1 / 3, 0.5, 0.5, 1, 1]
    )


def test_isotonic_fit_pools_tied_scores():
    # Score 1 has rate 1/2 and score 2 rate 0, so they pool to 1/4; score 3 has rate 1.
    edges, probabilities = isotonic_fit([2, 1, 3, 1, 2], [0, 1, 1, 0, 0])
    np.testing.assert_array_equal(edges, [-INF, 2, INF])
    np.testing.assert_allclose(probabilities, [0.25, 1.0])
    # Every tie gets one probability, the pooled rate of its band.
    np.testing.assert_allclose(
        isotonic_predict([1, 1, 2, 2, 3], edges, probabilities), [0.25] * 4 + [1]
    )


def test_isotonic_fit_on_constant_and_separated_scores():
    edges, probabilities = isotonic_fit([5, 5, 5, 5, 5], [0, 1, 1, 0, 1])
    np.testing.assert_array_equal(edges, [-INF, INF])
    np.testing.assert_allclose(probabilities, [0.6])
    edges, probabilities = isotonic_fit([1, 2, 3, 4], [0, 0, 1, 1])
    np.testing.assert_array_equal(edges, [-INF, 2, INF])
    np.testing.assert_array_equal(probabilities, [0.0, 1.0])
    # Running against the direction pools everything into one band.
    edges, probabilities = isotonic_fit([1, 2, 3, 4], [0, 0, 1, 1], increasing=False)
    np.testing.assert_array_equal(edges, [-INF, INF])
    np.testing.assert_allclose(probabilities, [0.5])


def test_isotonic_fit_on_extreme_scores():
    score = [-1e308, -1e300, 0.0, 1e300, 1e308]
    edges, probabilities = isotonic_fit(score, [0, 1, 0, 1, 1])
    np.testing.assert_array_equal(edges, [-INF, -1e308, 0.0, INF])
    np.testing.assert_allclose(probabilities, [0, 0.5, 1])
    values = [-INF, -1e308, -1e307, 0.0, 1e-300, INF]
    np.testing.assert_allclose(
        isotonic_predict(values, edges, probabilities), [0, 0, 0.5, 0.5, 1, 1]
    )


def test_isotonic_fit_decreasing_for_points():
    # Ascending scores 550, 600, 650, 700 have rates 1, 0, 1, 0; 600 and 650 pool to 1/2.
    points = [700, 650, 600, 550]
    target = [0, 1, 0, 1]
    edges, probabilities = isotonic_fit(points, target, increasing=False)
    np.testing.assert_array_equal(edges, [-INF, 550, 650, INF])
    np.testing.assert_allclose(probabilities, [1, 0.5, 0])
    negated = isotonic_fit([-p for p in points], target)
    np.testing.assert_allclose(negated[1], probabilities[::-1])


def _min_max_isotonic(score, target, weights):
    """Isotonic regression from its min-max formula, at each distinct score."""
    distinct = np.unique(score)
    total = np.array([weights[score == value].sum() for value in distinct])
    event = np.array([(weights * target)[score == value].sum() for value in distinct])
    keep = total > 0
    total, event = total[keep], event[keep]
    n = len(total)
    return distinct[keep], np.array(
        [
            max(
                min(event[j : k + 1].sum() / total[j : k + 1].sum() for k in range(i, n))
                for j in range(i + 1)
            )
            for i in range(n)
        ]
    )


@pytest.mark.parametrize("seed", range(6))
def test_isotonic_fit_matches_the_min_max_formula(seed):
    rng = np.random.default_rng(seed)
    score = rng.integers(0, 12, 40).astype(float)
    target = (rng.random(40) < 0.2 + 0.05 * score).astype(int)
    weights = rng.integers(0, 4, 40).astype(float)
    distinct, expected = _min_max_isotonic(score, target, weights)
    edges, probabilities = isotonic_fit(score, target, weights=weights)
    fitted = isotonic_predict(distinct, edges, probabilities)
    np.testing.assert_allclose(fitted, expected, rtol=1e-12, atol=1e-15)
    assert (np.diff(probabilities) > 0).all()
    # The weighted mean of the fitted probabilities is the weighted event rate.
    assert isotonic_predict(score, edges, probabilities) @ weights == pytest.approx(
        target @ weights, rel=1e-12
    )


def test_isotonic_fit_matches_scikit_learn_on_bank_data():
    data = Dataset.bank_data()
    edges, probabilities = isotonic_fit(data["duration"], data["target"])
    # scikit-learn's IsotonicRegression gives the same 26 rates at the fitted durations.
    assert len(probabilities) == 26
    np.testing.assert_allclose(probabilities[:4], [0, 1 / 480, 4 / 237, 7 / 199], rtol=1e-14)
    np.testing.assert_allclose(probabilities[-3:], [53 / 94, 5 / 7, 4 / 5], rtol=1e-14)
    np.testing.assert_array_equal(edges[1:5], [29, 75, 90, 143])
    np.testing.assert_array_equal(edges[-4:-1], [986, 1531, 1994])
    # Interior edges are observed durations, and the bands are those of pandas.cut.
    assert np.isin(edges[1:-1], data["duration"]).all()
    codes = pd.cut(data["duration"], edges).cat.codes.to_numpy()
    np.testing.assert_array_equal(
        isotonic_predict(data["duration"], edges, probabilities).to_numpy(), probabilities[codes]
    )
    table = calibration_table(pd.Series(codes), data["target"], probabilities[codes])
    np.testing.assert_allclose(table["event_rate"], table["mean_probability"], rtol=1e-12)


def test_isotonic_integer_weights_equal_repeated_rows():
    data = Dataset.bank_data()
    weights = np.random.default_rng(3).integers(0, 3, len(data))
    repeated = np.repeat(np.arange(len(data)), weights)
    weighted = isotonic_fit(data["balance"], data["target"], weights=weights)
    rows = isotonic_fit(data["balance"].to_numpy()[repeated], data["target"].to_numpy()[repeated])
    np.testing.assert_array_equal(weighted[0], rows[0])
    np.testing.assert_allclose(weighted[1], rows[1], rtol=1e-13)
    # A score with zero weight leaves no trace: here it would otherwise be a band of its own.
    edges, probabilities = isotonic_fit([1, 2, 3], [0, 1, 1], weights=[1, 0, 1])
    np.testing.assert_array_equal(edges, [-INF, 1, INF])
    np.testing.assert_array_equal(probabilities, [0, 1])


def test_isotonic_on_series_with_other_indices_and_nullable_dtypes():
    score = pd.Series([0.4, 0.1, 0.3, 0.2], index=[13, 11, 12, 10], dtype="Float64")
    target = pd.Series([1, 0, 1, 0], index=[13, 11, 12, 10], dtype="Int64")
    edges, probabilities = isotonic_fit(
        score, target, weights=pd.Series([1, 1, 1, 1], index=score.index)
    )
    np.testing.assert_array_equal(edges, [-INF, 0.2, INF])
    np.testing.assert_array_equal(probabilities, [0, 1])
    with pytest.raises(ValueError, match="weights must have the same index as score"):
        isotonic_fit(score, target, weights=pd.Series([1, 1, 1, 1]))
    predicted = isotonic_predict(
        pd.Series([0.2, None], index=["a", "b"], dtype="Float64"), edges, probabilities
    )
    assert predicted.index.tolist() == ["a", "b"]
    np.testing.assert_array_equal(predicted, [0, np.nan])
    assert isinstance(isotonic_predict(0.25, edges, probabilities), float)


@pytest.mark.parametrize(
    "fit,score,target,options,message",
    [
        (isotonic_fit, [1, 2], [0, 1], {"increasing": "yes"}, "increasing must be True or False"),
        (isotonic_fit, [1, 2], [1, 1], {}, "target must contain both 0 and 1, got only 1"),
        (isotonic_fit, [1, np.nan], [0, 1], {}, "score must be finite, got 1 missing"),
        (platt_fit, [1, 2], [0, 1], {"weights": [1, 0]}, "weights must give both target classes"),
        (platt_fit, [1, 2], [0, 1], {"weights": [1]}, "weights must have one weight per value"),
        (platt_fit, [1, 2], [0, 2], {}, "target must contain only 0 and 1, got 2"),
        (platt_fit, [1, 2, 3], [0, 1], {}, "score and target must have the same length"),
    ],
)
def test_invalid_calibration_fits(fit, score, target, options, message):
    with pytest.raises(ValueError, match=message):
        fit(score, target, **options)


@pytest.mark.parametrize(
    "edges,probabilities,message",
    [
        ([0, 1, 2], [0.1, 0.2], r"edges must increase from -inf to inf, got \[0.0, 1.0, 2.0\]"),
        ([-INF, 1, 1, INF], [0.1, 0.2, 0.3], "edges must increase from -inf to inf"),
        ([-INF, np.nan, INF], [0.1, 0.2], "edges must increase from -inf to inf"),
        ([-INF], [], "edges must increase from -inf to inf"),
        ([-INF, 1, INF], [0.1], r"one value for each of the 2 band\(s\), got shape \(1,\)"),
        ([-INF, 1, INF], [0.1, 1.2], "probabilities must be between 0 and 1, got 1.2"),
        ([-INF, 1, INF], [0.1, np.nan], "probabilities must not have missing values"),
        ([-INF, 1, INF], ["a", "b"], "edges and probabilities must be numeric"),
    ],
)
def test_invalid_isotonic_bands(edges, probabilities, message):
    with pytest.raises(ValueError, match=message):
        isotonic_predict([0.5], edges, probabilities)


def test_calibration_fits_do_not_modify_inputs():
    score = pd.Series([0.3, 0.1, 0.2, 0.4, 0.4])
    target = pd.Series([0, 1, 0, 1, 1])
    weights = pd.Series([1.0, 2.0, 1.0, 1.0, 3.0])
    copies = score.copy(), target.copy(), weights.copy()
    platt_fit(score, target, weights=weights)
    edges, probabilities = isotonic_fit(score, target, weights=weights)
    before = edges.copy(), probabilities.copy()
    isotonic_predict(score, edges, probabilities)
    for original, copy in zip((score, target, weights), copies, strict=True):
        pd.testing.assert_series_equal(original, copy)
    np.testing.assert_array_equal(edges, before[0])
    np.testing.assert_array_equal(probabilities, before[1])
