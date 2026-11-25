"""IRB risk weights, the Vasicek one-factor model and the most prudent PDs of kivi.capital."""

import math

import numpy as np
import pandas as pd
import pytest

from kivi import _stats
from kivi.capital import (
    asset_correlation,
    capital_requirement,
    conditional_pd,
    most_prudent_pd,
    systematic_factor,
    vasicek_fit,
)

# Basel II (June 2006), Annex 5: the risk weights in percent of each PD for an effective maturity
# of 2.5 years. The columns are the corporate exposures with an LGD of 45%, the SMEs with annual
# sales of 5 million euros and an LGD of 45%, the residential mortgages with LGDs of 45% and 25%,
# the other retail exposures with LGDs of 45% and 85%, and the qualifying revolving retail
# exposures with LGDs of 45% and 85%. The table leaves out the 1.06 scaling factor.
ANNEX_5 = [
    (0.03, 14.44, 11.30, 4.15, 2.30, 4.45, 8.41, 0.98, 1.85),
    (0.05, 19.65, 15.39, 6.23, 3.46, 6.63, 12.52, 1.51, 2.86),
    (0.10, 29.65, 23.30, 10.69, 5.94, 11.16, 21.08, 2.71, 5.12),
    (0.25, 49.47, 39.01, 21.30, 11.83, 21.15, 39.96, 5.76, 10.88),
    (0.40, 62.72, 49.49, 29.94, 16.64, 28.42, 53.69, 8.41, 15.88),
    (0.50, 69.61, 54.91, 35.08, 19.49, 32.36, 61.13, 10.04, 18.97),
    (0.75, 82.78, 65.14, 46.46, 25.81, 40.10, 75.74, 13.80, 26.06),
    (1.00, 92.32, 72.40, 56.40, 31.33, 45.77, 86.46, 17.22, 32.53),
    (1.30, 100.95, 78.77, 67.00, 37.22, 50.80, 95.95, 21.02, 39.70),
    (1.50, 105.59, 82.11, 73.45, 40.80, 53.37, 100.81, 23.40, 44.19),
    (2.00, 114.86, 88.55, 87.94, 48.85, 57.99, 109.53, 28.92, 54.63),
    (2.50, 122.16, 93.43, 100.64, 55.91, 60.90, 115.03, 33.98, 64.18),
    (3.00, 128.44, 97.58, 111.99, 62.22, 62.79, 118.61, 38.66, 73.03),
    (4.00, 139.58, 105.04, 131.63, 73.13, 65.01, 122.80, 47.16, 89.08),
    (5.00, 149.86, 112.27, 148.22, 82.35, 66.42, 125.45, 54.75, 103.41),
    (6.00, 159.61, 119.48, 162.52, 90.29, 67.73, 127.94, 61.61, 116.37),
    (10.00, 193.09, 146.51, 204.41, 113.56, 75.54, 142.69, 83.89, 158.47),
    (15.00, 221.54, 171.91, 235.72, 130.96, 88.60, 167.36, 103.89, 196.23),
    (20.00, 238.23, 188.42, 253.12, 140.62, 100.28, 189.41, 117.99, 222.86),
]
ANNEX_5_COLUMNS = [
    ({"asset_class": "corporate"}, 0.45),
    ({"asset_class": "corporate", "turnover": 5.0}, 0.45),
    ({"asset_class": "mortgage"}, 0.45),
    ({"asset_class": "mortgage"}, 0.25),
    ({"asset_class": "other_retail"}, 0.45),
    ({"asset_class": "other_retail"}, 0.85),
    ({"asset_class": "revolving"}, 0.45),
    ({"asset_class": "revolving"}, 0.85),
]


def risk_weight(probability, lgd, **options):
    """The risk weight 12.5 K in percent."""
    return 1250 * capital_requirement(probability, lgd, **options)


@pytest.mark.parametrize("column", range(8))
def test_risk_weights_reproduce_annex_5_of_basel_ii(column):
    table = np.array(ANNEX_5)
    options, lgd = ANNEX_5_COLUMNS[column]
    weights = risk_weight(table[:, 0] / 100, lgd, **options)
    # The 152 printed weights are rounded to 0.01 percentage points; all lie within 0.0067.
    assert np.abs(weights - table[:, column + 1]).max() < 0.01


@pytest.mark.parametrize(
    "probability,lgd,options,expected",
    [
        (0.01, 0.45, {"asset_class": "corporate"}, 92.3168),
        (0.01, 0.45, {"asset_class": "other_retail"}, 45.7727),
        (0.01, 0.25, {"asset_class": "mortgage"}, 31.3327),
        (0.01, 0.80, {"asset_class": "revolving"}, 30.6207),
        (0.01, 0.45, {"asset_class": "corporate", "turnover": 5.0}, 72.3947),
        (0.01, 0.45, {"asset_class": "corporate", "financial": True}, 117.9494),
    ],
)
def test_risk_weights_of_the_source_check(probability, lgd, options, expected):
    # docs/design/kivi-1.5-capital-survival.md, section 2.1, recomputed with SciPy.
    assert round(risk_weight(probability, lgd, **options), 4) == expected


def test_the_worked_example_by_hand():
    # Corporate PD 1%, LGD 45%, M 2.5: R = 0.192784, b = 0.137486 and K = 0.073853.
    w = (1 - math.exp(-0.5)) / (1 - math.exp(-50))
    r = 0.12 * w + 0.24 * (1 - w)
    assert asset_correlation(0.01, asset_class="corporate") == pytest.approx(r, rel=1e-15)
    assert round(r, 6) == 0.192784
    b = (0.11852 - 0.05478 * math.log(0.01)) ** 2
    stressed = _stats.normal_cdf(
        (_stats.normal_quantile(0.01) + math.sqrt(r) * _stats.normal_quantile(0.999))
        / math.sqrt(1 - r)
    )
    k = 0.45 * (float(stressed) - 0.01) / (1 - 1.5 * b)
    assert capital_requirement(0.01, 0.45, asset_class="corporate") == pytest.approx(k, rel=1e-15)
    assert round(k, 6) == 0.073853


# Asset correlations.


@pytest.mark.parametrize(
    "asset_class,low,high",
    [("corporate", 0.24, 0.12), ("hvcre", 0.30, 0.12), ("other_retail", 0.16, 0.03)],
)
def test_correlation_falls_from_its_upper_to_its_lower_bound(asset_class, low, high):
    correlation = asset_correlation([1e-12, 0.01, 0.05, 1.0], asset_class=asset_class)
    assert correlation[0] == pytest.approx(low, rel=1e-10)
    assert correlation[-1] == pytest.approx(high, rel=1e-12)
    assert (np.diff(correlation) < 0).all()


@pytest.mark.parametrize("asset_class,value", [("mortgage", 0.15), ("revolving", 0.04)])
def test_retail_correlations_are_constant(asset_class, value):
    correlation = asset_correlation([0.001, 0.2, np.nan], asset_class=asset_class)
    assert correlation[:2].tolist() == [value, value] and np.isnan(correlation[2])


def test_turnover_lowers_the_corporate_correlation_linearly():
    base = asset_correlation(0.01, asset_class="corporate")
    turnover = [0.0, 5.0, 27.5, 50.0, 80.0, np.inf]
    adjusted = asset_correlation(0.01, asset_class="corporate", turnover=turnover)
    np.testing.assert_allclose(
        base - adjusted, [0.04, 0.04, 0.02, 0.0, 0.0, 0.0], rtol=1e-12, atol=1e-16
    )
    # China's range of 3 to 30, in tens of millions of yuan.
    china = asset_correlation(
        0.01, asset_class="corporate", turnover=[3.0, 16.5, 30.0], turnover_range=(3.0, 30.0)
    )
    np.testing.assert_allclose(base - china, [0.04, 0.02, 0.0], rtol=1e-12, atol=1e-16)


def test_financial_institutions_multiply_the_correlation_by_one_and_a_quarter():
    probability = np.array([0.0005, 0.01, 0.2])
    base = asset_correlation(probability, asset_class="corporate")
    np.testing.assert_allclose(
        asset_correlation(probability, asset_class="corporate", financial=True), 1.25 * base
    )
    # A turnover at or above the upper bound has no adjustment, so it may come with financial.
    above = asset_correlation(probability, asset_class="corporate", turnover=50.0, financial=True)
    np.testing.assert_allclose(above, 1.25 * base, rtol=1e-15)


def test_correlation_keeps_the_shape_of_its_inputs():
    probability = pd.Series([0.01, np.nan, 0.05], index=["a", "b", "c"], name="pd")
    result = asset_correlation(probability, asset_class="corporate")
    assert result.name == "correlation" and result.index.tolist() == ["a", "b", "c"]
    assert np.isnan(result["b"])
    turnover = pd.Series([5.0, 20.0, np.inf], index=["a", "b", "c"])
    by_turnover = asset_correlation(0.01, asset_class="corporate", turnover=turnover)
    assert isinstance(by_turnover, pd.Series) and by_turnover.name == "correlation"
    assert isinstance(asset_correlation(0.01, asset_class="mortgage"), float)
    grid = asset_correlation([[0.01], [0.02]], asset_class="corporate", turnover=[5.0, 10.0, 50.0])
    assert grid.shape == (2, 3)


# The capital requirement.


def test_capital_requirement_is_the_loss_at_the_level_minus_the_expected_loss():
    probability = np.array([0.0005, 0.003, 0.02, 0.15])
    lgd = np.array([0.45, 0.25, 0.8, 0.6])
    for asset_class in ("mortgage", "revolving", "other_retail"):
        rho = asset_correlation(probability, asset_class=asset_class)
        stressed = conditional_pd(probability, correlation=rho, level=0.999)
        expected = lgd * (stressed - probability)
        actual = capital_requirement(probability, lgd, asset_class=asset_class)
        np.testing.assert_allclose(actual, expected, rtol=1e-14)


@pytest.mark.parametrize("asset_class", ["corporate", "hvcre"])
def test_maturity_adjustment(asset_class):
    probability = np.array([0.0003, 0.01, 0.1])
    b = (0.11852 - 0.05478 * np.log(probability)) ** 2
    at_one = capital_requirement(probability, 0.45, asset_class=asset_class, maturity=1.0)
    standard = capital_requirement(probability, 0.45, asset_class=asset_class)
    np.testing.assert_allclose(at_one / standard, 1 - 1.5 * b, rtol=1e-13)
    # Not clipped to [1, 5]: a day and ten years follow the same line.
    maturity = [1 / 365, 2.5, 10.0]
    actual = capital_requirement(0.01, 0.45, asset_class=asset_class, maturity=maturity)
    slope = (0.11852 - 0.05478 * math.log(0.01)) ** 2
    expected = standard[1] * (1 + (np.array(maturity) - 2.5) * slope)
    np.testing.assert_allclose(actual, expected, rtol=1e-13)


def test_risk_weight_rises_again_below_a_pd_of_about_one_in_a_hundred_thousand():
    probability = np.geomspace(2.93e-6, 1e-3, 400)
    weights = risk_weight(probability, 0.45, asset_class="corporate")
    lowest = probability[np.argmin(weights)]
    assert 5e-6 < lowest < 1.5e-5
    # docs/design/kivi-1.5-capital-survival.md, section 2.2: 46% at 3e-6 and 145% at 2.95e-6.
    assert round(risk_weight(3e-6, 0.45, asset_class="corporate")) == 46
    assert round(risk_weight(2.95e-6, 0.45, asset_class="corporate")) == 145


@pytest.mark.parametrize("asset_class", ["corporate", "hvcre"])
def test_pds_where_the_maturity_adjustment_breaks_down_raise(asset_class):
    message = (
        rf"probability must be above 2.927e-06 for asset_class '{asset_class}'.*got 2.9e-06; "
        r"apply the PD floor first"
    )
    with pytest.raises(ValueError, match=message):
        capital_requirement([0.01, 2.9e-6], 0.45, asset_class=asset_class)
    assert capital_requirement(2.93e-6, 0.45, asset_class=asset_class) > 0
    # The bound is where 1 - 1.5 b reaches 0.
    bound = math.exp((0.11852 - math.sqrt(2 / 3)) / 0.05478)
    assert f"{bound:.4g}" == "2.927e-06"


def test_retail_classes_take_any_pd_without_the_maturity_adjustment():
    assert 0 < capital_requirement(1e-9, 0.45, asset_class="mortgage") < 1e-6


def test_correlation_and_level_replace_those_of_the_formula():
    probability, lgd = np.array([0.002, 0.03]), 0.4
    economic = capital_requirement(
        probability, lgd, asset_class="corporate", correlation=0.2, level=0.9997
    )
    stressed = conditional_pd(probability, correlation=0.2, level=0.9997)
    b = (0.11852 - 0.05478 * np.log(probability)) ** 2
    np.testing.assert_allclose(economic, lgd * (stressed - probability) / (1 - 1.5 * b), rtol=1e-13)
    retail = capital_requirement(probability, lgd, asset_class="mortgage", correlation=0.2)
    np.testing.assert_allclose(
        retail,
        lgd * (conditional_pd(probability, correlation=0.2, level=0.999) - probability),
        rtol=1e-13,
    )


def test_scaling_multiplies_the_requirement():
    base = capital_requirement([0.01, 0.05], 0.45, asset_class="corporate")
    scaled = capital_requirement([0.01, 0.05], 0.45, asset_class="corporate", scaling=1.06)
    np.testing.assert_allclose(scaled, 1.06 * base, rtol=1e-15)


def test_missing_pds_and_lgds_give_missing_requirements():
    result = capital_requirement(
        [0.01, np.nan, 0.02], [0.45, 0.45, np.nan], asset_class="corporate"
    )
    assert result[0] > 0 and np.isnan(result[1:]).all()
    assert np.isnan(capital_requirement(np.nan, 0.45, asset_class="other_retail"))
    assert np.isnan(capital_requirement(0.01, [np.nan], asset_class="mortgage")).all()


def test_capital_requirement_keeps_the_index_of_any_series():
    index = pd.Index([10, 20, 30], name="loan")
    lgd = pd.Series([0.45, 0.25, 0.6], index=index, name="lgd")
    by_lgd = capital_requirement([0.01, 0.02, 0.03], lgd, asset_class="corporate")
    assert by_lgd.name == "capital_requirement" and by_lgd.index.equals(index)
    maturity = pd.Series([1.0, 2.5, 5.0], index=index)
    by_maturity = capital_requirement(0.01, 0.45, asset_class="corporate", maturity=maturity)
    assert isinstance(by_maturity, pd.Series) and by_maturity.index.equals(index)
    expected = capital_requirement(0.01, 0.45, asset_class="corporate", maturity=[1.0, 2.5, 5.0])
    np.testing.assert_array_equal(by_maturity.to_numpy(), expected)
    with pytest.raises(ValueError, match="probability and lgd must have identical indices"):
        capital_requirement(pd.Series([0.01, 0.02, 0.03]), lgd, asset_class="corporate")
    with pytest.raises(ValueError, match="must broadcast to the length 3 of their Series index"):
        capital_requirement(0.01, lgd, asset_class="corporate", maturity=[[1.0], [2.0]])


@pytest.mark.parametrize(
    "probability,match",
    [
        (1.0, r"probability must be below 1, got 1.0: .* max\(0, lgd - elbe\)"),
        (0.0, "probability must be strictly between 0 and 1, got 0.0"),
        (1.5, "probability must be strictly between 0 and 1, got 1.5"),
        (True, r"probability must be below 1"),
    ],
)
def test_capital_requirement_rejects_pds_outside_the_open_interval(probability, match):
    with pytest.raises(ValueError, match=match):
        capital_requirement([0.01, probability], 0.45, asset_class="mortgage")


@pytest.mark.parametrize(
    "options,match",
    [
        ({"asset_class": "bank"}, "asset_class must be one of 'corporate', 'hvcre', .* got 'bank'"),
        ({"asset_class": None}, "asset_class must be one of .* got None"),
        ({"asset_class": "mortgage", "maturity": 1.0}, "maturity applies only to the asset"),
        ({"asset_class": "corporate", "maturity": 0.0}, "maturity must be positive and finite"),
        ({"asset_class": "corporate", "maturity": np.inf}, "maturity must be positive and finite"),
        ({"asset_class": "hvcre", "maturity": [1.0, np.nan]}, "maturity must not have missing"),
        ({"asset_class": "corporate", "maturity": True}, "maturity must be real and numeric"),
        ({"asset_class": "hvcre", "turnover": 5.0}, "turnover applies only to asset_class 'corp"),
        ({"asset_class": "revolving", "financial": True}, "financial applies only to asset_class"),
        ({"asset_class": "corporate", "financial": 1}, "financial must be True or False, got 1"),
        (
            {"asset_class": "corporate", "turnover": [10.0, np.inf], "financial": True},
            r"financial=True must not come with a turnover below the upper bound 50.0 .* got 10.0",
        ),
        ({"asset_class": "corporate", "turnover": np.nan}, "turnover must not have missing values"),
        ({"asset_class": "corporate", "turnover": -1.0}, "turnover must not be negative, got -1.0"),
        ({"asset_class": "corporate", "turnover": True}, "turnover must be real and numeric"),
        (
            {"asset_class": "corporate", "turnover": 5.0, "turnover_range": (5.0, 5.0)},
            r"turnover_range must be two finite numbers .* got \(5.0, 5.0\)",
        ),
        ({"asset_class": "corporate", "turnover_range": (-1.0, 5.0)}, "turnover_range must be"),
        ({"asset_class": "corporate", "turnover_range": (5.0, np.inf)}, "turnover_range must be"),
        ({"asset_class": "corporate", "turnover_range": (True, 50)}, "turnover_range must be"),
        ({"asset_class": "corporate", "turnover_range": "ab"}, "turnover_range must be"),
        ({"asset_class": "corporate", "turnover_range": (5.0,)}, "turnover_range must be"),
        (
            {"asset_class": "corporate", "correlation": 0.2, "turnover": 5.0},
            "turnover and financial adjust the correlation",
        ),
        (
            {"asset_class": "corporate", "correlation": 0.2, "financial": True},
            "turnover and financial adjust the correlation",
        ),
        ({"asset_class": "corporate", "correlation": 1.0}, "correlation must be from 0 to below 1"),
        ({"asset_class": "corporate", "correlation": -0.1}, "correlation must be from 0 to below"),
        ({"asset_class": "corporate", "correlation": np.nan}, "correlation must not have missing"),
        ({"asset_class": "corporate", "level": 1.0}, "level must be a number strictly between"),
        ({"asset_class": "corporate", "level": True}, "level must be a number strictly between"),
        ({"asset_class": "corporate", "level": "0.9"}, "level must be a number strictly between"),
        ({"asset_class": "corporate", "scaling": 0.0}, "scaling must be positive and finite"),
        ({"asset_class": "corporate", "scaling": True}, "scaling must be positive and finite"),
        ({"asset_class": "corporate", "scaling": np.inf}, "scaling must be positive and finite"),
    ],
)
def test_capital_requirement_rejects_invalid_options(options, match):
    with pytest.raises(ValueError, match=match):
        capital_requirement([0.01, 0.02], 0.45, **options)


@pytest.mark.parametrize(
    "lgd,match",
    [
        (1.2, "lgd must be between 0 and 1, got 1.2"),
        (-0.1, "lgd must be between 0 and 1, got -0.1"),
        (True, "lgd must be real and numeric"),
        ("0.45", "lgd must be real and numeric"),
        ([0.45, 0.25, 0.6], r"probability, lgd cannot be broadcast together: probability \(2,\)"),
    ],
)
def test_capital_requirement_rejects_invalid_lgds(lgd, match):
    with pytest.raises(ValueError, match=match):
        capital_requirement([0.01, 0.02], lgd, asset_class="corporate")


def test_financial_institution_with_an_sme_turnover_raises_also_in_asset_correlation():
    with pytest.raises(ValueError, match="financial=True must not come with a turnover"):
        asset_correlation(0.01, asset_class="corporate", turnover=20.0, financial=True)


def test_asset_correlation_takes_the_closed_unit_interval():
    assert asset_correlation([0.0, 1.0], asset_class="corporate").tolist() == pytest.approx(
        [0.24, 0.12], rel=1e-15
    )
    with pytest.raises(ValueError, match="probability must be between 0 and 1, got 1.5"):
        asset_correlation([0.5, 1.5], asset_class="corporate")


# The one-factor model.


def test_conditional_pd_at_a_level_is_the_pd_at_the_factor_of_that_quantile():
    probability = np.array([0.001, 0.02, 0.3])
    level = np.array([0.5, 0.9, 0.999])
    by_level = conditional_pd(probability[:, None], correlation=0.15, level=level)
    by_factor = conditional_pd(
        probability[:, None], correlation=0.15, factor=-_stats.normal_quantile(level)
    )
    np.testing.assert_array_equal(by_level, by_factor)
    # At the median of the factor the PD is below the long-run PD, which averages the tails.
    assert (by_level[:, 0] < probability).all() and (np.diff(by_level, axis=1) > 0).all()


def test_conditional_pd_averages_to_the_long_run_pd():
    nodes, weights = np.polynomial.hermite_e.hermegauss(120)
    for probability, rho in [(0.001, 0.05), (0.02, 0.12), (0.2, 0.3)]:
        mean = conditional_pd(probability, correlation=rho, factor=nodes) @ weights
        assert mean / math.sqrt(2 * math.pi) == pytest.approx(probability, rel=1e-11)


def test_conditional_pd_without_correlation_is_the_long_run_pd():
    np.testing.assert_allclose(
        conditional_pd([0.001, 0.4], correlation=0.0, factor=[3.0, -3.0]), [0.001, 0.4], rtol=1e-14
    )


def test_conditional_pd_keeps_series_and_missing_values():
    factor = pd.Series([1.0, np.nan, -1.0], index=[2021, 2022, 2023], name="z")
    result = conditional_pd(0.02, correlation=0.12, factor=factor)
    assert result.name == "conditional_pd" and result.index.tolist() == [2021, 2022, 2023]
    assert np.isnan(result[2022]) and result[2021] < 0.02 < result[2023]
    assert isinstance(conditional_pd(0.02, correlation=0.12, level=0.99), float)
    assert np.isnan(conditional_pd(np.nan, correlation=0.12, level=0.99))


@pytest.mark.parametrize(
    "options,match",
    [
        ({"correlation": 0.1}, "give exactly one of factor and level"),
        ({"correlation": 0.1, "factor": 0.0, "level": 0.9}, "give exactly one of factor and"),
        ({"correlation": 0.1, "factor": np.inf}, "factor must be finite, got inf"),
        ({"correlation": 0.1, "level": 1.0}, "level must be strictly between 0 and 1, got 1.0"),
        ({"correlation": 1.0, "level": 0.9}, "correlation must be from 0 to below 1, got 1.0"),
        ({"correlation": np.nan, "level": 0.9}, "correlation must not have missing values"),
        ({"correlation": 0.1, "factor": True}, "factor must be real and numeric"),
    ],
)
def test_conditional_pd_rejects_invalid_arguments(options, match):
    with pytest.raises(ValueError, match=match):
        conditional_pd(0.01, **options)


def test_conditional_pd_rejects_pds_outside_the_open_interval():
    with pytest.raises(ValueError, match="probability must be strictly between 0 and 1, got 0.0"):
        conditional_pd([0.0, 0.5], correlation=0.1, factor=0.0)


def test_systematic_factor_inverts_conditional_pd():
    rng = np.random.default_rng(0)
    probability = rng.uniform(0.001, 0.3, 200)
    rho = rng.uniform(0.01, 0.5, 200)
    factor = rng.normal(scale=2, size=200)
    rate = conditional_pd(probability, correlation=rho, factor=factor)
    np.testing.assert_allclose(
        systematic_factor(rate, probability, correlation=rho), factor, rtol=1e-9, atol=1e-11
    )
    back = conditional_pd(
        probability, correlation=rho, factor=systematic_factor(rate, probability, correlation=rho)
    )
    np.testing.assert_allclose(back, rate, rtol=1e-12)


def test_systematic_factor_by_hand():
    # (Phi^-1(0.01) - sqrt(0.88) Phi^-1(0.03)) / sqrt(0.12).
    expected = (-2.3263478740408408 - math.sqrt(0.88) * -1.880793608151251) / math.sqrt(0.12)
    assert systematic_factor(0.03, 0.01, correlation=0.12) == pytest.approx(expected, rel=1e-14)
    rates = pd.Series([0.004, np.nan], index=["2020", "2021"])
    result = systematic_factor(rates, 0.01, correlation=0.12)
    assert result.name == "factor" and np.isnan(result["2021"]) and result["2020"] > 0


@pytest.mark.parametrize(
    "rate,correlation,match",
    [
        (0.0, 0.1, "default_rate must be strictly between 0 and 1, got 0.0"),
        (1.0, 0.1, "default_rate must be strictly between 0 and 1, got 1.0"),
        (0.02, 0.0, "correlation must be strictly between 0 and 1, got 0.0"),
        (0.02, 1.0, "correlation must be strictly between 0 and 1, got 1.0"),
        (True, 0.1, "default_rate must be real and numeric"),
    ],
)
def test_systematic_factor_rejects_invalid_arguments(rate, correlation, match):
    with pytest.raises(ValueError, match=match):
        systematic_factor([0.01, rate], 0.01, correlation=correlation)


def test_vasicek_fit_recovers_the_parameters_of_a_long_series():
    rng = np.random.default_rng(1)
    factor = rng.standard_normal(20000)
    rates = conditional_pd(0.02, correlation=0.1, factor=factor)
    probability, correlation = vasicek_fit(rates)
    assert probability == pytest.approx(0.02, rel=0.03)
    assert correlation == pytest.approx(0.1, rel=0.05)
    assert isinstance(probability, float) and isinstance(correlation, float)


def test_vasicek_fit_of_a_constant_series_has_no_correlation():
    assert vasicek_fit(pd.Series([0.02, 0.02, 0.02])) == (0.02, 0.0)


@pytest.mark.parametrize(
    "rates,match",
    [
        ([0.01], "default_rate must have at least two values, got 1"),
        ([[0.01, 0.02]], "default_rate must be one-dimensional, got 2 dimensions"),
        ([0.01, np.nan, 0.02], "default_rate must not have missing values, got 1"),
        ([0.01, 1.2], "default_rate must be between 0 and 1, got 1.2"),
        ([0.0, 0.0, 0.0], "default_rate must have a mean strictly between 0 and 1, got 0.0"),
        ([1.0, 1.0], "default_rate must have a mean strictly between 0 and 1, got 1.0"),
        ([0.0, 1.0, 0.0, 1.0], r"default_rate must have a variance below mean \* \(1 - mean\)"),
        ([True, False], "default_rate must be real and numeric"),
    ],
)
def test_vasicek_fit_rejects_invalid_series(rates, match):
    with pytest.raises(ValueError, match=match):
        vasicek_fit(rates)


# Most prudent estimation.

# Pluto and Tasche (arXiv:cond-mat/0411699v2), tables 1-9, in percent: grades A, B and C with 100,
# 400 and 300 obligors, at the levels 50%, 75%, 90%, 95%, 99% and 99.9%. Tables 1-3 have no
# defaults, tables 4-6 none in A, two in B and one in C, both with independent defaults, and
# tables 7-9 no defaults with a correlation of 0.12.
PLUTO_TASCHE_LEVELS = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
PLUTO_TASCHE = [
    ([0, 0, 0], 0.0, [[0.09, 0.17, 0.29, 0.37, 0.57, 0.86], [0.10, 0.20, 0.33, 0.43, 0.66, 0.98],
                     [0.23, 0.46, 0.76, 0.99, 1.52, 2.28]]),
    ([0, 2, 1], 0.0, [[0.46, 0.65, 0.83, 0.97, 1.25, 1.62], [0.52, 0.73, 0.95, 1.10, 1.43, 1.85],
                     [0.56, 0.90, 1.29, 1.57, 2.19, 3.04]]),
    ([0, 0, 0], 0.12, [[0.15, 0.40, 0.86, 1.31, 2.65, 5.29], [0.17, 0.45, 0.96, 1.45, 2.92, 5.77],
                      [0.37, 0.92, 1.89, 2.78, 5.30, 9.84]]),
]  # fmt: skip
# The printed cells that kivi, SciPy's beta quantile and adaptive quadrature all round otherwise:
# exactly 0.6378%, 2.6563%, 5.7650% and 0.9252%.
PLUTO_TASCHE_MISPRINTS = {(1, 0, 1): 0.64, (2, 0, 4): 2.66, (2, 1, 5): 5.76, (2, 2, 1): 0.93}


@pytest.mark.parametrize("table", range(3))
def test_most_prudent_pd_reproduces_the_tables_of_pluto_and_tasche(table):
    event, correlation, printed = PLUTO_TASCHE[table]
    bounds = np.array(
        [
            most_prudent_pd(event, [100, 400, 300], level=level, correlation=correlation)
            for level in PLUTO_TASCHE_LEVELS
        ]
    ).T
    expected = np.array(printed)
    for (number, grade, level), value in PLUTO_TASCHE_MISPRINTS.items():
        if number == table:
            assert abs(expected[grade, level] - value) == pytest.approx(0.01)
            expected[grade, level] = value
    np.testing.assert_array_equal(np.round(100 * bounds, 2), expected)


def test_most_prudent_pd_without_defaults_is_the_closed_form():
    # 1 - (1 - q)**(1 / n) for the pooled obligors 800, 700 and 300: 0.2874%, 0.3284%, 0.7646%.
    bounds = most_prudent_pd([0, 0, 0], [100, 400, 300], level=0.9)
    expected = -np.expm1(np.log(0.1) / np.array([800.0, 700.0, 300.0]))
    np.testing.assert_allclose(bounds, expected, rtol=1e-15)
    assert np.round(100 * bounds, 4).tolist() == [0.2874, 0.3284, 0.7646]


def test_correlation_and_level_raise_the_bounds():
    event, total = [0, 1, 2], [5000, 3000, 1000]
    independent = most_prudent_pd(event, total, level=0.9)
    assert (most_prudent_pd(event, total, level=0.9, correlation=0.1) > independent).all()
    assert (most_prudent_pd(event, total, level=0.99) > independent).all()


def test_a_grade_whose_pooled_obligors_all_defaulted_or_that_pools_none_gets_one():
    assert most_prudent_pd([0, 3], [10, 3]).tolist()[1] == 1.0
    assert most_prudent_pd([0, 0], [10, 0], correlation=0.2).tolist()[1] == 1.0
    assert most_prudent_pd([0, 0], [0, 0]).tolist() == [1.0, 1.0]


def test_most_prudent_pd_keeps_the_index_of_a_series():
    total = pd.Series([200, 300, 100], index=["A", "B", "C"], name="obligors")
    result = most_prudent_pd([0, 1, 0], total, level=0.9)
    assert result.name == "most_prudent_pd" and result.index.tolist() == ["A", "B", "C"]
    np.testing.assert_array_equal(result, most_prudent_pd([0, 1, 0], [200, 300, 100], level=0.9))


@pytest.mark.parametrize(
    "event,total,options,match",
    [
        ([0, 1], [10, 20, 30], {}, "event and total must have the same length, got 2 and 3"),
        ([0, 11], [10, 10], {}, "event must not exceed total, got 11 events among 10"),
        ([0, -1], [10, 10], {}, "event must be a non-negative whole number, got -1.0"),
        ([0, 1.5], [10, 10], {}, "event must be a non-negative whole number, got 1.5"),
        ([0, 1], [10, np.nan], {}, "total must not have missing values, got 1"),
        ([0, 1], [10, np.inf], {}, "total must be a non-negative whole number, got inf"),
        ([], [], {}, "event must not be empty"),
        ([[0, 1]], [[10, 10]], {}, "event must be one-dimensional, got 2 dimensions"),
        ([0, 1], [10, 10], {"level": 1.0}, "level must be a number strictly between 0 and 1"),
        ([0, 1], [10, 10], {"correlation": 1.0}, "correlation must be a number from 0 to below"),
        ([0, 1], [10, 10], {"correlation": True}, "correlation must be a number from 0 to below"),
        ([True, False], [10, 10], {}, "event must be real and numeric"),
    ],
)
def test_most_prudent_pd_rejects_invalid_counts(event, total, options, match):
    with pytest.raises(ValueError, match=match):
        most_prudent_pd(event, total, **options)


def test_most_prudent_pd_rejects_series_with_different_indices():
    with pytest.raises(ValueError, match="event and total must have identical indices"):
        most_prudent_pd(pd.Series([0, 1], index=[1, 2]), pd.Series([10, 10], index=[2, 3]))
