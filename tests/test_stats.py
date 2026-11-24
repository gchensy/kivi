"""The normal quantile of arrays, the beta quantile and the bivariate normal distribution."""

import math

import numpy as np
import pandas as pd
import pytest

from kivi import _stats

# The normal quantile of arrays.


def test_normal_quantile_of_an_array_is_the_scalar_call_of_each_element():
    probability = np.array([[1e-300, 0.02425, 0.3, 0.3], [0.5, 0.500000001, 0.975, 1 - 1e-12]])
    result = _stats.normal_quantile(probability)
    assert isinstance(result, np.ndarray) and result.shape == (2, 4) and result.dtype == float
    expected = [[_stats.normal_quantile(p) for p in row] for row in probability.tolist()]
    assert result.tolist() == expected


@pytest.mark.parametrize(
    "probability,expected",
    [(0.975, 1.959963984540054), (1e-10, -6.361340902404056), (0.5, 0.0)],
)
def test_normal_quantile_of_a_scalar_is_still_a_float(probability, expected):
    # Reference values from scipy.stats.norm.ppf, as in tests/test_metrics.py.
    result = _stats.normal_quantile(probability)
    assert type(result) is float
    assert result == pytest.approx(expected, rel=1e-14, abs=0)


def test_normal_quantile_of_an_array_gives_nan_for_missing_values():
    result = _stats.normal_quantile(pd.Series([0.5, np.nan, 0.975]))
    assert isinstance(result, np.ndarray)
    assert result[0] == 0.0 and np.isnan(result[1])
    assert result[2] == _stats.normal_quantile(0.975)
    assert _stats.normal_quantile([]).shape == (0,)


@pytest.mark.parametrize(
    "probability,shown",
    [([0.2, 1.0], "1.0"), ([0.0], "0.0"), ([np.nan, -0.5], "-0.5"), ([[0.5], [2.0]], "2.0")],
)
def test_normal_quantile_of_an_array_rejects_probabilities_outside_the_open_interval(
    probability, shown
):
    with pytest.raises(ValueError, match=rf"probability must be between 0 and 1, got {shown}$"):
        _stats.normal_quantile(probability)


def test_normal_quantile_of_a_missing_scalar_still_raises():
    with pytest.raises(ValueError, match="probability must be between 0 and 1, got nan"):
        _stats.normal_quantile(np.nan)


def test_normal_quantile_of_an_array_matches_scipy_in_the_tails():
    special = pytest.importorskip("scipy.special")
    probability = np.r_[
        np.logspace(-300, -1, 300),
        np.linspace(0.01, 0.99, 197),
        1 - np.logspace(-2, -16, 60),
    ]
    actual = _stats.normal_quantile(probability)
    np.testing.assert_allclose(actual, special.ndtri(probability), rtol=1e-14, atol=0)


# The beta quantile.


def test_beta_quantile_of_zero_defaults_is_the_closed_form():
    # Pluto and Tasche (2005): no defaults among 800, 700 and 300 obligors at a level of 90%
    # bound the PD by 1 - 0.1**(1 / n), that is 0.2874%, 0.3284% and 0.7646%.
    total = np.array([800.0, 700.0, 300.0])
    bound = _stats.beta_quantile(1.0, total, 0.9)
    np.testing.assert_allclose(bound, -np.expm1(np.log(0.1) / total), rtol=1e-15, atol=0)
    assert np.round(100 * bound, 4).tolist() == [0.2874, 0.3284, 0.7646]


def test_beta_quantile_with_one_as_second_shape_is_a_power():
    probability = np.array([1e-300, 0.25, 0.5, 1 - 1e-12])
    for a in (0.3, 2.0, 50.0):
        expected = np.exp(np.log(probability) / a)
        np.testing.assert_allclose(_stats.beta_quantile(a, 1.0, probability), expected, rtol=1e-15)


@pytest.mark.parametrize("a", [0.3, 0.9, 1.5, 4.0, 30.0, 250.0, 1000.0])
@pytest.mark.parametrize("b", [0.3, 0.9, 2.5, 7.0, 40.0, 400.0])
def test_beta_quantile_inverts_the_incomplete_beta_function(a, b):
    probability = np.array([1e-200, 1e-30, 1e-6, 0.01, 0.2, 0.5, 0.8, 0.99, 1 - 1e-6, 1 - 1e-13])
    x = _stats.beta_quantile(a, b, probability)
    assert ((x > 0) & (x < 1)).all() and (np.diff(x) >= 0).all()
    # Leave out the quantiles below the smallest normal number or above the largest number below
    # 1, which double precision cannot hold.
    held = (x > np.finfo(float).tiny) & (x < np.nextafter(1.0, 0.0))
    a, b, probability, x = (np.broadcast_to(v, x.shape)[held] for v in (a, b, probability, x))
    # The residual in the tail with the smaller probability, over the density, is the distance
    # to the root. Rounding 1 - x moves a small x by up to one unit of 1 in the upper tail.
    upper = probability > 0.5
    tail = np.where(upper, _stats.betainc(b, a, 1 - x), _stats.betainc(a, b, x))
    residual = np.where(upper, (1 - probability) - tail, tail - probability)
    density = np.exp(_stats._log_front(a, b, x) - np.log(x) - np.log1p(-x))
    distance = np.abs(residual) / density
    assert (distance <= 1e-12 * np.minimum(x, 1 - x) + np.spacing(1.0)).all()


def test_beta_quantile_keeps_the_end_points_missing_values_and_the_broadcast_shape():
    result = _stats.beta_quantile(2.0, [3.0, 0.5], [[0.0], [1.0], [np.nan]])
    assert result.shape == (3, 2)
    assert result[0].tolist() == [0.0, 0.0] and result[1].tolist() == [1.0, 1.0]
    assert np.isnan(result[2]).all()


def test_beta_quantile_stops_at_the_smallest_normal_number_and_below_one():
    # The quantiles are about 1e-1000 and 1 - 1e-150, which double precision cannot hold.
    assert _stats.beta_quantile(0.1, 5.0, 1e-100) == np.finfo(float).tiny
    assert _stats.beta_quantile(1000.0, 0.1, 1 - 1e-15) == np.nextafter(1.0, 0.0)


@pytest.mark.parametrize("a,b", [(0.0, 1.0), (2.0, -1.0), (np.inf, 2.0), (2.0, np.nan)])
def test_beta_quantile_rejects_shapes_that_are_not_finite_and_positive(a, b):
    with pytest.raises(ValueError, match="shape parameters must be finite and positive"):
        _stats.beta_quantile(a, b, 0.5)


@pytest.mark.parametrize("probability", [-0.1, 1.5])
def test_beta_quantile_rejects_probabilities_outside_the_unit_interval(probability):
    message = rf"probability must be between 0 and 1, got {probability}"
    with pytest.raises(ValueError, match=message):
        _stats.beta_quantile(2.0, 3.0, [0.5, probability])


def test_beta_quantile_reports_a_search_that_does_not_converge(monkeypatch):
    monkeypatch.setattr(_stats, "_BETA_ITERATIONS", 1)
    with pytest.raises(ArithmeticError, match="the beta quantile did not converge"):
        _stats.beta_quantile(2.0, 3.0, 0.4)


BETA_SHAPES = [0.1, 0.5, 0.9, 1.5, 2.0, 5.0, 30.0, 200.0, 1e3, 1e4, 1e5, 1e6]
BETA_PROBABILITIES = [1e-300, 1e-100, 1e-30, 1e-12, 1e-6, 1e-3, 0.01, 0.05, 0.1, 0.3, 0.5]
BETA_PROBABILITIES += [0.7, 0.9, 0.95, 0.99, 0.999, 1 - 1e-6, 1 - 1e-12, 1 - 1e-15]


def test_beta_quantile_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    # SciPy warns that its search does not converge for a = 0.5, b = 3 at 1e-10.
    a, b = (grid.ravel() for grid in np.meshgrid([0.6, 1.5, 4.0, 25.0, 300.0], [0.7, 3.0, 60.0]))
    probability = np.array([1e-10, 1e-4, 0.05, 0.5, 0.95, 1 - 1e-4, 1 - 1e-10])[:, None]
    actual = _stats.beta_quantile(a, b, probability)
    np.testing.assert_allclose(actual, stats.beta.ppf(probability, a, b), rtol=1e-12, atol=0)


def test_beta_quantile_solves_scipys_incomplete_beta_function_at_extreme_arguments():
    # SciPy's quantile misses the root at some of these arguments, such as a = 0.5, b = 2 at
    # 1e-12, so the residual of SciPy's incomplete beta function at kivi's quantile, divided by
    # the density, measures the error of the quantile.
    special = pytest.importorskip("scipy.special")
    stats = pytest.importorskip("scipy.stats")
    a, b, probability = (
        grid.ravel() for grid in np.meshgrid(BETA_SHAPES, BETA_SHAPES, BETA_PROBABILITIES)
    )
    x = _stats.beta_quantile(a, b, probability)
    upper = probability > 0.5
    tail = np.where(upper, special.betaincc(a, b, x), special.betainc(a, b, x))
    residual = np.where(upper, (1 - probability) - tail, tail - probability)
    # Leave out quantiles that double precision cannot hold, and tails where SciPy underflows.
    shown = (x > 1e-300) & (1 - x > 1e-15) & (tail > 1e-280)
    assert shown.mean() > 0.85
    error = np.abs(residual[shown]) / stats.beta.pdf(x[shown], a[shown], b[shown]) / x[shown]
    # The incomplete beta function loses digits as the shapes grow, and the quantile with it.
    moderate = np.maximum(a, b)[shown] <= 1e4
    assert error[moderate].max() < 1e-12
    assert error.max() < 1e-10


# The bivariate normal distribution function.


@pytest.mark.parametrize("rho", [-1.0, -0.99, -0.925, -0.6, 0.0, 0.3, 0.924, 0.925, 0.9999, 1.0])
def test_bivariate_normal_cdf_at_the_origin_is_the_closed_form(rho):
    expected = 0.25 + math.asin(rho) / (2 * math.pi)
    assert _stats.bivariate_normal_cdf(0.0, 0.0, rho) == pytest.approx(expected, rel=1e-15, abs=0)


def test_bivariate_normal_cdf_without_correlation_is_the_product_of_the_margins():
    x, y = np.array([-7.5, -1.0, 0.3, 2.0]), np.array([0.5, -2.5, 0.3, 9.0])
    expected = _stats.normal_cdf(x) * _stats.normal_cdf(y)
    assert _stats.bivariate_normal_cdf(x, y, 0.0).tolist() == expected.tolist()


def test_bivariate_normal_cdf_with_perfect_correlation():
    x, y = np.array([-3.0, 0.4, 1.2, -0.5]), np.array([-1.0, 0.4, -2.0, 0.6])
    assert (
        _stats.bivariate_normal_cdf(x, y, 1.0).tolist()
        == _stats.normal_cdf(np.minimum(x, y)).tolist()
    )
    reversed_ = np.maximum(_stats.normal_cdf(x) + _stats.normal_cdf(y) - 1, 0)
    np.testing.assert_allclose(_stats.bivariate_normal_cdf(x, y, -1.0), reversed_, atol=2e-16)


def test_bivariate_normal_cdf_with_infinite_and_missing_limits():
    x = [-np.inf, 1.0, np.inf, 0.5, np.inf, np.nan, 0.5]
    y = [0.3, -np.inf, -1.0, np.inf, np.inf, 0.0, 0.5]
    rho = [0.5, 0.5, 0.95, -0.3, 0.2, 0.1, np.nan]
    result = _stats.bivariate_normal_cdf(x, y, rho)
    assert result[:2].tolist() == [0.0, 0.0]
    assert result[2] == _stats.normal_cdf(-1.0) and result[3] == _stats.normal_cdf(0.5)
    assert result[4] == 1.0
    assert np.isnan(result[5:]).all()


def test_bivariate_normal_cdf_is_symmetric_and_adds_up_to_a_margin():
    rng = np.random.default_rng(0)
    x, y = rng.normal(scale=2.5, size=(2, 300))
    rho = rng.uniform(-1, 1, size=300)
    np.testing.assert_array_equal(
        _stats.bivariate_normal_cdf(x, y, rho), _stats.bivariate_normal_cdf(y, x, rho)
    )
    # P(X <= x, Y <= y) + P(X <= x, Y > y) = P(X <= x), and -Y has the correlation -rho.
    total = _stats.bivariate_normal_cdf(x, y, rho) + _stats.bivariate_normal_cdf(x, -y, -rho)
    np.testing.assert_allclose(total, _stats.normal_cdf(x), rtol=0, atol=4e-16)


def test_bivariate_normal_cdf_is_continuous_where_the_method_switches():
    x, y = np.meshgrid([-6.0, -2.0, -0.4, 0.0, 1.5], [-5.0, -1.0, 0.2, 3.0])
    for switch in (0.925, -0.925):
        below = _stats.bivariate_normal_cdf(x, y, np.nextafter(switch, 0.0))
        np.testing.assert_allclose(below, _stats.bivariate_normal_cdf(x, y, switch), atol=2e-16)


def test_bivariate_normal_cdf_rejects_correlations_outside_the_unit_interval():
    with pytest.raises(ValueError, match="the correlation must lie between -1 and 1"):
        _stats.bivariate_normal_cdf([0.0, 0.0], [0.0, 0.0], [0.5, 1.5])


def test_bivariate_normal_cdf_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    count = 300
    x = np.r_[rng.uniform(-9, 4, count), [-8.0, -37.0, 6.0, 0.0, 30.0]]
    y = np.r_[rng.uniform(-9, 4, count), [-8.0, 2.0, 6.0, -40.0, 30.0]]
    near = rng.uniform(0.9, 1 - 1e-6, count // 3) * rng.choice([-1, 1], count // 3)
    rho = np.r_[rng.uniform(-1, 1, count - len(near)), near, [0.999999, -0.5, -0.999, 0.97, 0.3]]
    expected = [
        stats.multivariate_normal.cdf([a, b], cov=[[1, r], [r, 1]])
        for a, b, r in zip(x, y, rho, strict=True)
    ]
    actual = _stats.bivariate_normal_cdf(x, y, rho)
    # Both are accurate to about 1e-16 absolute, SciPy's also in its far tails.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=4e-16)


def test_bivariate_normal_cdf_keeps_its_relative_accuracy_for_positive_correlations():
    # Plackett's identity from rho = 0, integrated by adaptive quadrature: with rho >= 0 the
    # integrand is positive, so the reference keeps its relative accuracy in the tails. The
    # diagonal x = y is the case of kivi.capital.vasicek_fit.
    integrate = pytest.importorskip("scipy.integrate")
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(2)
    x = np.r_[rng.uniform(-8, 3, 150), np.linspace(-6, 2, 30)]
    y = np.r_[rng.uniform(-8, 3, 150), np.linspace(-6, 2, 30)]
    rho = np.r_[rng.uniform(0, 1 - 1e-6, 150), rng.uniform(0, 0.999, 30)]

    def reference(a, b, r):
        def density(theta):
            exponent = -(a * a + b * b - 2 * a * b * math.sin(theta)) / (2 * math.cos(theta) ** 2)
            return math.exp(exponent) / (2 * math.pi)

        integral, _ = integrate.quad(density, 0, math.asin(r), epsabs=0, epsrel=5e-14, limit=200)
        return stats.norm.cdf(a) * stats.norm.cdf(b) + integral

    expected = np.array([reference(a, b, r) for a, b, r in zip(x, y, rho, strict=True)])
    shown = expected > 1e-10
    assert shown.mean() > 0.6
    actual = _stats.bivariate_normal_cdf(x, y, rho)
    np.testing.assert_allclose(actual[shown], expected[shown], rtol=5e-14, atol=0)
