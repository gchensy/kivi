"""Distribution functions for the calibration tests and the metrics, using only NumPy and math.

The regularized incomplete beta and gamma functions follow the series and
continued fractions of Press et al., "Numerical Recipes", 3rd ed., sections
6.2 and 6.4, with the continued fractions evaluated by the modified Lentz
method. For large parameters the factor in front of the beta continued
fraction is computed from Stirling's series around the mean, as in DiDonato
and Morris (ACM TOMS 18, 1992), instead of from differences of log-gamma
values, which lose about ``log10(a + b)`` digits. The normal quantile starts
from the rational approximation of P. J. Acklam (2003), whose relative error
is below 1.2e-9, and refines it with a step of Halley's method. The normal
distribution function is ``0.5 erfc(-x / sqrt(2))``. The tail of Student's t
distribution with ``df`` degrees of freedom beyond ``|t|`` is ``I_x(df / 2, 1 / 2) / 2``
at ``x = df / (df + t**2)``, and near the centre ``1 / 2 - I_y(1 / 2, df / 2) / 2`` at
``y = 1 - x`` (Abramowitz and Stegun, 26.5.27 and 26.7.1). From 100 degrees of freedom
on, the continued fractions lose about ``log10(df)`` digits where ``x`` is close to the
mean of its beta distribution, so there the tail is an expansion in incomplete gamma
functions of half-integer order instead, which start from ``erfc``.
"""

import math
from fractions import Fraction

import numpy as np
import numpy.typing as npt

from . import _validation

_EPS = 1e-15
_TINY = 1e-300
_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
# From here on, Stirling's series to z**-9 gives log-gamma to about 2e-14.
_STIRLING_FROM = 10.0

_lgamma = np.vectorize(math.lgamma, otypes=[float])
_erfc = np.vectorize(math.erfc, otypes=[float])


def betainc(a: npt.ArrayLike, b: npt.ArrayLike, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the regularized incomplete beta function I_x(a, b), elementwise.

    ``a`` and ``b`` must be positive; ``x`` is clipped to [0, 1] and a missing
    ``x`` gives NaN. This is the CDF of Beta(a, b) at ``x``, and the binomial
    tail P(X >= k) for X ~ Bin(n, p) equals I_p(k, n - k + 1).
    """
    a, b, x = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (a, b, x)))
    result = np.where(np.isnan(x), np.nan, np.where(x >= 1, 1.0, 0.0))
    inside = (x > 0) & (x < 1)
    if not inside.any():
        return result
    a, b, x = a[inside], b[inside], x[inside]
    # The continued fraction converges fast below the mean; above it, use the symmetry
    # I_x(a, b) = 1 - I_{1-x}(b, a).
    flip = x > (a + 1) / (a + b + 2)
    a, b = np.where(flip, b, a), np.where(flip, a, b)
    x = np.where(flip, 1 - x, x)
    value = np.exp(_log_front(a, b, x)) * _beta_fraction(a, b, x) / a
    result[inside] = np.where(flip, 1 - value, value)
    return result


def _log_front(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return ln[x**a (1 - x)**b / B(a, b)] for 0 < x < 1."""
    result = np.empty_like(x)
    both = (a >= _STIRLING_FROM) & (b >= _STIRLING_FROM)
    one = ~both & (np.maximum(a, b) >= _STIRLING_FROM)
    neither = ~both & ~one
    if both.any():
        # Around the mean p0 = a / (a + b) the first-order terms of a ln(x / p0) and
        # b ln((1 - x) / q0) cancel exactly, so only ln(1 + t) - t remains.
        a_, b_, x_ = a[both], b[both], x[both]
        total = a_ + b_
        p0, q0 = a_ / total, b_ / total
        result[both] = (
            a_ * _log1pmx((x_ - p0) / p0)
            + b_ * _log1pmx((p0 - x_) / q0)
            + 0.5 * (np.log(a_) + np.log(b_) - np.log(total))
            - _HALF_LOG_2PI
            + _stirling_correction(total)
            - _stirling_correction(a_)
            - _stirling_correction(b_)
        )
    if one.any():
        # ln Gamma(large + small) - ln Gamma(large) without subtracting two large values.
        a_, b_, x_ = a[one], b[one], x[one]
        small, large = np.minimum(a_, b_), np.maximum(a_, b_)
        gamma_ratio = (
            (large - 0.5) * np.log1p(small / large)
            + small * np.log(large + small)
            - small
            + _stirling_correction(large + small)
            - _stirling_correction(large)
        )
        result[one] = a_ * np.log(x_) + b_ * np.log1p(-x_) - _lgamma(small) + gamma_ratio
    if neither.any():
        a_, b_, x_ = a[neither], b[neither], x[neither]
        result[neither] = (
            a_ * np.log(x_) + b_ * np.log1p(-x_) + _lgamma(a_ + b_) - _lgamma(a_) - _lgamma(b_)
        )
    return result


def _stirling_correction(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return ln Gamma(z) - [(z - 1/2) ln z - z + ln(2 pi) / 2] for z >= 10."""
    w = 1 / (z * z)
    return (1 / 12 + w * (-1 / 360 + w * (1 / 1260 + w * (-1 / 1680 + w / 1188)))) / z


def _log1pmx(t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return ln(1 + t) - t, without cancellation for small t."""
    with np.errstate(divide="ignore"):  # t = -1 when 1 - x underflows; the limit is -inf
        result = np.log1p(t) - t
    small = np.abs(t) < 0.25
    if small.any():
        # -t**2/2 + t**3/3 - ... by Horner's rule; 30 terms reach double precision.
        u = t[small]
        series = np.full_like(u, -1 / 30)
        for k in range(29, 1, -1):
            series = (-1) ** (k + 1) / k + u * series
        result[small] = u * u * series
    return result


def _beta_fraction(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Evaluate the continued fraction of I_x(a, b) for x below the mean."""
    # The worst case needs O(sqrt(max(a, b))) terms.
    max_terms = 200 + 20 * math.ceil(math.sqrt(float(np.max(np.maximum(a, b)))))
    c = np.ones_like(x)
    d = 1 / _guard(1 - (a + b) * x / (a + 1))
    h = d.copy()
    active = np.ones(x.shape, dtype=bool)
    for m in range(1, max_terms + 1):
        even = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        odd = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        for term in (even, odd):
            d = 1 / _guard(1 + term * d)
            c = _guard(1 + term / c)
            step = np.where(active, c * d, 1.0)
            h *= step
        active &= np.abs(step - 1) >= _EPS
        if not active.any():
            return h
    raise ArithmeticError("the incomplete beta function did not converge")


def normal_cdf(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the distribution function of the standard normal distribution, elementwise.

    A missing ``x`` gives NaN. The lower tail keeps its relative accuracy far
    from the mean, so compute an upper tail ``1 - Phi(x)`` as ``normal_cdf(-x)``.
    """
    values = np.asarray(x, dtype=float)
    return np.asarray(0.5 * _erfc(-values / math.sqrt(2)), dtype=float)


def student_t_sf(t: npt.ArrayLike, dof: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the upper tail P(T > t) of Student's t distribution, elementwise.

    ``dof`` must be finite and positive and may be fractional; ``t`` may be
    infinite, and a missing ``t`` gives NaN.
    """
    t, dof = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(dof, dtype=float))
    if not (np.isfinite(dof) & (dof > 0)).all():
        raise ValueError("the degrees of freedom must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):  # t**2 overflows for huge t
        ratio = t * t / dof
        # x + y = 1, each computed without cancellation.
        x = 1 / (1 + ratio)
        y = np.where(np.isinf(ratio), 1.0, ratio / (1 + ratio))
        expansion = (dof >= _T_EXPANSION_DOF) & (ratio > 0) & (np.log1p(ratio) <= 1)
    a = dof / 2
    # Pass betainc the argument that lies below the mean of its distribution, so that it is not
    # recomputed as 1 minus the other one, which loses the digits of a t close to 0.
    central = ~expansion & (y < 1.5 / (a + 2.5))
    tails = ~expansion & ~central
    # The other entries get x = 0, which betainc does not evaluate.
    within = betainc(0.5, np.where(central, a, 1.0), np.where(central, y, 0))
    tail = 0.5 * betainc(np.where(tails, a, 1.0), 0.5, np.where(tails, x, 0))
    # P(|T| < |t|) = I_y(1/2, df/2) near the centre; P(T > |t|) = I_x(df/2, 1/2) / 2 in the tails.
    upper = np.where(central, 0.5 - 0.5 * within, tail)
    if expansion.any():
        upper[expansion] = _t_tail_expansion(np.abs(t[expansion]), dof[expansion])
    return np.where(t > 0, upper, 1 - upper)


# From here on, student_t_sf uses the expansion of _t_tail_expansion where ln(1 + t**2 / df) <= 1.
_T_EXPANSION_DOF = 100.0


def _t_expansion_coefficients(terms: int) -> tuple[float, ...]:
    """Return the coefficients g_k of (v / (2 sinh(v / 2)))**(1/2) = sum_k g_k v**(2k).

    The power series of 2 sinh(v / 2) / v = sum_n (v**2 / 4)**n / (2n + 1)! is raised to the
    power -1/2 with exact fractions, by the recurrence of J. C. P. Miller for the powers of a
    series.
    """
    series = [Fraction(1, 4**n * math.factorial(2 * n + 1)) for n in range(terms)]
    power = [Fraction(1)]
    for n in range(1, terms):
        products = ((Fraction(k, 2) - n) * series[k] * power[n - k] for k in range(1, n + 1))
        power.append(sum(products, Fraction(0)) / n)
    return tuple(float(c) for c in power)


# The coefficients fall by a factor of about 50 per term, so 13 terms reach double precision for
# v0 <= 1.
_T_EXPANSION = _t_expansion_coefficients(13)


def _t_tail_expansion(
    t: npt.NDArray[np.float64], dof: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return P(T > t) for t > 0 and many degrees of freedom, as a series in erfc.

    With a = df / 2 and m = a - 1/4, the substitution x = exp(-v) in the integral of
    I_x(a, 1/2) gives

        P(T > t) = Gamma(a + 1/2) / (2 sqrt(pi) Gamma(a))
                   * integral from v0 to inf of exp(-m v) v**(-1/2) g(v) dv,

    with v0 = ln(1 + t**2 / df) and g(v) = (v / (2 sinh(v / 2)))**(1/2), an even function that
    is analytic for |v| < 2 pi. Integrating its power series term by term gives

        P(T > t) = R erfc(sqrt(z)) / 2 * [1 + sum_k g_k m**(-2k) G_k],  z = m v0,

    with R = Gamma(a + 1/2) / (Gamma(a) sqrt(m)) and G_k = Gamma(2k + 1/2, z) / Gamma(1/2, z),
    which the recurrence Gamma(s + 1, z) = s Gamma(s, z) + z**s exp(-z) gives from
    Gamma(1/2, z) = sqrt(pi) erfc(sqrt(z)). The terms are about g_k v0**(2k) for a large z and
    g_k m**(-2k) Gamma(2k + 1/2) / sqrt(pi) for a small one, so the series converges fast for
    v0 <= 1 and m >= 50. The part of the integral beyond v = 2 pi, where the power series of g
    diverges, is below exp(-2 pi m) and does not count.
    """
    a = dof / 2
    m = a - 0.25
    z = m * np.log1p(t * t / dof)
    upper = _erfc(np.sqrt(z))
    # ln R = a ln(1 + 1 / (2a)) - 1/2 - ln(1 - 1 / (4a)) / 2 from Stirling's series, with the
    # first two terms as a [ln(1 + u) - u] at u = 1 / (2a), which does not cancel.
    log_ratio = (
        a * _log1pmx(0.5 / a)
        - 0.5 * np.log1p(-0.25 / a)
        + _stirling_correction(a + 0.5)
        - _stirling_correction(a)
    )
    # exp(-z) / Gamma(1/2, z); erfc underflows only together with the result.
    seen = upper > 0
    step = np.where(seen, np.exp(-z) / (math.sqrt(math.pi) * np.where(seen, upper, 1.0)), 0.0)
    ratio = np.ones_like(z)
    total = np.ones_like(z)
    order, power = 0.5, np.sqrt(z)
    for k, coefficient in enumerate(_T_EXPANSION[1:], start=1):
        for _ in range(2):
            ratio = order * ratio + step * power
            power = power * z
            order += 1
        total += coefficient * m ** (-2 * k) * ratio
    return 0.5 * np.exp(log_ratio) * upper * total


def chi2_sf(statistic: float, dof: int) -> float:
    """Return the upper tail probability of a chi-square distribution."""
    if math.isnan(statistic):
        raise ValueError("the chi-square statistic must not be NaN")
    if statistic <= 0:
        return 1.0
    return gammaincc(dof / 2, statistic / 2)


def gammaincc(a: float, x: float) -> float:
    """Return the regularized upper incomplete gamma function Q(a, x) for a > 0."""
    if math.isnan(x):
        raise ValueError("x must not be NaN")
    if x <= 0:
        return 1.0
    if math.isinf(x):
        return 0.0
    log_front = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1:
        # Series for the lower function P(a, x) = 1 - Q(a, x).
        term = total = 1 / a
        for n in range(1, 10_000):
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * _EPS:
                return 1 - total * math.exp(log_front)
    else:
        # Continued fraction for Q(a, x).
        b = x + 1 - a
        c = 1 / _TINY
        d = 1 / b
        h = d
        for n in range(1, 10_000):
            term = -n * (n - a)
            b += 2
            d = 1 / _guard_scalar(term * d + b)
            c = _guard_scalar(b + term / c)
            h *= c * d
            if abs(c * d - 1) < _EPS:
                return h * math.exp(log_front)
    raise ArithmeticError("the incomplete gamma function did not converge")


def normal_quantile(probability: float) -> float:
    """Return the quantile of the standard normal distribution, for 0 < probability < 1."""
    if not 0 < probability < 1:
        raise ValueError(
            f"probability must be between 0 and 1, got {_validation.shown_argument(probability)}"
        )
    if probability > 0.5:
        # 1 - p is exact for p >= 0.5, and the lower tail keeps full relative accuracy.
        return -normal_quantile(1 - probability)
    if probability < _ACKLAM_LOW:
        q = math.sqrt(-2 * math.log(probability))
        x = _horner(_ACKLAM_C, q) / _horner(_ACKLAM_D, q)
        if x * x / 2 > 700:  # below about 1e-305 the density underflows; keep the approximation
            return x
        # The residual of the normal CDF, 0.5 erfc(-x / sqrt(2)), which is accurate for x <= 0.
        error = 0.5 * math.erfc(-x / math.sqrt(2)) - probability
    else:
        q = probability - 0.5
        x = q * _horner(_ACKLAM_A, q * q) / _horner(_ACKLAM_B, q * q)
        # Compare with p - 1/2, which is exact near the median, so that the residual keeps its
        # relative accuracy as x approaches 0.
        error = 0.5 * math.erf(x / math.sqrt(2)) - q
    # One step of Halley's method.
    u = error * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


# Coefficients from the highest power down; the denominators end with their constant 1.
_ACKLAM_LOW = 0.02425
_ACKLAM_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
    1.0,
)
_ACKLAM_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
    1.0,
)


def _horner(coefficients: tuple[float, ...], x: float) -> float:
    """Evaluate a polynomial whose coefficients run from the highest power down."""
    result = 0.0
    for coefficient in coefficients:
        result = result * x + coefficient
    return result


def _guard(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.where(np.abs(values) < _TINY, _TINY, values)


def _guard_scalar(value: float) -> float:
    return _TINY if abs(value) < _TINY else value
