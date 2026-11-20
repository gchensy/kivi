"""Distribution functions for the calibration tests, using only NumPy and math.

The regularized incomplete beta and gamma functions follow the series and
continued fractions of Press et al., "Numerical Recipes", 3rd ed., sections
6.2 and 6.4, with the continued fractions evaluated by the modified Lentz
method. For large parameters the factor in front of the beta continued
fraction is computed from Stirling's series around the mean, as in DiDonato
and Morris (ACM TOMS 18, 1992), instead of from differences of log-gamma
values, which lose about ``log10(a + b)`` digits.
"""

import math

import numpy as np
import numpy.typing as npt

_EPS = 1e-15
_TINY = 1e-300
_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
# From here on, Stirling's series to z**-9 gives log-gamma to about 2e-14.
_STIRLING_FROM = 10.0

_lgamma = np.vectorize(math.lgamma, otypes=[float])


def betainc(a: npt.ArrayLike, b: npt.ArrayLike, x: npt.ArrayLike) -> np.ndarray:
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


def _log_front(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
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


def _stirling_correction(z: np.ndarray) -> np.ndarray:
    """Return ln Gamma(z) - [(z - 1/2) ln z - z + ln(2 pi) / 2] for z >= 10."""
    w = 1 / (z * z)
    return (1 / 12 + w * (-1 / 360 + w * (1 / 1260 + w * (-1 / 1680 + w / 1188)))) / z


def _log1pmx(t: np.ndarray) -> np.ndarray:
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


def _beta_fraction(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
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


def _guard(values: np.ndarray) -> np.ndarray:
    return np.where(np.abs(values) < _TINY, _TINY, values)


def _guard_scalar(value: float) -> float:
    return _TINY if abs(value) < _TINY else value
