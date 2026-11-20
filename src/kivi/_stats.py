"""Distribution functions for the calibration tests, using only NumPy and math.

The regularized incomplete beta and gamma functions follow the series and
continued fractions of Press et al., "Numerical Recipes", 3rd ed., sections
6.2 and 6.4, with the continued fractions evaluated by the modified Lentz
method.
"""

import math

import numpy as np
import numpy.typing as npt

_EPS = 1e-15
_TINY = 1e-300

_lgamma = np.vectorize(math.lgamma, otypes=[float])


def betainc(a: npt.ArrayLike, b: npt.ArrayLike, x: npt.ArrayLike) -> np.ndarray:
    """Return the regularized incomplete beta function I_x(a, b), elementwise.

    ``a`` and ``b`` must be positive; ``x`` is clipped to [0, 1]. This is the
    CDF of Beta(a, b) at ``x``, and the binomial tail P(X >= k) for
    X ~ Bin(n, p) equals I_p(k, n - k + 1).
    """
    a, b, x = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (a, b, x)))
    result = np.where(x >= 1, 1.0, 0.0)
    inside = (x > 0) & (x < 1)
    if not inside.any():
        return result
    a, b, x = a[inside], b[inside], x[inside]
    # The continued fraction converges fast below the mean; above it, use the symmetry
    # I_x(a, b) = 1 - I_{1-x}(b, a).
    flip = x > (a + 1) / (a + b + 2)
    a, b = np.where(flip, b, a), np.where(flip, a, b)
    x = np.where(flip, 1 - x, x)
    log_front = _lgamma(a + b) - _lgamma(a) - _lgamma(b) + a * np.log(x) + b * np.log1p(-x)
    value = np.exp(log_front) * _beta_fraction(a, b, x) / a
    result[inside] = np.where(flip, 1 - value, value)
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
    if statistic <= 0:
        return 1.0
    return gammaincc(dof / 2, statistic / 2)


def gammaincc(a: float, x: float) -> float:
    """Return the regularized upper incomplete gamma function Q(a, x) for a > 0."""
    if x <= 0:
        return 1.0
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
