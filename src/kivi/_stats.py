"""Distribution functions for the calibration tests, the metrics and capital, using NumPy and math.

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
functions of half-integer order instead, which start from ``erfc``. The beta quantile
inverts the incomplete beta function by a safeguarded Newton method, and the bivariate
normal distribution function integrates Plackett's identity by Gauss-Legendre quadrature
as A. Genz (2004) does.
"""

import math
import typing
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


def beta_quantile(
    a: npt.ArrayLike, b: npt.ArrayLike, probability: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Return the quantile of the beta distribution Beta(a, b), elementwise.

    ``a`` and ``b`` must be finite and positive and ``probability`` between 0
    and 1; 0 gives 0, 1 gives 1 and a missing probability gives NaN. The
    result ``x`` solves ``betainc(a, b, x) = probability``, in closed form for
    ``a = 1`` or ``b = 1``. Otherwise Newton's method starts from the
    approximation of Numerical Recipes, 3rd ed., section 6.14.6, safeguarded
    as in its ``rtsafe``: a step that would leave the bracket of the root or
    not halve the previous step bisects the bracket instead, geometrically in
    ``x`` below 1/2 and in ``1 - x`` above, so that a quantile near 0 or 1 is
    reached in a few bisections. The residual ``I_x(a, b) - probability`` is
    computed from the tail that `betainc` evaluates without taking ``1 - x``:
    the lower tail below its switch near the mean and the upper tail
    ``I_{1-x}(b, a)`` above, which keeps the digits of a probability close
    to 1. A small ``x`` above the switch, as for a few defaults among many
    obligors, makes ``1 - x`` round, and the first-order correction by the
    density restores the digits that this costs.
    """
    a, b, p = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (a, b, probability)))
    if not (np.isfinite(a) & (a > 0) & np.isfinite(b) & (b > 0)).all():
        raise ValueError("the beta distribution's shape parameters must be finite and positive")
    outside = (p < 0) | (p > 1)
    if outside.any():
        raise ValueError(
            f"probability must be between 0 and 1, got {_validation.shown(p[outside][0])}"
        )
    result = np.where(np.isnan(p), np.nan, np.where(p >= 1, 1.0, 0.0))
    inside = (p > 0) & (p < 1)
    # I_x(1, b) = 1 - (1 - x)**b and I_x(a, 1) = x**a.
    first = inside & (a == 1)
    result[first] = -np.expm1(np.log1p(-p[first]) / b[first])
    second = inside & ~first & (b == 1)
    result[second] = np.exp(np.log(p[second]) / a[second])
    other = inside & ~first & ~second
    if other.any():
        result[other] = _beta_root(a[other], b[other], p[other])
    return result


# Newton's method converges quadratically: after a step this small relative to x or 1 - x,
# whichever is smaller, the error left is below double precision.
_BETA_STEP = 1e-10
# Enough for the bisections from the whole bracket down to adjacent numbers and the Newton steps.
_BETA_ITERATIONS = 200
# The bracket that a bisection splits: the smallest normal number and the largest number below 1.
_BETA_LOW = float(np.finfo(float).tiny)
_BETA_HIGH = float(np.nextafter(1.0, 0.0))


def _beta_root(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], p: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Solve I_x(a, b) = p for 0 < p < 1 by Newton's method, safeguarded by bisection."""
    x = np.clip(_beta_start(a, b, p), _BETA_LOW, _BETA_HIGH)
    low, high = np.zeros_like(x), np.ones_like(x)
    # The previous step, or the bracket at first, which the next Newton step must halve.
    previous = high - low
    todo = np.arange(len(x))
    for _ in range(_BETA_ITERATIONS):
        a_, b_, p_, x_ = a[todo], b[todo], p[todo], x[todo]
        y = 1 - x_
        with np.errstate(over="ignore"):
            # The density exp(ln[x**a (1 - x)**b / B(a, b)]) / (x (1 - x)), from x alone.
            density = np.exp(_log_front(a_, b_, x_) - np.log(x_) - np.log1p(-x_))
        # I_x(a, b) - p, from the upper tail I_{1-x}(b, a) where betainc would take 1 - x itself.
        # y = 1 - x misses 1 - x by (1 - y) - x, which is exact, and the first term of the Taylor
        # series in it restores the digits of a small x: b times that error in the upper tail.
        above = x_ > (a_ + 1) / (a_ + b_ + 2)
        tail = betainc(np.where(above, b_, a_), np.where(above, a_, b_), np.where(above, y, x_))
        error = (1 - y) - x_
        with np.errstate(invalid="ignore"):  # an infinite density where 1 - x is exact
            upper = tail + np.where(error == 0, 0.0, error * density)
        residual = np.where(above, (1 - p_) - upper, tail - p_)
        low[todo] = np.where(residual < 0, x_, low[todo])
        high[todo] = np.where(residual > 0, x_, high[todo])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            step = residual / density
        new = x_ - step
        newton = (new > low[todo]) & (new < high[todo]) & (np.abs(step) <= previous[todo] / 2)
        new = np.where(newton, new, _beta_split(low[todo], high[todo]))
        previous[todo] = np.abs(new - x_)
        x[todo] = new
        # A bracket that no number splits is as narrow as it gets.
        small = np.minimum(new, 1 - new)
        done = (residual == 0) | (newton & (np.abs(step) <= _BETA_STEP * small)) | (new == x_)
        todo = todo[~done]
        if not len(todo):
            return x
    raise ArithmeticError("the beta quantile did not converge")


def _beta_split(
    low: npt.NDArray[np.float64], high: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return a point between low and high: the geometric mean of x or of 1 - x near 0 or 1."""
    low, high = np.maximum(low, _BETA_LOW), np.minimum(high, _BETA_HIGH)
    middle = (low + high) / 2
    # Square roots before the products, which could underflow.
    near_zero = np.sqrt(low) * np.sqrt(high)
    near_one = 1 - np.sqrt(1 - low) * np.sqrt(1 - high)
    return np.where(high <= 0.5, near_zero, np.where(low >= 0.5, near_one, middle))


def _beta_start(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], p: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return the starting point of Numerical Recipes for the quantile of Beta(a, b)."""
    x = np.empty_like(p)
    large = (a >= 1) & (b >= 1)
    if large.any():
        a_, b_, p_ = a[large], b[large], p[large]
        # Abramowitz and Stegun 26.5.22, with the upper normal quantile y of 26.2.22.
        t = np.sqrt(-2 * np.log(np.minimum(p_, 1 - p_)))
        y = t - (2.30753 + 0.27061 * t) / (1 + t * (0.99229 + 0.04481 * t))
        y = np.where(p_ < 0.5, y, -y)
        shape = (y * y - 3) / 6
        h = 2 / (1 / (2 * a_ - 1) + 1 / (2 * b_ - 1))
        w = y * np.sqrt(shape + h) / h - (1 / (2 * b_ - 1) - 1 / (2 * a_ - 1)) * (
            shape + 5 / 6 - 2 / (3 * h)
        )
        with np.errstate(over="ignore"):
            x[large] = a_ / (a_ + b_ * np.exp(2 * w))
    small = ~large
    if small.any():
        a_, b_, p_ = a[small], b[small], p[small]
        # The two tails of the density as powers of x and 1 - x, scaled to the same total.
        first = np.exp(a_ * np.log(a_ / (a_ + b_))) / a_
        second = np.exp(b_ * np.log(b_ / (a_ + b_))) / b_
        total = first + second
        with np.errstate(divide="ignore"):
            x[small] = np.where(
                p_ < first / total,
                np.exp(np.log(a_ * total * p_) / a_),
                -np.expm1(np.log(b_ * total * (1 - p_)) / b_),
            )
    return x


def normal_cdf(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the distribution function of the standard normal distribution, elementwise.

    A missing ``x`` gives NaN. The lower tail keeps its relative accuracy far
    from the mean, so compute an upper tail ``1 - Phi(x)`` as ``normal_cdf(-x)``.
    """
    values = np.asarray(x, dtype=float)
    return np.asarray(0.5 * _erfc(-values / math.sqrt(2)), dtype=float)


def bivariate_normal_cdf(
    x: npt.ArrayLike, y: npt.ArrayLike, rho: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Return P(X <= x, Y <= y) for standard normal X and Y with correlation rho, elementwise.

    ``rho`` must lie in [-1, 1], ``x`` and ``y`` may be infinite, and a missing
    argument gives NaN. The algorithm is that of A. Genz, "Numerical
    computation of rectangular bivariate and trivariate normal and t
    probabilities", Statistics and Computing 14 (2004), which refines Drezner
    and Wesolowsky (1990): it integrates Plackett's identity
    ``d/d rho P(X <= x, Y <= y) = phi_2(x, y; rho)`` by Gauss-Legendre
    quadrature, from rho = 0 for ``|rho| < 0.925`` and from rho = 1 or -1
    above, where the closed form of the first terms of a series takes out the
    singularity of the density. The absolute error is below 2e-16. With
    rho >= 0 nothing cancels, and probabilities above 1e-10 keep a relative
    error of about 1e-14.
    """
    x, y, rho = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (x, y, rho)))
    if ((rho < -1) | (rho > 1)).any():
        raise ValueError("the correlation must lie between -1 and 1")
    result = np.full(x.shape, np.nan)
    known = ~(np.isnan(x) | np.isnan(y) | np.isnan(rho))
    # An infinite limit leaves one variable or none.
    empty = known & ((x == -np.inf) | (y == -np.inf))
    result[empty] = 0.0
    only_y = known & ~empty & (x == np.inf)
    result[only_y] = normal_cdf(y[only_y])
    only_x = known & ~empty & ~only_y & (y == np.inf)
    result[only_x] = normal_cdf(x[only_x])
    finite = known & np.isfinite(x) & np.isfinite(y)
    # Genz computes P(X > h, Y > k), which is the lower probability at h = -x and k = -y.
    h, k, r = -x[finite], -y[finite], rho[finite]
    value = np.empty(h.shape)
    near = np.abs(r) >= _GENZ_NEAR_ONE
    value[~near] = _upper_normal_from_zero(h[~near], k[~near], r[~near])
    value[near] = _upper_normal_from_one(h[near], k[near], r[near])
    result[finite] = np.clip(value, 0.0, 1.0)
    return result


# From this correlation on, bivariate_normal_cdf integrates from a correlation of 1 or -1.
_GENZ_NEAR_ONE = 0.925
# Gauss-Legendre nodes on (-1, 1) and their weights. Genz uses 20; with 30, a correlation from 0 to
# 0.925 keeps the relative error near 1e-14 instead of 1e-12 for probabilities above 1e-10.
_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(30)
# Genz leaves out the terms of an exponent below this, which keeps exp from overflowing.
_GENZ_EXPONENT = -100.0


def _upper_normal_from_zero(
    h: npt.NDArray[np.float64], k: npt.NDArray[np.float64], r: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return P(X > h, Y > k) for |r| < 0.925, integrating from a correlation of 0.

    With r = sin(theta), Plackett's identity gives Phi(-h) Phi(-k) plus the
    integral over theta from 0 to asin(r) of
    exp(-(h**2 + k**2 - 2 h k sin(theta)) / (2 cos(theta)**2)) / (2 pi), which is smooth.
    """
    half = np.arcsin(r)[:, None] / 2
    sine = np.sin(half * (1 + _LEGENDRE_NODES))
    exponent = (sine * (h * k)[:, None] - ((h * h + k * k) / 2)[:, None]) / (1 - sine * sine)
    integral = half[:, 0] * (np.exp(exponent) * _LEGENDRE_WEIGHTS).sum(axis=1)
    return integral / (2 * math.pi) + normal_cdf(-h) * normal_cdf(-k)


def _upper_normal_from_one(
    h: npt.NDArray[np.float64], k: npt.NDArray[np.float64], r: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return P(X > h, Y > k) for |r| >= 0.925, integrating from a correlation of 1 or -1.

    For r > 0, P(X > h, Y > k) = Phi(-max(h, k)) minus the integral of the
    density from r to 1, and the substitution s**2 = 1 - rho**2 turns that
    integral into one over s from 0 to sqrt(1 - r**2) of
    exp(-(h - k)**2 / (2 s**2) - h k / 2) e(s**2) / (2 pi), with
    e(u) = exp(-(h k / 2) u / (1 + sqrt(1 - u))**2) / sqrt(1 - u). The terms
    1 + c u + 5 c d u**2 of the Taylor series of e are integrated in closed form
    and the rest by Gauss-Legendre quadrature. A negative r uses the same
    integral for -k and P(X > h, Y > k) = Phi(-h) - P(X > h, -Y > -k).
    """
    negative = r < 0
    k = np.where(negative, -k, k)
    hk = h * k
    bvn = np.zeros_like(h)
    # 1 - r**2 without cancellation; it is 0 for |r| = 1, where the integral vanishes.
    square = (1 - np.abs(r)) * (1 + np.abs(r))
    inner = square > 0
    if inner.any():
        bvn[inner] = _genz_integral(h[inner], k[inner], hk[inner], square[inner])
    edge = normal_cdf(-np.maximum(h, k))
    # For r < 0 and h < k, Phi(-h) - Phi(-k), taken from the tail with the smaller values.
    band = np.where(h < 0, normal_cdf(k) - normal_cdf(h), normal_cdf(-h) - normal_cdf(-k))
    return np.where(negative, np.where(h >= k, -bvn, band - bvn), bvn + edge)


def _genz_integral(
    h: npt.NDArray[np.float64],
    k: npt.NDArray[np.float64],
    hk: npt.NDArray[np.float64],
    square: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return minus the integral of `_upper_normal_from_one` for 0 < 1 - r**2 = square."""
    bound = np.sqrt(square)
    bs = (h - k) ** 2
    c = (4 - hk) / 8
    d = (12 - hk) / 80
    with np.errstate(over="ignore", invalid="ignore"):
        exponent = -(bs / square + hk) / 2
        closed = np.where(
            exponent > _GENZ_EXPONENT,
            bound
            * np.exp(exponent)
            * (1 - c * (bs - square) * (1 - d * bs) / 3 + c * d * square * square),
            0.0,
        )
        b = np.sqrt(bs)
        closed -= np.where(
            hk > _GENZ_EXPONENT,
            np.exp(-hk / 2)
            * math.sqrt(2 * math.pi)
            * normal_cdf(-b / bound)
            * b
            * (1 - c * bs * (1 - d * bs) / 3),
            0.0,
        )
        half = bound[:, None] / 2
        u = (half * (1 + _LEGENDRE_NODES)) ** 2
        exponent = -(bs[:, None] / u + hk[:, None]) / 2
        root = np.sqrt(1 - u)
        series = 1 + c[:, None] * u * (1 + 5 * d[:, None] * u)
        exact = np.exp(-(hk[:, None] / 2) * u / (1 + root) ** 2) / root
        terms = np.where(exponent > _GENZ_EXPONENT, np.exp(exponent) * (series - exact), 0.0)
    numeric = half[:, 0] * (terms * _LEGENDRE_WEIGHTS).sum(axis=1)
    return (numeric - closed) / (2 * math.pi)


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


@typing.overload
def normal_quantile(probability: float) -> float: ...
@typing.overload
def normal_quantile(probability: npt.ArrayLike) -> float | npt.NDArray[np.float64]: ...
def normal_quantile(probability: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
    """Return the quantile of the standard normal distribution.

    A scalar must lie strictly between 0 and 1 and gives a float. An array
    gives an array of its shape, in which a missing value gives NaN and every
    other value must lie strictly between 0 and 1; each element is exactly the
    float that the scalar gives.
    """
    if np.ndim(probability) == 0:
        return _normal_quantile(typing.cast(float, probability))
    values = np.asarray(probability, dtype=float)
    known = ~np.isnan(values)
    outside = known & ~((values > 0) & (values < 1))
    if outside.any():
        raise ValueError(
            f"probability must be between 0 and 1, got {_validation.shown(values[outside][0])}"
        )
    result = np.full(values.shape, np.nan)
    # Each distinct value once: the probabilities of a portfolio often repeat those of its grades.
    distinct, position = np.unique(values[known], return_inverse=True)
    quantiles = np.array([_normal_quantile(value) for value in distinct.tolist()], dtype=float)
    result[known] = quantiles[position.ravel()]
    return result


def _normal_quantile(probability: float) -> float:
    """Return the quantile of the standard normal distribution, for 0 < probability < 1."""
    if not 0 < probability < 1:
        raise ValueError(
            f"probability must be between 0 and 1, got {_validation.shown_argument(probability)}"
        )
    if probability > 0.5:
        # 1 - p is exact for p >= 0.5, and the lower tail keeps full relative accuracy.
        return -_normal_quantile(1 - probability)
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
