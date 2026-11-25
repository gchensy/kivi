"""Regulatory capital of the IRB approach, the Vasicek one-factor model and most prudent PDs.

``asset_correlation`` and ``capital_requirement`` evaluate the risk-weight
functions of the internal ratings-based (IRB) approach of the Basel
framework (CRE31) and of China's 2023 capital rules (annex 6), whose
parameters a keyword sets where the two differ. ``conditional_pd``,
``systematic_factor`` and ``vasicek_fit`` work with the one-factor model
behind those functions, and ``most_prudent_pd`` gives the upper bounds of
Pluto and Tasche for the PDs of a portfolio with few defaults.

Probabilities lie strictly between 0 and 1. The value-by-value functions
broadcast their inputs, give NaN for a missing PD, LGD, default rate or
factor, and raise for a missing argument that decides the formula, such as
a turnover.
"""

import math
import typing
from collections import abc

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import _stats, _validation

__all__ = [
    "asset_correlation",
    "capital_requirement",
    "conditional_pd",
    "systematic_factor",
    "vasicek_fit",
    "most_prudent_pd",
]

_AssetClass: typing.TypeAlias = typing.Literal[
    "corporate", "hvcre", "mortgage", "revolving", "other_retail"
]
_ASSET_CLASSES = ("corporate", "hvcre", "mortgage", "revolving", "other_retail")
_RETAIL = ("mortgage", "revolving", "other_retail")
# The PD below which 1 - 1.5 b <= 0 for b = (0.11852 - 0.05478 ln PD)**2, about 2.927e-6.
_SMALLEST_PD = math.exp((0.11852 - math.sqrt(2 / 3)) / 0.05478)


def _factor_rule(panels: int, nodes: int, bound: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the nodes and weights of composite Gauss-Legendre quadrature against phi(z).

    The rule covers [-bound, bound] with equal panels; the standard normal mass beyond is
    about 2e-19 for a bound of 9.
    """
    points, weights = np.polynomial.legendre.leggauss(nodes)
    edges = np.linspace(-bound, bound, panels + 1)
    middle, half = (edges[:-1] + edges[1:]) / 2, (edges[1:] - edges[:-1]) / 2
    z = (middle[:, None] + half[:, None] * points).ravel()
    return z, (half[:, None] * weights).ravel() * np.exp(-z * z / 2) / math.sqrt(2 * math.pi)


# The factor rule of most_prudent_pd. P(D <= d) is a steep function of the factor near the level,
# which Gauss-Hermite quadrature resolves poorly: 160 nodes miss the PD of a level of 0.99 by up
# to 1e-4, relatively, where these 512 nodes come within 1e-13 of adaptive quadrature.
_FACTOR_NODES, _FACTOR_WEIGHTS = _factor_rule(64, 8, 9.0)


@typing.overload
def asset_correlation(
    probability: pd.Series,
    *,
    asset_class: _AssetClass,
    turnover: npt.ArrayLike | None = ...,
    turnover_range: tuple[float, float] = ...,
    financial: bool = ...,
) -> pd.Series: ...
@typing.overload
def asset_correlation(
    probability: npt.ArrayLike,
    *,
    asset_class: _AssetClass,
    turnover: pd.Series,
    turnover_range: tuple[float, float] = ...,
    financial: bool = ...,
) -> pd.Series: ...
@typing.overload
def asset_correlation(
    probability: float,
    *,
    asset_class: _AssetClass,
    turnover: float | None = ...,
    turnover_range: tuple[float, float] = ...,
    financial: bool = ...,
) -> float: ...
@typing.overload
def asset_correlation(
    probability: _validation.ListLike,
    *,
    asset_class: _AssetClass,
    turnover: float | _validation.ListLike | None = ...,
    turnover_range: tuple[float, float] = ...,
    financial: bool = ...,
) -> npt.NDArray[np.float64]: ...
@typing.overload
def asset_correlation(
    probability: npt.ArrayLike,
    *,
    asset_class: _AssetClass,
    turnover: npt.ArrayLike | None = ...,
    turnover_range: tuple[float, float] = ...,
    financial: bool = ...,
) -> float | pd.Series | npt.NDArray[np.float64]: ...
def asset_correlation(
    probability: npt.ArrayLike,
    *,
    asset_class: _AssetClass,
    turnover: npt.ArrayLike | None = None,
    turnover_range: tuple[float, float] = (5.0, 50.0),
    financial: bool = False,
) -> float | pd.Series | npt.NDArray[np.float64]:
    r"""Compute the asset correlation of the IRB risk-weight function.

    The asset correlation :math:`R` is the share of the variance of a
    borrower's assets that the economy as a whole explains. The IRB
    approach fixes it by asset class, falling with the PD for corporate
    exposures and for other retail exposures.

    Parameters
    ----------
    probability : array_like
        Probabilities of default, from 0 to 1. Missing values give missing
        correlations.
    asset_class : {"corporate", "hvcre", "mortgage", "revolving", "other_retail"}
        ``"corporate"`` for corporate, sovereign and bank exposures,
        ``"hvcre"`` for high-volatility commercial real estate,
        ``"mortgage"`` for residential mortgages, ``"revolving"`` for
        qualifying revolving retail exposures and ``"other_retail"`` for
        other retail exposures.
    turnover : array_like, optional
        Annual sales of each corporate borrower, in the units of
        `turnover_range`, such as millions of euros. Values below the upper
        bound of the range lower the correlation, as for small and
        medium-sized enterprises; ``numpy.inf`` marks a borrower without
        that adjustment. Only for ``asset_class="corporate"``; must not be
        missing.
    turnover_range : tuple of float, default (5.0, 50.0)
        The lower and upper bound of the turnover adjustment. The default is
        the Basel range in millions of euros; China's rules use (3.0, 30.0)
        in tens of millions of yuan.
    financial : bool, default False
        Whether the borrowers are financial institutions whose correlation is
        multiplied by 1.25. Only for ``asset_class="corporate"``, and not
        together with a turnover below the upper bound of `turnover_range`.

    Returns
    -------
    float, pandas.Series or numpy.ndarray
        Correlations with the broadcast shape of `probability` and
        `turnover`. If an input is a Series, the result is a Series with its
        index, named ``correlation``; scalar inputs give a float.

    Raises
    ------
    ValueError
        If `asset_class` is not one of the five classes, `probability` is not
        real and numeric or is outside [0, 1], or the inputs cannot be
        broadcast together or have different Series indices. Also if
        `turnover` or `financial` is given for another class than
        ``"corporate"``, `turnover` is missing, negative or not numeric,
        `turnover_range` is not two finite numbers with
        ``0 <= lower < upper``, `financial` is not a boolean, or `financial`
        comes with a turnover below the upper bound.

    See Also
    --------
    capital_requirement : The capital requirement that uses the correlation.
    conditional_pd : The PD of the one-factor model given the state of the economy.

    Notes
    -----
    For corporate exposures, and with weights of 35 instead of 50 and the
    bounds 0.03 and 0.16 for other retail exposures,

    .. math::

       w = \frac{1 - e^{-50\,\mathrm{PD}}}{1 - e^{-50}}, \qquad
       R = 0.12\,w + 0.24\,(1 - w).

    High-volatility commercial real estate uses 0.30 instead of 0.24,
    residential mortgages :math:`R = 0.15` and qualifying revolving retail
    exposures :math:`R = 0.04`. A turnover :math:`S` with the range
    :math:`[l, u]` lowers a corporate correlation by
    :math:`0.04\,(1 - (\min(\max(S, l), u) - l) / (u - l))`, which is 0.04
    at :math:`S \le l` and 0 at :math:`S \ge u`, and `financial` multiplies
    it by 1.25 (Basel CRE31.5–31.16; China, annex 6 and annex 8).

    Examples
    --------
    >>> from kivi.capital import asset_correlation
    >>> asset_correlation([0.0003, 0.01, 0.2], asset_class="corporate").round(4)
    array([0.2382, 0.1928, 0.12  ])
    >>> print(f"{asset_correlation(0.01, asset_class='corporate', turnover=5.0):.4f}")
    0.1528
    """
    asset_class = _checked_asset_class(asset_class)
    lower, upper = _checked_adjustment(asset_class, turnover, turnover_range, financial)
    inputs: dict[str, npt.ArrayLike] = {"probability": probability}
    if turnover is not None:
        inputs["turnover"] = turnover
    arrays, template = _broadcast(inputs)
    values = arrays["probability"]
    _check_unit(values, "probability")
    if turnover is not None:
        _check_turnover(arrays["turnover"], upper, financial)
    correlation = _correlation(values, asset_class, arrays.get("turnover"), lower, upper, financial)
    return _shaped(template, correlation, "correlation")


@typing.overload
def capital_requirement(
    probability: pd.Series,
    lgd: npt.ArrayLike,
    *,
    asset_class: _AssetClass,
    maturity: npt.ArrayLike | None = ...,
    turnover: npt.ArrayLike | None = ...,
    turnover_range: tuple[float, float] = ...,
    financial: bool = ...,
    correlation: npt.ArrayLike | None = ...,
    level: float = ...,
    scaling: float = ...,
) -> pd.Series: ...
@typing.overload
def capital_requirement(
    probability: npt.ArrayLike,
    lgd: pd.Series,
    *,
    asset_class: _AssetClass,
    maturity: npt.ArrayLike | None = ...,
    turnover: npt.ArrayLike | None = ...,
    turnover_range: tuple[float, float] = ...,
    financial: bool = ...,
    correlation: npt.ArrayLike | None = ...,
    level: float = ...,
    scaling: float = ...,
) -> pd.Series: ...
@typing.overload
def capital_requirement(
    probability: npt.ArrayLike,
    lgd: npt.ArrayLike,
    *,
    asset_class: _AssetClass,
    maturity: pd.Series,
    turnover: npt.ArrayLike | None = ...,
    turnover_range: tuple[float, float] = ...,
    financial: bool = ...,
    correlation: npt.ArrayLike | None = ...,
    level: float = ...,
    scaling: float = ...,
) -> pd.Series: ...
@typing.overload
def capital_requirement(
    probability: npt.ArrayLike,
    lgd: npt.ArrayLike,
    *,
    asset_class: _AssetClass,
    maturity: npt.ArrayLike | None = ...,
    turnover: pd.Series,
    turnover_range: tuple[float, float] = ...,
    financial: bool = ...,
    correlation: npt.ArrayLike | None = ...,
    level: float = ...,
    scaling: float = ...,
) -> pd.Series: ...
@typing.overload
def capital_requirement(
    probability: npt.ArrayLike,
    lgd: npt.ArrayLike,
    *,
    asset_class: _AssetClass,
    maturity: npt.ArrayLike | None = ...,
    turnover: npt.ArrayLike | None = ...,
    turnover_range: tuple[float, float] = ...,
    financial: bool = ...,
    correlation: pd.Series,
    level: float = ...,
    scaling: float = ...,
) -> pd.Series: ...
@typing.overload
def capital_requirement(
    probability: float,
    lgd: float,
    *,
    asset_class: _AssetClass,
    maturity: float | None = ...,
    turnover: float | None = ...,
    turnover_range: tuple[float, float] = ...,
    financial: bool = ...,
    correlation: float | None = ...,
    level: float = ...,
    scaling: float = ...,
) -> float: ...
@typing.overload
def capital_requirement(
    probability: _validation.ListLike,
    lgd: float | _validation.ListLike,
    *,
    asset_class: _AssetClass,
    maturity: float | _validation.ListLike | None = ...,
    turnover: float | _validation.ListLike | None = ...,
    turnover_range: tuple[float, float] = ...,
    financial: bool = ...,
    correlation: float | _validation.ListLike | None = ...,
    level: float = ...,
    scaling: float = ...,
) -> npt.NDArray[np.float64]: ...
@typing.overload
def capital_requirement(
    probability: npt.ArrayLike,
    lgd: npt.ArrayLike,
    *,
    asset_class: _AssetClass,
    maturity: npt.ArrayLike | None = ...,
    turnover: npt.ArrayLike | None = ...,
    turnover_range: tuple[float, float] = ...,
    financial: bool = ...,
    correlation: npt.ArrayLike | None = ...,
    level: float = ...,
    scaling: float = ...,
) -> float | pd.Series | npt.NDArray[np.float64]: ...
def capital_requirement(
    probability: npt.ArrayLike,
    lgd: npt.ArrayLike,
    *,
    asset_class: _AssetClass,
    maturity: npt.ArrayLike | None = None,
    turnover: npt.ArrayLike | None = None,
    turnover_range: tuple[float, float] = (5.0, 50.0),
    financial: bool = False,
    correlation: npt.ArrayLike | None = None,
    level: float = 0.999,
    scaling: float = 1.0,
) -> float | pd.Series | npt.NDArray[np.float64]:
    r"""Compute the capital requirement K of the IRB approach, as a share of the exposure.

    K covers the unexpected loss of an exposure that is not in default: the
    loss at the `level` quantile of the one-factor model minus the expected
    loss. The risk-weighted assets are ``12.5 * K * EAD`` and the risk weight
    ``12.5 * K``.

    Parameters
    ----------
    probability : array_like
        Probabilities of default strictly between 0 and 1, after any PD
        floor. Missing values give missing requirements.
    lgd : array_like
        Loss given default as a share of the exposure, from 0 to 1. Missing
        values give missing requirements.
    asset_class : {"corporate", "hvcre", "mortgage", "revolving", "other_retail"}
        The asset class, as in `asset_correlation`. It also decides whether
        the maturity adjustment applies: for ``"corporate"`` and ``"hvcre"``,
        not for the three retail classes.
    maturity : array_like, optional
        Effective maturity in years, positive and finite, for
        ``"corporate"`` and ``"hvcre"`` only. The default is 2.5 years. It
        is not clipped to the range of 1 to 5 years, which has exceptions
        for short-term exposures, and must not be missing.
    turnover : array_like, optional
        Annual sales of corporate borrowers, as in `asset_correlation`.
    turnover_range : tuple of float, default (5.0, 50.0)
        The bounds of the turnover adjustment, as in `asset_correlation`.
    financial : bool, default False
        Whether the 1.25 multiplier for financial institutions applies, as in
        `asset_correlation`.
    correlation : array_like, optional
        Asset correlations from 0 to 1 (exclusive) that replace those of the
        formula, as for economic capital; then `turnover` and `financial`
        must not be given. Must not be missing.
    level : float, default 0.999
        The confidence level of the one-factor model, strictly between 0
        and 1.
    scaling : float, default 1.0
        A positive factor applied to K. Only the Basel II framework of 2006
        multiplied the risk-weighted assets by 1.06; the current Basel
        framework and China's 2023 rules do not.

    Returns
    -------
    float, pandas.Series or numpy.ndarray
        Capital requirements with the broadcast shape of the inputs. If an
        input is a Series, the result is a Series with its index, named
        ``capital_requirement``; scalar inputs give a float.

    Raises
    ------
    ValueError
        If an input is not real and numeric, the inputs cannot be broadcast
        together or have different Series indices, `probability` is outside
        the open interval (0, 1), or is at or below about 2.927e-6 for
        ``"corporate"`` or ``"hvcre"``, or `lgd` is outside [0, 1]. Also if
        `maturity` is given for a retail class, missing, not positive or not
        finite, `correlation` is missing, outside [0, 1) or given together
        with `turnover` or `financial`, `level` is outside (0, 1), `scaling`
        is not positive and finite, or an argument is invalid as in
        `asset_correlation`.

    See Also
    --------
    asset_correlation : The correlation of each asset class.
    conditional_pd : The PD at the level of the one-factor model.
    kivi.strategy.expected_loss : The expected loss PD x LGD x EAD that K leaves out.

    Notes
    -----
    With the standard normal distribution function :math:`\Phi`, its
    quantile function :math:`\Phi^{-1}`, the correlation :math:`R` of
    `asset_correlation` and the level :math:`q`,

    .. math::

       K = \mathrm{LGD} \left[ \Phi\!\left(
       \frac{\Phi^{-1}(\mathrm{PD}) + \sqrt{R}\,\Phi^{-1}(q)}{\sqrt{1 - R}}
       \right) - \mathrm{PD} \right] \frac{1 + (M - 2.5)\,b}{1 - 1.5\,b},
       \qquad b = (0.11852 - 0.05478 \ln \mathrm{PD})^2,

    where the maturity factor applies to ``"corporate"`` and ``"hvcre"``
    only (Basel CRE31.5–31.16; China, annex 6). Then K is multiplied by
    `scaling`.

    The maturity factor divides by :math:`1 - 1.5\,b`, which reaches 0 at a
    PD of about 2.927e-6, so these PDs raise an error; well before that,
    below a PD of about 1e-5, the factor grows so fast that the risk weight
    rises again as the PD falls. The PD floors of 0.05% (Basel CRE32.4,
    China article 91) stay above both; only sovereign exposures have none.

    A defaulted exposure (PD = 1) raises an error too: its requirement is
    :math:`\max(0, \mathrm{LGD} - \mathrm{ELBE})` with the best estimate of
    its expected loss (Basel CRE31.3; China, annex 6, part II).

    Examples
    --------
    >>> from kivi.capital import capital_requirement
    >>> k = capital_requirement(0.01, 0.45, asset_class="corporate")
    >>> print(f"K = {k:.6f}, risk weight = {12.5 * k:.4%}")
    K = 0.073853, risk weight = 92.3168%
    >>> rw = 12.5 * capital_requirement([0.01, 0.01], [0.45, 0.25], asset_class="other_retail")
    >>> rw.round(4)
    array([0.4577, 0.2543])
    """
    asset_class = _checked_asset_class(asset_class)
    retail = asset_class in _RETAIL
    if retail and maturity is not None:
        raise ValueError(
            f"maturity applies only to the asset classes 'corporate' and 'hvcre', got asset_class "
            f"{asset_class!r}"
        )
    lower, upper = _checked_adjustment(asset_class, turnover, turnover_range, financial)
    if correlation is not None and (turnover is not None or financial):
        raise ValueError(
            "turnover and financial adjust the correlation of the formula and cannot be given "
            "together with correlation"
        )
    confidence = _checked_level(level)
    factor = _checked_scaling(scaling)
    inputs: dict[str, npt.ArrayLike] = {"probability": probability, "lgd": lgd}
    for name, value in (
        ("maturity", maturity),
        ("turnover", turnover),
        ("correlation", correlation),
    ):
        if value is not None:
            inputs[name] = value
    arrays, template = _broadcast(inputs)
    values = arrays["probability"]
    if (values == 1).any():
        raise ValueError(
            "probability must be below 1, got 1.0: the capital requirement of a defaulted "
            "exposure is max(0, lgd - elbe), with the best estimate elbe of its expected loss"
        )
    _check_open_unit(values, "probability")
    _check_unit(arrays["lgd"], "lgd")
    if turnover is not None:
        _check_turnover(arrays["turnover"], upper, financial)
    if correlation is None:
        r = _correlation(values, asset_class, arrays.get("turnover"), lower, upper, financial)
    else:
        r = arrays["correlation"]
        _check_correlation(r, "correlation", positive=False)
    quantile = _quantile(values)
    shift = np.sqrt(r) * _stats.normal_quantile(confidence)
    requirement = arrays["lgd"] * (_stats.normal_cdf((quantile + shift) / np.sqrt(1 - r)) - values)
    if not retail:
        requirement = requirement * _maturity_factor(values, arrays.get("maturity"), asset_class)
    return _shaped(template, requirement * factor, "capital_requirement")


@typing.overload
def conditional_pd(
    probability: pd.Series,
    *,
    correlation: npt.ArrayLike,
    factor: npt.ArrayLike | None = ...,
    level: npt.ArrayLike | None = ...,
) -> pd.Series: ...
@typing.overload
def conditional_pd(
    probability: npt.ArrayLike,
    *,
    correlation: pd.Series,
    factor: npt.ArrayLike | None = ...,
    level: npt.ArrayLike | None = ...,
) -> pd.Series: ...
@typing.overload
def conditional_pd(
    probability: npt.ArrayLike,
    *,
    correlation: npt.ArrayLike,
    factor: pd.Series,
    level: npt.ArrayLike | None = ...,
) -> pd.Series: ...
@typing.overload
def conditional_pd(
    probability: npt.ArrayLike,
    *,
    correlation: npt.ArrayLike,
    factor: npt.ArrayLike | None = ...,
    level: pd.Series,
) -> pd.Series: ...
@typing.overload
def conditional_pd(
    probability: float,
    *,
    correlation: float,
    factor: float | None = ...,
    level: float | None = ...,
) -> float: ...
@typing.overload
def conditional_pd(
    probability: _validation.ListLike,
    *,
    correlation: float | _validation.ListLike,
    factor: float | _validation.ListLike | None = ...,
    level: float | _validation.ListLike | None = ...,
) -> npt.NDArray[np.float64]: ...
@typing.overload
def conditional_pd(
    probability: npt.ArrayLike,
    *,
    correlation: npt.ArrayLike,
    factor: npt.ArrayLike | None = ...,
    level: npt.ArrayLike | None = ...,
) -> float | pd.Series | npt.NDArray[np.float64]: ...
def conditional_pd(
    probability: npt.ArrayLike,
    *,
    correlation: npt.ArrayLike,
    factor: npt.ArrayLike | None = None,
    level: npt.ArrayLike | None = None,
) -> float | pd.Series | npt.NDArray[np.float64]:
    r"""Compute the PD of the one-factor model given the state of the economy.

    In the one-factor model of Vasicek, behind the IRB risk-weight functions,
    a borrower defaults when its assets fall below a threshold set by its
    long-run PD, and the assets of all borrowers share one systematic
    factor. Given that factor, defaults are independent and occur with the
    conditional PD. It turns a through-the-cycle PD into a point-in-time PD
    for a scenario of the economy, and gives the stressed PD of
    `capital_requirement` at its level.

    Parameters
    ----------
    probability : array_like
        Long-run (unconditional) probabilities of default, strictly between
        0 and 1. Missing values give missing PDs.
    correlation : array_like
        Asset correlations from 0 to 1 (exclusive), such as those of
        `asset_correlation`. Must not be missing.
    factor : array_like, optional
        Values of the standard normal systematic factor, finite; positive
        values stand for a better economy than on average and give PDs below
        the long-run PD. Missing values give missing PDs.
    level : array_like, optional
        Instead of `factor`, the probability strictly between 0 and 1 that
        the economy is better than the scenario: the factor is the lower
        ``1 - level`` quantile, ``-Phi^-1(level)``. Missing values give
        missing PDs. Exactly one of `factor` and `level` must be given.

    Returns
    -------
    float, pandas.Series or numpy.ndarray
        Conditional PDs with the broadcast shape of the inputs. If an input
        is a Series, the result is a Series with its index, named
        ``conditional_pd``; scalar inputs give a float.

    Raises
    ------
    ValueError
        If not exactly one of `factor` and `level` is given, an input is not
        real and numeric, the inputs cannot be broadcast together or have
        different Series indices, `probability` or `level` is outside the
        open interval (0, 1), `correlation` is missing or outside [0, 1), or
        `factor` is infinite.

    See Also
    --------
    systematic_factor : The factor that a default rate implies, the inverse of this function.
    vasicek_fit : Estimate the long-run PD and the correlation from default rates.
    capital_requirement : The capital requirement, which uses the PD at the level 0.999.

    Notes
    -----
    With the standard normal distribution function :math:`\Phi`, the
    correlation :math:`\rho` and the factor :math:`z`,

    .. math::

       \mathrm{PD}(z) = \Phi\!\left(
       \frac{\Phi^{-1}(\mathrm{PD}) - \sqrt{\rho}\,z}{\sqrt{1 - \rho}} \right),

    and `level` :math:`q` stands for :math:`z = -\Phi^{-1}(q)`, so that
    ``level=0.999`` gives the PD of the IRB formula. Averaged over the
    standard normal :math:`z`, the conditional PD is the long-run PD.

    Examples
    --------
    >>> from kivi.capital import conditional_pd
    >>> conditional_pd(0.01, correlation=0.12, factor=[2.0, 0.0, -2.0]).round(5)
    array([0.00064, 0.00657, 0.04081])
    >>> print(f"{conditional_pd(0.01, correlation=0.12, level=0.999):.4%}")
    9.0326%
    """
    if (factor is None) == (level is None):
        raise ValueError("give exactly one of factor and level")
    inputs: dict[str, npt.ArrayLike] = {"probability": probability, "correlation": correlation}
    if factor is not None:
        inputs["factor"] = factor
    else:
        inputs["level"] = typing.cast(npt.ArrayLike, level)
    arrays, template = _broadcast(inputs)
    values, rho = arrays["probability"], arrays["correlation"]
    _check_open_unit(values, "probability")
    _check_correlation(rho, "correlation", positive=False)
    if factor is not None:
        z = arrays["factor"]
        if np.isinf(z).any():
            raise ValueError(f"factor must be finite, got {_validation.shown(z[np.isinf(z)][0])}")
    else:
        _check_open_unit(arrays["level"], "level")
        z = -_quantile(arrays["level"])
    probabilities = _stats.normal_cdf((_quantile(values) - np.sqrt(rho) * z) / np.sqrt(1 - rho))
    return _shaped(template, probabilities, "conditional_pd")


@typing.overload
def systematic_factor(
    default_rate: pd.Series, probability: npt.ArrayLike, *, correlation: npt.ArrayLike
) -> pd.Series: ...
@typing.overload
def systematic_factor(
    default_rate: npt.ArrayLike, probability: pd.Series, *, correlation: npt.ArrayLike
) -> pd.Series: ...
@typing.overload
def systematic_factor(
    default_rate: npt.ArrayLike, probability: npt.ArrayLike, *, correlation: pd.Series
) -> pd.Series: ...
@typing.overload
def systematic_factor(default_rate: float, probability: float, *, correlation: float) -> float: ...
@typing.overload
def systematic_factor(
    default_rate: _validation.ListLike,
    probability: float | _validation.ListLike,
    *,
    correlation: float | _validation.ListLike,
) -> npt.NDArray[np.float64]: ...
@typing.overload
def systematic_factor(
    default_rate: npt.ArrayLike, probability: npt.ArrayLike, *, correlation: npt.ArrayLike
) -> float | pd.Series | npt.NDArray[np.float64]: ...
def systematic_factor(
    default_rate: npt.ArrayLike, probability: npt.ArrayLike, *, correlation: npt.ArrayLike
) -> float | pd.Series | npt.NDArray[np.float64]:
    r"""Compute the systematic factor that an observed default rate implies.

    This inverts `conditional_pd`: it gives the value of the systematic
    factor under which the conditional PD equals the default rate, such as
    the default rate of a portfolio in each year. A negative factor marks a
    year worse than on average.

    Parameters
    ----------
    default_rate : array_like
        Observed default rates, strictly between 0 and 1. Missing values
        give missing factors.
    probability : array_like
        Long-run probabilities of default, strictly between 0 and 1, such as
        the first result of `vasicek_fit`. Missing values give missing
        factors.
    correlation : array_like
        Asset correlations strictly between 0 and 1, such as the second
        result of `vasicek_fit`. Must not be missing.

    Returns
    -------
    float, pandas.Series or numpy.ndarray
        Factors with the broadcast shape of the inputs. If an input is a
        Series, the result is a Series with its index, named ``factor``;
        scalar inputs give a float.

    Raises
    ------
    ValueError
        If an input is not real and numeric, the inputs cannot be broadcast
        together or have different Series indices, `default_rate` or
        `probability` is outside the open interval (0, 1), or `correlation`
        is missing or outside (0, 1). A default rate of 0 or 1 has no finite
        factor.

    See Also
    --------
    conditional_pd : The PD given the factor, the inverse of this function.
    vasicek_fit : Estimate the long-run PD and the correlation from default rates.

    Notes
    -----
    With the default rate :math:`\mathrm{DR}`,

    .. math::

       z = \frac{\Phi^{-1}(\mathrm{PD}) - \sqrt{1 - \rho}\,\Phi^{-1}(\mathrm{DR})}{\sqrt{\rho}}.

    Examples
    --------
    >>> from kivi.capital import conditional_pd, systematic_factor
    >>> systematic_factor([0.004, 0.01, 0.03], 0.01, correlation=0.12).round(4)
    array([ 0.4663, -0.4158, -1.6224])
    >>> print(f"{conditional_pd(0.01, correlation=0.12, factor=-1.6224):.4f}")
    0.0300
    """
    inputs = {"default_rate": default_rate, "probability": probability, "correlation": correlation}
    arrays, template = _broadcast(inputs)
    rates, values, rho = arrays["default_rate"], arrays["probability"], arrays["correlation"]
    _check_open_unit(rates, "default_rate")
    _check_open_unit(values, "probability")
    _check_correlation(rho, "correlation", positive=True)
    factors = (_quantile(values) - np.sqrt(1 - rho) * _quantile(rates)) / np.sqrt(rho)
    return _shaped(template, factors, "factor")


def vasicek_fit(default_rate: npt.ArrayLike) -> tuple[float, float]:
    r"""Estimate the long-run PD and the asset correlation from a series of default rates.

    In the one-factor model, the default rate of a large portfolio varies
    from year to year with the systematic factor, and its variance grows
    with the asset correlation. The method of moments matches the mean and
    the variance of the observed default rates. Default rates of 0 count,
    as in years without defaults.

    Parameters
    ----------
    default_rate : array_like
        Default rates of a portfolio in successive periods, such as years,
        from 0 to 1, at least two and without missing values.

    Returns
    -------
    probability : float
        The long-run PD, the mean of the default rates.
    correlation : float
        The asset correlation, from 0 to 1. A constant series gives 0.

    Raises
    ------
    ValueError
        If `default_rate` is not real and numeric, not one-dimensional, has
        fewer than two values, missing values or values outside [0, 1], its
        mean is 0 or 1, or its variance is at least ``mean * (1 - mean)``,
        the variance of a correlation of 1.

    See Also
    --------
    conditional_pd : The PD given the factor, with the fitted parameters.
    systematic_factor : The factor of each period, with the fitted parameters.

    Notes
    -----
    With the mean :math:`m` and the variance :math:`s^2` of the default
    rates (with ``ddof=1``), the correlation :math:`\rho` solves

    .. math::

       \Phi_2\!\left(\Phi^{-1}(m), \Phi^{-1}(m); \rho\right) - m^2 = s^2,

    where :math:`\Phi_2` is the bivariate standard normal distribution
    function: two borrowers of the portfolio default together with
    probability :math:`\Phi_2`. Its left side rises with :math:`\rho`, from 0
    to :math:`m (1 - m)`, and is solved by bisection. The moments describe a
    portfolio large enough for its default rate to equal the conditional PD,
    so the binomial noise of a small portfolio raises the estimate.

    Examples
    --------
    >>> from kivi.capital import vasicek_fit
    >>> probability, correlation = vasicek_fit([0.004, 0.012, 0.0, 0.031, 0.008, 0.015])
    >>> print(f"PD = {probability:.4f}, correlation = {correlation:.4f}")
    PD = 0.0117, correlation = 0.0994
    """
    rates = _validation.as_real(default_rate, "default_rate")
    if rates.ndim != 1:
        raise ValueError(f"default_rate must be one-dimensional, got {rates.ndim} dimensions")
    if len(rates) < 2:
        raise ValueError(f"default_rate must have at least two values, got {len(rates)}")
    missing = np.isnan(rates)
    if missing.any():
        raise ValueError(f"default_rate must not have missing values, got {missing.sum()}")
    _check_unit(rates, "default_rate")
    mean = float(rates.mean())
    if not 0 < mean < 1:
        raise ValueError(f"default_rate must have a mean strictly between 0 and 1, got {mean!r}")
    variance = float(rates.var(ddof=1))
    if variance >= mean * (1 - mean):
        raise ValueError(
            f"default_rate must have a variance below mean * (1 - mean) = {mean * (1 - mean)!r}, "
            f"the variance of a correlation of 1, got {variance!r}"
        )
    if variance == 0:
        return mean, 0.0
    threshold = float(_stats.normal_quantile(mean))
    # Phi_2 at rho = 0 is exactly the square of the margin that bivariate_normal_cdf adds to.
    square = float(_stats.normal_cdf(threshold)) ** 2
    # Bisection ends when no number lies between the two ends.
    low, middle, high = 0.0, 0.5, 1.0
    while low < middle < high:
        joint = float(_stats.bivariate_normal_cdf(threshold, threshold, middle))
        if joint - square < variance:
            low = middle
        else:
            high = middle
        middle = (low + high) / 2
    return mean, middle


@typing.overload
def most_prudent_pd(
    event: pd.Series, total: npt.ArrayLike, *, level: float = ..., correlation: float = ...
) -> pd.Series: ...
@typing.overload
def most_prudent_pd(
    event: npt.ArrayLike, total: pd.Series, *, level: float = ..., correlation: float = ...
) -> pd.Series: ...
@typing.overload
def most_prudent_pd(
    event: _validation.ListLike,
    total: _validation.ListLike,
    *,
    level: float = ...,
    correlation: float = ...,
) -> npt.NDArray[np.float64]: ...
@typing.overload
def most_prudent_pd(
    event: npt.ArrayLike, total: npt.ArrayLike, *, level: float = ..., correlation: float = ...
) -> pd.Series | npt.NDArray[np.float64]: ...
def most_prudent_pd(
    event: npt.ArrayLike, total: npt.ArrayLike, *, level: float = 0.95, correlation: float = 0.0
) -> pd.Series | npt.NDArray[np.float64]:
    r"""Compute the most prudent upper bounds of Pluto and Tasche for the PD of each grade.

    A portfolio with few or no defaults, such as banks or large corporates,
    gives default rates too small to calibrate a PD on. The most prudent
    estimate of a grade is the largest PD that the defaults observed in it
    and in all riskier grades do not reject at the confidence `level`,
    assuming that the PDs do not fall from one grade to the next riskier one.

    Parameters
    ----------
    event : array_like
        Number of defaults in each grade, a non-negative whole number, with
        the grades from the least to the most risky. Must not be missing.
    total : array_like
        Number of obligors in each grade, a whole number not below `event`,
        matched by position or, for two Series, by identical indices.
    level : float, default 0.95
        The confidence level, strictly between 0 and 1.
    correlation : float, default 0.0
        The asset correlation of the one-factor model, from 0 to 1
        (exclusive). With 0, defaults are independent; a positive
        correlation makes the same defaults compatible with higher PDs.

    Returns
    -------
    pandas.Series or numpy.ndarray
        The upper bound of the PD of each grade. If an input is a Series, the
        result is a Series with its index, named ``most_prudent_pd``.

    Raises
    ------
    ValueError
        If `event` or `total` is not real and numeric, not one-dimensional,
        empty or of different lengths, their Series indices differ, a count
        is missing or not a non-negative whole number, `event` exceeds
        `total`, `level` is outside (0, 1) or `correlation` is outside
        [0, 1).

    See Also
    --------
    kivi.calibration.binomial_test : Test the PD of each grade against its default count.
    conditional_pd : The PD of the one-factor model given the factor.

    Notes
    -----
    Grade :math:`k` pools the obligors :math:`n_k = \sum_{i \ge k} N_i` and
    the defaults :math:`d_k = \sum_{i \ge k} D_i` of itself and all riskier
    grades. Its bound :math:`p_k` is the largest PD for which
    :math:`P(D \le d_k) \ge 1 - q` at the level :math:`q`. With independent
    defaults, :math:`D` is binomial with :math:`n_k` and :math:`p_k`, so
    :math:`p_k` is the :math:`q` quantile of the beta distribution with
    :math:`d_k + 1` and :math:`n_k - d_k`, and without defaults
    :math:`p_k = 1 - (1 - q)^{1/n_k}`. With a correlation :math:`\rho`, the
    defaults are binomial given the factor,

    .. math::

       P(D \le d_k) = \int P\left(\mathrm{Bin}(n_k, \mathrm{PD}(z)) \le d_k\right)
       \varphi(z) \, dz,

    with the conditional PD of `conditional_pd`, integrated by composite
    Gauss-Legendre quadrature over the factor from -9 to 9 (64 panels of 8
    nodes) and solved by bisection. A grade whose pooled obligors all
    defaulted, or which pools none, gets 1. The bounds are those of one
    period; with factors correlated over several years they need simulation,
    which kivi leaves out. Pluto and Tasche [1]_ compute the same bounds for
    their examples, with a correlation of 0.12 for the correlated one.

    References
    ----------
    .. [1] K. Pluto and D. Tasche, "Estimating probabilities of default for
       low default portfolios", arXiv:cond-mat/0411699, 2004; also
       "Thinking positively", Risk 18 (8), 2005, pp. 72-78.

    Examples
    --------
    >>> from kivi.capital import most_prudent_pd
    >>> (100 * most_prudent_pd([0, 0, 0], [100, 400, 300], level=0.9)).round(2)
    array([0.29, 0.33, 0.76])
    """
    _validation.shared_index({"event": event, "total": total})
    events, totals = _counts(event, "event"), _counts(total, "total")
    if len(events) != len(totals):
        raise ValueError(
            f"event and total must have the same length, got {len(events)} and {len(totals)}"
        )
    invalid = events > totals
    if invalid.any():
        raise ValueError(
            f"event must not exceed total, got {int(events[invalid][0])} events among "
            f"{int(totals[invalid][0])}"
        )
    confidence = _checked_level(level)
    rho = _checked_scalar_correlation(correlation)
    pooled_total = np.cumsum(totals[::-1])[::-1]
    pooled_event = np.cumsum(events[::-1])[::-1]
    bound = np.ones(len(events))
    some = pooled_event < pooled_total
    if some.any() and rho == 0:
        bound[some] = _stats.beta_quantile(
            pooled_event[some] + 1, pooled_total[some] - pooled_event[some], confidence
        )
    elif some.any():
        bound[some] = _correlated_bound(pooled_event[some], pooled_total[some], confidence, rho)
    series = [values for values in (event, total) if isinstance(values, pd.Series)]
    if series:
        return pd.Series(bound, index=series[0].index, name="most_prudent_pd")
    return bound


def _correlated_bound(
    event: npt.NDArray[np.float64], total: npt.NDArray[np.float64], level: float, rho: float
) -> npt.NDArray[np.float64]:
    """Solve P(D <= event) = 1 - level for the PD of the one-factor model, by bisection."""
    shift = math.sqrt(rho) * _FACTOR_NODES
    scale = math.sqrt(1 - rho)
    # Bisect the normal quantile of the PD, which conditional_pd shifts by the factor.
    # Bisection ends when no number lies between the two ends.
    low = np.full(len(event), -40.0)
    high = np.full(len(event), 40.0)
    middle = (low + high) / 2
    while ((low < middle) & (middle < high)).any():
        stuck = (middle == low) | (middle == high)
        conditional = _stats.normal_cdf((middle[:, None] - shift) / scale)
        # P(Bin(n, p) <= d) = 1 - I_p(d + 1, n - d) at each node.
        within = 1 - _stats.betainc((event + 1)[:, None], (total - event)[:, None], conditional)
        # A sum over the nodes, not a matrix product, so that the result is the same on every CPU.
        compatible = (within * _FACTOR_WEIGHTS).sum(axis=1) > 1 - level
        low = np.where(compatible & ~stuck, middle, low)
        high = np.where(~compatible & ~stuck, middle, high)
        middle = (low + high) / 2
    return _stats.normal_cdf(middle)


def _quantile(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the standard normal quantile of each value, with NaN for a missing one."""
    # A zero-dimensional array would take the scalar path, which raises for a missing value.
    quantiles = _stats.normal_quantile(np.atleast_1d(values))
    return np.asarray(quantiles, dtype=float).reshape(values.shape)


def _checked_asset_class(asset_class: object) -> _AssetClass:
    """Return the asset class, or raise a ValueError that lists the five classes."""
    if not isinstance(asset_class, str) or asset_class not in _ASSET_CLASSES:
        names = ", ".join(repr(name) for name in _ASSET_CLASSES)
        raise ValueError(
            f"asset_class must be one of {names}, got {_validation.shown_argument(asset_class)}"
        )
    return typing.cast(_AssetClass, asset_class)


def _checked_adjustment(
    asset_class: str, turnover: object, turnover_range: object, financial: object
) -> tuple[float, float]:
    """Check the corporate adjustments and return the bounds of the turnover range."""
    if not isinstance(financial, (bool, np.bool_)):
        raise ValueError(
            f"financial must be True or False, got {_validation.shown_argument(financial)}"
        )
    for name, given in (("turnover", turnover is not None), ("financial", bool(financial))):
        if given and asset_class != "corporate":
            raise ValueError(
                f"{name} applies only to asset_class 'corporate', got asset_class {asset_class!r}"
            )
    bounds = turnover_range if isinstance(turnover_range, abc.Sequence) else ()
    valid = (
        len(bounds) == 2
        and all(
            _validation.is_real(bound)
            and not isinstance(bound, (bool, np.bool_))
            and math.isfinite(bound)
            for bound in bounds
        )
        and 0 <= bounds[0] < bounds[1]
    )
    if not valid:
        raise ValueError(
            "turnover_range must be two finite numbers lower and upper with 0 <= lower < upper, "
            f"got {_validation.shown_argument(turnover_range)}"
        )
    return float(bounds[0]), float(bounds[1])


def _check_turnover(turnover: npt.NDArray[np.float64], upper: float, financial: bool) -> None:
    """Require turnovers that are not missing and not negative, and none below upper with FIs."""
    missing = np.isnan(turnover)
    if missing.any():
        raise ValueError(
            f"turnover must not have missing values, got {missing.sum()}; pass numpy.inf for "
            "borrowers without the turnover adjustment"
        )
    if (turnover < 0).any():
        raise ValueError(
            f"turnover must not be negative, got {_validation.shown(turnover[turnover < 0][0])}"
        )
    if financial and (turnover < upper).any():
        raise ValueError(
            "financial=True must not come with a turnover below the upper bound "
            f"{upper!r} of turnover_range, got {_validation.shown(turnover[turnover < upper][0])}: "
            "the formula for financial institutions has no turnover adjustment"
        )


def _correlation(
    probability: npt.NDArray[np.float64],
    asset_class: str,
    turnover: npt.NDArray[np.float64] | None,
    lower: float,
    upper: float,
    financial: bool,
) -> npt.NDArray[np.float64]:
    """Return the correlation of the risk-weight function of an asset class."""
    if asset_class == "mortgage":
        return np.where(np.isnan(probability), np.nan, 0.15)
    if asset_class == "revolving":
        return np.where(np.isnan(probability), np.nan, 0.04)
    if asset_class == "other_retail":
        # (1 - exp(-35 PD)) / (1 - exp(-35)), without cancellation for a small PD.
        weight = np.expm1(-35 * probability) / math.expm1(-35)
        return 0.03 * weight + 0.16 * (1 - weight)
    weight = np.expm1(-50 * probability) / math.expm1(-50)
    correlation = 0.12 * weight + (0.24 if asset_class == "corporate" else 0.30) * (1 - weight)
    if turnover is not None:
        size = np.clip(turnover, lower, upper)
        correlation = correlation - 0.04 * (1 - (size - lower) / (upper - lower))
    return 1.25 * correlation if financial else correlation


def _maturity_factor(
    probability: npt.NDArray[np.float64],
    maturity: npt.NDArray[np.float64] | None,
    asset_class: str,
) -> npt.NDArray[np.float64]:
    """Return (1 + (M - 2.5) b) / (1 - 1.5 b), after checking the maturity and the PD."""
    if maturity is None:
        maturity = np.full(probability.shape, 2.5)
    if np.isnan(maturity).any():
        raise ValueError(
            f"maturity must not have missing values, got {np.isnan(maturity).sum()}; fill them "
            "with the maturity that the rules prescribe, such as 2.5 years"
        )
    invalid = ~(np.isfinite(maturity) & (maturity > 0))
    if invalid.any():
        raise ValueError(
            f"maturity must be positive and finite, got {_validation.shown(maturity[invalid][0])}"
        )
    slope = (0.11852 - 0.05478 * np.log(probability)) ** 2
    denominator = 1 - 1.5 * slope
    invalid = denominator <= 0
    if invalid.any():
        raise ValueError(
            f"probability must be above {_SMALLEST_PD:.4g} for asset_class {asset_class!r}, where "
            "the maturity adjustment divides by 1 - 1.5 b <= 0, got "
            f"{_validation.shown(probability[invalid][0])}; apply the PD floor first, such as "
            "numpy.maximum(probability, 0.0005)"
        )
    return (1 + (maturity - 2.5) * slope) / denominator


def _broadcast(
    inputs: abc.Mapping[str, npt.ArrayLike],
) -> tuple[dict[str, npt.NDArray[np.float64]], npt.ArrayLike]:
    """Broadcast value-by-value inputs; also return the input whose shape the result takes.

    Booleans count as 0 and 1 in probabilities, as elsewhere in kivi, and are rejected
    otherwise. A Series among the inputs sets the index of the result.
    """
    _validation.shared_index(inputs)
    arrays = {
        name: _validation.as_real(values, name, booleans=name == "probability")
        for name, values in inputs.items()
    }
    try:
        shape = np.broadcast_shapes(*(array.shape for array in arrays.values()))
    except ValueError:
        shapes = ", ".join(f"{name} {array.shape}" for name, array in arrays.items())
        names = ", ".join(arrays)
        raise ValueError(f"{names} cannot be broadcast together: {shapes}") from None
    series = [values for values in inputs.values() if isinstance(values, pd.Series)]
    template = series[0] if series else next(iter(inputs.values()))
    if series and shape != (len(series[0]),):
        raise ValueError(
            f"{', '.join(arrays)} must broadcast to the length {len(series[0])} of their Series "
            f"index, got shape {shape}"
        )
    return {name: np.broadcast_to(array, shape) for name, array in arrays.items()}, template


def _shaped(
    template: npt.ArrayLike, values: npt.NDArray[np.float64], name: str
) -> float | pd.Series | npt.NDArray[np.float64]:
    """Return values shaped like the input, with a Series named after the result."""
    result = _validation.like_input(template, np.asarray(values, dtype=float))
    if isinstance(result, pd.Series):
        result.name = name
    return result


def _check_unit(values: npt.NDArray[np.float64], name: str) -> None:
    """Require values from 0 to 1; missing values pass."""
    outside = (values < 0) | (values > 1)
    if outside.any():
        raise ValueError(
            f"{name} must be between 0 and 1, got {_validation.shown(values[outside][0])}"
        )


def _check_open_unit(values: npt.NDArray[np.float64], name: str) -> None:
    """Require values strictly between 0 and 1; missing values pass."""
    outside = (values <= 0) | (values >= 1)
    if outside.any():
        raise ValueError(
            f"{name} must be strictly between 0 and 1, got {_validation.shown(values[outside][0])}"
        )


def _check_correlation(values: npt.NDArray[np.float64], name: str, *, positive: bool) -> None:
    """Require correlations from 0, or above 0 when positive, to below 1, without missing ones."""
    missing = np.isnan(values)
    if missing.any():
        raise ValueError(f"{name} must not have missing values, got {missing.sum()}")
    outside = ((values <= 0) if positive else (values < 0)) | (values >= 1)
    if outside.any():
        interval = "strictly between 0 and 1" if positive else "from 0 to below 1"
        raise ValueError(f"{name} must be {interval}, got {_validation.shown(values[outside][0])}")


def _checked_level(level: object) -> float:
    """Return a confidence level strictly between 0 and 1 as a float."""
    if (
        isinstance(level, (bool, np.bool_))
        or not _validation.is_real(level)
        or not 0 < float(level) < 1
    ):
        raise ValueError(
            f"level must be a number strictly between 0 and 1, got "
            f"{_validation.shown_argument(level)}"
        )
    return float(level)


def _checked_scaling(scaling: object) -> float:
    """Return a positive and finite scaling factor as a float."""
    if (
        isinstance(scaling, (bool, np.bool_))
        or not _validation.is_real(scaling)
        or not (math.isfinite(scaling) and scaling > 0)
    ):
        raise ValueError(
            f"scaling must be positive and finite, got {_validation.shown_argument(scaling)}"
        )
    return float(scaling)


def _checked_scalar_correlation(correlation: object) -> float:
    """Return a correlation from 0 to below 1 as a float."""
    if (
        isinstance(correlation, (bool, np.bool_))
        or not _validation.is_real(correlation)
        or not 0 <= float(correlation) < 1
    ):
        raise ValueError(
            "correlation must be a number from 0 to below 1, got "
            f"{_validation.shown_argument(correlation)}"
        )
    return float(correlation)


def _counts(values: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Return non-negative whole counts of a one-dimensional, non-empty input as floats."""
    counts = _validation.as_real(values, name)
    if counts.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {counts.ndim} dimensions")
    if not len(counts):
        raise ValueError(f"{name} must not be empty")
    missing = np.isnan(counts)
    if missing.any():
        raise ValueError(f"{name} must not have missing values, got {missing.sum()}")
    invalid = ~np.isfinite(counts) | (counts < 0) | (counts != np.round(counts))
    if invalid.any():
        raise ValueError(
            f"{name} must be a non-negative whole number, got "
            f"{_validation.shown(counts[invalid][0])}"
        )
    return counts
