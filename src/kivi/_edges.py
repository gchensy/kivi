"""Edge fitting shared by ``kivi.binning`` and the functions that band scores.

``fit_edges`` fits the edges of ``kivi.binning.bin_edges``. ``band_edges``
gives the bands of ``kivi.metrics.gains_table``, ``kivi.monitoring`` and
``kivi.reject_inference``: quantile edges for an integer, or explicit edges.
"""

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import _special, _validation


def fit_edges(
    values: npt.ArrayLike,
    bins: int,
    *,
    method: str = "quantile",
    special: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
    advice: str = "consider passing it in special",
    stacklevel: int = 3,
) -> npt.NDArray[np.float64]:
    """Fit the edges of `kivi.binning.bin_edges`.

    `advice` ends the warning about tied edges. Callers without a `special`
    argument pass their own, and a `stacklevel` that points the warning at
    the line that called them.
    """
    _validation.check_count(bins, "bins")
    if method not in ("quantile", "uniform"):
        raise ValueError(f"method must be 'quantile' or 'uniform', got {_validation.shown(method)}")
    fitted, mass = observed(
        _validation.as_series(values, "values"),
        _special.special_values(special),
        _validation.as_weights(weights, values),
    )
    if method == "quantile":
        probabilities = np.linspace(0, 1, bins + 1)
        if mass is None:
            edges = np.quantile(fitted, probabilities)
        else:
            edges = weighted_quantile(fitted, mass, probabilities)
    else:
        edges = np.linspace(fitted.min(), fitted.max(), bins + 1)
    edges = np.r_[-np.inf, np.unique(edges)[1:-1], np.inf]
    if len(edges) - 1 < bins:
        levels, inverse = np.unique(fitted, return_inverse=True)
        totals = np.bincount(inverse, weights=mass)
        top = totals.argmax()
        warnings.warn(
            f"tied edges leave {len(edges) - 1} of the {bins} requested bins; the value "
            f"{levels[top]:g} makes up {totals[top] / totals.sum():.0%} of the fitted values, "
            f"{advice}",
            UserWarning,
            stacklevel=stacklevel,
        )
    return edges


def band_edges(
    values: np.ndarray, bins: int | npt.ArrayLike, weights: np.ndarray | None
) -> npt.NDArray[np.float64]:
    """Increasing band edges: quantile edges for an integer ``bins``, else ``bins`` itself."""
    message = "bins must be a positive integer or at least two increasing edges, got {}"
    if _validation.is_integral(bins) and not isinstance(bins, bool):
        if bins < 1:
            raise ValueError(message.format(_validation.shown_argument(bins)))
        # The edges of bin_edges; the warning for tied quantiles points at the caller of
        # gains_table and gives advice that applies to it.
        return fit_edges(
            values,
            int(bins),
            weights=weights,
            advice="pass fewer bins or explicit edges",
            stacklevel=4,
        )
    try:
        edges = np.asarray(bins, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(message.format(_validation.shown_argument(bins))) from None
    if edges.ndim != 1 or len(edges) < 2 or not (np.diff(edges) > 0).all():
        raise ValueError(message.format(_validation.shown_argument(bins)))
    return edges


def observed(
    values: pd.Series,
    special: pd.Index | None = None,
    weights: np.ndarray | None = None,
    *,
    is_special: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Finite numeric values and their weights, without missing, special or zero-weight values.

    ``is_special`` is the special mask when the caller has already computed it.
    """
    keep = values.notna().to_numpy(dtype=bool, copy=True)
    excluded = ["missing"]
    if special is not None:
        if is_special is None:
            is_special = _special.special_positions(values, special) >= 0
        keep &= ~is_special
        excluded.append("special")
    if weights is not None:
        keep &= weights > 0
        excluded.append("zero-weight")
    fitted = pd.to_numeric(values[keep], errors="raise").to_numpy(dtype=float)
    if not len(fitted):
        names = (
            ", ".join(excluded[:-1]) + " and " + excluded[-1] if len(excluded) > 1 else "missing"
        )
        raise ValueError(f"reference values have no finite observations apart from {names} values")
    if not np.isfinite(fitted).all():
        hint = "; pass infinite codes in special" if special is not None else ""
        raise ValueError(f"reference values must contain finite observations only{hint}")
    return fitted, None if weights is None else weights[keep]


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """Weighted quantiles by the inverted CDF, as NumPy's ``method="inverted_cdf"``.

    A probability within a relative 1e-10 of a step of the CDF selects that
    step, so rounding in ``probabilities`` (``np.linspace(0, 1, 6)[3]`` is
    0.6000000000000001) neither skips a value nor depends on the weight scale.
    """
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    targets = (np.asarray(probabilities) - 1e-10) * cumulative[-1]
    index = np.searchsorted(cumulative, targets, side="left")
    return values[order][np.minimum(index, len(values) - 1)]
