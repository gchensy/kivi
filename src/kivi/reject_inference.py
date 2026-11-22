"""Reject inference: outcomes for rejected applicants, to develop a model on all applicants.

A scorecard developed only on approved applicants learns from the outcomes
that the previous approval policy let it observe. Each function here
supplies what is missing for the rejected applicants, as inferred rows or as
weights, in a form that the ``weights`` arguments of kivi accept:

- `fuzzy_augmentation` splits each reject into a bad and a good row, weighted
  by its PD;
- `parceling` labels the rejects of each score band at the band's bad rate;
- `reweighting` weights the accepted applicants to stand in for the rejects
  of their score band.
"""

import math
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import _edges, _validation

__all__ = ["fuzzy_augmentation", "parceling", "reweighting"]


def fuzzy_augmentation(probability: npt.ArrayLike, *, factor: float = 1.0) -> pd.DataFrame:
    r"""Split each rejected applicant into a bad and a good row with fractional weights.

    The PD of a reject under the model fitted on the accepted applicants
    alone gives the weight of its bad row, and the rest of its weight of 1
    goes to its good row. Concatenate the rows with the accepted applicants
    at weight 1 and refit the model with weights.

    Parameters
    ----------
    probability : array_like
        PD of each rejected applicant under the accepts-only model, between
        0 and 1.
    factor : float, default 1
        Positive multiplier of the PDs. A factor above 1 assumes that the
        rejects are riskier than accepted applicants with the same PD, as
        when the approval used information that the model does not have.

    Returns
    -------
    pandas.DataFrame
        Two rows per rejected applicant, the bad row first, with the columns
        ``target``, 1 and then 0, and ``weight``, the PD times `factor`
        clipped to [0, 1] and its complement. The index repeats the index of
        `probability`, or its positions when it is not a Series. When
        `probability` is a Series with the unique index of the rejects,
        ``rejects.loc[result.index]`` gives the features of every row.

    Raises
    ------
    ValueError
        If `probability` is empty, not one-dimensional or not numeric, or
        has a value that is missing or outside [0, 1]; or if `factor` is not
        a positive finite number.

    See Also
    --------
    parceling : Infer the outcomes from the bad rate of a score band.
    reweighting : Weight the accepted applicants instead.
    kivi.selection.logistic_regression : Refit the model with ``weights``.

    Notes
    -----
    A reject with PD :math:`p` and the factor :math:`k` gets the rows

    .. math::

       (y = 1,\ w = \min(k p, 1)) \quad \text{and} \quad
       (y = 0,\ w = 1 - \min(k p, 1)).

    Clipping keeps both weights in [0, 1]: when :math:`k p \ge 1`, the reject
    counts as bad with its whole weight and its good row has weight 0, which
    kivi's ``weights`` arguments ignore. Each reject keeps a total weight of
    1, so the weighted bad rate of its rows is the mean of
    :math:`\min(k p_i, 1)`.

    With ``factor=1``, refitting the same logistic regression on the same
    encoding returns the coefficients of the accepts-only model exactly: at
    those coefficients, the weight of each reject's bad row equals its
    fitted PD, so the inferred rows leave the score equations unchanged. The
    refit only narrows the standard errors, although the inferred rows add
    no information: with these weights, the standard errors of
    `kivi.selection.logistic_regression` are too small. The model changes
    only with a factor other than 1,
    with another encoding, such as WOE refitted on the augmented sample, or
    with other variables.

    References
    ----------
    .. [1] N. Siddiqi, "Credit Risk Scorecards: Developing and Implementing
       Intelligent Credit Scoring", Wiley, 2006.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.reject_inference import fuzzy_augmentation
    >>> probability = pd.Series([0.1, 0.4, 0.7], index=["a", "b", "c"])
    >>> fuzzy_augmentation(probability)
       target  weight
    a       1     0.1
    a       0     0.9
    b       1     0.4
    b       0     0.6
    c       1     0.7
    c       0     0.3
    >>> fuzzy_augmentation(probability, factor=1.5).round(2)
       target  weight
    a       1    0.15
    a       0    0.85
    b       1    0.60
    b       0    0.40
    c       1    1.00
    c       0    0.00
    """
    series, values = _scores(probability, "probability")
    outside = (values < 0) | (values > 1)
    if outside.any():
        raise ValueError(f"probability must be between 0 and 1, got {float(values[outside][0])!r}")
    _check_factor(factor)
    bad = np.minimum(factor * values, 1.0)
    return _inferred(series.index, bad)


def parceling(
    score: npt.ArrayLike,
    target: npt.ArrayLike,
    reject_score: npt.ArrayLike,
    *,
    bins: int | npt.ArrayLike = 10,
    factor: float = 1.0,
    expected: bool = False,
    rng: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    r"""Label rejected applicants bad or good at the bad rate of their score band.

    The scores of the accepted and the rejected applicants are divided into
    bands. In each band, the bad rate of the rejects is `factor` times the
    observed bad rate of the accepted applicants, at most 1, and that share
    of the band's rejects, chosen at random, is labelled bad.

    Parameters
    ----------
    score : array_like
        Finite numeric scores of the accepted applicants, such as their PD
        under the accepts-only model. The direction does not matter.
    target : array_like
        Observed outcomes of the accepted applicants, 1 for bad and 0 for
        good, with both present. If `score` and `target` are both pandas
        Series, they must have identical indices; otherwise they are matched
        by position.
    reject_score : array_like
        Finite numeric scores of the rejected applicants, on the scale of
        `score`.
    bins : int or array_like, default 10
        Number of quantile bands, fitted with `kivi.binning.bin_edges` on the
        scores of accepted and rejected applicants together, so tied scores
        can leave fewer bands; or increasing band edges, closed on the
        right, that hold every score.
    factor : float, default 1
        Positive multiplier of the accepted applicants' bad rate in each
        band. A factor above 1 assumes that the rejects are riskier than
        accepted applicants with the same score.
    expected : bool, default False
        Return two rows per reject, weighted with the band's inferred bad
        rate and its complement as in `fuzzy_augmentation`, instead of a
        random label.
    rng : {None, int, numpy.random.Generator}, optional
        Seed or generator passed to `numpy.random.default_rng` to draw the
        labels. The global NumPy random state is not used, so the same seed
        gives the same labels. Not used with ``expected=True``.

    Returns
    -------
    pandas.DataFrame
        The columns ``target`` and ``weight``, indexed like `reject_score`,
        or by position when it is not a Series. One row per rejected
        applicant with its label and the weight 1, or, with
        ``expected=True``, two rows per rejected applicant as returned by
        `fuzzy_augmentation`.

    Raises
    ------
    ValueError
        If `score` and `target` are invalid as for
        `kivi.metrics.gains_table`; if `reject_score` is empty, not
        one-dimensional or not numeric, or has a missing or infinite value;
        if `bins` is neither a positive integer nor at least two increasing
        edges, or a score lies outside the edges; if a band holds rejected
        applicants but no accepted one, so that its bad rate is unknown; or
        if `factor` is not a positive finite number or `expected` is not a
        boolean.

    Warns
    -----
    UserWarning
        If tied scores leave fewer quantile bands than `bins`.

    See Also
    --------
    fuzzy_augmentation : Infer the outcomes from each reject's own PD.
    reweighting : Weight the accepted applicants of each band instead.
    kivi.metrics.gains_table : Bad rates by score band.

    Notes
    -----
    In a band with :math:`a` accepted applicants of whom :math:`b` are bad,
    and :math:`n` rejects, the inferred bad rate is
    :math:`r = \min(k b / a, 1)` for the factor :math:`k`. The number of bad
    rejects is :math:`r n` rounded down or up at random, up with a
    probability equal to its fractional part, so it is :math:`r n` on
    average and never further from it than 1. With ``expected=True``, each
    reject instead gets a bad row with the weight :math:`r` and a good row
    with :math:`1 - r`, which removes the noise of the draw.

    Parceling uses the score only: within a band, it treats every reject
    alike, whatever its features. A reject's inferred bad rate depends on
    the accepted applicants of its band, which are few in the bands where
    the approval rate is low.

    References
    ----------
    .. [1] N. Siddiqi, "Credit Risk Scorecards: Developing and Implementing
       Intelligent Credit Scoring", Wiley, 2006.

    Examples
    --------
    The accepted applicants have the bad rate 1/4 in the low band and 3/4
    in the high band. Three of the four rejects of the high band are
    therefore labelled bad, and the one reject of the low band is bad with
    the probability 1/4:

    >>> from kivi.reject_inference import parceling
    >>> score = [0.1, 0.2, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9]
    >>> target = [0, 0, 1, 0, 1, 0, 1, 1]
    >>> reject_score = [0.25, 0.65, 0.75, 0.85, 0.95]
    >>> parceling(score, target, reject_score, bins=[0, 0.5, 1], rng=0)
       target  weight
    0       0     1.0
    1       0     1.0
    2       1     1.0
    3       1     1.0
    4       1     1.0

    With ``expected=True``, every reject instead gets the bad rate of its
    band as the weight of its bad row:

    >>> parceling(score, target, reject_score, bins=[0, 0.5, 1], expected=True).head(4)
       target  weight
    0       1    0.25
    0       0    0.75
    1       1    0.75
    1       0    0.25
    """
    _check_factor(factor)
    if not isinstance(expected, (bool, np.bool_)):
        raise ValueError(f"expected must be True or False, got {_validation.shown(expected)}")
    values, labels, _ = _validation.scores_and_labels(score, target, "score")
    rejects, reject_values = _scores(reject_score, "reject_score")
    edges, accepted, rejected = _bands(values, reject_values, bins, "its bad rate is unknown")
    size = len(edges) - 1
    total = np.bincount(accepted, minlength=size)
    bad = np.bincount(accepted, weights=labels, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.minimum(factor * bad / total, 1.0)
    if expected:
        return _inferred(rejects.index, rate[rejected])
    generator = np.random.default_rng(rng)
    inferred = np.zeros(len(reject_values), dtype=np.int64)
    for band in np.flatnonzero(np.bincount(rejected, minlength=size)):
        members = np.flatnonzero(rejected == band)
        count = rate[band] * len(members)
        bads = math.floor(count)
        bads += int(generator.random() < count - bads)
        inferred[generator.permutation(members)[:bads]] = 1
    return pd.DataFrame(
        {"target": inferred, "weight": np.ones(len(inferred))}, index=rejects.index.copy()
    )


def reweighting(
    score: npt.ArrayLike, reject_score: npt.ArrayLike, *, bins: int | npt.ArrayLike = 10
) -> pd.Series | npt.NDArray[np.float64]:
    r"""Weight accepted applicants to stand in for the rejected applicants of their score band.

    In each band of the score, every accepted applicant gets the weight
    (accepted + rejected) / accepted, the inverse of the band's approval
    rate. The weighted accepted applicants then add up to all the
    applicants of their band, so a model refitted with these weights gives
    each band its share of the applicant population.

    Parameters
    ----------
    score : array_like
        Finite numeric scores of the accepted applicants, such as their PD
        under the accepts-only model. The direction does not matter.
    reject_score : array_like
        Finite numeric scores of the rejected applicants, on the scale of
        `score`.
    bins : int or array_like, default 10
        Number of quantile bands, fitted with `kivi.binning.bin_edges` on the
        scores of accepted and rejected applicants together, so tied scores
        can leave fewer bands; or increasing band edges, closed on the
        right, that hold every score.

    Returns
    -------
    pandas.Series or numpy.ndarray
        Weight of each accepted applicant, at least 1. A Series named
        ``weight`` with the index of `score` when `score` is a Series,
        otherwise an array.

    Raises
    ------
    ValueError
        If `score` or `reject_score` is empty, not one-dimensional or not
        numeric, or has a missing or infinite value; if `bins` is neither a
        positive integer nor at least two increasing edges, or a score lies
        outside the edges; or if a band holds rejected applicants but no
        accepted one.

    Warns
    -----
    UserWarning
        If tied scores leave fewer quantile bands than `bins`.

    See Also
    --------
    fuzzy_augmentation : Infer outcomes for the rejects instead.
    parceling : Label the rejects of each band at its bad rate.
    kivi.selection.logistic_regression : Refit the model with ``weights``.

    Notes
    -----
    With :math:`a` accepted and :math:`n` rejected applicants in a band, each
    accepted applicant of the band gets the weight

    .. math::

       w = \frac{a + n}{a}.

    Reweighting assumes that, within a band, the rejects default like the
    accepted applicants. It changes how much each band weighs in the fit,
    but adds no outcomes, so the rejects are predicted to be as risky as
    accepted applicants with the same score. It cannot represent applicants
    that no accepted applicant resembles: a band of rejects without accepted
    applicants raises an error. Bands of a score that decided the approval
    with a hard cut-off have no accepted applicants beyond the cut-off, so
    use a score on which accepted and rejected applicants overlap, such as
    the PD of the accepts-only model.

    The weights add up to more than the number of accepted applicants, so
    the standard errors of a fit with them are too small, as if the
    outcomes of all applicants had been observed.

    References
    ----------
    .. [1] N. Siddiqi, "Credit Risk Scorecards: Developing and Implementing
       Intelligent Credit Scoring", Wiley, 2006.

    Examples
    --------
    The low band has 4 accepted applicants and 1 reject, the high band 2 and 4:

    >>> from kivi.reject_inference import reweighting
    >>> score = [0.1, 0.2, 0.2, 0.3, 0.6, 0.7]
    >>> reweighting(score, [0.25, 0.65, 0.75, 0.85, 0.95], bins=[0, 0.5, 1])
    array([1.25, 1.25, 1.25, 1.25, 3.  , 3.  ])
    """
    _, values = _scores(score, "score")
    _, reject_values = _scores(reject_score, "reject_score")
    edges, accepted, rejected = _bands(
        values, reject_values, bins, "no accepted applicant can stand in for them"
    )
    size = len(edges) - 1
    total = np.bincount(accepted, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        weight = (total + np.bincount(rejected, minlength=size)) / total
    result = _validation.like_input(score, weight[accepted])
    return result.rename("weight") if isinstance(result, pd.Series) else result  # type: ignore[return-value]  # one-dimensional, never a float


def _scores(values: npt.ArrayLike, name: str) -> tuple[pd.Series, npt.NDArray[np.float64]]:
    """Return `values` as a Series and as finite floats, or raise naming `name`."""
    series = _validation.as_series(values, name)
    if not len(series):
        raise ValueError(f"{name} must not be empty")
    dtype = series.dtype
    if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_complex_dtype(dtype):
        raise ValueError(f"{name} must be real and numeric, got dtype {dtype}")
    array = series.to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(array)
    if not finite.all():
        raise ValueError(
            f"{name} must be finite, got {(~finite).sum()} missing or infinite value(s) "
            f"such as {float(array[~finite][0])!r}"
        )
    return series, array


def _check_factor(factor: float) -> None:
    if (
        isinstance(factor, bool)
        or not _validation.is_real(factor)
        or not math.isfinite(factor)
        or factor <= 0
    ):
        raise ValueError(
            f"factor must be a positive finite number, got {_validation.shown(factor)}"
        )


def _bands(
    values: np.ndarray, reject_values: np.ndarray, bins: int | npt.ArrayLike, consequence: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the band edges and the band of every accepted and rejected score."""
    pooled = np.concatenate([values, reject_values])
    if not _validation.is_integral(bins) or isinstance(bins, bool):
        edges = _edges.band_edges(pooled, bins, None)
    else:
        # Warn about tied scores here instead, in terms of bands and at the caller of
        # parceling or reweighting.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            edges = _edges.band_edges(pooled, bins, None)
        if len(edges) - 1 < bins:
            levels, counts = np.unique(pooled, return_counts=True)
            top = int(counts.argmax())
            warnings.warn(
                f"tied scores leave {len(edges) - 1} of the {bins} requested bands; the score "
                f"{float(levels[top])!r} makes up {counts[top] / len(pooled):.0%} of the scores, "
                "pass fewer bins or explicit edges",
                UserWarning,
                stacklevel=3,
            )
    bands = []
    for name, scores in (("score", values), ("reject_score", reject_values)):
        band = np.searchsorted(edges, scores, side="left") - 1
        outside = (band < 0) | (band >= len(edges) - 1)
        if outside.any():
            raise ValueError(
                f"{name} has {outside.sum()} value(s) outside the edges in bins, such as "
                f"{float(scores[outside][0])!r}; use infinite outer edges"
            )
        bands.append(band)
    accepted, rejected = bands
    size = len(edges) - 1
    unmatched = (np.bincount(accepted, minlength=size) == 0) & (
        np.bincount(rejected, minlength=size) > 0
    )
    if unmatched.any():
        band = int(np.argmax(unmatched))  # type: ignore[assignment]  # a band number from here on
        count = int((rejected == band).sum())
        raise ValueError(
            f"the score band ({float(edges[band])!r}, {float(edges[band + 1])!r}] holds {count} "
            f"rejected applicant(s) but no accepted one, so {consequence}; use fewer bins or "
            "wider edges"
        )
    return edges, accepted, rejected


def _inferred(index: pd.Index, bad: np.ndarray) -> pd.DataFrame:
    """Two rows per reject, bad then good, weighted with `bad` and its complement."""
    return pd.DataFrame(
        {
            "target": np.tile(np.array([1, 0], dtype=np.int64), len(bad)),
            "weight": np.column_stack([bad, 1 - bad]).ravel(),
        },
        index=index.repeat(2),
    )
