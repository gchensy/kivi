"""Credit strategy: cut-offs, swap sets and expected loss.

``cutoff_table`` shows what each candidate cut-off of a score approves and
how many events it lets through. ``swap_set`` compares the decisions of a
current and a new score at their cut-offs. ``expected_loss`` and
``expected_loss_table`` turn PDs, LGDs and exposures into the expected loss
of each exposure and of each grade.

As in ``kivi.metrics``, a score is higher for a higher risk, such as a PD,
so negate scorecard points before passing them in. An applicant is approved
when the score is at or below the cut-off.
"""

import math
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._counts import group_counts, group_rows, ratio
from ._validation import (
    as_real,
    as_series,
    as_weights,
    finite_scores,
    is_real_dtype,
    known_outcomes,
    like_input,
    scores_and_labels,
    shared_index,
)
from .binning import _fit_edges

__all__ = ["cutoff_table", "swap_set", "expected_loss", "expected_loss_table"]

_DECISIONS = ("approve", "decline")


def cutoff_table(
    score: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    cutoffs: int | npt.ArrayLike = 10,
    loss: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    """Summarize the approved book and the rejected events at each cut-off.

    A cut-off approves every observation whose score is at or below it, so
    a higher cut-off approves more applicants and lets more events through.
    Each row shows the size and the event rate of the approved book and the
    shares of events and non-events that the cut-off rejects.

    Parameters
    ----------
    score : array_like
        Finite numeric scores that are higher for a higher risk, such as a
        PD. Negate scorecard points, which are higher for a lower risk.
    target : array_like
        Binary labels, with 1 indicating the event. Both classes must be
        present. If `score` and `target` are both pandas Series, they must
        have identical indices; otherwise they are matched by position.
    cutoffs : int or array_like, default 10
        Number of quantile cut-offs, or increasing cut-off values. An
        integer `n` uses the edges of ``kivi.binning.bin_edges(score,
        bins=n)`` above the lowest, so the rows approve about 1/n, 2/n and
        so on up to all of the sample, and tied scores can leave fewer rows.
        Pass explicit values as a sequence, also for a single cut-off, such
        as ``[0.1]``. With `weights`, an integer gives weighted quantiles,
        as ``bin_edges(score, bins=n, weights=weights)`` does.
    loss : array_like, optional
        Non-negative loss of each observation, matched like `target`, such
        as its expected loss from `expected_loss` or the loss realized on a
        defaulted account. The table then has a column ``approved_loss``.
    weights : array_like, optional
        Non-negative weight of each observation, such as the inverse of its
        sampling rate, matched like `target`. Counts and losses become sums
        of weights. On the same cut-offs, integer weights give the same
        table as repeating rows; an integer `cutoffs` can give other
        cut-offs, because weighted quantiles use the inverted CDF.

    Returns
    -------
    pandas.DataFrame
        One row per cut-off, from the lowest to the highest, indexed by the
        cut-off value. The columns are described in the Notes.

    Raises
    ------
    ValueError
        If the inputs are empty or not one-dimensional, their indices or
        lengths do not match, `target` has a label other than 0 and 1 or
        lacks one of them, `score` is not real and numeric or has a missing
        or infinite value, or `cutoffs` is neither a positive integer nor a
        sequence of increasing numbers. Also if `loss` is not numeric or
        does not give each observation one finite, non-negative loss, or
        `weights` is not numeric, does not give each observation one finite,
        non-negative weight, has an index other than that of `score`, or
        leaves a class without weight.

    Warns
    -----
    UserWarning
        If `cutoffs` is an integer and tied scores leave fewer cut-offs than
        requested, as in `kivi.binning.bin_edges`.

    See Also
    --------
    swap_set : Compare the decisions of two scores at their cut-offs.
    expected_loss : Expected loss of each exposure, to pass as `loss`.
    kivi.metrics.gains_table : Event rates of the bands between cut-offs.

    Notes
    -----
    A score equal to the cut-off is approved, so tied scores always get the
    same decision. With the cut-offs of an integer `cutoffs`, the approved
    book of a row consists of the bands of ``kivi.metrics.gains_table`` with
    ``bins=cutoffs`` up to that cut-off.

    The columns are:

    - ``approved``: observations with a score at or below the cut-off, or
      the sum of their weights;
    - ``approval_rate``: ``approved`` divided by the whole sample;
    - ``approved_event``: events among the approved observations;
    - ``approved_event_rate``: ``approved_event`` divided by ``approved``,
      the bad rate of the approved book, NaN when nothing is approved;
    - ``rejected_event_share`` and ``rejected_non_event_share``: shares of
      all events and of all non-events with a score above the cut-off;
    - ``approved_loss``: with `loss`, the sum of the losses of the
      approved observations, weighted with `weights`.

    A higher cut-off raises the approval rate and lowers both rejected
    shares. ``rejected_event_share`` is the share of bad outcomes the
    cut-off avoids, and ``rejected_non_event_share`` the share of good
    business it gives up.

    Examples
    --------
    >>> from kivi.strategy import cutoff_table
    >>> score = [0.1, 0.2, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9]
    >>> target = [0, 0, 1, 0, 0, 1, 1, 1]
    >>> table = cutoff_table(score, target, cutoffs=[0.2, 0.5, 1.0])
    >>> table[["approved", "approval_rate", "approved_event", "approved_event_rate"]]
            approved  approval_rate  approved_event  approved_event_rate
    cutoff
    0.2            3          0.375               1             0.333333
    0.5            5          0.625               1             0.200000
    1.0            8          1.000               4             0.500000
    >>> table[["rejected_event_share", "rejected_non_event_share"]]
            rejected_event_share  rejected_non_event_share
    cutoff
    0.2                     0.75                       0.5
    0.5                     0.75                       0.0
    1.0                     0.00                       0.0
    """
    shared_index({"score": score, "target": target, "loss": loss, "weights": weights})
    values, labels, mass = scores_and_labels(score, target, "score", weights=weights)
    points = _cutoff_values(values, cutoffs, mass)
    amounts = None if loss is None else _loss_amounts(loss, score)
    order = np.argsort(values, kind="stable")
    labels = labels[order]
    if mass is None:
        event_mass = labels
        non_event_mass = 1 - labels
    else:
        event_mass = labels * mass[order]
        non_event_mass = (1 - labels) * mass[order]
    # Observations at or below each cut-off: tied scores are all approved.
    approved_rows = np.searchsorted(values[order], points, side="right")
    cum_event = np.r_[0, np.cumsum(event_mass)]
    cum_non_event = np.r_[0, np.cumsum(non_event_mass)]
    approved_event = cum_event[approved_rows]
    approved_non_event = cum_non_event[approved_rows]
    approved = approved_event + approved_non_event
    total_event, total_non_event = cum_event[-1], cum_non_event[-1]
    columns = {
        "approved": approved,
        "approval_rate": approved / (total_event + total_non_event),
        "approved_event": approved_event,
        "approved_event_rate": ratio(approved_event, approved),
        "rejected_event_share": (total_event - approved_event) / total_event,
        "rejected_non_event_share": (total_non_event - approved_non_event) / total_non_event,
    }
    if amounts is not None:
        weighted = amounts if mass is None else amounts * mass
        columns["approved_loss"] = np.r_[0.0, np.cumsum(weighted[order])][approved_rows]
    return pd.DataFrame(columns, index=pd.Index(points, name="cutoff"))


def swap_set(
    current: npt.ArrayLike,
    new: npt.ArrayLike,
    target: npt.ArrayLike | None = None,
    *,
    current_cutoff: float,
    new_cutoff: float,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    """Compare the decisions of a current and a new score at their cut-offs.

    Each score approves the observations at or below its cut-off. The four
    rows are the observations both scores approve, the swap-outs that only
    the current score approves, the swap-ins that only the new score
    approves, and those both decline.

    Parameters
    ----------
    current : array_like
        Finite numeric scores of the current model, higher for a higher
        risk, such as a PD. Negate scorecard points.
    new : array_like
        Finite numeric scores of the new model, of the same length as
        `current` and higher for a higher risk. The two scores need not be
        on the same scale.
    target : array_like, optional
        Binary labels, with 1 indicating the event, and missing values for
        unknown outcomes, such as the applicants the current score declined.
        Without `target`, the table only counts the observations.
    current_cutoff : float
        Cut-off of the current score. Observations with a `current` score at
        or below it are approved.
    new_cutoff : float
        Cut-off of the new score, applied in the same way to `new`.
    weights : array_like, optional
        Non-negative weight of each observation, such as the inverse of its
        sampling rate. Counts become sums of weights, so integer weights
        give the same table as repeating rows.

    Returns
    -------
    pandas.DataFrame
        Four rows indexed by the decisions of the two scores, ``current``
        and ``new``, each ``"approve"`` or ``"decline"``. The columns are
        ``total`` and ``share``, and with `target` also ``known``,
        ``event`` and ``event_rate``; see the Notes.

    Raises
    ------
    ValueError
        If the inputs are empty or not one-dimensional, their lengths
        differ, or they are Series with different indices; `current` or
        `new` is not real and numeric or has a missing or infinite value;
        `target` has a value other than 0, 1 and missing; a cut-off is not
        a number or is missing; or `weights` is not numeric, does not give
        each observation one finite, non-negative weight with a positive
        total, or has an index other than that of `current`.

    See Also
    --------
    cutoff_table : Approval and event rates over a range of cut-offs.
    kivi.reject_inference.fuzzy_augmentation : Infer the outcomes of declined applicants.

    Notes
    -----
    The rows ``("approve", "decline")`` and ``("decline", "approve")`` are
    the swap-outs and the swap-ins. Every row is reported, also when it is
    empty. The columns are:

    - ``total``: observations in the row, or the sum of their weights;
    - ``share``: ``total`` divided by the whole sample;
    - ``known``: observations in the row with a known outcome;
    - ``event``: events in the row;
    - ``event_rate``: ``event`` divided by ``known``, NaN without known
      outcomes.

    The outcomes of the applicants the current score declined are usually
    unknown, because they never received a loan. The rows of current
    declines, including the swap-ins, then have no ``event_rate``, and
    the swap set alone cannot show whether the new score accepts better
    applicants than it rejects. Reject inference estimates the missing
    outcomes; pass the inferred labels, with weights for an augmented
    sample, to complete the table.

    Examples
    --------
    >>> from kivi.strategy import swap_set
    >>> current = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    >>> new = [0.1, 0.3, 0.2, 0.6, 0.4, 0.7, 0.5, 0.8]
    >>> target = [0, 0, 1, 0, 1, 1, 0, 1]
    >>> swap_set(current, new, target, current_cutoff=0.4, new_cutoff=0.4)
                     total  share  known  event  event_rate
    current new
    approve approve      3  0.375      3      1    0.333333
            decline      1  0.125      1      0    0.000000
    decline approve      1  0.125      1      1    1.000000
            decline      3  0.375      3      2    0.666667

    The current declines usually have no known outcome:

    >>> import numpy as np
    >>> known = np.where(np.array(current) <= 0.4, target, np.nan)
    >>> swap_set(current, new, known, current_cutoff=0.4, new_cutoff=0.4)
                     total  share  known  event  event_rate
    current new
    approve approve      3  0.375      3      1    0.333333
            decline      1  0.125      1      0    0.000000
    decline approve      1  0.125      0      0         NaN
            decline      3  0.375      0      0         NaN
    """
    shared_index({"current": current, "new": new, "target": target, "weights": weights})
    raw_current = current
    current, new = as_series(current, "current"), as_series(new, "new")
    if len(current) != len(new):
        raise ValueError(
            f"current and new must have the same length, got {len(current)} and {len(new)}"
        )
    if not len(current):
        raise ValueError("current and new must not be empty")
    current, new = finite_scores(current, "current"), finite_scores(new, "new")
    declined = 2 * (current > _cutoff(current_cutoff, "current_cutoff")) + (
        new > _cutoff(new_cutoff, "new_cutoff")
    )
    mass = as_weights(weights, raw_current, of="current")
    total = np.bincount(declined, weights=mass, minlength=4)
    columns = {"total": total, "share": total / total.sum()}
    if target is not None:
        known, event = known_outcomes(target, len(current), "current")
        if mass is None:
            columns["known"] = np.bincount(declined[known], minlength=4)
            columns["event"] = np.bincount(declined[event], minlength=4)
        else:
            columns["known"] = np.bincount(declined, weights=known * mass, minlength=4)
            columns["event"] = np.bincount(declined, weights=event * mass, minlength=4)
        columns["event_rate"] = ratio(columns["event"], columns["known"])
    index = pd.MultiIndex.from_product([_DECISIONS, _DECISIONS], names=["current", "new"])
    return pd.DataFrame(columns, index=index)


def expected_loss(
    probability: npt.ArrayLike, lgd: npt.ArrayLike, ead: npt.ArrayLike
) -> float | pd.Series | npt.NDArray[np.float64]:
    r"""Compute the expected loss of each exposure from its PD, LGD and EAD.

    The expected loss is the product of the probability of default, the
    loss given default and the exposure at default.

    Parameters
    ----------
    probability : array_like
        Probabilities of default, from 0 to 1. Missing values give missing
        losses.
    lgd : array_like
        Loss given default as a share of the exposure, from 0 to 1, as one
        value for all exposures or one per exposure.
    ead : array_like
        Exposure at default, such as the outstanding balance, finite and
        non-negative, as one value or one per exposure.

    Returns
    -------
    float, pandas.Series or numpy.ndarray
        Expected losses, in the units of `ead`, with the broadcast shape of
        the inputs. If an input is a Series, the result is a Series with its
        index and name; scalar inputs give a float.

    Raises
    ------
    ValueError
        If an input is not real and numeric, the inputs cannot be broadcast
        together, Series inputs have different indices, `probability` or
        `lgd` has a value outside [0, 1], or `ead` has a negative or
        infinite value.

    See Also
    --------
    expected_loss_table : Exposure and expected loss by grade.
    cutoff_table : Pass the expected losses as `loss` to sum them by cut-off.
    kivi.scorecard.score_to_pd : The PD implied by a scorecard score.

    Notes
    -----
    For an exposure with probability of default :math:`\mathrm{PD}`, loss
    given default :math:`\mathrm{LGD}` and exposure at default
    :math:`\mathrm{EAD}`,

    .. math::

       \mathrm{EL} = \mathrm{PD} \times \mathrm{LGD} \times \mathrm{EAD}.

    The PD refers to a horizon, usually one year, and the expected loss
    refers to the same horizon. Missing values are not checked and give a
    missing loss.

    Examples
    --------
    >>> from kivi.strategy import expected_loss
    >>> expected_loss([0.02, 0.1, 0.25], 0.45, [1000, 500, 200])
    array([ 9. , 22.5, 22.5])
    """
    inputs = {"probability": probability, "lgd": lgd, "ead": ead}
    shared_index(inputs)
    arrays, template = _loss_inputs(inputs)
    return like_input(template, arrays["probability"] * arrays["lgd"] * arrays["ead"])


def expected_loss_table(
    groups: npt.ArrayLike,
    probability: npt.ArrayLike,
    lgd: npt.ArrayLike,
    ead: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Summarize exposure and expected loss by grade or other group.

    Each row adds up the exposures and expected losses of a group, such as
    a grade of a master scale, and gives their ratio, the expected loss
    rate.

    Parameters
    ----------
    groups : array_like
        Grade or group labels. Missing values form a group.
    probability : array_like
        Probability of default of each exposure, from 0 to 1, without
        missing values.
    lgd : array_like
        Loss given default, from 0 to 1, as one value for all exposures or
        one per exposure.
    ead : array_like
        Exposure at default, finite and non-negative, as one value or one
        per exposure. Series inputs must have identical indices; otherwise
        the inputs are matched by position.
    weights : array_like, optional
        Non-negative weight of each exposure, such as the inverse of its
        sampling rate, matched like `probability`. Counts, exposures and
        losses become sums of weights, so integer weights give the same
        groups and sums as repeating rows, and groups whose weights sum to
        zero are left out. Groups that are not categorical keep the order in
        which they first appear, also in rows of zero weight, so their order
        can differ from that of repeated rows.

    Returns
    -------
    pandas.DataFrame
        One row per group with the columns described in the Notes.

    Raises
    ------
    ValueError
        If `groups` or `probability` is empty or not one-dimensional, an
        input does not have one value per exposure or a Series index other
        than that of the others, `probability`, `lgd` or `ead` is not real
        and numeric or has a missing value, `probability` or `lgd` has a
        value outside [0, 1], or `ead` has a negative or infinite value.
        Also if `weights` is not numeric, does not give each exposure one
        finite, non-negative weight with a positive total, or has an index
        other than that of `groups`.

    See Also
    --------
    expected_loss : Expected loss of each exposure.
    kivi.calibration.calibration_table : Observed and predicted default rates by grade.

    Notes
    -----
    Groups are counted as in `kivi.woe.woe_table`. Rows follow the order of
    the categories when `groups` is categorical, such as the result of
    ``pandas.cut``, with missing values last. Otherwise they follow the order
    in which the groups first appear. Categories that are never observed are
    omitted.

    For the exposures :math:`i` of a group, with weights :math:`w_i`, the
    columns are:

    - ``total``: the number of exposures, :math:`\sum_i w_i`;
    - ``exposure``: :math:`\sum_i w_i \mathrm{EAD}_i`;
    - ``exposure_weighted_probability``:
      :math:`\sum_i w_i \mathrm{EAD}_i \mathrm{PD}_i / \sum_i w_i \mathrm{EAD}_i`;
    - ``expected_loss``:
      :math:`\sum_i w_i \mathrm{PD}_i \mathrm{LGD}_i \mathrm{EAD}_i`;
    - ``expected_loss_rate``: ``expected_loss`` divided by ``exposure``.

    The two ratios are NaN for a group without exposure. With one LGD for
    all exposures, ``expected_loss_rate`` is the LGD times
    ``exposure_weighted_probability``. The portfolio's expected loss is the
    sum of the ``expected_loss`` column, and its expected loss rate is that
    sum divided by the sum of ``exposure``.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.strategy import expected_loss_table
    >>> grades = pd.Series(["A", "A", "B", "B", "B"])
    >>> probability = pd.Series([0.01, 0.03, 0.1, 0.2, 0.2])
    >>> ead = pd.Series([1000.0, 3000.0, 500.0, 0.0, 1500.0])
    >>> table = expected_loss_table(grades, probability, 0.5, ead)
    >>> table[["total", "exposure", "expected_loss", "expected_loss_rate"]]
         total  exposure  expected_loss  expected_loss_rate
    bin
    A        2    4000.0           50.0              0.0125
    B        3    2000.0          175.0              0.0875
    >>> table["exposure_weighted_probability"]
    bin
    A    0.025
    B    0.175
    Name: exposure_weighted_probability, dtype: float64
    """
    shared_index(
        {"groups": groups, "probability": probability, "lgd": lgd, "ead": ead, "weights": weights}
    )
    raw_groups = groups
    groups = as_series(groups, "groups")
    length = len(as_series(probability, "probability"))
    if length != len(groups):
        raise ValueError(
            f"groups and probability must have the same length, got {len(groups)} and {length}"
        )
    if not len(groups):
        raise ValueError("groups and probability must not be empty")
    arrays, _ = _loss_inputs(
        {"probability": probability, "lgd": lgd, "ead": ead}, length=len(groups)
    )
    mass = as_weights(weights, raw_groups, of="groups")
    counts = group_counts(groups, np.zeros(len(groups), dtype=np.intp), mass)
    rows = group_rows(groups, counts.index)
    # Only exposures of zero weight lie in a group that group_counts left out.
    kept = rows >= 0
    exposure = arrays["ead"] if mass is None else mass * arrays["ead"]

    def group_sum(values: np.ndarray) -> np.ndarray:
        return np.bincount(rows[kept], weights=values[kept], minlength=len(counts))

    group_exposure = group_sum(exposure)
    group_loss = group_sum(exposure * arrays["probability"] * arrays["lgd"])
    return pd.DataFrame(
        {
            "total": counts.to_numpy()[:, 0],
            "exposure": group_exposure,
            "exposure_weighted_probability": ratio(
                group_sum(exposure * arrays["probability"]), group_exposure
            ),
            "expected_loss": group_loss,
            "expected_loss_rate": ratio(group_loss, group_exposure),
        },
        index=counts.index,
    )


def _cutoff_values(values: np.ndarray, cutoffs, weights: np.ndarray | None) -> np.ndarray:
    """Increasing cut-offs: quantile edges above the lowest for an integer, else the values."""
    message = "cutoffs must be a positive integer or a sequence of increasing numbers, got {!r}"
    if isinstance(cutoffs, Integral) and not isinstance(cutoffs, bool):
        if cutoffs < 1:
            raise ValueError(message.format(cutoffs))
        # The edges of bin_edges above the lowest; the last one, infinity, approves everyone.
        # The warning for tied quantiles points at the caller of cutoff_table.
        edges = _fit_edges(
            values,
            int(cutoffs),
            weights=weights,
            advice="pass fewer cutoffs or explicit cut-off values",
            stacklevel=4,
        )
        return edges[1:]
    if np.ndim(cutoffs) != 1:
        raise ValueError(message.format(cutoffs))
    series = cutoffs if isinstance(cutoffs, pd.Series) else pd.Series(cutoffs)
    if not len(series) or not is_real_dtype(series.dtype):
        raise ValueError(message.format(cutoffs))
    array = series.to_numpy(dtype=float, na_value=np.nan)
    if np.isnan(array).any() or not (np.diff(array) > 0).all():
        raise ValueError(message.format(cutoffs))
    return array


def _cutoff(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or math.isnan(float(value)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    return float(value)


def _loss_amounts(loss, score) -> np.ndarray:
    amounts = as_real(as_series(loss, "loss"), "loss")
    if len(amounts) != len(score):
        raise ValueError(
            f"loss must have one value per value of score, got {len(amounts)} for {len(score)}"
        )
    invalid = ~np.isfinite(amounts) | (amounts < 0)
    if invalid.any():
        raise ValueError(
            f"loss must be finite and non-negative, got {float(amounts[invalid][0])!r}"
        )
    return amounts


def _loss_inputs(inputs: dict, length: int | None = None) -> tuple[dict, object]:
    """Broadcast and check PD, LGD and EAD; also return the input whose shape the result takes.

    With ``length``, for a table, every input must broadcast to that length and must not have
    missing values.
    """
    # Booleans count as 0 and 1 in probabilities, as elsewhere in kivi, but not in LGDs or EADs.
    arrays = {
        name: as_real(values, name, booleans=name == "probability")
        for name, values in inputs.items()
    }
    if length is not None:
        for name, array in arrays.items():
            if array.ndim > 1 or (array.ndim == 1 and len(array) not in (1, length)):
                raise ValueError(
                    f"{name} must be one value or one per exposure, got shape {array.shape} "
                    f"for {length} exposures"
                )
            # Count before broadcasting, so a missing scalar counts once.
            missing = np.isnan(array)
            if missing.any():
                raise ValueError(f"{name} must not have missing values, got {missing.sum()}")
        shape = (length,)
    else:
        try:
            shape = np.broadcast_shapes(*(array.shape for array in arrays.values()))
        except ValueError:
            shapes = ", ".join(f"{name} {array.shape}" for name, array in arrays.items())
            raise ValueError(
                f"probability, lgd and ead cannot be broadcast together: {shapes}"
            ) from None
    arrays = {name: np.broadcast_to(array, shape) for name, array in arrays.items()}
    for name in ("probability", "lgd"):
        outside = (arrays[name] < 0) | (arrays[name] > 1)
        if outside.any():
            raise ValueError(
                f"{name} must be between 0 and 1, got {float(arrays[name][outside][0])!r}"
            )
    invalid = np.isinf(arrays["ead"]) | (arrays["ead"] < 0)
    if invalid.any():
        raise ValueError(
            f"ead must be finite and non-negative, got {float(arrays['ead'][invalid][0])!r}"
        )
    series = [values for values in inputs.values() if isinstance(values, pd.Series)]
    template = series[0] if series else inputs["probability"]
    if series and shape != (len(template),):
        raise ValueError(
            f"probability, lgd and ead must broadcast to the length {len(template)} of their "
            f"Series index, got shape {shape}"
        )
    return arrays, template
