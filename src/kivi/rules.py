"""Policy rules: the hits of each rule and the waterfall of a rule set.

``rule_table`` counts what each rule of a set hits, the hits that no other
rule of the set shares and, with a target, the events among them.
``rule_set_table`` applies the rules one after the other, as a decision
waterfall: each row shows what a rule adds to the rules before it and what
the rules up to it leave approved.

A rule is a column of booleans, or of 0 and 1, that is True where the rule
hits, that is where it declines or flags an applicant. Missing values in a
rule raise, because whether a rule hits when its data are missing is part
of the rule. Missing labels in the target are unknown outcomes, such as
those of declined applicants: they count in the hits but not in the known
outcomes and events, as in ``kivi.strategy.swap_set``.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import _counts, _validation

__all__ = ["rule_table", "rule_set_table"]


def rule_table(
    rules: pd.DataFrame,
    target: npt.ArrayLike | None = None,
    *,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Count the hits of each rule, those it alone hits and, with a target, their events.

    Each rule is evaluated on every row, whatever the other rules do, so an
    applicant that two rules hit counts in both rows. The unique hits are
    the applicants that only this rule of the set hits, the declines that
    rest on this rule alone.

    Parameters
    ----------
    rules : pandas.DataFrame
        One column per rule, named after the rule, and one row per
        applicant. Each column holds booleans, or 0 and 1, with True or 1
        where the rule hits, such as a decline or a referral. Missing values
        are not allowed: decide first whether the rule hits when its data
        are missing, for example with ``fillna``.
    target : array_like, optional
        Binary labels, with 1 indicating the event, and missing values for
        unknown outcomes, such as those of declined applicants. Without
        `target`, the table only counts the hits. If `target` is a Series,
        it must have the index of `rules`; otherwise it is matched by
        position.
    weights : array_like, optional
        Non-negative frequency weight of each row, such as the inverse of
        its sampling rate, matched like `target`. Counts become sums of
        weights, so integer weights give the same table as repeating rows.

    Returns
    -------
    pandas.DataFrame
        One row per rule, in the order of the columns of `rules`, indexed by
        the rule names in an index named ``rule``. The columns are
        ``hits``, ``hit_rate`` and ``unique_hits``, and with `target` also
        ``known``, ``event``, ``event_rate``, ``lift``, ``recall`` and
        ``unique_event``; see the Notes.

    Raises
    ------
    ValueError
        If `rules` is not a DataFrame, has no rows or no columns, more than
        one level of column names or a repeated name, or a column with a
        missing value or a value other than True, False, 0 and 1. Also if
        `target` is not one-dimensional, does not give each row one label or
        has a label other than 0, 1 and missing, `weights` is not numeric or
        does not give each row one finite, non-negative weight with a
        positive total, or a Series argument has an index other than that of
        `rules`.

    See Also
    --------
    rule_set_table : Apply the rules in order, as a decision waterfall.
    kivi.strategy.cutoff_table : The same summary for the cut-offs of a score.
    kivi.strategy.swap_set : Compare the decisions of two scores.

    Notes
    -----
    For a rule that hits the rows :math:`H`, with weights :math:`w_i` and
    the known outcomes :math:`K` and events :math:`E` of all rows, the
    columns are:

    - ``hits``: :math:`\sum_{i \in H} w_i`, the number of rows the rule
      hits without weights;
    - ``hit_rate``: ``hits`` divided by the whole sample;
    - ``unique_hits``: the rows that this rule hits and no other rule of
      `rules` does;
    - ``known``: the hits with a known outcome, :math:`H \cap K`;
    - ``event``: the events among the hits, :math:`H \cap E`;
    - ``event_rate``: ``event`` divided by ``known``, NaN when no hit has a
      known outcome;
    - ``lift``: ``event_rate`` divided by the event rate of all known
      outcomes, :math:`\sum_{E} w_i / \sum_{K} w_i`, so a rule whose hits
      default twice as often as the sample has a lift of 2. It is NaN when
      the sample has no known event;
    - ``recall``: ``event`` divided by all known events, the share of the
      events that the rule hits, NaN without known events;
    - ``unique_event``: the events among the unique hits.

    The rows do not add up, because one row can hit under several rules:
    `rule_set_table` counts each hit once. The overlap of two rules is the
    product of their columns, ``rules.astype(int).T @ rules.astype(int)``
    for every pair, with the hits on the diagonal.

    The outcome of a declined applicant is usually unknown. Pass missing
    labels for it: the applicant then counts in ``hits``, ``hit_rate`` and
    ``unique_hits`` but not in ``known``, ``event`` and the rates, so a rule
    that has declined every applicant it hits has no known outcome at all.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.rules import rule_table
    >>> rules = pd.DataFrame({
    ...     "high_debt": [1, 1, 0, 0, 0, 1, 0, 0],
    ...     "young": [0, 1, 1, 0, 0, 0, 0, 0],
    ...     "inquiries": [0, 0, 1, 1, 0, 0, 0, 0],
    ... })
    >>> target = [1, 1, 0, 1, 0, 0, 0, 0]
    >>> table = rule_table(rules, target)
    >>> table[["hits", "hit_rate", "unique_hits", "known", "event"]]
               hits  hit_rate  unique_hits  known  event
    rule
    high_debt     3     0.375            2      3      2
    young         2     0.250            0      2      1
    inquiries     2     0.250            1      2      1
    >>> table[["event_rate", "lift", "recall", "unique_event"]].round(3)
               event_rate   lift  recall  unique_event
    rule
    high_debt       0.667  1.778   0.667             1
    young           0.500  1.333   0.333             0
    inquiries       0.500  1.333   0.333             1

    The outcomes of the applicants that a rule declined are often unknown:

    >>> observed = [None, None, 0, 1, 0, None, 0, 0]
    >>> rule_table(rules, observed)[["hits", "known", "event", "event_rate"]]
               hits  known  event  event_rate
    rule
    high_debt     3      0      0         NaN
    young         2      1      0         0.0
    inquiries     2      2      1         0.5
    """
    hits, mass, outcomes = _inputs(rules, target, weights)
    total = _total(np.ones(len(hits), dtype=bool), mass)
    hit = _sums(hits, mass)
    unique = hits & (hits.sum(axis=1) == 1)[:, None]
    columns = {"hits": hit, "hit_rate": hit / total, "unique_hits": _sums(unique, mass)}
    if outcomes is not None:
        known, event = outcomes
        known_hits = _sums(hits & known[:, None], mass)
        event_hits = _sums(hits & event[:, None], mass)
        total_event = _total(event, mass)
        event_rate = _counts.ratio(event_hits, known_hits)
        overall = _counts.ratio(np.array([total_event]), np.array([_total(known, mass)]))
        columns["known"] = known_hits
        columns["event"] = event_hits
        columns["event_rate"] = event_rate
        columns["lift"] = _counts.ratio(event_rate, np.full(len(event_rate), overall[0]))
        columns["recall"] = _counts.ratio(event_hits, np.full(len(event_hits), total_event))
        columns["unique_event"] = _sums(unique & event[:, None], mass)
    return pd.DataFrame(columns, index=rules.columns.rename("rule"))


def rule_set_table(
    rules: pd.DataFrame,
    target: npt.ArrayLike | None = None,
    *,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    """Apply the rules in order and summarize what each adds and what they leave approved.

    The rules form a decision waterfall: the first rule declines the rows it
    hits, the second those of the remaining rows that it hits, and so on.
    Each row shows the new hits of a rule, those that no earlier rule hit,
    and the book that the rules up to it leave approved, with the columns
    of `kivi.strategy.cutoff_table`.

    Parameters
    ----------
    rules : pandas.DataFrame
        One column per rule, in the order in which the rules apply, and one
        row per applicant. Each column holds booleans, or 0 and 1, with
        True or 1 where the rule hits. Missing values are not allowed:
        decide first whether the rule hits when its data are missing.
    target : array_like, optional
        Binary labels, with 1 indicating the event, and missing values for
        unknown outcomes, such as those of declined applicants. Without
        `target`, the table only counts the hits and the approved rows. If
        `target` is a Series, it must have the index of `rules`; otherwise
        it is matched by position.
    weights : array_like, optional
        Non-negative frequency weight of each row, such as the inverse of
        its sampling rate, matched like `target`. Counts become sums of
        weights, so integer weights give the same table as repeating rows.

    Returns
    -------
    pandas.DataFrame
        One row per rule, in the order of the columns of `rules`, indexed by
        the rule names in an index named ``rule``. The columns are
        ``new_hits``, and with `target` ``new_event``, then ``approved`` and
        ``approval_rate``, and with `target` also ``approved_event``,
        ``approved_event_rate``, ``rejected_event_share`` and
        ``rejected_non_event_share``; see the Notes.

    Raises
    ------
    ValueError
        If `rules` is not a DataFrame, has no rows or no columns, more than
        one level of column names or a repeated name, or a column with a
        missing value or a value other than True, False, 0 and 1. Also if
        `target` is not one-dimensional, does not give each row one label or
        has a label other than 0, 1 and missing, `weights` is not numeric or
        does not give each row one finite, non-negative weight with a
        positive total, or a Series argument has an index other than that of
        `rules`.

    See Also
    --------
    rule_table : The hits of each rule on its own.
    kivi.strategy.cutoff_table : The same cumulative columns for the cut-offs of a score.

    Notes
    -----
    The new hits of a rule are the rows that it hits and that no rule
    before it hits. The columns are:

    - ``new_hits``: the new hits of the rule, or the sum of their weights;
    - ``new_event``: the events among the new hits;
    - ``approved``: the rows that no rule up to this one hits;
    - ``approval_rate``: ``approved`` divided by the whole sample;
    - ``approved_event``: the events among the approved rows;
    - ``approved_event_rate``: ``approved_event`` divided by the approved
      rows with a known outcome, NaN when none is known;
    - ``rejected_event_share`` and ``rejected_non_event_share``: the shares
      of all known events and of all known non-events that a rule up to
      this one hits, NaN when the sample has no known event or non-event.

    Every hit counts once, under the first rule that hits it, so the new
    hits of all rules add up to the rows that any rule hits. The new hits of
    a rule depend on the rules before it, and moving a rule forward gives it
    more. The last row describes the whole set: its cumulative columns do
    not depend on the order of the rules.

    With every outcome known, the cumulative columns of a row equal those of
    ``kivi.strategy.cutoff_table(score, target, cutoffs=[0])`` for a score
    that is 1 where a rule up to that row hits and 0 elsewhere. With unknown
    outcomes, ``approved`` and ``approval_rate`` still count every row,
    while the event rate and the shares count only the known outcomes.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.rules import rule_set_table
    >>> rules = pd.DataFrame({
    ...     "high_debt": [1, 1, 0, 0, 0, 1, 0, 0],
    ...     "young": [0, 1, 1, 0, 0, 0, 0, 0],
    ...     "inquiries": [0, 0, 1, 1, 0, 0, 0, 0],
    ... })
    >>> target = [1, 1, 0, 1, 0, 0, 0, 0]
    >>> table = rule_set_table(rules, target)
    >>> table[["new_hits", "new_event", "approved", "approval_rate"]]
               new_hits  new_event  approved  approval_rate
    rule
    high_debt         3          2         5          0.625
    young             1          0         4          0.500
    inquiries         1          1         3          0.375
    >>> table[["approved_event", "approved_event_rate"]]
               approved_event  approved_event_rate
    rule
    high_debt               1                 0.20
    young                   1                 0.25
    inquiries               0                 0.00
    >>> table[["rejected_event_share", "rejected_non_event_share"]].round(3)
               rejected_event_share  rejected_non_event_share
    rule
    high_debt                 0.667                       0.2
    young                     0.667                       0.4
    inquiries                 1.000                       0.4

    Applying ``inquiries`` first gives it both of its hits and leaves
    ``high_debt`` two new ones, while the last row stays the same:

    >>> rule_set_table(rules[["inquiries", "young", "high_debt"]], target)[
    ...     ["new_hits", "new_event", "approved", "approved_event"]
    ... ]
               new_hits  new_event  approved  approved_event
    rule
    inquiries         2          1         6               2
    young             1          1         5               1
    high_debt         2          1         3               0
    """
    hits, mass, outcomes = _inputs(rules, target, weights)
    count = hits.shape[1]
    # The position of the first rule that hits each row, or the number of rules if none does.
    first = np.where(hits.any(axis=1), hits.argmax(axis=1), count)
    total = _total(np.ones(len(hits), dtype=bool), mass)

    def by_rule(selected: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return the new rows of each rule and the rows left after it, of the selected rows."""
        if mass is None:
            codes = first if selected is None else first[selected]
            sums: np.ndarray = np.bincount(codes, minlength=count + 1)
        else:
            sums = np.bincount(
                first, weights=mass if selected is None else selected * mass, minlength=count + 1
            )
        # The rows that no rule up to rule j hits: those first hit after it, or never.
        left = np.cumsum(sums[::-1])[::-1][1:]
        return sums[:count], left

    new_hits, approved = by_rule()
    columns = {"new_hits": new_hits}
    if outcomes is not None:
        known, event = outcomes
        non_event = known & ~event
        new_event, approved_event = by_rule(event)
        approved_known = by_rule(known)[1]
        approved_non_event = by_rule(non_event)[1]
        total_event, total_non_event = _total(event, mass), _total(non_event, mass)
        columns["new_event"] = new_event
    columns["approved"] = approved
    columns["approval_rate"] = approved / total
    if outcomes is not None:
        columns["approved_event"] = approved_event
        columns["approved_event_rate"] = _counts.ratio(approved_event, approved_known)
        columns["rejected_event_share"] = _counts.ratio(
            total_event - approved_event, np.full(count, total_event)
        )
        columns["rejected_non_event_share"] = _counts.ratio(
            total_non_event - approved_non_event, np.full(count, total_non_event)
        )
    return pd.DataFrame(columns, index=rules.columns.rename("rule"))


def _inputs(
    rules: pd.DataFrame, target: npt.ArrayLike | None, weights: npt.ArrayLike | None
) -> tuple[
    npt.NDArray[np.bool_],
    npt.NDArray[np.float64] | None,
    tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]] | None,
]:
    """Return the hits of every rule as columns, the weights, the known outcomes and events."""
    if not isinstance(rules, pd.DataFrame):
        raise ValueError(
            f"rules must be a DataFrame with one column per rule, got {type(rules).__name__}"
        )
    _validation.shared_index({"rules": rules, "target": target, "weights": weights}, frames=True)
    if not len(rules.columns):
        raise ValueError("rules must have at least one column")
    if isinstance(rules.columns, pd.MultiIndex):
        raise ValueError(f"rules must have one level of column names, got {rules.columns.nlevels}")
    if rules.columns.has_duplicates:
        repeated = _validation.shown(rules.columns[rules.columns.duplicated()])
        raise ValueError(f"rules must have unique column names, got {repeated} twice")
    if not len(rules):
        raise ValueError("rules must have at least one row")
    columns = []
    for name in rules.columns:
        values = rules[name]
        argument = f"rules column {_validation.shown(name)}"
        missing = values.isna().to_numpy(dtype=bool)
        if missing.any():
            raise ValueError(
                f"{argument} must not have missing values, got {missing.sum()}; decide whether "
                "the rule hits when its data are missing"
            )
        columns.append(_validation.as_flags(values, argument))
    hits = np.column_stack(columns)
    outcomes = None
    if target is not None:
        length = len(_validation.as_series(target, "target"))
        if length != len(rules):
            raise ValueError(
                f"target must have one label per row of rules, got {length} for {len(rules)}"
            )
        outcomes = _validation.known_outcomes(target, len(rules), "rules")
    if weights is not None:
        length = len(_validation.as_series(weights, "weights"))
        if length != len(rules):
            raise ValueError(
                f"weights must have one weight per row of rules, got {length} for {len(rules)}"
            )
    mass = _validation.as_weights(weights, rules, of="rules")
    return hits, mass, outcomes


def _sums(selected: np.ndarray, mass: np.ndarray | None) -> np.ndarray:
    """Column sums of a boolean matrix: counts without weights, else sums of the weights."""
    if mass is None:
        return selected.sum(axis=0, dtype=np.int64)
    return np.where(selected, mass[:, None], 0.0).sum(axis=0)


def _total(selected: np.ndarray, mass: np.ndarray | None) -> float:
    """Return the number of selected rows, or the sum of their weights."""
    if mass is None:
        return int(selected.sum())
    return float(mass[selected].sum())
