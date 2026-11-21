"""Portfolio analysis of loan panels.

``vintage_table`` follows the cumulative bad rate of each cohort as it ages.
``transition_matrix`` counts the movements of loans or obligors between
delinquency buckets or rating grades, and ``roll_rates`` reduces it to the
share of each bucket that rolls to the next worse one.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._validation import as_flags, as_series, as_weights, first_value, shared_index

__all__ = ["vintage_table", "transition_matrix", "roll_rates"]


def vintage_table(
    loan: npt.ArrayLike,
    cohort: npt.ArrayLike,
    month_on_book: npt.ArrayLike,
    bad: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Tabulate the cumulative bad rate of each cohort by month on book.

    A vintage table follows each cohort of loans, usually the loans
    originated in the same month, as it ages. Its entry for a cohort and a
    month on book is the share of the cohort's loans that were bad at or
    before that month on book, so every row is non-decreasing.

    Parameters
    ----------
    loan : array_like
        Identifier of the loan of each row of the panel. A loan has at most
        one row per month on book, and its rows need not be sorted.
    cohort : array_like
        Cohort of each row, such as the month of origination. It must be the
        same on every row of a loan. Cohorts are sorted, and categorical
        cohorts follow the order of their categories.
    month_on_book : array_like
        Whole number of months since origination of each row.
    bad : array_like
        Whether the loan is bad at that month on book, as booleans or 0 and
        1. Pass the bad definition as a condition on the status, such as
        ``status == "90+"``, or ``status >= "60-89"`` for an ordered
        categorical.
    weights : array_like, optional
        Non-negative weight of each loan, such as the amount lent, repeated
        on every row of the loan. The table then reports the share of the
        cohort's weight in loans that were bad, and cohorts whose weights
        sum to zero are left out.

    Returns
    -------
    pandas.DataFrame
        Cumulative bad rates with one row per cohort and one column per
        distinct month on book of the panel. An entry is NaN outside the
        months on book at which the cohort is observed, from the smallest to
        the largest month on book of its rows.

    Raises
    ------
    ValueError
        If the inputs are empty, not one-dimensional or of different lengths,
        or Series with different indices. Also if `loan` or `cohort` has a
        missing value, a loan has more than one cohort or more than one row
        for a month on book, `month_on_book` has a value that is not a whole
        number, `bad` has a value other than 0 and 1, or `weights` is not
        valid or differs between the rows of a loan.

    See Also
    --------
    transition_matrix : Movements between states from one month to another.
    roll_rates : Share of each delinquency bucket that rolls to the next.
    kivi.datasets.make_loan_panel : Synthetic loan panel to try this on.

    Notes
    -----
    Let :math:`L_c` be the loans of cohort :math:`c`, :math:`\tau_i` the first
    month on book at which loan :math:`i` is bad (infinite if it never is) and
    :math:`w_i` its weight, 1 without `weights`. The entry for month on book
    :math:`k` is

    .. math::

       V_{c,k} = \frac{\sum_{i \in L_c} w_i \, [\tau_i \le k]}
                      {\sum_{i \in L_c} w_i}.

    The denominator includes every loan of the cohort, also loans that left
    the panel early, for example because they were repaid; such a loan
    counts as bad only if it was bad before it left. A loan that is bad once
    stays counted as bad, even if it later cures.

    A cohort is observed from the smallest to the largest month on book of
    its rows, and its entries outside that range are NaN, not 0. Later
    months on book are missing because the cohort is too young to show
    whether its loans will go bad, and earlier ones because the panel starts
    after the cohort was originated. Loans that were bad only before the
    panel starts are not counted, so start the panel at origination where
    possible. Comparing the cohorts at the same month on book compares them
    at the same age, which separates a change in the quality of new lending
    from the seasoning of the portfolio.

    Two cases need care. If every loan of a cohort leaves the panel before
    it ends, for example because all of them were repaid, the cohort's last
    entry is final, yet its later months on book are NaN; to fill them, keep
    a row for each month end after a loan left, with a status that is not
    bad, such as ``"closed"``. And a cohort that spans several origination
    months, such as a quarter, mixes loans of different ages: at its largest
    months on book, its youngest loans count in the denominator although
    they have not been observed that long, which biases those entries down.
    Monthly cohorts avoid this.

    Examples
    --------
    >>> from kivi.vintage import vintage_table
    >>> loan = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    >>> cohort = ["2024-01"] * 6 + ["2024-02"] * 4
    >>> month_on_book = [1, 2, 3, 1, 2, 3, 1, 2, 1, 2]
    >>> status = ["current", "90+", "current", "current", "current", "current",
    ...           "90+", "90+", "current", "current"]
    >>> bad = [value == "90+" for value in status]
    >>> vintage_table(loan, cohort, month_on_book, bad)
    month_on_book    1    2    3
    cohort
    2024-01        0.0  0.5  0.5
    2024-02        0.5  0.5  NaN

    Loan 1 cured after month on book 2 but stays counted as bad, and the
    2024-02 cohort has not reached month on book 3.
    """
    names = ("loan", "cohort", "month_on_book", "bad")
    shared_index(
        {
            "loan": loan,
            "cohort": cohort,
            "month_on_book": month_on_book,
            "bad": bad,
            "weights": weights,
        }
    )
    raw_loan = loan
    loan, cohort, month_on_book, bad = (
        as_series(values, name)
        for values, name in zip((loan, cohort, month_on_book, bad), names, strict=True)
    )
    lengths = [len(loan), len(cohort), len(month_on_book), len(bad)]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"loan, cohort, month_on_book and bad must have the same length, got "
            f"{', '.join(map(str, lengths[:-1]))} and {lengths[-1]}"
        )
    if not lengths[0]:
        raise ValueError("loan, cohort, month_on_book and bad must not be empty")
    mass = as_weights(weights, raw_loan, "weights", "loan")

    loan_code, loan_label = pd.factorize(loan)
    _check_not_missing(loan_code, "loan")
    cohort_code, cohort_label = pd.factorize(cohort, sort=True)
    _check_not_missing(cohort_code, "cohort")
    mob = _whole_numbers(month_on_book, "month_on_book")
    flag = as_flags(bad, "bad")

    # The cohort and the weight of each loan come from its first row; all rows must agree.
    n_loans = len(loan_label)
    _, first_row = np.unique(loan_code, return_index=True)
    loan_cohort = cohort_code[first_row]
    conflict = np.flatnonzero(loan_cohort[loan_code] != cohort_code)
    if len(conflict):
        row = conflict[0]
        first = cohort.iloc[first_row[loan_code[row]]]
        raise ValueError(
            f"each loan must have one cohort, got {first_value(first)!r} and "
            f"{first_value(cohort.iloc[row])!r} for loan {first_value(loan.iloc[row])!r}"
        )
    loan_mass = np.ones(n_loans)
    if mass is not None:
        loan_mass = mass[first_row]
        conflict = np.flatnonzero(loan_mass[loan_code] != mass)
        if len(conflict):
            row = conflict[0]
            raise ValueError(
                f"weights must be the same on every row of a loan, got "
                f"{float(loan_mass[loan_code[row]])!r} and {float(mass[row])!r} for loan "
                f"{first_value(loan.iloc[row])!r}"
            )

    # Rows by loan and month on book: duplicates are adjacent, and so is each loan's first bad row.
    order = np.lexsort((mob, loan_code))
    sorted_loan, sorted_mob = loan_code[order], mob[order]
    repeated = np.flatnonzero((np.diff(sorted_loan) == 0) & (np.diff(sorted_mob) == 0))
    if len(repeated):
        row = order[repeated[0]]
        raise ValueError(
            f"each loan must have at most one row per month on book, got two rows for loan "
            f"{first_value(loan.iloc[row])!r} at month on book {int(mob[row])}"
        )
    columns = np.unique(mob)
    # Column position of each loan's first bad month on book; loans never bad get len(columns).
    first_bad = np.full(n_loans, len(columns))
    bad_loan, first = np.unique(sorted_loan[flag[order]], return_index=True)
    first_bad[bad_loan] = np.searchsorted(columns, sorted_mob[flag[order]][first])

    n_cohorts, width = len(cohort_label), len(columns) + 1
    bad_mass = np.bincount(
        loan_cohort * width + first_bad, weights=loan_mass, minlength=n_cohorts * width
    ).reshape(n_cohorts, width)
    total = bad_mass.sum(axis=1)
    cumulative = np.cumsum(bad_mass[:, :-1], axis=1)
    # Each cohort is observed between its smallest and its largest month on book.
    youngest, oldest = np.full(n_cohorts, columns[-1]), np.full(n_cohorts, columns[0])
    np.minimum.at(youngest, cohort_code, mob)
    np.maximum.at(oldest, cohort_code, mob)
    keep = total > 0
    rate = cumulative[keep] / total[keep, None]
    outside = (columns[None, :] < youngest[keep, None]) | (columns[None, :] > oldest[keep, None])
    rate[outside] = np.nan
    return pd.DataFrame(
        rate,
        index=cohort_label[keep].rename("cohort"),
        columns=pd.Index(columns, name="month_on_book"),
    )


def transition_matrix(
    before: npt.ArrayLike,
    after: npt.ArrayLike,
    *,
    states: npt.ArrayLike | None = None,
    normalize: bool = False,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Count the movements of entities between two states.

    Each entity, such as a loan or an obligor, is observed in one state
    before and one state after, for example its delinquency bucket at two
    consecutive month ends or its rating grade at two annual reviews. Row
    :math:`i`, column :math:`j` counts the entities that moved from state
    :math:`i` to state :math:`j`. With `normalize`, each row holds the
    shares of the states after, the migration matrix of a rating system.

    Parameters
    ----------
    before, after : array_like
        State of each entity at the first and at the second observation. If
        both are Series, they must have identical indices; otherwise they are
        matched by position. Neither may contain missing values: drop the
        entities that were not observed twice, or give the entities that left
        a state such as ``"closed"``.
    states : array_like, optional
        All states in order, for delinquency buckets and rating grades from
        the best to the worst. States without entities still get a row and a
        column. By default, the categories of categorical inputs, which must
        then be the same for both, or else the sorted distinct values.
        Sorting puts labels such as ``"current"`` after ``"90+"``, so pass
        `states` or ordered categoricals for delinquency buckets.
    normalize : bool, default False
        If True, divide each row by its total, so that it holds the shares of
        the state's entities that moved to each state. A row without entities
        is then NaN.
    weights : array_like, optional
        Non-negative weight of each entity, such as its balance at the first
        observation, matched like `after`. The counts are then sums of
        weights.

    Returns
    -------
    pandas.DataFrame
        One row per state before and one column per state after, in the
        order of `states`. Without `weights` and `normalize`, the entries are
        integer counts.

    Raises
    ------
    ValueError
        If the inputs are empty, not one-dimensional, of different lengths or
        Series with different indices, have a missing value or a value that
        is not one of `states`, or `weights` is not valid. Also if `states`
        is not one-dimensional or has duplicates, or is not given while
        `before` and `after` are categoricals with different categories.

    See Also
    --------
    roll_rates : Share of each state that moved to the next worse state.
    vintage_table : Cumulative bad rates by cohort and month on book.

    Notes
    -----
    With :math:`n_{ij}` entities moving from state :math:`i` to state
    :math:`j`, the normalized entry

    .. math::

       p_{ij} = \frac{n_{ij}}{\sum_k n_{ik}}

    estimates the probability of moving to :math:`j` from :math:`i`. If the
    movements follow a time-homogeneous Markov chain, the :math:`h`-th power
    of the normalized matrix gives the probabilities over :math:`h` periods.

    Examples
    --------
    >>> from kivi.vintage import transition_matrix
    >>> before = ["A", "A", "A", "B", "B", "C"]
    >>> after = ["A", "A", "B", "B", "C", "C"]
    >>> transition_matrix(before, after, states=["A", "B", "C", "D"])
    after   A  B  C  D
    before
    A       2  1  0  0
    B       0  1  1  0
    C       0  0  1  0
    D       0  0  0  0
    >>> transition_matrix(before, after, normalize=True).round(3)
    after       A      B    C
    before
    A       0.667  0.333  0.0
    B       0.000  0.500  0.5
    C       0.000  0.000  1.0
    """
    labels, before_code, after_code, mass = _transitions(before, after, states, weights)
    size = len(labels)
    counts = np.bincount(
        before_code * size + after_code, weights=mass, minlength=size * size
    ).reshape(size, size)
    if normalize:
        total = counts.sum(axis=1, keepdims=True)
        counts = np.divide(counts, total, out=np.full((size, size), np.nan), where=total > 0)
    return pd.DataFrame(counts, index=labels.rename("before"), columns=labels.rename("after"))


def roll_rates(
    before: npt.ArrayLike,
    after: npt.ArrayLike,
    *,
    states: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> pd.Series:
    r"""Compute the share of each state's entities that rolled to the next state.

    Roll rates summarize the movements of delinquency between two month
    ends: the roll rate of ``30-59`` days past due is the share of the loans
    at 30 to 59 days that are at 60 to 89 days a month later. The states must
    be ordered from the best to the worst, so that the next state is the next
    worse one.

    Parameters
    ----------
    before, after : array_like
        State of each entity at the first and at the second observation, as
        in `transition_matrix`. Neither may contain missing values.
    states : array_like, optional
        All states from the best to the worst. Required unless `before` and
        `after` are ordered categoricals with the same categories, whose
        order is then used. A sorted order would be wrong for most labels,
        for example ``"current"`` would come after ``"90+"``.
    weights : array_like, optional
        Non-negative weight of each entity, such as its balance at the first
        observation, matched like `after`. The rates are then shares of
        weight, the balance roll rates.

    Returns
    -------
    pandas.Series
        Roll rate of every state except the worst, which has no next state,
        indexed by the state. A state without entities has a NaN rate.

    Raises
    ------
    ValueError
        In the same cases as `transition_matrix`, and if `states` is not
        given while `before` and `after` are not ordered categoricals.

    See Also
    --------
    transition_matrix : The movements between all pairs of states.
    vintage_table : Cumulative bad rates by cohort and month on book.

    Notes
    -----
    With :math:`n_{ij}` entities moving from state :math:`i` to state
    :math:`j`, the roll rate of state :math:`i` is

    .. math::

       r_i = \frac{n_{i,i+1}}{\sum_k n_{ik}}.

    Entities that improved, stayed or moved more than one state worse count
    in the denominator only. A loan can fall at most one delinquency bucket
    further behind from one month end to the next, so the product of the
    roll rates estimates the share of current loans that reach the worst
    bucket after one month per roll, if the movements follow a Markov chain.

    Examples
    --------
    >>> from kivi.vintage import roll_rates
    >>> states = ["current", "30", "60", "90+"]
    >>> before = ["current"] * 4 + ["30"] * 3 + ["60"] * 2 + ["90+"]
    >>> after = ["current", "current", "current", "30",
    ...          "current", "30", "60", "60", "90+", "90+"]
    >>> roll_rates(before, after, states=states).round(3)
    before
    current    0.250
    30         0.333
    60         0.500
    Name: roll_rate, dtype: float64
    """
    ordered = all(
        isinstance(getattr(values, "dtype", None), pd.CategoricalDtype) and values.dtype.ordered  # type: ignore[union-attr]  # a CategoricalDtype
        for values in (before, after)
    )
    if states is None and not ordered:
        raise ValueError(
            "roll_rates needs the states from the best to the worst: pass states, or pass "
            "before and after as ordered categoricals"
        )
    labels, before_code, after_code, mass = _transitions(before, after, states, weights)
    size = len(labels)
    total = np.bincount(before_code, weights=mass, minlength=size)[:-1]
    forward = after_code == before_code + 1
    rolled = np.bincount(
        before_code[forward], weights=None if mass is None else mass[forward], minlength=size
    )[:-1]
    rate = np.divide(rolled, total, out=np.full(len(total), np.nan), where=total > 0)
    return pd.Series(rate, index=labels[:-1].rename("before"), name="roll_rate")


def _transitions(
    before: npt.ArrayLike,
    after: npt.ArrayLike,
    states: npt.ArrayLike | None,
    weights: npt.ArrayLike | None,
) -> tuple[pd.Index, npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.float64] | None]:
    """States as an Index, the state code of each entity before and after, and the weights."""
    shared_index({"before": before, "after": after, "weights": weights})
    raw_after = after
    before, after = as_series(before, "before"), as_series(after, "after")
    if len(before) != len(after):
        raise ValueError(
            f"before and after must have the same length, got {len(before)} and {len(after)}"
        )
    if not len(before):
        raise ValueError("before and after must not be empty")
    mass = as_weights(weights, raw_after, "weights", "after")
    for values, name in ((before, "before"), (after, "after")):
        missing = values.isna().to_numpy(dtype=bool)
        if missing.any():
            raise ValueError(
                f"{name} must not contain missing values, got {missing.sum()}; drop the entities "
                f"that were not observed twice or give them a state of their own"
            )
    labels = _state_labels(before, after, states)
    before_code = _state_codes(before, labels, "before")
    return labels, before_code, _state_codes(after, labels, "after"), mass


def _state_labels(before: pd.Series, after: pd.Series, states: npt.ArrayLike | None) -> pd.Index:
    """Return the states in order: as given, the shared categories or the sorted values."""
    if states is not None:
        if np.ndim(states) != 1:
            raise ValueError(f"states must be one-dimensional, got {states!r}")
        labels = pd.Index(states)  # type: ignore[arg-type]  # takes any array_like
        if labels.has_duplicates:
            repeated = first_value(labels[labels.duplicated()])
            raise ValueError(f"states must not contain duplicates, got {repeated!r} more than once")
        return labels
    categories = [
        values.dtype.categories
        for values in (before, after)
        if isinstance(values.dtype, pd.CategoricalDtype)
    ]
    if len(categories) == 2 and not categories[0].equals(categories[1]):
        raise ValueError(
            "before and after must have the same categories when states is not given, got "
            f"{list(categories[0])} and {list(categories[1])}"
        )
    if categories:
        return categories[0]
    values = pd.Index(pd.concat([before, after], ignore_index=True).unique())
    try:
        return values.sort_values()
    except TypeError:
        first, second = first_value(values), first_value(values[1:])
        raise ValueError(
            f"before and after have values that cannot be sorted, such as {first!r} and "
            f"{second!r}; pass states to give their order"
        ) from None


def _state_codes(values: pd.Series, labels: pd.Index, name: str) -> npt.NDArray[np.intp]:
    """Position of each value in ``labels``; a value that is not a state raises."""
    # Categorical codes avoid hashing the labels, which warns for intervals on NumPy < 2.2.
    if isinstance(values.dtype, pd.CategoricalDtype) and values.dtype.categories.equals(labels):
        codes = values.cat.codes.to_numpy()
    else:
        codes = labels.get_indexer(values)  # type: ignore[arg-type]  # takes any array_like
    unknown = codes < 0
    if unknown.any():
        raise ValueError(
            f"{name} must contain only values in states, got {first_value(values[unknown])!r}"
        )
    return codes.astype(np.intp)


def _check_not_missing(codes: np.ndarray, name: str) -> None:
    missing = codes < 0
    if missing.any():
        raise ValueError(f"{name} must not contain missing values, got {missing.sum()}")


def _whole_numbers(values: pd.Series, name: str) -> npt.NDArray[np.int64]:
    """Finite whole numbers as int64, from any real numeric dtype including nullable ones."""
    dtype = values.dtype
    if (
        not pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_complex_dtype(dtype)
    ):
        raise ValueError(f"{name} must contain whole numbers, got dtype {dtype}")
    array = values.to_numpy(dtype=float, na_value=np.nan)
    invalid = ~np.isfinite(array) | (array != np.round(array))
    if invalid.any():
        raise ValueError(f"{name} must contain whole numbers, got {float(array[invalid][0])!r}")
    return array.astype(np.int64)
