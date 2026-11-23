"""Portfolio analysis of loan panels.

``delinquency_panel`` builds a monthly loan panel from the instalment
schedules and the payments of the loans. ``vintage_table`` follows the
cumulative bad rate of each cohort as it ages. ``transition_matrix`` counts
the movements of loans or obligors between delinquency buckets or rating
grades, and ``roll_rates`` reduces it to the share of each bucket that rolls
to the next worse one.
"""

import math
import numbers

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import _validation

__all__ = ["delinquency_panel", "vintage_table", "transition_matrix", "roll_rates"]

# The delinquency buckets of make_loan_panel and the days past due at which each one starts.
_STATUSES = ("current", "1-29", "30-59", "60-89", "90+")
_BUCKET_DAYS = np.array([1, 30, 60, 90])
# Amounts within this share of the cumulative amount due count as paid, so that rounding in
# floating-point sums does not leave an instalment paid in pieces unpaid.
_RELATIVE_SLACK = 1e-10


def delinquency_panel(
    loan: npt.ArrayLike,
    due_date: npt.ArrayLike,
    amount_due: npt.ArrayLike,
    payment_loan: npt.ArrayLike,
    payment_date: npt.ArrayLike,
    payment_amount: npt.ArrayLike,
    *,
    end: object,
    principal: npt.ArrayLike | None = None,
    tolerance: float = 0.0,
) -> pd.DataFrame:
    r"""Build a monthly delinquency panel from instalment schedules and payments.

    At every month end, a loan is behind by the instalments that have fallen
    due and are not paid yet. This function applies the payments of each
    loan to its instalments, the oldest first, and gives one row per loan
    and month end with the days past due, the delinquency status and the
    amounts past due and outstanding: the loan panel that `vintage_table`,
    `transition_matrix` and `roll_rates` take.

    Parameters
    ----------
    loan : array_like
        Loan of each instalment of the schedule. The instalments need not be
        sorted.
    due_date : array_like
        Due date of each instalment, as dates or as strings that
        `pandas.to_datetime` reads. A time of day is ignored.
    amount_due : array_like
        Amount of each instalment, finite and non-negative, such as the
        principal and the interest due on that date.
    payment_loan : array_like
        Loan of each payment. Every one must be a loan of the schedule, and a
        loan without payments has paid nothing. The payments need not be
        sorted, and their Series may have any index.
    payment_date : array_like
        Date of each payment, read like `due_date`.
    payment_amount : array_like
        Amount of each payment, finite and non-negative. Net a reversal
        against the payment that it reverses.
    end : date or month
        Last month of the panel, such as ``"2026-01"``, a date in that month
        or a `pandas.Period`. A string is read as `pandas.Period` reads it,
        and a period longer than a month, such as ``"2025Q4"``, ends in its
        last month. The payments must be complete up to its last day.
    principal : array_like, optional
        Principal part of each instalment, between 0 and its amount due,
        matched to the schedule like `amount_due`. The panel then gets a
        ``balance`` column.
    tolerance : float, default 0.0
        Largest unpaid amount of an instalment that still counts as paid,
        such as a few cents of rounding. It decides only which instalments
        count as paid; the amounts in the panel stay exact.

    Returns
    -------
    pandas.DataFrame
        One row per loan and month end, sorted by loan and then by month,
        with the columns described in the Notes.

    Raises
    ------
    ValueError
        If the inputs of the schedule, or those of the payments, are not
        one-dimensional, have different lengths or are Series with different
        indices, or if the schedule is empty. Also if a loan or a date is
        missing, a date cannot be read, an amount is missing, negative,
        infinite or not a number, `principal` exceeds `amount_due`, a payment
        belongs to a loan that is not in the schedule, `end` is not a date or
        a month or comes before the month of every due date, or `tolerance`
        is not a finite, non-negative number.

    See Also
    --------
    vintage_table : Cumulative bad rates by cohort and month on book.
    transition_matrix : Movements between delinquency buckets.
    kivi.datasets.make_loan_panel : Synthetic loan panel with the same statuses.

    Notes
    -----
    A loan has a row at the end of every month from the month of its first
    due date to `end`. It leaves the panel after the month end at which all
    its instalments are paid, also when it is repaid early, while a loan
    that is still behind after its last due date keeps rolling until `end`.

    At each month end :math:`t`, only the payments made on or before
    :math:`t` count. They go to the instalments in the order of their due
    dates, the oldest unpaid one first and instalments due on the same day
    in the order of the schedule, so an amount paid before an instalment
    falls due counts towards it. With :math:`C_k` the total amount due by
    instalment :math:`k` of a loan in that order, and :math:`P_t` the total
    paid by :math:`t`, instalment :math:`k` counts as paid at :math:`t` if

    .. math::

       C_k - P_t \le \mathrm{tolerance},

    that is, once at most `tolerance` remains unpaid of it and the
    instalments before it together. A partial payment leaves an instalment
    unpaid until then, and an unpaid remainder within `tolerance` is still
    owed: the next payments go to it first. Amounts within a relative
    :math:`10^{-10}` of :math:`C_k` count as paid, so that the rounding of
    sums does not leave unpaid an instalment paid in parts.

    If :math:`d` is the due date of the oldest instalment that is not paid
    at :math:`t`, the loan is :math:`t - d` days past due when :math:`d` is
    before :math:`t`, and 0 days otherwise; an instalment is not past due on
    its due date.

    The columns are:

    - ``loan``: the loan, with the labels of `loan`;
    - ``month``: the month, a monthly `pandas.Period`, observed on its last
      day;
    - ``month_on_book``: months since the month before the loan's first due
      date, so 1 in the month of the first due date. This is the month on
      book of `kivi.datasets.make_loan_panel`, whose first instalment falls
      due one month after origination;
    - ``days_past_due``: the days past due, as defined above;
    - ``status``: the days past due as an ordered categorical with the
      categories ``current`` (0 days), ``1-29``, ``30-59``, ``60-89`` and
      ``90+`` (90 days or more), the statuses of
      `kivi.datasets.make_loan_panel`;
    - ``overdue_amount``: the unpaid amount of the instalments due before
      the month end;
    - ``outstanding``: the unpaid amount of all instalments, past due or not;
    - ``balance``, only with `principal`: the unpaid principal of all
      instalments. A payment pays the interest of an instalment, its amount
      due less its principal, before its principal.

    Days past due at month ends do not always move one bucket a month. An
    instalment due on 31 January and still unpaid is 29 days past due at the
    end of February and 60 days at the end of March, so it skips ``30-59``.

    Examples
    --------
    >>> from kivi.vintage import delinquency_panel
    >>> loan = ["A"] * 3 + ["B"] * 3
    >>> due_date = ["2024-01-15", "2024-02-15", "2024-03-15"] * 2
    >>> amount_due = [100.0] * 6
    >>> payment_loan = ["A", "A", "B"]
    >>> payment_date = ["2024-01-15", "2024-03-20", "2024-01-12"]
    >>> payment_amount = [100.0, 200.0, 200.0]
    >>> panel = delinquency_panel(
    ...     loan, due_date, amount_due, payment_loan, payment_date, payment_amount, end="2024-05"
    ... )
    >>> panel.drop(columns="month_on_book")
      loan    month  days_past_due   status  overdue_amount  outstanding
    0    A  2024-01              0  current             0.0        200.0
    1    A  2024-02             14     1-29           100.0        200.0
    2    A  2024-03              0  current             0.0          0.0
    3    B  2024-01              0  current             0.0        100.0
    4    B  2024-02              0  current             0.0        100.0
    5    B  2024-03             16     1-29           100.0        100.0
    6    B  2024-04             46    30-59           100.0        100.0
    7    B  2024-05             77    60-89           100.0        100.0

    Loan A missed its February instalment and paid it with the March one on
    20 March, after the end of February, so it was 14 days past due at the
    end of February and is repaid at the end of March. Loan B paid two
    instalments in January, which keeps it current in February, and has not
    paid the last one, so it keeps rolling after its last due date.
    """
    schedule = {"loan": loan, "due_date": due_date, "amount_due": amount_due}
    if principal is not None:
        schedule["principal"] = principal
    payments = {
        "payment_loan": payment_loan,
        "payment_date": payment_date,
        "payment_amount": payment_amount,
    }
    _validation.shared_index(schedule)
    _validation.shared_index(payments)
    columns = {name: _validation.as_series(values, name) for name, values in schedule.items()}
    paid = {name: _validation.as_series(values, name) for name, values in payments.items()}
    for group in (columns, paid):
        lengths = [len(values) for values in group.values()]
        if len(set(lengths)) > 1:
            names = list(group)
            raise ValueError(
                f"{', '.join(names[:-1])} and {names[-1]} must have the same length, got "
                f"{', '.join(map(str, lengths[:-1]))} and {lengths[-1]}"
            )
    if not len(columns["loan"]):
        raise ValueError("loan, due_date and amount_due must not be empty")
    end_month = _end_month(end)
    if (
        isinstance(tolerance, (bool, np.bool_))
        or not _validation.is_real(tolerance)
        or not (math.isfinite(tolerance) and tolerance >= 0)
    ):
        shown = _validation.shown_argument(tolerance)
        raise ValueError(f"tolerance must be finite and non-negative, got {shown}")

    loan_code, loan_label = pd.factorize(columns["loan"], sort=True)
    _check_not_missing(loan_code, "loan")
    due_day = _days(columns["due_date"], "due_date")
    due = _amounts(columns["amount_due"], "amount_due")
    if principal is not None:
        part = _amounts(columns["principal"], "principal")
        above = np.flatnonzero(part > due)
        if len(above):
            row = above[0]
            raise ValueError(
                f"principal must not exceed amount_due, got {float(part[row])!r} for an amount "
                f"due of {float(due[row])!r}"
            )
    pay_loan = paid["payment_loan"]
    missing = pay_loan.isna().to_numpy(dtype=bool)
    if missing.any():
        raise ValueError(f"payment_loan must not contain missing values, got {missing.sum()}")
    pay_code = loan_label.get_indexer(pay_loan)  # type: ignore[arg-type]  # takes any array_like
    unknown = pay_code < 0
    if unknown.any():
        raise ValueError(
            "payment_loan must contain only loans of the schedule, got "
            f"{_validation.shown(pay_loan[unknown])}"
        )
    pay_day = _days(paid["payment_date"], "payment_date")
    pay_amount = _amounts(paid["payment_amount"], "payment_amount")

    # Instalments by loan and due date, and the total due by each instalment within its loan.
    n_loans = len(loan_label)
    order = np.lexsort((due_day, loan_code))
    inst_loan, inst_day, due = loan_code[order], due_day[order], due[order]
    cumulative_due = _cumulative(due, inst_loan)
    inst_count = np.bincount(inst_loan, minlength=n_loans)
    inst_start = np.cumsum(inst_count) - inst_count
    inst_last = inst_start + inst_count - 1
    # An instalment is settled once the total paid reaches this, and counts as paid once it
    # reaches this less the tolerance.
    exact = cumulative_due * (1 - _RELATIVE_SLACK)

    # Payments by loan and date, and the total paid by each payment within its loan.
    pay_order = np.lexsort((pay_day, pay_code))
    pay_loan_sorted, pay_day_sorted = pay_code[pay_order], pay_day[pay_order]
    cumulative_paid = _cumulative(pay_amount[pay_order], pay_loan_sorted)
    pay_count = np.bincount(pay_loan_sorted, minlength=n_loans)
    pay_start = np.cumsum(pay_count) - pay_count

    # Each loan has a row at every month end from the month of its first due date to end, or
    # to the month of the payment after which all its instalments count as paid.
    first_month = _months(inst_day[inst_start])
    last_month = np.full(n_loans, end_month)
    repaid_at = exact[inst_last] - tolerance
    short = _count_at_most(
        pay_loan_sorted,
        cumulative_paid,
        np.arange(n_loans),
        np.nextafter(repaid_at, -np.inf),
        n_loans,
    )
    repaid = short < pay_count
    last_month[repaid] = _months(pay_day_sorted[(pay_start + short)[repaid]])
    last_month = np.maximum(last_month, first_month)
    last_month[repaid_at <= 0] = first_month[repaid_at <= 0]
    months = np.maximum(np.minimum(last_month, end_month) - first_month + 1, 0)
    if not months.any():
        earliest = _month_label(int(first_month.min()))
        raise ValueError(
            f"end must not be before the month of the earliest due date, got "
            f"{_month_label(end_month)!r} before {earliest!r}"
        )
    row_loan = np.repeat(np.arange(n_loans), months)
    row_month = (
        first_month[row_loan]
        + np.arange(len(row_loan))
        - np.repeat(np.cumsum(months) - months, months)
    )
    row_day = _month_end_days(row_month)

    # Total paid by each month end.
    made = _count_at_most(pay_loan_sorted, pay_day_sorted, row_loan, row_day, n_loans)
    total_paid = np.zeros(len(row_loan))
    some = made > 0
    total_paid[some] = cumulative_paid[(pay_start[row_loan] + made - 1)[some]]

    # Instalments settled and counted as paid at each month end, and instalments due before
    # it; each count is the position within the loan of the first one that is not.
    settled = _count_at_most(inst_loan, exact, row_loan, total_paid, n_loans)
    counted = _count_at_most(inst_loan, exact - tolerance, row_loan, total_paid, n_loans)
    fallen_due = _count_at_most(inst_loan, inst_day, row_loan, row_day - 1, n_loans)
    start, size = inst_start[row_loan], inst_count[row_loan]

    late = counted < fallen_due
    days_past_due = np.zeros(len(row_loan), dtype=np.int64)
    days_past_due[late] = row_day[late] - inst_day[(start + counted)[late]]
    overdue = np.zeros(len(row_loan))
    owed = settled < fallen_due
    overdue[owed] = cumulative_due[(start + fallen_due - 1)[owed]] - total_paid[owed]
    outstanding = np.zeros(len(row_loan))
    open_ = settled < size
    outstanding[open_] = cumulative_due[inst_last[row_loan][open_]] - total_paid[open_]
    status = np.searchsorted(_BUCKET_DAYS, days_past_due, side="right")
    panel = {
        "loan": loan_label[row_loan],
        "month": _month_periods(row_month),
        "month_on_book": row_month - first_month[row_loan] + 1,
        "days_past_due": days_past_due,
        "status": pd.Categorical.from_codes(
            status, dtype=pd.CategoricalDtype(list(_STATUSES), ordered=True)
        ),
        "overdue_amount": overdue,
        "outstanding": outstanding,
    }
    if principal is not None:
        # The unpaid principal: the principal of every instalment after the first one that is
        # not settled, and of that one the part of its unpaid amount that is not interest.
        part = part[order]
        cumulative_part = _cumulative(part, inst_loan)
        balance = np.zeros(len(row_loan))
        first = (start + settled)[open_]
        remainder = np.minimum(cumulative_due[first] - total_paid[open_], due[first])
        balance[open_] = (
            cumulative_part[inst_last[row_loan][open_]]
            - cumulative_part[first]
            + np.minimum(part[first], remainder)
        )
        panel["balance"] = balance
    return pd.DataFrame(panel)


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
        Whole, non-negative number of months since origination of each row.
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
        for a month on book, `month_on_book` has a value that is not a whole,
        non-negative number, `bad` has a value other than 0 and 1, or
        `weights` is not valid or differs between the rows of a loan.

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
    _validation.shared_index(
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
        _validation.as_series(values, name)
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
    mass = _validation.as_weights(weights, raw_loan, "weights", "loan")

    loan_code, loan_label = pd.factorize(loan)
    _check_not_missing(loan_code, "loan")
    cohort_code, cohort_label = pd.factorize(cohort, sort=True)
    _check_not_missing(cohort_code, "cohort")
    mob = _whole_numbers(month_on_book, "month_on_book")
    if (mob < 0).any():
        raise ValueError(
            f"month_on_book must not be negative, got {int(mob[mob < 0][0])}; count the months "
            "from origination"
        )
    flag = _validation.as_flags(bad, "bad")

    # The cohort and the weight of each loan come from its first row; all rows must agree.
    n_loans = len(loan_label)
    _, first_row = np.unique(loan_code, return_index=True)
    loan_cohort = cohort_code[first_row]
    conflict = np.flatnonzero(loan_cohort[loan_code] != cohort_code)
    if len(conflict):
        row = conflict[0]
        first = cohort.iloc[first_row[loan_code[row]]]
        raise ValueError(
            f"each loan must have one cohort, got {_validation.shown(first)} and "
            f"{_validation.shown(cohort.iloc[row])} for loan {_validation.shown(loan.iloc[row])}"
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
                f"{_validation.shown(loan.iloc[row])}"
            )

    # Rows by loan and month on book: duplicates are adjacent, and so is each loan's first bad row.
    order = np.lexsort((mob, loan_code))
    sorted_loan, sorted_mob = loan_code[order], mob[order]
    repeated = np.flatnonzero((np.diff(sorted_loan) == 0) & (np.diff(sorted_mob) == 0))
    if len(repeated):
        row = order[repeated[0]]
        raise ValueError(
            f"each loan must have at most one row per month on book, got two rows for loan "
            f"{_validation.shown(loan.iloc[row])} at month on book {int(mob[row])}"
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
        `states` or ordered categoricals for delinquency buckets. A state
        must not be missing.
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
        is not one-dimensional or has duplicates or missing values, or is
        not given while `before` and `after` are categoricals with different
        categories, or if `normalize` is not a boolean.

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
    if not isinstance(normalize, (bool, np.bool_)):
        # pandas.crosstab takes "index", "columns" and "all", which would all give row shares.
        raise ValueError(
            f"normalize must be True or False, got {_validation.shown_argument(normalize)}"
        )
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
    _validation.shared_index({"before": before, "after": after, "weights": weights})
    raw_after = after
    before, after = _validation.as_series(before, "before"), _validation.as_series(after, "after")
    if len(before) != len(after):
        raise ValueError(
            f"before and after must have the same length, got {len(before)} and {len(after)}"
        )
    if not len(before):
        raise ValueError("before and after must not be empty")
    mass = _validation.as_weights(weights, raw_after, "weights", "after")
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
            raise ValueError(
                f"states must be one-dimensional, got {_validation.shown_argument(states)}"
            )
        labels = pd.Index(states)  # type: ignore[arg-type]  # takes any array_like
        if labels.hasnans:
            raise ValueError(
                f"states must not contain missing values, got {np.count_nonzero(labels.isna())}"
            )
        if labels.has_duplicates:
            repeated = _validation.shown(labels[labels.duplicated()])
            raise ValueError(f"states must not contain duplicates, got {repeated} more than once")
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
        first, second = _validation.shown(values), _validation.shown(values[1:])
        raise ValueError(
            f"before and after have values that cannot be sorted, such as {first} and "
            f"{second}; pass states to give their order"
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
            f"{name} must contain only values in states, got {_validation.shown(values[unknown])}"
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


def _end_month(end: object) -> int:
    """Return the month of ``end`` as months since January 1970, raising a ValueError."""
    # NumPy shows a missing datetime64 as None.
    missing_date = isinstance(end, np.datetime64) and np.isnat(end)
    shown = "NaT" if missing_date else _validation.shown_argument(end)
    message = f"end must be a date or a month, got {shown}"
    if isinstance(end, (numbers.Number, np.bool_)) or _validation.is_missing(end):
        # pandas.Timestamp reads a number as nanoseconds since 1970.
        raise ValueError(message)
    try:
        if isinstance(end, str):
            # pandas.Period reads "2024Q1" as a quarter, where pandas.Timestamp warns on
            # pandas 3 and reads it as the first day of the quarter.
            period = pd.Period(end)
        elif isinstance(end, pd.Period):
            period = end
        else:
            timestamp = pd.Timestamp(end)  # type: ignore[arg-type]  # checked by pandas
            if timestamp.tz is not None:
                timestamp = timestamp.tz_localize(None)
            period = pd.Period(timestamp, freq="M")
    except (TypeError, ValueError):
        raise ValueError(message) from None
    # An empty string and "NaT" give NaT.
    if _validation.is_missing(period):
        raise ValueError(message)
    return int(period.asfreq("M").ordinal)


def _month_label(month: int) -> str:
    """Return a month counted from January 1970 as a string such as ``"2024-01"``."""
    return str(np.datetime64(int(month), "M"))


def _months(days: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Months since January 1970 of days since 1 January 1970."""
    return days.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)


def _month_end_days(months: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Days since 1 January 1970 of the last day of months since January 1970."""
    return (months + 1).astype("datetime64[M]").astype("datetime64[D]").astype(np.int64) - 1


def _month_periods(months: npt.NDArray[np.int64]) -> pd.PeriodIndex:
    """Monthly Periods of months since January 1970."""
    return pd.DatetimeIndex(months.astype("datetime64[M]").astype("datetime64[ns]")).to_period("M")


def _days(values: pd.Series, name: str) -> npt.NDArray[np.int64]:
    """Days since 1 January 1970 of dates or strings of dates, without the time of day."""
    if not len(values):
        return np.empty(0, dtype=np.int64)
    dtype = values.dtype
    if (
        pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or isinstance(dtype, pd.PeriodDtype)
    ):
        # pandas.to_datetime reads numbers as nanoseconds since 1970.
        raise ValueError(f"{name} must hold dates, got dtype {dtype}")
    missing = values.isna().to_numpy(dtype=bool)
    if missing.any():
        raise ValueError(f"{name} must not contain missing values, got {missing.sum()}")
    try:
        dates = pd.to_datetime(values)
    except (TypeError, ValueError):
        failed = pd.to_datetime(values, errors="coerce").isna()
        value = values[failed] if failed.any() else values
        raise ValueError(f"{name} must hold dates, got {_validation.shown(value)}") from None
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    return dates.to_numpy().astype("datetime64[D]").astype(np.int64)


def _amounts(values: pd.Series, name: str) -> npt.NDArray[np.float64]:
    """Finite, non-negative amounts as floats; missing values and booleans raise."""
    array = _validation.as_real(values, name)
    missing = np.isnan(array)
    if missing.any():
        raise ValueError(f"{name} must not contain missing values, got {missing.sum()}")
    invalid = ~(np.isfinite(array) & (array >= 0))
    if invalid.any():
        raise ValueError(
            f"{name} must be finite and non-negative, got {_validation.shown(array[invalid][0])}"
        )
    return array


def _cumulative(
    values: npt.NDArray[np.float64], groups: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    """Return the running sums of values within each group, in the order of the rows.

    Each group is summed on its own, so that a small loan after large ones
    keeps the precision of its own amounts.
    """
    return np.asarray(pd.Series(values).groupby(groups).cumsum(), dtype=np.float64)


def _count_at_most(
    key_group: npt.NDArray[np.intp],
    keys: npt.NDArray[np.generic],
    query_group: npt.NDArray[np.intp],
    queries: npt.NDArray[np.generic],
    n_groups: int,
) -> npt.NDArray[np.intp]:
    """For each query, the number of keys of its group that are at most the query."""
    n_keys = len(keys)
    # Sorted by group, then value, with keys before queries on ties, so that every key at most
    # a query comes before it.
    order = np.lexsort(
        (
            np.r_[np.zeros(n_keys, dtype=bool), np.ones(len(queries), dtype=bool)],
            np.r_[keys, queries],
            np.r_[key_group, query_group],
        )
    )
    is_key = order < n_keys
    keys_before = np.cumsum(is_key) - is_key
    position = np.empty(len(order), dtype=np.intp)
    position[order] = np.arange(len(order))
    group_size = np.bincount(key_group, minlength=n_groups)
    group_start = np.cumsum(group_size) - group_size
    return keys_before[position[n_keys:]] - group_start[query_group]
