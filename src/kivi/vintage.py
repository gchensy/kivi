"""Portfolio analysis of loan panels.

``delinquency_panel`` builds a monthly loan panel from the instalment
schedules and the payments of the loans. ``vintage_table`` follows the
cumulative bad rate of each cohort as it ages, and ``default_rate_table``
gives the one-year default rate of each grade from a series of start
months, as PD calibration uses it. ``transition_matrix`` counts
the movements of loans or obligors between delinquency buckets or rating
grades, and ``roll_rates`` reduces it to the share of each bucket that rolls
to the next worse one. ``migration_bandwidth`` and ``migration_test`` summarize
and test the migrations between rating grades as the European Central Bank asks
in the validation of PD models.
"""

import math
import numbers
import typing
from collections import abc

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import _stats, _validation

__all__ = [
    "delinquency_panel",
    "vintage_table",
    "default_rate_table",
    "transition_matrix",
    "roll_rates",
    "migration_bandwidth",
    "migration_test",
]

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
    origination: pd.Series | abc.Mapping[typing.Any, typing.Any] | None = None,
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
    origination : pandas.Series or mapping, optional
        Origination date of each loan, as a Series indexed by loan or a
        mapping from loan to date, read like `due_date`. It must give a date
        for every loan of the schedule and may hold other loans too. The
        cohorts are then the origination months, and the months on book
        count from them. By default, both come from the first due dates.
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
        with the columns described in the Notes: those of
        `kivi.datasets.make_loan_panel` that a schedule gives, in the same
        order, and the days past due and the amounts owed.

    Raises
    ------
    ValueError
        If the inputs of the schedule, or those of the payments, are not
        one-dimensional, have different lengths or are Series with different
        indices, or if the schedule is empty. Also if a loan or a date is
        missing, a date cannot be read, an amount is missing, negative,
        infinite or not a number, `principal` exceeds `amount_due`, a payment
        belongs to a loan that is not in the schedule, `end` is not a date or
        a month or comes before the month of every due date, `origination`
        is not a Series or a mapping, repeats a loan, lacks a loan of the
        schedule or its date, or comes after the first due date of its loan,
        or `tolerance` is not a finite, non-negative number.

    See Also
    --------
    vintage_table : Cumulative bad rates by cohort and month on book.
    transition_matrix : Movements between delinquency buckets.
    kivi.datasets.make_loan_panel : Synthetic loan panel with the same statuses.

    Notes
    -----
    A loan has a row at the end of every month from the month of its first
    due date to `end`. It leaves the panel after the month end at which all
    its instalments count as paid, also when it is repaid early, while a
    loan that is still behind after its last due date keeps rolling until
    `end`. Payments after `end` are not used, and neither are amounts paid
    beyond the total due, such as an overpayment or a payment after the loan
    was repaid: there is no credit balance.

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
    owed: the next payments go to it first. A loan therefore leaves the
    panel once at most `tolerance` of it is unpaid, so a `tolerance` of at
    least its total due makes a loan that pays nothing leave in its first
    month. Amounts within a relative :math:`10^{-10}` of :math:`C_k` count
    as paid, so that the rounding of sums does not leave unpaid an
    instalment paid in parts. Instalments due on the same day add up, so a
    schedule row that a faulty join repeats doubles the amount due.

    If :math:`d` is the due date of the oldest instalment that is not paid
    at :math:`t`, the loan is :math:`t - d` days past due when :math:`d` is
    before :math:`t`, and 0 days otherwise; an instalment is not past due on
    its due date.

    The columns are:

    - ``loan_id``: the loan, with the labels of `loan`;
    - ``cohort``: the origination month of the loan with `origination`, and
      otherwise the month before its first due date, a monthly
      `pandas.Period`;
    - ``month``: the month, a monthly `pandas.Period`, observed on its last
      day;
    - ``month_on_book``: ``month - cohort`` in months. Without
      `origination`, it is 1 in the month of the first due date, as in
      `kivi.datasets.make_loan_panel`, whose first instalment falls due one
      month after origination. With `origination`, a loan whose first
      instalment falls due later, after a grace period, has its first row
      at a later month on book;
    - ``days_past_due``: the days past due, as defined above;
    - ``overdue_amount``: the unpaid amount of the instalments due before
      the month end;
    - ``outstanding``: the unpaid amount of all instalments, past due or not;
    - ``balance``, only with `principal`: the unpaid principal of all
      instalments. A payment pays the interest of an instalment, its amount
      due less its principal, before its principal;
    - ``status``: the days past due as an ordered categorical with the
      categories ``current`` (0 days), ``1-29``, ``30-59``, ``60-89`` and
      ``90+`` (90 days or more), the statuses of
      `kivi.datasets.make_loan_panel`.

    Without `origination`, loans originated in the same month but with
    grace periods of different lengths fall into different cohorts. With
    it, a cohort whose loans all have a grace period starts at a later month
    on book in `vintage_table`, which leaves the earlier ones NaN.

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
    >>> panel.drop(columns=["cohort", "month_on_book"])
      loan_id    month  days_past_due  overdue_amount  outstanding   status
    0       A  2024-01              0             0.0        200.0  current
    1       A  2024-02             14           100.0        200.0     1-29
    2       A  2024-03              0             0.0          0.0  current
    3       B  2024-01              0             0.0        100.0  current
    4       B  2024-02              0             0.0        100.0  current
    5       B  2024-03             16           100.0        100.0     1-29
    6       B  2024-04             46           100.0        100.0    30-59
    7       B  2024-05             77           100.0        100.0    60-89

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
    origin_day = None if origination is None else _origination_days(origination, loan_label)

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
    first_day = inst_day[inst_start]
    first_month = _months(first_day)
    if origin_day is None:
        cohort = first_month - 1
    else:
        early = np.flatnonzero(first_day < origin_day)
        if len(early):
            first = early[0]
            shown = _validation.shown(loan_label[first])
            raise ValueError(
                f"due_date must not be before the origination of its loan, got "
                f"{_day_label(first_day[first])!r} for loan {shown}, originated on "
                f"{_day_label(origin_day[first])!r}"
            )
        cohort = _months(origin_day)
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
        "loan_id": loan_label[row_loan],
        "cohort": _month_periods(cohort[row_loan]),
        "month": _month_periods(row_month),
        "month_on_book": row_month - cohort[row_loan],
        "days_past_due": days_past_due,
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
    panel["status"] = pd.Categorical.from_codes(
        status, dtype=pd.CategoricalDtype(list(_STATUSES), ordered=True)
    )
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


def default_rate_table(
    loan: npt.ArrayLike,
    month: npt.ArrayLike,
    grade: npt.ArrayLike,
    defaulted: npt.ArrayLike,
    *,
    horizon: int = 12,
    starts: npt.ArrayLike | None = None,
    probability: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Tabulate the one-year default rate of each grade by the cohort approach.

    Each start month forms a cohort of the loans that are not in default at
    its end, grouped by their grade at that month. The default rate of a
    grade is the share of its cohort that is in default in at least one
    month of the `horizon` months that follow, as the one-year default rate
    of IRB PD calibration and validation counts it. Loans that leave the
    panel during those months stay in the cohort.

    Parameters
    ----------
    loan : array_like
        Identifier of the loan of each row of a monthly panel, or of the
        obligor where the definition of default applies to obligors. A loan
        has at most one row per month, and its rows need not be sorted.
    month : array_like
        Month of each row, for the status at its end: monthly periods, or
        datetimes, which stand for their month on their own calendar, such
        as the ``month`` of `delinquency_panel`.
    grade : array_like
        Grade or pool of each row. Only the row of a start month counts, so
        later migrations do not move a loan to another grade. Categorical
        grades follow the order of their categories and other grades are
        sorted. A missing grade forms a row of its own, listed last.
    defaulted : array_like
        Whether the loan is in default in that month, as booleans or 0 and
        1, such as ``status == "90+"``.
    horizon : int, default 12
        Number of months that follow each start month, 12 for the one-year
        default rate.
    starts : array_like, optional
        Start months of the cohorts, as periods, dates or strings, which are
        read as a `pandas.Period` and stand for its last month, so
        ``"2024Q2"`` is June 2024. The `horizon` months after each one must
        lie in the panel. By default, every quarter end from the first month
        of the panel whose `horizon` months end at the last month of the
        panel or before.
    probability : array_like, optional
        PD of each row, between 0 and 1. The table then gives the mean PD
        of each cohort at its start month. Only the rows of the cohorts are
        checked for missing values, so defaulted rows may leave it empty.
    weights : array_like, optional
        Non-negative frequency weight of each row, such as the inverse of a
        sampling rate; the weight of the start row of a loan counts. Groups
        whose weights sum to zero are left out. Do not weight by exposure for
        PD calibration, which counts obligors (Basel CRE36.78).

    Returns
    -------
    pandas.DataFrame
        One row per start month and grade with obligors in the cohort, with
        a MultiIndex of ``start``, monthly periods, and ``grade``, sorted by
        start month and then by grade. The columns are:

        - ``total``: the loans of the cohort, or their weight;
        - ``event``: those in default in at least one of the `horizon`
          months after the start month;
        - ``exited``: those not in default in those months whose last row
          is before the last of them, such as loans repaid, sold or written
          off; they count in ``total``;
        - ``event_rate``: ``event / total``, the default rate;
        - ``mean_probability``: the mean PD of the cohort at the start
          month, only with `probability`.

        The counts are integers without `weights` and sums of weights with
        them.

    Raises
    ------
    ValueError
        If the inputs are empty, not one-dimensional or of different lengths,
        or Series with different indices. Also if `loan` or `month` has a
        missing value, a loan has two rows in one month, `month` does not
        hold monthly periods or datetimes, `defaulted` has a value other
        than 0 and 1, `horizon` is not a positive integer, `starts` is not a
        list of distinct months whose `horizon` months lie in the panel, the
        panel covers no quarter end and the `horizon` months after it, or
        `probability` or `weights` is not valid.

    See Also
    --------
    vintage_table : Cumulative bad rates by origination cohort and month on book.
    transition_matrix : Movements between grades from one month to another.
    kivi.calibration.jeffreys_test : Test the PD of each grade against its default rate.
    kivi.calibration.central_tendency_shift : Shift PDs to a long-run average default rate.
    kivi.capital.vasicek_fit : Long-run PD and asset correlation from yearly default rates.

    Notes
    -----
    Let :math:`C_{r,g}` be the loans that have a row at start month
    :math:`r`, are not in default in it and have grade :math:`g` there, and
    :math:`w_i` the weight of the row of loan :math:`i` at :math:`r`, 1
    without `weights`. With :math:`h` the `horizon`, the default rate is

    .. math::

       \mathrm{DR}_{r,g} = \frac{\sum_{i \in C_{r,g}} w_i \,
           [\text{loan } i \text{ is in default in a month } m,\ r < m \le r + h]}
           {\sum_{i \in C_{r,g}} w_i}.

    This is the one-year default rate of the EBA guidelines on PD estimation
    [1]_ (paragraphs 73 to 77) and of the ECB's validation reporting [2]_
    (section 2.3): the denominator holds every obligor not in default at the
    start, the numerator those with at least one default in the year, and an
    obligor that defaults twice counts once. Loans that migrate to another
    grade or model, are repaid, sold or written off during the year stay in
    the denominator, and their defaults count as long as the panel keeps
    their rows (paragraph 76). A loan without a row in some months of the
    window counts as not in default in them, so it is not censored as in a
    survival curve, which gives a higher rate when many loans leave: the
    ``exited`` column shows how many did, for the analysis of the bias that
    paragraphs 76 and 80 ask for. A missing grade is reported on its own row,
    as paragraph 75 asks for obligors without a rating at the start; the
    ECB's reporting leaves them out (section 2.5.1).

    The guidelines evaluate the one-year default rates at least quarterly
    (paragraph 78), which the default `starts` do, and leave the choice
    between overlapping windows, such as these, and non-overlapping ones,
    one start month a year, to an analysis of their bias (paragraph 80).
    The observed average default rate is the arithmetic mean of the
    one-year default rates (paragraph 81), and the long-run average is that
    mean over a period that is representative of the likely range of
    default rates, at least five years (paragraphs 82 to 86).

    Month-end statuses miss a default that starts and ends within one month,
    so `defaulted` should flag a month in which the loan was in default at
    any time, including the probation period after it. Leave out technical
    defaults and other defaults recorded in error.

    References
    ----------
    .. [1] European Banking Authority, "Guidelines on PD estimation, LGD
       estimation and the treatment of defaulted exposures",
       EBA/GL/2017/16, section 5.3, 2017.
    .. [2] European Central Bank, "Instructions for reporting the validation
       results of internal models: IRB Pillar I models for credit risk",
       February 2019.

    Examples
    --------
    >>> import pandas as pd
    >>> from kivi.vintage import default_rate_table
    >>> months = [f"2024-{number:02d}" for number in range(1, 10)]
    >>> rows = (
    ...     [(1, month, "A", month in ("2024-05", "2024-06")) for month in months]
    ...     + [(2, month, "A", False) for month in months[:4]]
    ...     + [(3, month, "B" if month < "2024-08" else "A", False) for month in months]
    ...     + [(4, month, "B", month >= "2024-08") for month in months]
    ...     + [(5, month, "A", False) for month in months[4:]]
    ... )
    >>> panel = pd.DataFrame(rows, columns=["loan", "month", "grade", "defaulted"])
    >>> panel["month"] = pd.PeriodIndex(panel["month"], freq="M")
    >>> default_rate_table(
    ...     panel["loan"], panel["month"], panel["grade"], panel["defaulted"], horizon=3
    ... )
                   total  event  exited  event_rate
    start   grade
    2024-03 A          2      1       1         0.5
            B          2      0       0         0.0
    2024-06 A          1      0       0         0.0
            B          2      1       0         0.5

    Loan 1 defaults in May and is not in the June cohort; loan 2 is repaid
    after April and counts as not in default; loan 3 stays in grade B
    although it moves to A in August; loan 5 enters the panel in May.
    """
    inputs = {
        "loan": loan,
        "month": month,
        "grade": grade,
        "defaulted": defaulted,
        "probability": probability,
        "weights": weights,
    }
    _validation.shared_index(inputs)
    raw_loan = loan
    names = ("loan", "month", "grade", "defaulted")
    loan, month, grade, defaulted = (
        _validation.as_series(values, name)
        for values, name in zip((loan, month, grade, defaulted), names, strict=True)
    )
    lengths = [len(loan), len(month), len(grade), len(defaulted)]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"loan, month, grade and defaulted must have the same length, got "
            f"{', '.join(map(str, lengths[:-1]))} and {lengths[-1]}"
        )
    if not lengths[0]:
        raise ValueError("loan, month, grade and defaulted must not be empty")
    horizon = _validation.check_count(horizon, "horizon")
    mass = _validation.as_weights(weights, raw_loan, "weights", "loan")
    loan_code, loan_label = pd.factorize(loan)
    _check_not_missing(loan_code, "loan")
    months = _panel_months(month, "month")
    flag = _validation.as_flags(defaulted, "defaulted")
    chance = None
    if probability is not None:
        chance = _validation.as_real(
            _validation.as_series(probability, "probability"), "probability", booleans=True
        )
        if len(chance) != len(loan):
            raise ValueError(
                f"probability must have one value per row of loan, got {len(chance)} for "
                f"{len(loan)}"
            )
        outside = (chance < 0) | (chance > 1)
        if outside.any():
            raise ValueError(
                f"probability must be between 0 and 1, got {_validation.shown(chance[outside][0])}"
            )

    # Rows by loan and month: two rows of a loan in one month are adjacent.
    order = np.lexsort((months, loan_code))
    sorted_loan, sorted_month = loan_code[order], months[order]
    repeated = np.flatnonzero((np.diff(sorted_loan) == 0) & (np.diff(sorted_month) == 0))
    if len(repeated):
        row = order[repeated[0]]
        raise ValueError(
            f"each loan must have at most one row per month, got two rows for loan "
            f"{_validation.shown(loan.iloc[row])} in {_month_label(months[row])}"
        )
    first, last = int(months.min()), int(months.max())
    start_months = _start_months(starts, first, last, horizon)

    # The cohort rows: a row of a start month that is not in default.
    start_of = np.searchsorted(start_months, months)
    found = start_of < len(start_months)
    found[found] = start_months[start_of[found]] == months[found]
    rows = np.flatnonzero(found & ~flag)
    if chance is not None:
        unknown = np.isnan(chance[rows])
        if unknown.any():
            raise ValueError(
                f"probability must not be missing at the start month of a cohort, got "
                f"{unknown.sum()} missing value(s)"
            )
    # Defaults of a loan up to a month: a running count of defaulted rows in the order of loan
    # and month, looked up with a key that orders the rows the same way.
    span = last - first + horizon + 1
    key = sorted_loan.astype(np.int64) * span + (sorted_month - first)
    defaults_before = np.r_[0, np.cumsum(flag[order])]
    row_key = loan_code[rows].astype(np.int64) * span + (months[rows] - first)
    event = (
        defaults_before[np.searchsorted(key, row_key + horizon, side="right")]
        > defaults_before[np.searchsorted(key, row_key, side="right")]
    )
    last_month = np.full(len(loan_label), first, dtype=np.int64)
    np.maximum.at(last_month, loan_code, months)
    exited = ~event & (last_month[loan_code[rows]] < months[rows] + horizon)

    grade_code, grade_label = _grade_codes(grade)
    width = len(grade_label) + 1  # the last column of each start month is the missing grade
    row_grade = grade_code[rows]
    cell = start_of[rows] * width + np.where(row_grade < 0, width - 1, row_grade)
    size = len(start_months) * width
    row_mass = None if mass is None else mass[rows]

    def tally(selected: npt.NDArray[np.bool_]) -> np.ndarray:
        if row_mass is None:
            return np.bincount(cell[selected], minlength=size)
        # NumPy < 2 counts an empty selection as integers, also with weights.
        return np.bincount(cell[selected], weights=row_mass[selected], minlength=size).astype(float)

    everyone = np.ones(len(rows), dtype=bool)
    total = tally(everyone)
    keep = np.flatnonzero(total > 0)
    columns: dict[str, np.ndarray] = {
        "total": total[keep],
        "event": tally(event)[keep],
        "exited": tally(exited)[keep],
    }
    columns["event_rate"] = columns["event"] / columns["total"]
    if chance is not None:
        row_weight = np.ones(len(rows)) if row_mass is None else row_mass
        weight_sum = np.bincount(cell, weights=row_weight, minlength=size)[keep]
        mean = np.bincount(cell, weights=row_weight * chance[rows], minlength=size)[keep]
        columns["mean_probability"] = mean / weight_sum
    grade_position = keep % width
    # Levels and codes, not labels: building the index from labels would hash interval grades,
    # which warns on NumPy < 2.2.
    index = pd.MultiIndex(
        levels=[_month_periods(start_months), grade_label],  # type: ignore[list-item]  # an Index is a valid level
        codes=[
            (keep // width).tolist(),
            np.where(grade_position == width - 1, -1, grade_position).tolist(),
        ],
        names=["start", "grade"],
    ).remove_unused_levels()
    return pd.DataFrame(columns, index=index)


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


def migration_bandwidth(
    before: npt.ArrayLike,
    after: npt.ArrayLike,
    *,
    grades: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
) -> pd.Series:
    r"""Measure how far customers migrate between rating grades, upwards and downwards.

    The matrix weighted bandwidth (MWB) of the European Central Bank [1]_
    summarizes a migration matrix in two numbers: how far the customers who
    moved to worse grades moved, and how far those who moved to better
    grades moved, each relative to the furthest move from their grade in
    either direction. A stable rating system has small values.

    Parameters
    ----------
    before : array_like
        Grade of each customer at the beginning of the observation period.
        Every value must be one of `grades`.
    after : array_like
        Status of each customer at the end of the period: a grade, or any
        other label, such as ``"default"`` or ``"closed"``, for a customer
        who left the grades. If `before` and `after` are both Series, they
        must have identical indices; otherwise they are matched by
        position. Neither may contain missing values.
    grades : array_like
        The :math:`K` grades of the rating scale for non-defaulted
        customers, from the best to the worst, at least two.
    weights : array_like, optional
        Non-negative frequency weight of each customer, matched like
        `after`: integer weights give the same result as repeating rows.

    Returns
    -------
    pandas.Series
        ``upper``, the MWB of the migrations to worse grades, above the
        diagonal of the migration matrix, and ``lower``, that of the
        migrations to better grades, below it, in an index named
        ``statistic``. Each is between 0 and 1, and NaN when no customer
        moved in its direction.

    Raises
    ------
    ValueError
        If the inputs are empty, not one-dimensional, of different lengths
        or Series with different indices, or have a missing value, if
        `before` has a value that is not one of `grades`, or if `weights`
        is not valid. Also if `grades` is not one-dimensional, has fewer
        than two grades, duplicates or missing values.

    See Also
    --------
    migration_test : Test the monotonicity of the migration matrix.
    transition_matrix : The migration matrix itself.

    Notes
    -----
    Number the grades from 1, the best, to :math:`K`. With :math:`N_i`
    customers in grade :math:`i` at the beginning and :math:`N_{ij}` of
    them in grade :math:`j` at the end, :math:`p_{ij} = N_{ij} / N_i`, and

    .. math::

       \mathrm{MWB}_\mathrm{upper} = \frac{1}{M_u} \sum_{i=1}^{K-1}
       \sum_{j=i+1}^{K} |i - j| \, N_i \, p_{ij}, \qquad
       M_u = \sum_{i=1}^{K-1} \max(|i - K|, |i - 1|) \, N_i
       \sum_{j=i+1}^{K} p_{ij},

    with :math:`\mathrm{MWB}_\mathrm{lower}` and :math:`M_l` defined in
    the same way over :math:`j < i`, for :math:`i` from 2 to :math:`K`.
    :math:`N_i` counts the customers who defaulted or left as well, but
    :math:`N_i \, p_{ij} = N_{ij}` counts only those who reached a grade, so
    the customers who left change neither bandwidth. With weights, the
    counts are sums of weights.

    References
    ----------
    .. [1] European Central Bank, "Instructions for reporting the
       validation results of internal models: IRB Pillar I models for
       credit risk", February 2019, section 2.5.5.1.

    Examples
    --------
    >>> from kivi.vintage import migration_bandwidth
    >>> before = ["A", "A", "A", "B", "B", "B", "C", "C"]
    >>> after = ["A", "B", "C", "A", "B", "default", "B", "C"]
    >>> migration_bandwidth(before, after, grades=["A", "B", "C"]).round(4)
    statistic
    upper    0.7500
    lower    0.6667
    dtype: float64

    Of grade A, one customer moved one grade down and one two grades,
    where each could have moved two, so the upper MWB is (1 + 2) / (2 + 2).
    The customer of grade B who defaulted counts in neither bandwidth.
    """
    labels, counts, _ = _migrations(before, after, grades, weights)
    size = len(labels)
    grade, other = np.indices((size, size))
    distance = np.abs(grade - other)
    furthest = np.maximum(size - 1 - grade, grade)
    result = {}
    for name, side in (("upper", other > grade), ("lower", other < grade)):
        moved = np.where(side, counts, 0)
        numerator = float(np.sum(distance * moved))
        denominator = float(np.sum(furthest * moved))
        result[name] = numerator / denominator if denominator > 0 else math.nan
    return pd.Series(list(result.values()), index=pd.Index(list(result), name="statistic"))


def migration_test(
    before: npt.ArrayLike,
    after: npt.ArrayLike,
    *,
    grades: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Test whether the migration frequencies fall away from the diagonal of the matrix.

    In a stable rating system, few customers move far: the share of a grade
    that moves to another grade should fall the further that grade lies
    from it. The z tests of the European Central Bank [1]_ compare each
    off-diagonal cell of the migration matrix with its neighbour towards the
    diagonal. A small p-value means that more customers reached the cell
    than its neighbour, a sign that the portfolio or the rating system
    shifted.

    Parameters
    ----------
    before : array_like
        Grade of each customer at the beginning of the observation period.
        Every value must be one of `grades`.
    after : array_like
        Status of each customer at the end of the period: a grade, or any
        other label, such as ``"default"`` or ``"closed"``, for a customer
        who left the grades. If `before` and `after` are both Series, they
        must have identical indices; otherwise they are matched by
        position. Neither may contain missing values.
    grades : array_like
        The :math:`K` grades of the rating scale for non-defaulted
        customers, from the best to the worst, at least two.
    weights : array_like, optional
        Non-negative frequency weight of each customer, matched like
        `after`: integer weights give the same result as repeating rows.

    Returns
    -------
    pandas.DataFrame
        One row for each of the :math:`K (K - 1)` off-diagonal cells, in
        the order of `grades`, indexed by the grades ``before`` and
        ``after``. ``total`` is the number of customers in the grade before,
        those who left included; ``share`` the share of them in the grade
        after; ``statistic`` the z statistic; and ``p_value`` its one-sided
        p-value.

    Raises
    ------
    ValueError
        In the same cases as `migration_bandwidth`.

    See Also
    --------
    migration_bandwidth : How far the customers migrate.
    transition_matrix : The migration matrix itself.

    Notes
    -----
    With :math:`N_i` customers in grade :math:`i` at the beginning, those
    who defaulted or left included, and the shares :math:`p_{ij}` of them in
    grade :math:`j` at the end, each cell is compared with its neighbour
    :math:`n` towards the diagonal, :math:`n = j + 1` below the diagonal
    (:math:`j < i`) and :math:`n = j - 1` above it (:math:`j > i`):

    .. math::

       z_{ij} = \frac{p_{in} - p_{ij}}
       {\sqrt{\left[p_{ij} (1 - p_{ij}) + p_{in} (1 - p_{in})
       + 2 p_{ij} p_{in}\right] / N_i}},

    whose denominator is the standard deviation of :math:`p_{in} - p_{ij}`
    when the customers of grade :math:`i` migrate independently. Under the
    null hypothesis :math:`p_{ij} \le p_{in}`, the p-value is
    :math:`\Phi(z_{ij})`, so the cells that are fuller than their
    neighbour get small p-values. As the European Central Bank asks, the
    statistic and the p-value are NaN where they are not defined: for a
    grade without customers, and where both shares are 0 or one of them is
    1. With weights, the counts are sums of weights, so weights that are
    not frequencies, such as exposures, do not give standard normal
    statistics.

    References
    ----------
    .. [1] European Central Bank, "Instructions for reporting the
       validation results of internal models: IRB Pillar I models for
       credit risk", February 2019, section 2.5.5.2.

    Examples
    --------
    >>> from kivi.vintage import migration_test
    >>> before = ["A", "A", "A", "B", "B", "B", "C", "C"]
    >>> after = ["A", "B", "C", "A", "B", "default", "B", "C"]
    >>> migration_test(before, after, grades=["A", "B", "C"]).round(4)
                  total   share  statistic  p_value
    before after
    A      B          3  0.3333     0.0000   0.5000
           C          3  0.3333     0.0000   0.5000
    B      A          3  0.3333     0.0000   0.5000
           C          3  0.0000     1.2247   0.8897
    C      A          2  0.0000     1.4142   0.9214
           B          2  0.5000     0.0000   0.5000

    The customer of grade B who defaulted counts in its ``total``. No cell
    holds more customers than its neighbour towards the diagonal, so no
    p-value is small.
    """
    labels, counts, total = _migrations(before, after, grades, weights)
    size = len(labels)
    grade, other = (codes.ravel() for codes in np.indices((size, size)))
    off = grade != other
    grade, other = grade[off], other[off]
    neighbour = np.where(other < grade, other + 1, other - 1)
    share = np.divide(
        counts, total[:, None], out=np.full((size, size), np.nan), where=total[:, None] > 0
    )
    cell, near = share[grade, other], share[grade, neighbour]
    spread = cell * (1 - cell) + near * (1 - near) + 2 * cell * near
    defined = (total[grade] > 0) & (spread > 0)
    statistic = np.full(len(grade), np.nan)
    statistic[defined] = (near - cell)[defined] / np.sqrt(spread[defined] / total[grade][defined])
    return pd.DataFrame(
        {
            "total": total[grade],
            "share": cell,
            "statistic": statistic,
            "p_value": _stats.normal_cdf(statistic),
        },
        # Levels and codes, unlike from_arrays, do not hash the labels, which warns for
        # intervals on NumPy < 2.2.
        index=pd.MultiIndex(
            levels=[labels, labels],  # type: ignore[list-item]  # an Index is a sequence
            codes=[grade, other],
            names=["before", "after"],
        ),
    )


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
        return _ordered_labels(states, "states")
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


def _ordered_labels(values: npt.ArrayLike, name: str) -> pd.Index:
    """Return the states or grades given in ``values``, in order, as an Index."""
    if np.ndim(values) != 1:
        raise ValueError(
            f"{name} must be one-dimensional, got {_validation.shown_argument(values)}"
        )
    labels = pd.Index(values)  # type: ignore[arg-type]  # takes any array_like
    if labels.hasnans:
        raise ValueError(
            f"{name} must not contain missing values, got {np.count_nonzero(labels.isna())}"
        )
    if labels.has_duplicates:
        repeated = _validation.shown(labels[labels.duplicated()])
        raise ValueError(f"{name} must not contain duplicates, got {repeated} more than once")
    return labels


def _positions(values: pd.Series, labels: pd.Index) -> npt.NDArray[np.intp]:
    """Position of each value in ``labels``, or -1 for a value that is not one of them."""
    # Categorical codes avoid hashing the labels, which warns for intervals on NumPy < 2.2.
    if isinstance(values.dtype, pd.CategoricalDtype) and values.dtype.categories.equals(labels):
        return values.cat.codes.to_numpy().astype(np.intp)
    return labels.get_indexer(values).astype(np.intp)  # type: ignore[arg-type]  # any array_like


def _state_codes(
    values: pd.Series, labels: pd.Index, name: str, of: str = "states"
) -> npt.NDArray[np.intp]:
    """Position of each value in ``labels``, the argument ``of``; another value raises."""
    codes = _positions(values, labels)
    unknown = codes < 0
    if unknown.any():
        raise ValueError(
            f"{name} must contain only values in {of}, got {_validation.shown(values[unknown])}"
        )
    return codes


def _migrations(
    before: npt.ArrayLike,
    after: npt.ArrayLike,
    grades: npt.ArrayLike,
    weights: npt.ArrayLike | None,
) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """Return the grades, the counts between the grades and the count of each grade before.

    The counts are integers without weights and sums of weights with them.

    Entry (i, j) of the counts holds the customers that moved from grade i to grade j; the
    count of grade i also holds those whose status after is not a grade.
    """
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
                f"{name} must not contain missing values, got {missing.sum()}; give the customers "
                "that left the grades a status of their own, such as 'default' or 'closed'"
            )
    labels = _ordered_labels(grades, "grades")
    size = len(labels)
    if size < 2:
        raise ValueError(
            f"grades must hold at least two grades, got {_validation.shown_argument(grades)}"
        )
    before_code = _state_codes(before, labels, "before", "grades")
    after_code = _positions(after, labels)
    total = np.bincount(before_code, weights=mass, minlength=size)
    graded = after_code >= 0
    counts = np.bincount(
        before_code[graded] * size + after_code[graded],
        weights=None if mass is None else mass[graded],
        minlength=size * size,
    )
    return labels, counts.reshape(size, size), total


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


def _end_month(end: object, name: str = "end") -> int:
    """Return the month of ``end`` as months since January 1970, raising a ValueError.

    A string is read as a `pandas.Period` and stands for its last month, so
    ``"2024Q1"`` is March 2024. ``name`` is the argument shown in the error.
    """
    # NumPy shows a missing datetime64 as None.
    missing_date = isinstance(end, np.datetime64) and np.isnat(end)
    shown = "NaT" if missing_date else _validation.shown_argument(end)
    message = f"{name} must be a date or a month, got {shown}"
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


def _panel_months(values: pd.Series, name: str) -> npt.NDArray[np.int64]:
    """Months since January 1970 of monthly periods or datetimes, each on its own calendar."""
    dtype = values.dtype
    monthly = isinstance(dtype, pd.PeriodDtype) and dtype == pd.PeriodDtype("M")
    if not (monthly or pd.api.types.is_datetime64_any_dtype(dtype)):
        advice = (
            "convert other periods with .dt.asfreq('M'), which takes their last month"
            if isinstance(dtype, pd.PeriodDtype)
            else "convert strings with pandas.PeriodIndex(values, freq='M')"
        )
        raise ValueError(
            f"{name} must hold monthly periods or datetimes, got dtype {dtype}; {advice}"
        )
    missing = values.isna().to_numpy(dtype=bool)
    if missing.any():
        raise ValueError(f"{name} must not contain missing values, got {missing.sum()}")
    if getattr(dtype, "tz", None) is not None:
        values = values.dt.tz_localize(None)
    years = values.dt.year.to_numpy(dtype=np.int64)
    return years * 12 + values.dt.month.to_numpy(dtype=np.int64) - 1 - 1970 * 12


def _start_months(
    starts: npt.ArrayLike | None, first: int, last: int, horizon: int
) -> npt.NDArray[np.int64]:
    """Return the sorted start months, whose ``horizon`` months must lie in the panel."""
    if starts is None:
        months = np.arange(first, last - horizon + 1, dtype=np.int64)
        # Months count from January 1970, so March, June, September and December are 2 mod 3.
        months = months[months % 3 == 2]
        if not len(months):
            raise ValueError(
                f"the panel must cover a quarter end and the {_month_count(horizon)} after it, got "
                f"{_month_label(first)} to {_month_label(last)}; pass starts or a smaller horizon"
            )
        return months
    if isinstance(starts, (str, bytes)) or np.ndim(starts) != 1:
        raise ValueError(
            f"starts must be a list of months, got {_validation.shown_argument(starts)}"
        )
    values = list(typing.cast(abc.Iterable[object], starts))
    if not values:
        raise ValueError("starts must not be empty")
    months = np.array([_end_month(value, "starts") for value in values], dtype=np.int64)
    unique, count = np.unique(months, return_counts=True)
    if (count > 1).any():
        raise ValueError(
            f"starts must not repeat a month, got {_month_label(unique[count > 1][0])} more "
            "than once"
        )
    if unique[0] < first:
        raise ValueError(
            f"starts must not be before the first month of the panel, {_month_label(first)}, "
            f"got {_month_label(unique[0])}"
        )
    if unique[-1] + horizon > last:
        raise ValueError(
            f"starts must be followed by {_month_count(horizon)} of the panel, which ends in "
            f"{_month_label(last)}, got {_month_label(unique[-1])}; the default rate of an "
            "incomplete window would be too low"
        )
    return unique


def _grade_codes(grade: pd.Series) -> tuple[npt.NDArray[np.intp], pd.Index]:
    """Code of each grade in its order, -1 for a missing grade, and the grades.

    Categorical grades follow their categories and others are sorted; pandas
    sorts numbers before strings when a column holds both.
    """
    codes, labels = pd.factorize(grade, sort=True)
    return codes.astype(np.intp), labels


def _month_count(months: int) -> str:
    """Return a number of months as text, such as ``"1 month"`` or ``"12 months"``."""
    return f"{months} month" if months == 1 else f"{months} months"


def _origination_days(
    origination: pd.Series | abc.Mapping[typing.Any, typing.Any], loans: pd.Index
) -> npt.NDArray[np.int64]:
    """Origination day of each of ``loans``, from a Series indexed by loan or a mapping."""
    if isinstance(origination, pd.Series):
        dates = origination
        if dates.index.has_duplicates:
            repeated = _validation.shown(dates.index[dates.index.duplicated()])
            raise ValueError(f"origination must have one date per loan, got {repeated} twice")
    elif isinstance(origination, abc.Mapping):
        dates = pd.Series(list(origination.values()))
        dates.index = _validation.object_index(origination.keys())
    else:
        raise ValueError(
            "origination must be a Series indexed by loan or a mapping from loan to date, got "
            f"{type(origination).__name__}"
        )
    position = dates.index.get_indexer(loans)
    absent = position < 0
    if absent.any():
        raise ValueError(
            "origination must give the date of every loan of the schedule, got none for "
            f"{_validation.shown(loans[absent])}"
        )
    return _days(dates.iloc[position].reset_index(drop=True), "origination")


def _day_label(day: int) -> str:
    """Return a day counted from 1 January 1970 as a string such as ``"2024-01-31"``."""
    return str(np.datetime64(int(day), "D"))


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
