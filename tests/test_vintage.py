import datetime
import math
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from kivi.calibration import calibration_table
from kivi.datasets import make_loan_panel
from kivi.lifetime import survival_table
from kivi.vintage import (
    default_rate_table,
    delinquency_panel,
    migration_bandwidth,
    migration_test,
    roll_rates,
    transition_matrix,
    vintage_table,
)

# Cohort 2024-01: loan 1 is bad at month on book 2 and cures at 3; loan 2 is never bad.
# Cohort 2024-02: loan 3 is bad from month on book 1; loan 4 is never bad. 2024-02 is observed
# for two months only.
LOAN = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
COHORT = ["2024-01"] * 6 + ["2024-02"] * 4
MOB = [1, 2, 3, 1, 2, 3, 1, 2, 1, 2]
BAD = [False, True, False, False, False, False, True, True, False, False]
EXPECTED = pd.DataFrame(
    [[0.0, 0.5, 0.5], [0.5, 0.5, np.nan]],
    index=pd.Index(["2024-01", "2024-02"], name="cohort"),
    columns=pd.Index([1, 2, 3], name="month_on_book"),
)

STATES = ["current", "1-29", "30-59", "60-89", "90+"]
# Ten loans at two month ends: 4 current (1 rolls), 3 at 1-29 (1 cures, 1 stays, 1 rolls),
# 2 at 30-59 (1 rolls, 1 cures to current) and 1 at 90+.
BEFORE = ["current"] * 4 + ["1-29"] * 3 + ["30-59"] * 2 + ["90+"]
AFTER = ["current", "current", "current", "1-29", "current", "1-29", "30-59", "60-89", "current"]
AFTER = [*AFTER, "90+"]


@pytest.fixture(scope="module")
def panel():
    return make_loan_panel(n_loans=1200, n_months=18, rng=11)


def brute_force_vintage(panel, bad):
    """Ever-bad shares by cohort and month on book, one cell at a time."""
    loans = panel.groupby("loan_id")["cohort"].first()
    columns = np.sort(panel["month_on_book"].unique())
    rows = {}
    for cohort, members in loans.groupby(loans):
        rows_of_cohort = panel[panel["cohort"] == cohort]
        oldest = rows_of_cohort["month_on_book"].max()
        values = []
        for k in columns:
            bad_loans = rows_of_cohort.loc[
                bad[rows_of_cohort.index] & (rows_of_cohort.month_on_book <= k), "loan_id"
            ]
            values.append(bad_loans.nunique() / len(members) if k <= oldest else np.nan)
        rows[cohort] = values
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)


# vintage_table


def test_vintage_table_hand_example():
    table = vintage_table(LOAN, COHORT, MOB, BAD)
    pd.testing.assert_frame_equal(table, EXPECTED)


def test_vintage_table_cured_loan_stays_bad_and_young_cohorts_are_nan():
    table = vintage_table(LOAN, COHORT, MOB, BAD)
    # Loan 1 is no longer bad at month on book 3, but the cohort's rate does not fall back.
    assert table.loc["2024-01", 3] == 0.5
    # The 2024-02 cohort has not reached month on book 3: missing, not zero.
    assert np.isnan(table.loc["2024-02", 3])
    assert np.isnan(table.to_numpy()).sum() == 1


def test_vintage_table_ignores_row_order():
    order = np.random.default_rng(0).permutation(len(LOAN))
    shuffled = [np.asarray(values)[order] for values in (LOAN, COHORT, MOB, BAD)]
    pd.testing.assert_frame_equal(vintage_table(*shuffled), EXPECTED)


def test_vintage_table_accepts_series_with_a_shared_index():
    index = pd.Index([90, 10, 80, 20, 70, 30, 60, 40, 50, 0])
    inputs = [pd.Series(values, index=index) for values in (LOAN, COHORT, MOB, BAD)]
    pd.testing.assert_frame_equal(vintage_table(*inputs), EXPECTED)
    # Arrays and Series mix by position.
    pd.testing.assert_frame_equal(vintage_table(LOAN, inputs[1], MOB, inputs[3]), EXPECTED)


def test_vintage_table_rejects_series_with_different_indices():
    loan = pd.Series(LOAN)
    bad = pd.Series(BAD, index=range(1, 11))
    with pytest.raises(ValueError, match="loan and bad must have identical indices"):
        vintage_table(loan, COHORT, MOB, bad)
    weights = pd.Series(1.0, index=range(1, 11))
    with pytest.raises(ValueError, match="loan and weights must have identical indices"):
        vintage_table(loan, COHORT, MOB, BAD, weights=weights)


def test_vintage_table_single_loan():
    table = vintage_table([7, 7, 7], ["2025-03"] * 3, [1, 2, 3], [0, 1, 0])
    assert table.index.tolist() == ["2025-03"]
    assert table.to_numpy().tolist() == [[0.0, 1.0, 1.0]]


def test_vintage_table_counts_ties_once_per_loan():
    # Three of four loans go bad at month on book 2, and loan 1 has two bad rows.
    loan = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    month_on_book = [1, 2, 3] * 4
    bad = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0]
    table = vintage_table(loan, ["c"] * 12, month_on_book, bad)
    assert table.to_numpy().tolist() == [[0.0, 0.75, 0.75]]


def test_vintage_table_keeps_loans_that_left_in_the_denominator():
    # Loan 2 is repaid after month on book 1; the cohort is observed to month on book 3.
    loan = [1, 1, 1, 2, 3, 3, 3]
    month_on_book = [1, 2, 3, 1, 1, 2, 3]
    bad = [0, 0, 1, 0, 0, 0, 0]
    table = vintage_table(loan, ["c"] * 7, month_on_book, bad)
    np.testing.assert_allclose(table.to_numpy(), [[0, 0, 1 / 3]])


def test_vintage_table_uses_observed_months_on_book_as_columns():
    # Quarterly snapshots from month on book 0; months between snapshots are not columns.
    loan = [1, 1, 1, 2, 2, 3]
    cohort = ["a", "a", "a", "a", "a", "b"]
    month_on_book = [0, 3, 6, 0, 3, 0]
    bad = [0, 0, 1, 0, 1, 1]
    table = vintage_table(loan, cohort, month_on_book, bad)
    assert table.columns.tolist() == [0, 3, 6]
    np.testing.assert_allclose(table.to_numpy(), [[0, 0.5, 1], [1, np.nan, np.nan]])


def test_vintage_table_sorts_cohorts_and_follows_categories():
    periods = pd.Series(pd.PeriodIndex(["2024-03", "2024-01", "2024-03", "2024-01"], freq="M"))
    table = vintage_table([1, 2, 3, 4], periods, [1, 1, 1, 1], [1, 0, 0, 0])
    assert isinstance(table.index, pd.PeriodIndex)
    assert table.index.astype(str).tolist() == ["2024-01", "2024-03"]
    assert table[1].tolist() == [0.0, 0.5]
    # Categorical cohorts follow their categories; unused categories (empty cohorts) are left out.
    cohort = pd.Categorical(COHORT, categories=["2024-03", "2024-02", "2024-01"])
    table = vintage_table(LOAN, cohort, MOB, BAD)
    assert table.index.tolist() == ["2024-02", "2024-01"]
    np.testing.assert_allclose(table.to_numpy(), EXPECTED.to_numpy()[::-1])


def test_vintage_table_accepts_nullable_dtypes():
    table = vintage_table(
        pd.array(LOAN, dtype="Int64"),
        pd.array(COHORT, dtype="string"),
        pd.array(MOB, dtype="Int64"),
        pd.array(BAD, dtype="boolean"),
        weights=pd.array([1.0] * 10, dtype="Float64"),
    )
    np.testing.assert_array_equal(table.to_numpy(), EXPECTED.to_numpy())
    assert table.columns.tolist() == [1, 2, 3]
    # Integer flags and whole floats work too.
    table = vintage_table(LOAN, COHORT, np.array(MOB, dtype=float), np.array(BAD, dtype=int))
    pd.testing.assert_frame_equal(table, EXPECTED)


def test_vintage_table_weights_give_shares_of_the_amount():
    # Cohort 2024-01 lends 100 (loan 1, bad) and 300 (loan 2); 2024-02 lends 50 (bad) and 150.
    weights = [100] * 3 + [300] * 3 + [50] * 2 + [150] * 2
    table = vintage_table(LOAN, COHORT, MOB, BAD, weights=weights)
    np.testing.assert_allclose(table.to_numpy(), [[0, 0.25, 0.25], [0.25, 0.25, np.nan]])


def test_vintage_table_integer_weights_repeat_loans():
    panel = make_loan_panel(n_loans=300, n_months=8, rng=2)
    weights = panel["loan_id"] % 3
    bad = panel["status"] >= "60-89"
    weighted = vintage_table(
        panel["loan_id"], panel["cohort"], panel["month_on_book"], bad, weights=weights
    )
    repeated = panel.loc[panel.index.repeat(weights)]
    copy = repeated.groupby(level=0).cumcount()
    loan = repeated["loan_id"] * 10 + copy
    expected = vintage_table(
        loan.to_numpy(), repeated["cohort"], repeated["month_on_book"], bad[repeated.index]
    )
    pd.testing.assert_frame_equal(weighted, expected.reindex(weighted.index))


def test_vintage_table_is_nan_before_a_cohort_is_observed():
    # The panel starts at month on book 3 for cohort "b"; its first two months are unknown.
    loan = [1, 1, 1, 2, 2, 2]
    cohort = ["a", "a", "a", "b", "b", "b"]
    month_on_book = [1, 2, 3, 3, 4, 5]
    bad = [0, 1, 1, 0, 1, 1]
    table = vintage_table(loan, cohort, month_on_book, bad)
    np.testing.assert_allclose(
        table.to_numpy(), [[0, 1, 1, np.nan, np.nan], [np.nan, np.nan, 0, 1, 1]]
    )


def test_vintage_table_of_a_panel_that_starts_after_origination(panel):
    late = panel[panel["month"] >= pd.Period("2025-01", freq="M")]
    table = vintage_table(
        late["loan_id"], late["cohort"], late["month_on_book"], late["status"] >= "30-59"
    )
    row = table.loc[pd.Period("2024-06", freq="M")]
    # The 2024-06 cohort is observed from month on book 7, in January 2025, to 13, in July.
    assert row.loc[:6].isna().all() and row.loc[7:13].notna().all() and row.loc[14:].isna().all()
    cohort = late[late["cohort"] == pd.Period("2024-06", freq="M")]
    first_bad = cohort.loc[cohort["status"] >= "30-59"].groupby("loan_id")["month_on_book"].min()
    expected = (first_bad <= 9).sum() / cohort["loan_id"].nunique()
    assert row.loc[9] == pytest.approx(expected)


def test_vintage_table_horizon_of_a_cohort_that_was_repaid():
    # Cohort "a" is repaid after month on book 2 while "b" runs to 4: "a" is NaN afterwards,
    # unless the repaid loans keep rows with a status that is not bad.
    loan = [1, 1, 2, 2, 3, 3, 3, 3]
    cohort = ["a"] * 4 + ["b"] * 4
    month_on_book = [1, 2, 1, 2, 1, 2, 3, 4]
    bad = [0, 1, 0, 0, 0, 0, 0, 1]
    table = vintage_table(loan, cohort, month_on_book, bad)
    np.testing.assert_allclose(table.loc["a"], [0, 0.5, np.nan, np.nan])
    closed = vintage_table(
        [*loan, 1, 1, 2, 2],
        [*cohort, "a", "a", "a", "a"],
        [*month_on_book, 3, 4, 3, 4],
        [*bad, 0, 0, 0, 0],
    )
    np.testing.assert_allclose(closed.loc["a"], [0, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(closed.loc["b"], [0, 0, 0, 1])


def test_vintage_table_drops_cohorts_without_weight():
    weights = [1.0] * 6 + [0.0] * 4
    table = vintage_table(LOAN, COHORT, MOB, BAD, weights=weights)
    pd.testing.assert_frame_equal(table, EXPECTED.iloc[:1])


def test_vintage_table_matches_a_brute_force_count(panel):
    bad = panel["status"] == "90+"
    table = vintage_table(panel["loan_id"], panel["cohort"], panel["month_on_book"], bad)
    expected = brute_force_vintage(panel, bad)
    np.testing.assert_allclose(table.to_numpy(), expected.to_numpy())
    assert table.index.equals(pd.PeriodIndex(expected.index, name="cohort"))
    # Rows never fall, and each cohort is observed one month less than the one before.
    assert (table.diff(axis=1).iloc[:, 1:].fillna(0) >= 0).all().all()
    assert table.notna().sum(axis=1).tolist() == list(range(18, 0, -1))


def test_vintage_table_does_not_mutate_inputs(panel):
    copy = panel.copy()
    vintage_table(
        panel["loan_id"], panel["cohort"], panel["month_on_book"], panel["status"] == "90+"
    )
    pd.testing.assert_frame_equal(panel, copy)


def test_vintage_table_rejects_a_loan_in_two_cohorts():
    cohort = [*COHORT[:5], "2024-02", *COHORT[6:]]
    with pytest.raises(
        ValueError, match="each loan must have one cohort, got '2024-01' and '2024-02' for loan 2"
    ):
        vintage_table(LOAN, cohort, MOB, BAD)


def test_vintage_table_rejects_repeated_months_on_book():
    month_on_book = [1, 2, 3, 1, 2, 2, 1, 2, 1, 2]
    with pytest.raises(
        ValueError,
        match="at most one row per month on book, got two rows for loan 2 at month on book 2",
    ):
        vintage_table(LOAN, COHORT, month_on_book, BAD)


def test_vintage_table_rejects_weights_that_vary_within_a_loan():
    weights = [100, 100, 90, *[1] * 7]
    with pytest.raises(
        ValueError,
        match=r"weights must be the same on every row of a loan, got 100.0 and 90.0 for loan 1",
    ):
        vintage_table(LOAN, COHORT, MOB, BAD, weights=weights)


@pytest.mark.parametrize(
    ("month_on_book", "message"),
    [
        ([1.5, *MOB[1:]], "month_on_book must contain whole numbers, got 1.5"),
        ([np.nan, *MOB[1:]], "month_on_book must contain whole numbers, got nan"),
        ([np.inf, *MOB[1:]], "month_on_book must contain whole numbers, got inf"),
        (
            ["1", *map(str, MOB[1:])],
            "month_on_book must contain whole numbers, got dtype (object|str)",
        ),
        ([True] * 10, "month_on_book must contain whole numbers, got dtype bool"),
    ],
)
def test_vintage_table_rejects_months_on_book_that_are_not_whole_numbers(month_on_book, message):
    with pytest.raises(ValueError, match=message):
        vintage_table(LOAN, COHORT, month_on_book, BAD)


def test_vintage_table_rejects_negative_months_on_book():
    # -1 used to become a column of the table.
    with pytest.raises(
        ValueError,
        match="month_on_book must not be negative, got -1; count the months from origination",
    ):
        vintage_table(LOAN, COHORT, [-1, *MOB[1:]], BAD)
    shifted = vintage_table(LOAN, COHORT, [value - 1 for value in MOB], BAD)
    assert shifted.columns.tolist() == [0, 1, 2]
    np.testing.assert_array_equal(shifted.to_numpy(), EXPECTED.to_numpy())


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ([2, *[0] * 9], "bad must contain only booleans or 0 and 1, got 2"),
        ([np.nan, *[0.0] * 9], "bad must contain only booleans or 0 and 1, got nan"),
        (pd.array([None, *[False] * 9], dtype="boolean"), "got <NA>"),
        (["yes"] * 10, "bad must contain only booleans or 0 and 1, got 'yes'"),
    ],
)
def test_vintage_table_rejects_bad_flags_that_are_not_binary(bad, message):
    with pytest.raises(ValueError, match=message):
        vintage_table(LOAN, COHORT, MOB, bad)


def test_vintage_table_rejects_missing_loans_and_cohorts():
    with pytest.raises(ValueError, match="loan must not contain missing values, got 1"):
        vintage_table([np.nan, *LOAN[1:]], COHORT, MOB, BAD)
    with pytest.raises(ValueError, match="cohort must not contain missing values, got 2"):
        vintage_table(LOAN, [None, None, *COHORT[2:]], MOB, BAD)


def test_vintage_table_rejects_empty_or_mismatched_inputs():
    with pytest.raises(ValueError, match="must not be empty"):
        vintage_table([], [], [], [])
    with pytest.raises(
        ValueError,
        match="loan, cohort, month_on_book and bad must have the same length, got 10, 10, 9 and 10",
    ):
        vintage_table(LOAN, COHORT, MOB[:-1], BAD)
    with pytest.raises(ValueError, match="loan must be one-dimensional, got 2 dimensions"):
        vintage_table([LOAN], COHORT, MOB, BAD)
    with pytest.raises(ValueError, match="weights must have one weight per value of loan"):
        vintage_table(LOAN, COHORT, MOB, BAD, weights=[1.0] * 9)


# transition_matrix


def test_transition_matrix_hand_counts():
    matrix = transition_matrix(BEFORE, AFTER, states=STATES)
    expected = [
        [3, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
    ]
    assert matrix.to_numpy().tolist() == expected
    assert matrix.to_numpy().dtype == np.int64
    assert matrix.index.tolist() == STATES and matrix.index.name == "before"
    assert matrix.columns.tolist() == STATES and matrix.columns.name == "after"


def test_transition_matrix_row_shares():
    matrix = transition_matrix(BEFORE, AFTER, states=STATES, normalize=True)
    np.testing.assert_allclose(matrix.loc["current"], [0.75, 0.25, 0, 0, 0])
    np.testing.assert_allclose(matrix.loc["1-29"], [1 / 3, 1 / 3, 1 / 3, 0, 0])
    np.testing.assert_allclose(matrix.loc["30-59"], [0.5, 0, 0, 0.5, 0])
    # No loan was at 60-89: its row is missing rather than zero.
    assert matrix.loc["60-89"].isna().all()
    np.testing.assert_allclose(matrix.drop("60-89").sum(axis=1), 1)


@pytest.mark.parametrize("normalize", ["columns", "index", "all", 1, 0, None, np.int64(1)])
def test_transition_matrix_normalize_must_be_a_boolean(normalize):
    # "columns" used to give row shares, the same as True.
    shown = repr(normalize.item() if isinstance(normalize, np.generic) else normalize)
    with pytest.raises(
        ValueError, match=re.escape(f"normalize must be True or False, got {shown}")
    ):
        transition_matrix(BEFORE, AFTER, states=STATES, normalize=normalize)
    shares = transition_matrix(BEFORE, AFTER, states=STATES, normalize=np.True_)
    pd.testing.assert_frame_equal(
        shares, transition_matrix(BEFORE, AFTER, states=STATES, normalize=True)
    )


def test_transition_matrix_default_states():
    # Plain values are sorted, so delinquency labels need states or an ordered categorical.
    matrix = transition_matrix([3, 1, 2, 1], [2, 1, 3, 2])
    assert matrix.index.tolist() == [1, 2, 3]
    assert matrix.to_numpy().tolist() == [[1, 1, 0], [0, 0, 1], [0, 1, 0]]
    dtype = pd.CategoricalDtype(STATES, ordered=True)
    before = pd.Series(BEFORE, dtype=dtype)
    after = pd.Series(AFTER, dtype=dtype)
    pd.testing.assert_frame_equal(
        transition_matrix(before, after), transition_matrix(BEFORE, AFTER, states=STATES)
    )
    # One categorical input sets the states of both.
    pd.testing.assert_frame_equal(
        transition_matrix(before, AFTER), transition_matrix(BEFORE, AFTER, states=STATES)
    )


def test_transition_matrix_states_override_categories():
    dtype = pd.CategoricalDtype(STATES, ordered=True)
    before = pd.Series(BEFORE, dtype=dtype)
    after = pd.Series(AFTER, dtype=dtype)
    reverse = STATES[::-1]
    matrix = transition_matrix(before, after, states=reverse)
    assert matrix.index.tolist() == reverse
    assert matrix.loc["30-59", "60-89"] == 1


def test_transition_matrix_weights():
    # Balances at the first month end; the 60-89 row stays empty.
    weights = [100, 200, 300, 400, 50, 60, 70, 80, 90, 10]
    matrix = transition_matrix(BEFORE, AFTER, states=STATES, weights=weights)
    assert matrix.loc["current"].tolist() == [600, 400, 0, 0, 0]
    assert matrix.loc["1-29"].tolist() == [50, 60, 70, 0, 0]
    assert matrix.loc["30-59"].tolist() == [90, 0, 0, 80, 0]
    shares = transition_matrix(BEFORE, AFTER, states=STATES, weights=weights, normalize=True)
    np.testing.assert_allclose(shares.loc["current"], [0.6, 0.4, 0, 0, 0])


def test_transition_matrix_integer_weights_repeat_rows():
    weights = np.array([0, 1, 2, 3, 1, 0, 2, 1, 4, 2])
    weighted = transition_matrix(BEFORE, AFTER, states=STATES, weights=weights)
    repeated = transition_matrix(
        np.repeat(BEFORE, weights), np.repeat(AFTER, weights), states=STATES
    )
    np.testing.assert_array_equal(weighted.to_numpy(), repeated.to_numpy())


def test_transition_matrix_series_and_nullable_dtypes():
    index = pd.Index(list("jihgfedcba"))
    before = pd.Series(BEFORE, index=index, dtype="string")
    after = pd.Series(AFTER, index=index, dtype="string")
    matrix = transition_matrix(before, after, states=STATES)
    expected = transition_matrix(BEFORE, AFTER, states=STATES)
    np.testing.assert_array_equal(matrix.to_numpy(), expected.to_numpy())
    grades = transition_matrix(
        pd.array([1, 2, 2], dtype="Int64"), pd.array([2, 2, 3], dtype="Int64"), states=[1, 2, 3]
    )
    assert grades.to_numpy().tolist() == [[0, 1, 0], [0, 1, 1], [0, 0, 0]]


def test_transition_matrix_of_interval_bands():
    # Score bands from pandas.cut at two dates share their categories; no hashing of intervals.
    edges = [-np.inf, 500, 600, np.inf]
    before = pd.cut(pd.Series([450, 550, 650, 700]), edges)
    after = pd.cut(pd.Series([520, 480, 640, 590]), edges)
    matrix = transition_matrix(before, after)
    assert isinstance(matrix.index, pd.IntervalIndex)
    assert matrix.to_numpy().tolist() == [[0, 1, 0], [1, 0, 0], [0, 1, 1]]


def test_transition_matrix_ignores_order(panel):
    observed = panel.assign(after=panel.groupby("loan_id")["status"].shift(-1)).dropna()
    shuffled = observed.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(
        transition_matrix(observed["status"], observed["after"]),
        transition_matrix(shuffled["status"], shuffled["after"]),
    )


def test_transition_matrix_rejects_missing_values():
    with pytest.raises(ValueError, match="after must not contain missing values, got 1"):
        transition_matrix(BEFORE, [*AFTER[:-1], None], states=STATES)
    dtype = pd.CategoricalDtype(STATES)
    with pytest.raises(ValueError, match="before must not contain missing values, got 2"):
        transition_matrix(pd.Series([None, None, "90+"], dtype=dtype), ["90+"] * 3)


def test_transition_matrix_rejects_values_outside_the_states():
    with pytest.raises(ValueError, match="after must contain only values in states, got 'closed'"):
        transition_matrix(["current", "1-29"], ["current", "closed"], states=STATES)
    with pytest.raises(ValueError, match="before must contain only values in states, got 3"):
        transition_matrix([3, 1], [1, 1], states=[1, 2])


def test_transition_matrix_rejects_invalid_states():
    with pytest.raises(ValueError, match="states must not contain duplicates, got 'b' more than"):
        transition_matrix(["a"], ["b"], states=["a", "b", "b"])
    with pytest.raises(ValueError, match="states must be one-dimensional, got 'ab'"):
        transition_matrix(["a"], ["b"], states="ab")
    # A missing state used to give a row and a column labelled NaN.
    for states in (["a", "b", None], ["a", "b", np.nan], pd.Index([1.0, np.nan])):
        with pytest.raises(ValueError, match="states must not contain missing values, got 1"):
            transition_matrix(["a"], ["b"], states=states)
    before = pd.Categorical(["a"], categories=["a", "b"])
    after = pd.Categorical(["b"], categories=["b", "a"])
    with pytest.raises(ValueError, match="before and after must have the same categories"):
        transition_matrix(before, after)
    with pytest.raises(ValueError, match="values that cannot be sorted, such as 1 and 'a'"):
        transition_matrix([1, "a"], ["a", 1])
    assert transition_matrix([1, "a"], ["a", 1], states=["a", 1]).to_numpy().tolist() == [
        [0, 1],
        [1, 0],
    ]


def test_transition_matrix_rejects_empty_or_mismatched_inputs():
    with pytest.raises(ValueError, match="before and after must not be empty"):
        transition_matrix([], [])
    with pytest.raises(ValueError, match="before and after must have the same length, got 2 and 1"):
        transition_matrix(["a", "b"], ["a"])
    with pytest.raises(ValueError, match="before and after must have identical indices"):
        transition_matrix(pd.Series(["a"]), pd.Series(["a"], index=[5]))
    with pytest.raises(ValueError, match="weights must be finite and non-negative"):
        transition_matrix(["a"], ["a"], weights=[-1.0])


def test_transition_matrix_checks_the_index_of_weights_against_every_series():
    before = pd.Series(["a", "b"], index=[1, 2])
    weights = pd.Series([1.0, 2.0], index=[7, 8])
    with pytest.raises(ValueError, match="before and weights must have identical indices"):
        transition_matrix(before, ["a", "b"], weights=weights)
    with pytest.raises(ValueError, match="before and weights must have identical indices"):
        roll_rates(before, ["a", "b"], states=["a", "b"], weights=weights)
    matrix = transition_matrix(before, ["b", "b"], weights=weights.set_axis([1, 2]))
    assert matrix.to_numpy().tolist() == [[0.0, 1.0], [0.0, 2.0]]


# roll_rates


def test_roll_rates_hand_example():
    rates = roll_rates(BEFORE, AFTER, states=STATES)
    assert rates.name == "roll_rate"
    assert rates.index.tolist() == STATES[:-1] and rates.index.name == "before"
    np.testing.assert_allclose(rates.to_numpy(), [0.25, 1 / 3, 0.5, np.nan])


def test_roll_rates_with_balances():
    weights = [100, 200, 300, 400, 50, 60, 70, 80, 90, 10]
    rates = roll_rates(BEFORE, AFTER, states=STATES, weights=weights)
    np.testing.assert_allclose(rates.to_numpy(), [0.4, 70 / 180, 80 / 170, np.nan])


def test_roll_rates_count_only_the_next_state():
    # From A: one stays, one moves to B, one jumps to C, one improves to Z.
    rates = roll_rates(["A", "A", "A", "A"], ["A", "B", "C", "Z"], states=["Z", "A", "B", "C"])
    np.testing.assert_allclose(rates.to_numpy(), [np.nan, 0.25, np.nan])


def test_roll_rates_with_a_single_state_is_empty():
    rates = roll_rates(["a", "a"], ["a", "a"], states=["a"])
    assert rates.empty and rates.name == "roll_rate"


def test_roll_rates_chain_to_the_four_month_flow_into_ninety_plus(panel):
    # Delinquency worsens by at most one bucket a month, so the only four-month path from current
    # to 90+ runs through every bucket and its probability is the product of the roll rates.
    status = panel["status"]
    after = status.groupby(panel["loan_id"]).shift(-1)
    observed = after.notna()
    rates = roll_rates(status[observed], after[observed])
    matrix = transition_matrix(status[observed], after[observed], normalize=True)
    flow = np.linalg.matrix_power(matrix.to_numpy(), 4)[0, -1]
    assert rates.prod() == pytest.approx(flow, rel=1e-12)
    np.testing.assert_allclose(rates, np.diagonal(matrix.to_numpy(), offset=1))


def test_roll_rates_rejects_what_transition_matrix_rejects():
    with pytest.raises(ValueError, match="after must contain only values in states, got 'x'"):
        roll_rates(["a"], ["x"], states=["a", "b"])
    with pytest.raises(ValueError, match="before must not contain missing values, got 1"):
        roll_rates([None, "a"], ["a", "a"], states=["a"])
    with pytest.raises(ValueError, match="states must not contain missing values, got 1"):
        roll_rates(["a"], ["a"], states=["a", None])


def test_roll_rates_need_the_order_of_the_states():
    # Sorted, "current" would come after "90+": it would be the worst state and drop out, and
    # the 90+ -> current cure would be reported as a roll.
    before = ["current", "current", "60-89", "90+", "90+"]
    after = ["current", "1-29", "90+", "current", "90+"]
    message = "roll_rates needs the states from the best to the worst: pass states"
    with pytest.raises(ValueError, match=message):
        roll_rates(before, after)
    with pytest.raises(ValueError, match=message):
        roll_rates(pd.Series(before).astype("category"), pd.Series(after).astype("category"))
    with pytest.raises(ValueError, match=message):
        roll_rates(pd.Series(before, dtype=pd.CategoricalDtype(STATES, ordered=True)), after)
    rates = roll_rates(before, after, states=STATES)
    np.testing.assert_allclose(rates.to_numpy(), [0.5, np.nan, np.nan, 1.0])
    dtype = pd.CategoricalDtype(STATES, ordered=True)
    pd.testing.assert_series_equal(
        roll_rates(pd.Series(before, dtype=dtype), pd.Series(after, dtype=dtype)), rates
    )


# delinquency_panel

DUE = ["2024-01-15", "2024-02-15", "2024-03-15"]


def status_of(days):
    """The delinquency bucket of a number of days past due."""
    for low, label in ((90, "90+"), (60, "60-89"), (30, "30-59"), (1, "1-29")):
        if days >= low:
            return label
    return "current"


COLUMNS = ["loan_id", "cohort", "month", "month_on_book", "days_past_due", "overdue_amount"]
COLUMNS += ["outstanding", "status"]


def expected_panel(rows, balance=None):
    """Expected panel from (loan, first month, days past due, overdue amount, outstanding) rows.

    Consecutive rows of a loan are consecutive months from its first month, and the cohort is
    the month before it, as without origination dates.
    """
    frame = pd.DataFrame(rows, columns=["loan", "first", "days", "overdue", "outstanding"])
    step = frame.groupby("loan", sort=False).cumcount()
    first = pd.PeriodIndex(frame["first"], freq="M")
    panel = pd.DataFrame(
        {
            "loan_id": frame["loan"],
            "cohort": first - 1,
            "month": first + step.to_numpy(),
            "month_on_book": (step + 1).astype(np.int64),
            "days_past_due": frame["days"].astype(np.int64),
            "overdue_amount": frame["overdue"].astype(float),
            "outstanding": frame["outstanding"].astype(float),
        }
    )
    if balance is not None:
        panel["balance"] = np.asarray(balance, dtype=float)
    panel["status"] = pd.Categorical(
        [status_of(days) for days in frame["days"]], categories=STATES, ordered=True
    )
    return panel


def one_loan(dates=DUE, amount=100.0, name="A"):
    """Schedule arguments of one loan with one instalment of ``amount`` on each date."""
    return [name] * len(dates), list(dates), [amount] * len(dates)


def brute_force_panel(
    loan,
    due_date,
    amount_due,
    payment_loan,
    payment_date,
    payment_amount,
    *,
    end,
    origination=None,
    principal=None,
    tolerance=0.0,
):
    """The panel month end by month end, paying the instalments one at a time."""
    schedule = pd.DataFrame(
        {
            "loan": loan,
            "due": pd.to_datetime(pd.Series(due_date)),
            "amount": amount_due,
            "principal": np.nan if principal is None else principal,
        }
    )
    payments = pd.DataFrame(
        {
            "loan": payment_loan,
            "date": pd.to_datetime(pd.Series(payment_date)),
            "amount": payment_amount,
        }
    )
    last = pd.Period(end, freq="M")
    rows = []
    for name in sorted(schedule["loan"].unique()):
        # A stable sort keeps instalments due on the same day in the order of the schedule.
        own = schedule[schedule["loan"] == name].sort_values("due", kind="stable")
        paid = payments[payments["loan"] == name]
        month = own["due"].iloc[0].to_period("M")
        if origination is None:
            cohort = month - 1
        else:
            cohort = pd.Timestamp(origination[name]).to_period("M")
        while month <= last:
            day = month.end_time.normalize()
            left = paid.loc[paid["date"] <= day, "amount"].sum()
            unpaid = []
            for amount in own["amount"]:
                share = min(amount, left)
                left -= share
                unpaid.append(amount - share)
            unpaid = np.array(unpaid)
            # Instalment k counts as paid when it and the older ones lack at most tolerance.
            counted = np.cumsum(unpaid) <= tolerance
            fallen = (own["due"] < day).to_numpy()
            behind = np.flatnonzero(~counted)
            days = 0
            if len(behind) and fallen[behind[0]]:
                days = (day - own["due"].iloc[behind[0]]).days
            row = {
                "loan_id": name,
                "cohort": cohort,
                "month": month,
                "month_on_book": (month - cohort).n,
                "days_past_due": days,
                "overdue_amount": unpaid[fallen].sum(),
                "outstanding": unpaid.sum(),
            }
            if principal is not None:
                row["balance"] = np.minimum(own["principal"].to_numpy(), unpaid).sum()
            row["status"] = status_of(days)
            rows.append(row)
            if counted.all():
                break
            month += 1
    panel = pd.DataFrame(rows)
    panel["cohort"] = pd.PeriodIndex(panel["cohort"], freq="M")
    panel["month"] = pd.PeriodIndex(panel["month"], freq="M")
    panel["status"] = pd.Categorical(panel["status"], categories=STATES, ordered=True)
    return panel.astype({"month_on_book": np.int64, "days_past_due": np.int64})


def random_book(seed, n_loans=40):
    """Schedules and payments in whole amounts, so that every sum is exact."""
    rng = np.random.default_rng(seed)
    start = np.datetime64("2024-01-01")
    loan, due_date, amount_due, principal = [], [], [], []
    payment_loan, payment_date, payment_amount = [], [], []
    origination = {}
    for name in rng.permutation(n_loans) * 7:
        first = start + rng.integers(0, 120)
        # Up to three months between origination and the first due date, sometimes none.
        origination[name] = first - rng.integers(0, 100)
        # Some instalments fall due on the same day, and some months have none.
        gaps = rng.choice([0, 28, 29, 30, 31, 61], size=rng.integers(1, 7))
        gaps[0] = 0
        dates = first + np.cumsum(gaps)
        amounts = rng.integers(0, 150, size=len(dates)).astype(float)
        loan += [name] * len(dates)
        due_date += list(dates)
        amount_due += list(amounts)
        principal += list(np.floor(amounts * rng.uniform(0, 1, len(dates))))
        n_payments = rng.integers(0, 9)
        payment_loan += [name] * n_payments
        payment_date += list(first + rng.integers(-30, 200, size=n_payments))
        payment_amount += list(rng.integers(0, 260, size=n_payments).astype(float))
    order = rng.permutation(len(loan))
    pay_order = rng.permutation(len(payment_loan))
    schedule = [np.asarray(values)[order] for values in (loan, due_date, amount_due, principal)]
    payments = [
        np.asarray(values)[pay_order] for values in (payment_loan, payment_date, payment_amount)
    ]
    end = pd.Period("2024-03", freq="M") + int(rng.integers(0, 12))
    return schedule, payments, end, origination


def test_delinquency_panel_on_time_payments_stay_current():
    panel = delinquency_panel(*one_loan(), ["A"] * 3, DUE, [100.0] * 3, end="2024-06")
    expected = expected_panel(
        [("A", "2024-01", 0, 0, 200), ("A", "2024-01", 0, 0, 100), ("A", "2024-01", 0, 0, 0)]
    )
    # The loan leaves the panel after the month end at which it is repaid.
    pd.testing.assert_frame_equal(panel, expected)


def test_delinquency_panel_missed_instalments_are_caught_up():
    dates = [*DUE, "2024-04-15"]
    loan, due_date, amount_due = one_loan(dates)
    loan += ["B"] * 4
    due_date += dates
    amount_due += [100.0] * 4
    # A misses February and pays it on 10 March; B misses February and March and pays both on
    # 2 April. The payments after a month end count as unpaid at that month end.
    payment_loan = ["A", "A", "A", "B", "B", "B"]
    payment_date = ["2024-01-15", "2024-03-10", "2024-04-15", "2024-01-15", "2024-04-02"]
    payment_date += ["2024-04-15"]
    payment_amount = [100.0, 200.0, 100.0, 100.0, 200.0, 100.0]
    panel = delinquency_panel(
        loan, due_date, amount_due, payment_loan, payment_date, payment_amount, end="2024-12"
    )
    expected = expected_panel(
        [
            ("A", "2024-01", 0, 0, 300),
            ("A", "2024-01", 14, 100, 300),
            ("A", "2024-01", 0, 0, 100),
            ("A", "2024-01", 0, 0, 0),
            ("B", "2024-01", 0, 0, 300),
            ("B", "2024-01", 14, 100, 300),
            # 31 March is 45 days after the February due date.
            ("B", "2024-01", 45, 200, 300),
            ("B", "2024-01", 0, 0, 0),
        ]
    )
    pd.testing.assert_frame_equal(panel, expected)


@pytest.mark.parametrize("tolerance", [0.5, 1, np.float64(10.0)])
def test_delinquency_panel_partial_payment_within_tolerance_counts_as_paid(tolerance):
    payments = ["A", "A"], DUE[:2], [99.5, 100.0]
    panel = delinquency_panel(*one_loan(DUE[:2]), *payments, end="2024-03", tolerance=tolerance)
    # The 0.5 left of January stays owed: February's payment goes to it first.
    expected = expected_panel([("A", "2024-01", 0, 0.5, 100.5), ("A", "2024-01", 0, 0.5, 0.5)])
    pd.testing.assert_frame_equal(panel, expected)


@pytest.mark.parametrize("tolerance", [0, 0.25, 0.49])
def test_delinquency_panel_partial_payment_beyond_tolerance_is_past_due(tolerance):
    payments = ["A", "A"], DUE[:2], [99.5, 100.0]
    panel = delinquency_panel(*one_loan(DUE[:2]), *payments, end="2024-03", tolerance=tolerance)
    expected = expected_panel(
        [
            ("A", "2024-01", 16, 0.5, 100.5),
            # 0.5 of February's instalment is unpaid, so it is past due from 15 February.
            ("A", "2024-01", 14, 0.5, 0.5),
            ("A", "2024-01", 45, 0.5, 0.5),
        ]
    )
    pd.testing.assert_frame_equal(panel, expected)


def test_delinquency_panel_shortfalls_within_tolerance_add_up():
    payments = ["A"] * 3, DUE, [99.5] * 3
    panel = delinquency_panel(*one_loan(), *payments, end="2024-04", tolerance=1)
    # 0.5, then 1.0 and then 1.5 is unpaid in total; 1.5 exceeds the tolerance.
    expected = expected_panel(
        [
            ("A", "2024-01", 0, 0.5, 200.5),
            ("A", "2024-01", 0, 1.0, 101.0),
            ("A", "2024-01", 16, 1.5, 1.5),
            ("A", "2024-01", 46, 1.5, 1.5),
        ]
    )
    pd.testing.assert_frame_equal(panel, expected)


def test_delinquency_panel_prepayment_keeps_the_loan_current():
    loan, due_date, amount_due = one_loan()
    loan += ["B"] * 3
    due_date += DUE
    amount_due += [100.0] * 3
    # A pays two instalments in January; B repays the loan before its first due date.
    payments = ["A", "B"], ["2024-01-10", "2024-01-05"], [200.0, 300.0]
    panel = delinquency_panel(loan, due_date, amount_due, *payments, end="2024-03")
    expected = expected_panel(
        [
            ("A", "2024-01", 0, 0, 100),
            ("A", "2024-01", 0, 0, 100),
            ("A", "2024-01", 16, 100, 100),
            ("B", "2024-01", 0, 0, 0),
        ]
    )
    pd.testing.assert_frame_equal(panel, expected)


def test_delinquency_panel_loan_past_maturity_keeps_rolling():
    panel = delinquency_panel(*one_loan(), [], [], [], end="2024-08")
    # Days since 15 January at the end of each month from January to August.
    days = [16, 45, 76, 106, 137, 167, 198, 229]
    overdue = [100, 200, 300, 300, 300, 300, 300, 300]
    rows = [("A", "2024-01", d, p, 300) for d, p in zip(days, overdue, strict=True)]
    pd.testing.assert_frame_equal(panel, expected_panel(rows))
    assert panel["status"].tolist() == ["1-29", "30-59", "60-89"] + ["90+"] * 5


def test_delinquency_panel_counts_payments_up_to_the_last_day_of_the_month():
    schedule = one_loan(["2024-02-15"])
    on_time = delinquency_panel(*schedule, ["A"], ["2024-02-29"], [100.0], end="2024-04")
    pd.testing.assert_frame_equal(on_time, expected_panel([("A", "2024-02", 0, 0, 0)]))
    late = delinquency_panel(*schedule, ["A"], ["2024-03-01"], [100.0], end="2024-04")
    expected = expected_panel([("A", "2024-02", 14, 100, 100), ("A", "2024-02", 0, 0, 0)])
    pd.testing.assert_frame_equal(late, expected)


def test_delinquency_panel_instalment_due_at_the_month_end_is_not_yet_past_due():
    panel = delinquency_panel(*one_loan(["2024-01-31"]), [], [], [], end="2024-04")
    # 0 days at the end of January, 29 at the end of February and 60 at the end of March: the
    # loan skips 30-59.
    rows = [("A", "2024-01", 0, 0, 100), ("A", "2024-01", 29, 100, 100)]
    rows += [("A", "2024-01", 60, 100, 100), ("A", "2024-01", 90, 100, 100)]
    pd.testing.assert_frame_equal(panel, expected_panel(rows))
    assert panel["status"].tolist() == ["current", "1-29", "60-89", "90+"]


def test_delinquency_panel_several_loans_in_any_order():
    loan = ["B", "A", "C", "A", "B", "C", "A", "B"]
    due_date = ["2024-02-10", "2024-03-15", "2024-01-20", "2024-01-15", "2024-01-10"]
    due_date += ["2024-02-20", "2024-02-15", "2024-03-10"]
    amount_due = [50.0, 100.0, 80.0, 100.0, 50.0, 80.0, 100.0, 50.0]
    payment_loan = ["C", "A", "B", "A", "C"]
    payment_date = ["2024-02-20", "2024-01-15", "2024-01-08", "2024-02-28", "2024-01-20"]
    payment_amount = [80.0, 100.0, 50.0, 200.0, 80.0]
    panel = delinquency_panel(
        loan, due_date, amount_due, payment_loan, payment_date, payment_amount, end="2024-04"
    )
    expected = expected_panel(
        [
            ("A", "2024-01", 0, 0, 200),
            ("A", "2024-01", 0, 0, 0),
            ("B", "2024-01", 0, 0, 100),
            ("B", "2024-01", 19, 50, 100),
            ("B", "2024-01", 50, 100, 100),
            ("B", "2024-01", 80, 100, 100),
            ("C", "2024-01", 0, 0, 80),
            ("C", "2024-01", 0, 0, 0),
        ]
    )
    pd.testing.assert_frame_equal(panel, expected)
    # The payments may have any index; the schedule's Series share one.
    index = pd.Index([5, 3, 9, 1, 7, 2, 8, 4])
    schedule = [pd.Series(values, index=index) for values in (loan, due_date, amount_due)]
    payments = [
        pd.Series(values, index=list("vwxyz"))
        for values in (payment_loan, payment_date, payment_amount)
    ]
    pd.testing.assert_frame_equal(delinquency_panel(*schedule, *payments, end="2024-04"), expected)


def test_delinquency_panel_sorts_loans_and_keeps_their_labels():
    loan = [10, 2, 33]
    panel = delinquency_panel(loan, ["2024-01-15"] * 3, [100.0] * 3, [], [], [], end="2024-01")
    assert panel["loan_id"].tolist() == [2, 10, 33]
    assert panel["loan_id"].dtype == np.int64
    mixed = delinquency_panel([2, "a", 1], ["2024-01-15"] * 3, [1.0] * 3, [], [], [], end="2024-01")
    assert mixed["loan_id"].tolist() == [1, 2, "a"]


def test_delinquency_panel_balance_pays_interest_before_principal():
    loan, due_date, amount_due = one_loan(DUE[:2], amount=110.0)
    payments = ["A", "A"], ["2024-01-15", "2024-02-20"], [50.0, 170.0]
    panel = delinquency_panel(
        loan, due_date, amount_due, *payments, end="2024-03", principal=[100.0, 100.0]
    )
    # 50 pays January's 10 of interest and 40 of principal: 60 + 100 of principal remain.
    expected = expected_panel(
        [("A", "2024-01", 16, 60, 170), ("A", "2024-01", 0, 0, 0)], balance=[160, 0]
    )
    pd.testing.assert_frame_equal(panel, expected)
    # 5 pays only interest, so all the principal remains.
    small = delinquency_panel(
        loan,
        due_date,
        amount_due,
        ["A"],
        ["2024-01-15"],
        [5.0],
        end="2024-01",
        principal=[100.0] * 2,
    )
    assert small["balance"].tolist() == [200.0]
    # Without principal, there is no balance.
    assert "balance" not in delinquency_panel(loan, due_date, amount_due, [], [], [], end="2024-01")


def test_delinquency_panel_vintage_table_of_the_status():
    # Cohort 2023-12 (first due in January): A pays on time, B never pays. Cohort 2024-01: C pays
    # its first instalment only, D misses two and repays everything on 1 April.
    loan, due_date, amount_due = [], [], []
    for name, first in (("A", 1), ("B", 1), ("C", 2), ("D", 2)):
        loan += [name] * 3
        due_date += [f"2024-{month:02d}-15" for month in range(first, first + 3)]
        amount_due += [100.0] * 3
    payment_loan = ["A", "A", "A", "C", "D"]
    payment_date = ["2024-01-15", "2024-02-15", "2024-03-15", "2024-02-15", "2024-04-01"]
    payment_amount = [100.0, 100.0, 100.0, 100.0, 300.0]
    panel = delinquency_panel(
        loan, due_date, amount_due, payment_loan, payment_date, payment_amount, end="2024-06"
    )
    assert panel.groupby("loan_id")["month_on_book"].max().to_dict() == {
        "A": 3,
        "B": 6,
        "C": 5,
        "D": 3,
    }
    assert (panel["cohort"] == panel["month"] - panel["month_on_book"]).all()
    table = vintage_table(
        panel["loan_id"], panel["cohort"], panel["month_on_book"], panel["status"] >= "30-59"
    )
    # B is 30-59 at month on book 2; D at 2 (45 days on 31 March) and C at 3 (46 days on 30 April).
    expected = pd.DataFrame(
        [[0.0, 0.5, 0.5, 0.5, 0.5, 0.5], [0.0, 0.5, 1.0, 1.0, 1.0, np.nan]],
        index=pd.PeriodIndex(["2023-12", "2024-01"], freq="M", name="cohort"),
        columns=pd.Index(range(1, 7), name="month_on_book"),
    )
    pd.testing.assert_frame_equal(table, expected)


@pytest.mark.parametrize("seed", range(6))
def test_delinquency_panel_matches_a_month_by_month_calculation(seed):
    schedule, payments, end, origination = random_book(seed)
    tolerance = [0.0, 3.0, 40.0][seed % 3]
    loan, due_date, amount_due, principal = schedule
    arguments = (loan, due_date, amount_due, *payments)
    options = {"end": end, "tolerance": tolerance, "principal": principal}
    if seed % 2:
        # As a mapping on odd seeds, without origination dates on even ones.
        options["origination"] = origination
    panel = delinquency_panel(*arguments, **options)
    pd.testing.assert_frame_equal(panel, brute_force_panel(*arguments, **options))
    assert (panel["overdue_amount"] <= panel["outstanding"]).all()
    assert (panel["balance"] <= panel["outstanding"]).all()


def test_delinquency_panel_has_the_columns_of_make_loan_panel_in_its_order():
    panel = delinquency_panel(*one_loan(), [], [], [], end="2024-01", principal=[90.0] * 3)
    assert panel.columns.tolist() == [*COLUMNS[:-1], "balance", "status"]
    simulated = make_loan_panel(n_loans=5, n_months=2, rng=0).columns.tolist()
    shared = [column for column in simulated if column in panel.columns]
    assert shared == ["loan_id", "cohort", "month", "month_on_book", "balance", "status"]
    assert [column for column in panel.columns if column in simulated] == shared
    assert panel["cohort"].dtype == pd.PeriodDtype("M")


def test_delinquency_panel_origination_sets_the_cohort_and_the_month_on_book():
    # E is originated a month before its first due date, F three months before (a grace
    # period), and G in the month of its first due date.
    loan = ["E", "E", "F", "F", "G"]
    due_date = ["2024-02-20", "2024-03-20", "2024-04-05", "2024-05-05", "2024-01-25"]
    payments = (loan, due_date, [50.0] * 5)
    dates = {"E": "2024-01-20", "F": "2024-01-05", "G": "2024-01-02"}
    by_due = delinquency_panel(loan, due_date, [50.0] * 5, *payments, end="2024-06")
    by_origination = delinquency_panel(
        loan, due_date, [50.0] * 5, *payments, end="2024-06", origination=dates
    )
    assert by_origination["cohort"].astype(str).tolist() == ["2024-01"] * 5
    assert by_origination["month_on_book"].tolist() == [1, 2, 3, 4, 0]
    assert by_due["cohort"].astype(str).tolist() == ["2024-01"] * 2 + ["2024-03"] * 2 + ["2023-12"]
    assert by_due["month_on_book"].tolist() == [1, 2, 1, 2, 1]
    # Only the cohort and the month on book change; the rows stay the same.
    changed = ["cohort", "month_on_book"]
    pd.testing.assert_frame_equal(
        by_origination.drop(columns=changed), by_due.drop(columns=changed)
    )
    # A Series indexed by loan works like a mapping; other loans and the order do not matter.
    series = pd.Series(pd.to_datetime([*dates.values(), "2020-01-01"]), index=[*dates, "Z"])
    pd.testing.assert_frame_equal(
        delinquency_panel(
            loan, due_date, [50.0] * 5, *payments, end="2024-06", origination=series[::-1]
        ),
        by_origination,
    )


def test_delinquency_panel_origination_matches_integer_loans():
    panel = delinquency_panel(
        [3, 1],
        ["2024-03-15", "2024-02-15"],
        [1.0, 1.0],
        [],
        [],
        [],
        end="2024-03",
        origination=pd.Series(["2023-11-30", "2024-01-15"], index=[3.0, 1.0]),
    )
    assert panel["loan_id"].tolist() == [1, 1, 3]
    assert panel["month_on_book"].tolist() == [1, 2, 4]


@pytest.mark.parametrize(
    "origination, message",
    [
        (
            ["2024-01-01"],
            "origination must be a Series indexed by loan or a mapping from loan to date, got list",
        ),
        (
            pd.DataFrame({"date": ["2024-01-01"]}, index=["A"]),
            "origination must be a Series "
            "indexed by loan or a mapping from loan to date, got DataFrame",
        ),
        (
            pd.Series(["2024-01-01", "2024-01-02"], index=["A", "A"]),
            "origination must have one date per loan, got 'A' twice",
        ),
        (
            {"B": "2024-01-01"},
            "origination must give the date of every loan of the schedule, got none for 'A'",
        ),
        ({}, "origination must give the date of every loan of the schedule, got none for 'A'"),
        ({"A": None}, "origination must not contain missing values, got 1"),
        (pd.Series([pd.NaT], index=["A"]), "origination must not contain missing values, got 1"),
        ({"A": 20240101}, "origination must hold dates, got dtype int64"),
        ({"A": "soon"}, "origination must hold dates, got 'soon'"),
        (
            {"A": "2024-01-16"},
            "due_date must not be before the origination of its loan, got "
            "'2024-01-15' for loan 'A', originated on '2024-01-16'",
        ),
    ],
)
def test_delinquency_panel_rejects_invalid_origination(origination, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        delinquency_panel(*one_loan(), [], [], [], end="2024-06", origination=origination)


def test_delinquency_panel_origination_on_the_first_due_date_is_valid():
    panel = delinquency_panel(*one_loan(), [], [], [], end="2024-01", origination={"A": DUE[0]})
    assert panel["month_on_book"].tolist() == [0]


def test_delinquency_panel_balance_roll_rates_run_as_in_the_guide():
    # The guide's roll-rate code for make_loan_panel, on a panel built from schedules.
    loan = np.repeat(["A", "B", "C", "D"], 3)
    due_date = DUE * 4
    principal = [94.0, 96.0, 98.0] * 4
    payment_loan = ["A", "A", "A", "B", "B", "C", "C", "D", "D"]
    payment_date = ["2024-01-15", "2024-02-15", "2024-03-14", "2024-01-15", "2024-03-20"]
    payment_date += ["2024-01-10", "2024-03-15", "2024-01-15", "2024-02-15"]
    payment_amount = [100.0, 100.0, 100.0, 100.0, 200.0, 200.0, 100.0, 100.0, 99.99]
    panel = delinquency_panel(
        loan,
        due_date,
        [100.0] * 12,
        payment_loan,
        payment_date,
        payment_amount,
        end="2024-06",
        principal=principal,
    )
    later = panel[["loan_id", "month_on_book", "status"]].assign(
        month_on_book=panel["month_on_book"] - 1
    )
    pairs = panel.merge(later, on=["loan_id", "month_on_book"], suffixes=("", "_next"))
    before, after = pairs["status"], pairs["status_next"]
    # Six pairs start current (B and D roll), two at 1-29 (D rolls), and D rolls on twice.
    np.testing.assert_allclose(roll_rates(before, after).to_numpy(), [2 / 6, 1 / 2, 1, 1])
    # By balance: B and D roll from current with 194 each of 876, and D from 1-29 with 98.01
    # of 194 + 98.01.
    np.testing.assert_allclose(
        roll_rates(before, after, weights=pairs["balance"]).to_numpy(),
        [388 / 876, 98.01 / 292.01, 1, 1],
    )


def test_delinquency_panel_tolerance_of_the_total_due_lets_an_unpaid_loan_leave():
    panel = delinquency_panel(*one_loan(), [], [], [], end="2024-06", tolerance=300)
    # The amounts stay exact: January's 100 is overdue, although it counts as paid.
    pd.testing.assert_frame_equal(panel, expected_panel([("A", "2024-01", 0, 100, 300)]))
    rolling = delinquency_panel(*one_loan(), [], [], [], end="2024-06", tolerance=299.5)
    assert len(rolling) == 6


def test_delinquency_panel_adds_up_instalments_due_on_the_same_day():
    # A schedule row repeated by a faulty join doubles the amount due.
    loan, due_date, amount_due = one_loan()
    doubled = delinquency_panel(
        loan + ["A"],
        due_date + [DUE[0]],
        amount_due + [100.0],
        ["A"],
        [DUE[0]],
        [100.0],
        end="2024-02",
    )
    expected = expected_panel([("A", "2024-01", 16, 100, 300), ("A", "2024-01", 45, 200, 300)])
    pd.testing.assert_frame_equal(doubled, expected)


def test_delinquency_panel_ignores_payments_after_end_and_after_repayment():
    expected = delinquency_panel(*one_loan(), ["A"], ["2024-01-15"], [300.0], end="2024-04")
    # Payments after the loan was repaid, and after end, leave no credit and add no rows.
    payments = ["A"] * 3, ["2024-01-15", "2024-02-01", "2024-09-01"], [300.0, 50.0, 500.0]
    pd.testing.assert_frame_equal(
        delinquency_panel(*one_loan(), *payments, end="2024-04"), expected
    )
    late = delinquency_panel(*one_loan(), ["A"], ["2024-05-02"], [300.0], end="2024-04")
    assert late["days_past_due"].tolist() == [16, 45, 76, 106]


def test_delinquency_panel_counts_an_instalment_paid_in_parts_despite_rounding():
    assert 0.1 + 0.7 < 0.8
    panel = delinquency_panel(
        *one_loan(["2024-01-15"], amount=0.8),
        ["A", "A"],
        ["2024-01-10"] * 2,
        [0.1, 0.7],
        end="2024-03",
    )
    pd.testing.assert_frame_equal(panel, expected_panel([("A", "2024-01", 0, 0, 0)]))


def test_delinquency_panel_overpayments_and_instalments_of_zero():
    loan, due_date, amount_due = one_loan(["2024-01-15", "2024-02-15", "2024-02-15"])
    amount_due = [100.0, 0.0, 100.0]
    panel = delinquency_panel(
        loan, due_date, amount_due, ["A"], ["2024-01-15"], [100.0], end="2024-03"
    )
    # The instalment of 0 is paid with nothing.
    expected = expected_panel(
        [
            ("A", "2024-01", 0, 0, 100),
            ("A", "2024-01", 14, 100, 100),
            ("A", "2024-01", 45, 100, 100),
        ]
    )
    pd.testing.assert_frame_equal(panel, expected)
    overpaid = delinquency_panel(
        loan, due_date, amount_due, ["A"], ["2024-01-15"], [500.0], end="2024-03"
    )
    pd.testing.assert_frame_equal(overpaid, expected_panel([("A", "2024-01", 0, 0, 0)]))


def test_delinquency_panel_leaves_out_loans_first_due_after_end():
    loan = ["A", "B"]
    panel = delinquency_panel(
        loan, ["2024-01-15", "2024-03-15"], [1.0, 1.0], [], [], [], end="2024-02"
    )
    assert panel["loan_id"].tolist() == ["A", "A"]
    message = re.escape(
        "end must not be before the month of the earliest due date, got '2023-12' before '2024-01'"
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        delinquency_panel(loan, ["2024-01-15", "2024-03-15"], [1.0, 1.0], [], [], [], end="2023-12")


@pytest.mark.parametrize(
    "end",
    [
        "2024-03",
        "2024-03-31",
        "2024-03-01 18:00",
        pd.Timestamp("2024-03-02"),
        pd.Timestamp("2024-03-02", tz="Asia/Shanghai"),
        datetime.date(2024, 3, 9),
        np.datetime64("2024-03-30"),
        pd.Period("2024-03", freq="M"),
        pd.Period("2024-03-15", freq="D"),
        pd.Period("2024Q1", freq="Q"),
        # Strings are read as pandas.Period reads them, without warnings: a quarter ends in its
        # last month, and a time zone is dropped.
        "2024Q1",
        "March 2024",
        "2024-03-15T10:00:00+08:00",
        datetime.datetime(2024, 3, 2, tzinfo=datetime.timezone.utc),
    ],
)
def test_delinquency_panel_end_is_a_month_or_a_date_in_it(end):
    panel = delinquency_panel(*one_loan(), [], [], [], end=end)
    assert panel["month"].astype(str).tolist() == ["2024-01", "2024-02", "2024-03"]


def test_delinquency_panel_end_of_a_year_is_its_december():
    panel = delinquency_panel(*one_loan(), [], [], [], end="2024")
    assert panel["month"].astype(str).tolist() == [f"2024-{month:02d}" for month in range(1, 13)]


def test_delinquency_panel_reads_dates_of_any_type():
    expected = delinquency_panel(*one_loan(), ["A"], ["2024-02-20"], [150.0], end="2024-04")
    loan, _, amount_due = one_loan()
    due = pd.to_datetime(pd.Series(DUE))
    variants = [
        (due.to_numpy(), np.array(["2024-02-20"], dtype="datetime64[D]")),
        (list(due), [pd.Timestamp("2024-02-20 23:30")]),
        ([day.date() for day in due], [datetime.datetime(2024, 2, 20, 8)]),
        (due.dt.tz_localize("Europe/Paris"), pd.Series(["2024-02-20T01:00"])),
        (pd.Series(DUE, dtype="string"), pd.Series(pd.to_datetime(["2024-02-20"]), dtype=object)),
    ]
    for due_date, payment_date in variants:
        panel = delinquency_panel(
            loan, due_date, amount_due, ["A"], payment_date, [150.0], end="2024-04"
        )
        pd.testing.assert_frame_equal(panel, expected)


def test_delinquency_panel_accepts_nullable_dtypes_and_empty_payments():
    expected = delinquency_panel(*one_loan(), [], [], [], end="2024-02")
    loan = pd.array(["A"] * 3, dtype="string")
    amount = pd.array([100, 100, 100], dtype="Int64")
    panel = delinquency_panel(loan, DUE, amount, [], [], [], end="2024-02")
    pd.testing.assert_frame_equal(panel, expected, check_dtype=False)
    for empty in (np.array([]), pd.Series([], dtype=object), np.array([], dtype="datetime64[ns]")):
        panel = delinquency_panel(*one_loan(), empty, empty, [], end="2024-02")
        pd.testing.assert_frame_equal(panel, expected)


def test_delinquency_panel_does_not_modify_its_inputs():
    loan, due_date, amount_due = (pd.Series(values) for values in one_loan())
    payments = [pd.Series(values) for values in (["A"], ["2024-01-20"], [100.0])]
    copies = [values.copy() for values in (loan, due_date, amount_due, *payments)]
    delinquency_panel(loan, due_date, amount_due, *payments, end="2024-03", principal=amount_due)
    for values, copy in zip((loan, due_date, amount_due, *payments), copies, strict=True):
        pd.testing.assert_series_equal(values, copy)


def test_delinquency_panel_has_the_statuses_of_make_loan_panel():
    panel = delinquency_panel(*one_loan(), [], [], [], end="2024-01")
    simulated = make_loan_panel(n_loans=5, n_months=2, rng=0)
    assert panel["status"].dtype == simulated["status"].dtype


def test_delinquency_panel_end_is_keyword_only_and_required():
    with pytest.raises(TypeError):
        delinquency_panel(*one_loan(), [], [], [])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        delinquency_panel(*one_loan(), [], [], [], "2024-01")  # type: ignore[misc]


SCHEDULE = one_loan()
NO_PAYMENTS = ([], [], [])


@pytest.mark.parametrize(
    "arguments, options, message",
    [
        (
            (["A"] * 3, DUE[:2], [1.0] * 3, *NO_PAYMENTS),
            {},
            "loan, due_date and amount_due must have the same length, got 3, 2 and 3",
        ),
        (
            (*SCHEDULE, *NO_PAYMENTS),
            {"principal": [1.0]},
            "loan, due_date, amount_due and principal must have the same length, got 3, 3, 3 and 1",
        ),
        (
            (*SCHEDULE, ["A"], ["2024-01-15"], [1.0, 2.0]),
            {},
            "payment_loan, payment_date and payment_amount must have the same length, got 1, 1 "
            "and 2",
        ),
        (([], [], [], *NO_PAYMENTS), {}, "loan, due_date and amount_due must not be empty"),
        (
            (np.array([["A"]]), ["2024-01-15"], [1.0], *NO_PAYMENTS),
            {},
            "loan must be one-dimensional, got 2 dimensions",
        ),
        (
            (["A", None, "B"], DUE, [1.0] * 3, *NO_PAYMENTS),
            {},
            "loan must not contain missing values, got 1",
        ),
        (
            (["A"] * 3, ["2024-01-15", None, "2024-03-15"], [1.0] * 3, *NO_PAYMENTS),
            {},
            "due_date must not contain missing values, got 1",
        ),
        (
            (["A"] * 3, ["2024-01-15", "soon", "2024-03-15"], [1.0] * 3, *NO_PAYMENTS),
            {},
            "due_date must hold dates, got 'soon'",
        ),
        (
            (["A"] * 3, [20240115, 20240215, 20240315], [1.0] * 3, *NO_PAYMENTS),
            {},
            "due_date must hold dates, got dtype int64",
        ),
        (
            (
                ["A"] * 2,
                pd.Series(pd.period_range("2024-01-15", periods=2)),
                [1.0] * 2,
                *NO_PAYMENTS,
            ),
            {},
            r"due_date must hold dates, got dtype period\[D\]",
        ),
        (
            (["A"] * 3, DUE, [1.0, np.nan, 1.0], *NO_PAYMENTS),
            {},
            "amount_due must not contain missing values, got 1",
        ),
        (
            (["A"] * 3, DUE, [1.0, -1.0, 1.0], *NO_PAYMENTS),
            {},
            "amount_due must be finite and non-negative, got -1.0",
        ),
        (
            (["A"] * 3, DUE, [True, False, True], *NO_PAYMENTS),
            {},
            "amount_due must be real and numeric, got dtype bool",
        ),
        (
            (["A"] * 3, DUE, ["1", "2", "3"], *NO_PAYMENTS),
            {},
            "amount_due must be real and numeric, got dtype (object|str) with the value '1'",
        ),
        (
            (*SCHEDULE, *NO_PAYMENTS),
            {"principal": [100.0, 120.0, 0.0]},
            "principal must not exceed amount_due, got 120.0 for an amount due of 100.0",
        ),
        (
            (*SCHEDULE, *NO_PAYMENTS),
            {"principal": [100.0, -1.0, 0.0]},
            "principal must be finite and non-negative, got -1.0",
        ),
        (
            (*SCHEDULE, ["A", "Z"], ["2024-01-15"] * 2, [1.0, 1.0]),
            {},
            "payment_loan must contain only loans of the schedule, got 'Z'",
        ),
        (
            (*SCHEDULE, ["A", None], ["2024-01-15"] * 2, [1.0, 1.0]),
            {},
            "payment_loan must not contain missing values, got 1",
        ),
        (
            (*SCHEDULE, ["A"], [pd.NaT], [1.0]),
            {},
            "payment_date must not contain missing values, got 1",
        ),
        (
            (*SCHEDULE, ["A"], ["2024-01-15"], [np.inf]),
            {},
            "payment_amount must be finite and non-negative, got inf",
        ),
        (
            (*SCHEDULE, ["A"], ["2024-01-15"], [-5]),
            {},
            "payment_amount must be finite and non-negative, got -5.0",
        ),
    ],
)
def test_delinquency_panel_rejects_invalid_inputs(arguments, options, message):
    options = {"end": "2024-06", **options}
    with pytest.raises(ValueError, match=f"^{message}$"):
        delinquency_panel(*arguments, **options)


def test_delinquency_panel_matches_the_series_of_each_input_by_index():
    loan, due_date, amount_due = (pd.Series(values) for values in SCHEDULE)
    other = due_date.set_axis([5, 6, 7])
    with pytest.raises(ValueError, match="^loan and due_date must have identical indices$"):
        delinquency_panel(loan, other, amount_due, *NO_PAYMENTS, end="2024-06")
    with pytest.raises(ValueError, match="^loan and principal must have identical indices$"):
        delinquency_panel(
            loan, due_date, amount_due, *NO_PAYMENTS, end="2024-06", principal=other.map(len)
        )
    payments = [pd.Series(values) for values in (["A"], ["2024-01-15"], [1.0])]
    payments[2].index = [9]
    with pytest.raises(
        ValueError, match="^payment_loan and payment_amount must have identical indices$"
    ):
        delinquency_panel(loan, due_date, amount_due, *payments, end="2024-06")


@pytest.mark.parametrize(
    "end, shown",
    [
        (None, "None"),
        (pd.NaT, "NaT"),
        (5, "5"),
        (np.int64(202401), "202401"),
        (True, "True"),
        ("soon", "'soon'"),
        ([2024, 1], r"\[2024, 1\]"),
        (np.timedelta64(3, "D"), r"datetime.timedelta\(days=3\)"),
        # pandas reads these as NaT instead of raising.
        ("", "''"),
        ("NaT", "'NaT'"),
        (np.datetime64("NaT", "D"), "NaT"),
        (" ", "' '"),
        ("2024-W11", "'2024-W11'"),
    ],
)
def test_delinquency_panel_end_must_be_a_date_or_a_month(end, shown):
    with pytest.raises(ValueError, match=f"^end must be a date or a month, got {shown}$"):
        delinquency_panel(*SCHEDULE, *NO_PAYMENTS, end=end)


@pytest.mark.parametrize(
    "tolerance, shown",
    [(-1, "-1"), (np.nan, "nan"), (np.inf, "inf"), (True, "True"), ("1", "'1'"), (None, "None")],
)
def test_delinquency_panel_tolerance_must_be_finite_and_non_negative(tolerance, shown):
    message = f"^tolerance must be finite and non-negative, got {shown}$"
    with pytest.raises(ValueError, match=message):
        delinquency_panel(*SCHEDULE, *NO_PAYMENTS, end="2024-06", tolerance=tolerance)


# The migration tests of the ECB validation report.

GRADES = ["A", "B", "C"]
MIGRATION_BEFORE = ["A", "A", "A", "B", "B", "B", "C", "C"]
MIGRATION_AFTER = ["A", "B", "C", "A", "B", "default", "B", "C"]


def bandwidths_by_hand(before, after, grades):
    """Upper and lower MWB by the formulas of the ECB, adding up N_i p_ij customer by customer."""
    number = {grade: position + 1 for position, grade in enumerate(grades)}
    size = len(grades)
    sums = {"upper": [0.0, 0.0], "lower": [0.0, 0.0]}
    for start, end in zip(before, after, strict=True):
        if end not in number:
            continue
        i, j = number[start], number[end]
        side = "upper" if j > i else "lower" if j < i else None
        if side:
            sums[side][0] += abs(i - j)
            sums[side][1] += max(abs(i - size), abs(i - 1))
    return {side: num / den if den else math.nan for side, (num, den) in sums.items()}


def migration_z_by_hand(before, after, grades):
    total = {grade: sum(start == grade for start in before) for grade in grades}
    share = {
        (i, j): sum(s == i and e == j for s, e in zip(before, after, strict=True)) / total[i]
        for i in grades
        for j in grades
        if total[i]
    }
    result = {}
    for a, i in enumerate(grades):
        for b, j in enumerate(grades):
            if a == b:
                continue
            if not total[i]:
                result[i, j] = math.nan
                continue
            near = grades[b + 1] if b < a else grades[b - 1]
            p, q = share[i, j], share[i, near]
            spread = p * (1 - p) + q * (1 - q) + 2 * p * q
            result[i, j] = (q - p) / math.sqrt(spread / total[i]) if spread else math.nan
    return result


def test_migration_bandwidth_matches_the_hand_calculation():
    result = migration_bandwidth(MIGRATION_BEFORE, MIGRATION_AFTER, grades=GRADES)
    assert list(result.index) == ["upper", "lower"]
    assert result.index.name == "statistic"
    assert result["upper"] == pytest.approx(3 / 4, rel=1e-15)
    assert result["lower"] == pytest.approx(2 / 3, rel=1e-15)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_migration_bandwidth_matches_the_formulas_on_random_migrations(seed):
    rng = np.random.default_rng(seed)
    grades = list("ABCDEF")
    before = rng.choice(grades, size=300)
    after = rng.choice(grades + ["default", "closed"], size=300)
    result = migration_bandwidth(before, after, grades=grades)
    expected = bandwidths_by_hand(before, after, grades)
    assert result["upper"] == pytest.approx(expected["upper"], rel=1e-13)
    assert result["lower"] == pytest.approx(expected["lower"], rel=1e-13)


def test_migration_bandwidth_ignores_customers_who_left():
    stayed = migration_bandwidth(MIGRATION_BEFORE, MIGRATION_AFTER, grades=GRADES)
    left = migration_bandwidth(
        MIGRATION_BEFORE + ["A", "C"], MIGRATION_AFTER + ["closed", "default"], grades=GRADES
    )
    pd.testing.assert_series_equal(left, stayed)


def test_migration_bandwidth_without_migrations_in_one_direction():
    result = migration_bandwidth(["A", "B", "B"], ["B", "B", "default"], grades=GRADES)
    assert result["upper"] == pytest.approx(1 / 2)
    assert math.isnan(result["lower"])
    # From the middle of five grades the furthest move is two grades in either direction.
    result = migration_bandwidth(["C", "C"], ["D", "B"], grades=list("ABCDE"))
    assert result.tolist() == [0.5, 0.5]


def test_migration_test_matches_the_hand_calculation():
    table = migration_test(MIGRATION_BEFORE, MIGRATION_AFTER, grades=GRADES)
    assert list(table.columns) == ["total", "share", "statistic", "p_value"]
    assert table.index.names == ["before", "after"]
    assert table.index.tolist() == [
        ("A", "B"),
        ("A", "C"),
        ("B", "A"),
        ("B", "C"),
        ("C", "A"),
        ("C", "B"),
    ]
    # The customer of grade B who defaulted counts in its total.
    assert table["total"].tolist() == [3, 3, 3, 3, 2, 2]
    assert table["total"].dtype == np.int64
    np.testing.assert_allclose(table["share"], [1 / 3, 1 / 3, 1 / 3, 0, 0, 1 / 2], rtol=1e-15)
    # B to C against B to B: (1/3 - 0) / sqrt((1/3)(2/3) / 3).
    assert table.loc[("B", "C"), "statistic"] == pytest.approx(math.sqrt(1.5), rel=1e-14)
    # C to A against C to B: (1/2 - 0) / sqrt((1/2)(1/2) / 2).
    assert table.loc[("C", "A"), "statistic"] == pytest.approx(math.sqrt(2), rel=1e-14)
    p_value = [0.5 * math.erfc(-z / math.sqrt(2)) for z in table["statistic"]]
    np.testing.assert_allclose(table["p_value"], p_value, rtol=1e-14)


@pytest.mark.parametrize("seed", [3, 4])
def test_migration_test_matches_the_formulas_on_random_migrations(seed):
    rng = np.random.default_rng(seed)
    grades = list("ABCDE")
    before = rng.choice(grades[:4], size=200)
    after = rng.choice(grades + ["default"], size=200, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05])
    table = migration_test(before, after, grades=grades)
    expected = migration_z_by_hand(list(before), list(after), grades)
    assert len(table) == 20
    for cell, value in expected.items():
        assert table.loc[cell, "statistic"] == pytest.approx(value, rel=1e-12, nan_ok=True)
    # Grade E has no customers before, so its row is undefined.
    assert table.loc["E", "statistic"].isna().all()
    assert (table.loc["E", "total"] == 0).all()


def test_migration_test_is_undefined_where_the_ecb_leaves_it_empty():
    with np.errstate(all="raise"):
        # Every customer of A stayed: A to B has shares 1 and 0, and A to C shares 0 and 0.
        table = migration_test(["A", "A", "B"], ["A", "A", "B"], grades=GRADES)
    assert table.loc[[("A", "B"), ("A", "C")], ["statistic", "p_value"]].isna().all(axis=None)
    assert table.loc["C", "total"].tolist() == [0, 0]


def test_migration_test_flags_a_cell_fuller_than_its_neighbour():
    # Of 100 customers in A, 30 skipped grade B and 5 moved to it.
    before = ["A"] * 100 + ["B"] * 10
    after = ["A"] * 60 + ["B"] * 5 + ["C"] * 30 + ["default"] * 5 + ["B"] * 10
    table = migration_test(before, after, grades=GRADES)
    p, q = 0.30, 0.05
    z = (q - p) / math.sqrt((p * (1 - p) + q * (1 - q) + 2 * p * q) / 100)
    assert table.loc[("A", "C"), "statistic"] == pytest.approx(z, rel=1e-13)
    assert table.loc[("A", "C"), "p_value"] < 1e-5


def test_migration_tests_integer_weights_equal_repeated_rows():
    rng = np.random.default_rng(5)
    before = rng.choice(GRADES, size=40)
    after = rng.choice(GRADES + ["closed"], size=40)
    weights = rng.integers(0, 4, size=40)
    rows = np.repeat(np.arange(40), weights)
    pd.testing.assert_series_equal(
        migration_bandwidth(before, after, grades=GRADES, weights=weights),
        migration_bandwidth(before[rows], after[rows], grades=GRADES),
        rtol=1e-13,
    )
    pd.testing.assert_frame_equal(
        migration_test(before, after, grades=GRADES, weights=weights),
        migration_test(before[rows], after[rows], grades=GRADES),
        check_dtype=False,
        rtol=1e-13,
    )


def test_migration_tests_take_categoricals_and_interval_grades():
    grades = pd.IntervalIndex.from_breaks([0, 0.01, 0.05, 1])
    before = pd.cut(pd.Series([0.005, 0.02, 0.5, 0.03]), grades)
    after = pd.cut(pd.Series([0.02, 0.02, 0.04, 0.6]), grades)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table = migration_test(before, after, grades=grades)
        result = migration_bandwidth(before, after, grades=grades)
    assert table.loc[(grades[0], grades[1]), "share"] == 1.0
    # Up one grade from the best grade, which could move two, and from the middle one.
    assert result["upper"] == pytest.approx(2 / 3, rel=1e-15)
    assert result["lower"] == pytest.approx(1 / 2, rel=1e-15)


@pytest.mark.parametrize(
    "before,after,grades,message",
    [
        (["A", "D"], ["A", "B"], GRADES, "before must contain only values in grades, got 'D'"),
        (["A", None], ["A", "B"], GRADES, "before must not contain missing values, got 1"),
        (["A", "B"], ["A", np.nan], GRADES, "after must not contain missing values, got 1"),
        (["A"], ["A"], ["A"], "grades must hold at least two grades, got ['A']"),
        (["A"], ["A"], pd.Series([], dtype=object), "grades must hold at least two grades, got []"),
        (["A"], ["A"], ["A", "A"], "grades must not contain duplicates, got 'A' more than once"),
        (["A"], ["A"], ["A", None], "grades must not contain missing values, got 1"),
        (["A"], ["A"], "AB", "grades must be one-dimensional, got 'AB'"),
        (["A", "B"], ["A"], GRADES, "before and after must have the same length, got 2 and 1"),
        ([], [], GRADES, "before and after must not be empty"),
    ],
)
def test_invalid_migration_inputs(before, after, grades, message):
    for function in (migration_bandwidth, migration_test):
        with pytest.raises(ValueError, match=re.escape(message)):
            function(before, after, grades=grades)


def test_migration_inputs_must_share_an_index():
    before, after = pd.Series(["A", "B"]), pd.Series(["A", "B"], index=[3, 4])
    with pytest.raises(ValueError, match="before and after must have identical indices"):
        migration_test(before, after, grades=GRADES)
    with pytest.raises(ValueError, match="weights must be finite and non-negative"):
        migration_bandwidth(["A", "B"], ["B", "A"], grades=GRADES, weights=[1, -1])


# One-year default rates. Loan 1 defaults in May and cures in July; loan 2 is repaid after
# April; loan 3 moves from grade B to A in August; loan 4 defaults from August; loan 5 enters
# the panel in May. With a horizon of 3 months the quarter ends March and June start cohorts.
RATE_MONTHS = [f"2024-{number:02d}" for number in range(1, 10)]
RATE_ROWS = (
    [(1, month, "A", month in ("2024-05", "2024-06")) for month in RATE_MONTHS]
    + [(2, month, "A", False) for month in RATE_MONTHS[:4]]
    + [(3, month, "B" if month < "2024-08" else "A", False) for month in RATE_MONTHS]
    + [(4, month, "B", month >= "2024-08") for month in RATE_MONTHS]
    + [(5, month, "A", False) for month in RATE_MONTHS[4:]]
)


def rate_panel():
    panel = pd.DataFrame(RATE_ROWS, columns=["loan", "month", "grade", "defaulted"])
    panel["month"] = pd.PeriodIndex(panel["month"], freq="M")
    return panel


def rate_columns(panel):
    return panel["loan"], panel["month"], panel["grade"], panel["defaulted"]


def rate_index(starts, grades):
    return pd.MultiIndex.from_arrays(
        [pd.PeriodIndex(starts, freq="M"), pd.Index(grades)], names=["start", "grade"]
    )


def test_default_rate_table_matches_hand_calculation():
    table = default_rate_table(*rate_columns(rate_panel()), horizon=3)
    expected = pd.DataFrame(
        {
            "total": [2, 2, 1, 2],
            "event": [1, 0, 0, 1],
            "exited": [1, 0, 0, 0],
            "event_rate": [0.5, 0.0, 0.0, 0.5],
        },
        index=rate_index(["2024-03", "2024-03", "2024-06", "2024-06"], ["A", "B", "A", "B"]),
    )
    pd.testing.assert_frame_equal(table, expected)


def test_default_rate_table_counts_each_cohort_loan_once():
    # Loan 1 defaults twice in the window, loan 2 is in default at the start and stays out of
    # the cohort, and loan 3 defaults only after the window.
    loan = [1] * 5 + [2] * 5 + [3] * 5
    month = pd.PeriodIndex(["2024-03", "2024-04", "2024-05", "2024-06", "2024-07"] * 3, freq="M")
    defaulted = [0, 1, 0, 1, 0] + [1, 0, 0, 0, 0] + [0, 0, 0, 0, 1]
    table = default_rate_table(loan, month, ["A"] * 15, defaulted, horizon=3, starts=["2024-03"])
    assert table.loc[(pd.Period("2024-03", "M"), "A")].tolist() == [2, 1, 0, 0.5]


def test_default_rate_table_takes_the_grade_of_the_start_month():
    table = default_rate_table(*rate_columns(rate_panel()), horizon=3, starts=["2024-06"])
    # Loan 3 moves to grade A in August but stays in the B cohort of June.
    assert table.loc[(pd.Period("2024-06", "M"), "B"), "total"] == 2


def test_default_rate_table_reports_missing_grades_last():
    panel = rate_panel()
    panel.loc[(panel["loan"] == 3) & (panel["month"] == pd.Period("2024-03", "M")), "grade"] = None
    grades = pd.Categorical(panel["grade"], categories=["B", "A"], ordered=True)
    table = default_rate_table(panel["loan"], panel["month"], grades, panel["defaulted"], horizon=3)
    assert table.index.get_level_values("grade").tolist()[:3] == ["B", "A", np.nan]
    assert table["total"].tolist() == [1, 2, 1, 2, 1]
    assert isinstance(table.index.levels[1], pd.CategoricalIndex)
    # Grades that are not categorical are sorted.
    reverse = default_rate_table(
        panel["loan"],
        panel["month"],
        panel["grade"].map({"A": 2, "B": 1}),
        panel["defaulted"],
        horizon=3,
    )
    assert reverse.index.get_level_values("grade")[:3].tolist()[:2] == [1.0, 2.0]


def test_default_rate_table_keeps_exits_in_the_cohort():
    # Two loans of three leave before the end of the window without a default.
    loan = [1, 1, 2, 3, 3, 3, 3]
    month = pd.PeriodIndex(
        ["2024-03", "2024-04", "2024-03", "2024-03", "2024-04", "2024-05", "2024-06"], freq="M"
    )
    table = default_rate_table(loan, month, ["A"] * 7, [False] * 7, horizon=3, starts=["2024-03"])
    assert table[["total", "event", "exited"]].iloc[0].tolist() == [3, 0, 2]


def brute_default_rates(loan, month, grade, defaulted, horizon, starts, weights=None):
    """One row per cohort loan: start, grade, event, exited and weight, by looping."""
    frame = pd.DataFrame(
        {
            "loan": loan,
            "month": month,
            "grade": grade,
            "defaulted": defaulted,
            "weight": 1.0 if weights is None else weights,
        }
    )
    records = []
    for start in starts:
        cohort = frame[(frame["month"] == start) & ~frame["defaulted"]]
        for row in cohort.itertuples():
            own = frame[frame["loan"] == row.loan]
            window = own[(own["month"] > start) & (own["month"] <= start + horizon)]
            event = bool(window["defaulted"].any())
            exited = not event and own["month"].max() < start + horizon
            records.append((start, row.grade, event, exited, row.weight))
    return pd.DataFrame(records, columns=["start", "grade", "event", "exited", "weight"])


def random_rate_panel(rng, loans=60, months=30):
    records = []
    for number in range(loans):
        first = int(rng.integers(0, months - 1))
        length = int(rng.integers(1, months - first + 1))
        grade = rng.choice(["A", "B", "C", None])
        for offset in range(length):
            if rng.random() < 0.15:  # a month without a row
                continue
            if rng.random() < 0.1:
                grade = rng.choice(["A", "B", "C", None])
            records.append((f"L{number}", first + offset, grade, bool(rng.random() < 0.1)))
    panel = pd.DataFrame(records, columns=["loan", "offset", "grade", "defaulted"])
    calendar = pd.period_range("2020-01", periods=months, freq="M")
    panel["month"] = calendar[panel["offset"].to_numpy()]
    return panel


@pytest.mark.parametrize("seed", range(8))
def test_default_rate_table_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    panel = random_rate_panel(rng)
    horizon = int(rng.integers(1, 13))
    first, last = panel["month"].min(), panel["month"].max()
    candidates = pd.period_range(first, last - horizon, freq="M")
    starts = list(rng.choice(candidates, size=min(5, len(candidates)), replace=False))
    table = default_rate_table(
        panel["loan"],
        panel["month"],
        panel["grade"],
        panel["defaulted"],
        horizon=horizon,
        starts=starts,
    )
    rows = brute_default_rates(
        panel["loan"], panel["month"], panel["grade"], panel["defaulted"], horizon, starts
    )
    expected = (
        rows.groupby(["start", "grade"], dropna=False, sort=False)
        .agg(total=("event", "size"), event=("event", "sum"), exited=("exited", "sum"))
        .reset_index()
    )
    expected["grade_order"] = expected["grade"].map({"A": 0, "B": 1, "C": 2}).fillna(3)
    expected = expected.sort_values(["start", "grade_order"]).drop(columns="grade_order")
    got = table.reset_index()
    assert got["start"].tolist() == expected["start"].tolist()
    assert got["grade"].fillna("missing").tolist() == expected["grade"].fillna("missing").tolist()
    for column in ("total", "event", "exited"):
        assert got[column].tolist() == expected[column].astype(np.int64).tolist()
    np.testing.assert_array_equal(got["event_rate"], got["event"] / got["total"])


def test_default_rate_table_default_starts_are_complete_quarter_ends():
    rng = np.random.default_rng(3)
    panel = random_rate_panel(rng, months=40)
    table = default_rate_table(panel["loan"], panel["month"], panel["grade"], panel["defaulted"])
    starts = table.index.get_level_values("start").unique()
    last = panel["month"].max()
    expected = [
        month
        for month in pd.period_range(panel["month"].min(), last, freq="M")
        if month.month % 3 == 0 and month + 12 <= last
    ]
    assert starts.tolist() == expected


def test_default_rate_table_integer_weights_equal_repeated_loans():
    rng = np.random.default_rng(5)
    panel = random_rate_panel(rng)
    times = pd.Series(
        rng.integers(0, 4, size=panel["loan"].nunique()), index=panel["loan"].unique()
    )
    weights = panel["loan"].map(times).to_numpy(dtype=float)
    weighted = default_rate_table(
        panel["loan"],
        panel["month"],
        panel["grade"],
        panel["defaulted"],
        horizon=6,
        weights=weights,
    )
    copies = [
        panel[panel["loan"] == loan].assign(loan=f"{loan}-{copy}")
        for loan, count in times.items()
        for copy in range(count)
    ]
    repeated = pd.concat(copies, ignore_index=True)
    plain = default_rate_table(
        repeated["loan"],
        repeated["month"],
        repeated["grade"],
        repeated["defaulted"],
        horizon=6,
        starts=weighted.index.get_level_values("start").unique(),
    )
    pd.testing.assert_frame_equal(weighted, plain.astype(float), check_index_type=False)


def test_default_rate_table_leaves_out_groups_without_weight():
    table = default_rate_table(
        *rate_columns(rate_panel()),
        horizon=3,
        starts=["2024-03"],
        weights=np.where(rate_panel()["grade"] == "B", 0.0, 1.5),
    )
    assert table.index.get_level_values("grade").tolist() == ["A"]
    assert table.dtypes.tolist() == [np.float64] * 4
    assert table["total"].tolist() == [3.0]


def test_default_rate_table_one_start_equals_calibration_table():
    panel = make_loan_panel(rng=0)
    start = pd.Period("2024-09", "M")
    defaulted = panel["status"] == "90+"
    probability = panel["month_on_book"] / 100
    table = default_rate_table(
        panel["loan_id"],
        panel["month"],
        panel["status"],
        defaulted,
        starts=[start],
        probability=probability,
    )
    rows = brute_default_rates(
        panel["loan_id"], panel["month"], panel["status"], defaulted, 12, [start]
    )
    cohort = panel[(panel["month"] == start) & ~defaulted]
    expected = calibration_table(
        cohort["status"].cat.remove_unused_categories(),
        rows["event"].to_numpy(dtype=int),
        probability[cohort.index].to_numpy(),
    )
    assert table.loc[start, ["total", "event"]].to_numpy().tolist() == (
        expected[["total", "event"]].to_numpy().tolist()
    )
    np.testing.assert_allclose(
        table.loc[start, "mean_probability"], expected["mean_probability"], rtol=1e-15
    )


def test_default_rate_table_without_exits_equals_kaplan_meier():
    # Every loan has a row in each month of the window, so no loan is censored.
    rng = np.random.default_rng(11)
    loans, months = 300, 13
    defaulted = rng.random((loans, months)) < 0.02
    defaulted[:, 0] = False
    panel = pd.DataFrame(
        {
            "loan": np.repeat(np.arange(loans), months),
            "month": np.tile(pd.period_range("2023-12", periods=months, freq="M"), loans),
            "defaulted": defaulted.ravel(),
        }
    )
    table = default_rate_table(
        panel["loan"], panel["month"], ["all"] * len(panel), panel["defaulted"]
    )
    first = np.where(defaulted[:, 1:].any(axis=1), defaulted[:, 1:].argmax(axis=1) + 1, 12)
    curve = survival_table(first, defaulted[:, 1:].any(axis=1).astype(int))
    assert table["event_rate"].iloc[0] == pytest.approx(
        curve["cumulative_pd"].iloc[-1], rel=1e-12, abs=0
    )
    assert table["exited"].iloc[0] == 0


def test_default_rate_table_on_the_simulated_loan_book():
    panel = make_loan_panel(rng=0)
    table = default_rate_table(
        panel["loan_id"], panel["month"], panel["status"], panel["status"] == "90+"
    )
    starts = table.index.get_level_values("start").unique()
    assert starts.tolist() == list(pd.period_range("2024-03", "2024-12", freq="Q-DEC").asfreq("M"))
    assert "90+" not in table.index.get_level_values("grade")
    portfolio = table.groupby(level="start")[["total", "event"]].sum()
    rate = portfolio["event"] / portfolio["total"]
    assert rate.round(4).tolist() == [0.0683, 0.0626, 0.0564, 0.052]
    current = table.xs("current", level="grade")
    assert current.loc[pd.Period("2024-12", "M"), ["total", "event", "exited"]].tolist() == [
        2146,
        73,
        413,
    ]


def test_default_rate_table_reads_datetimes_and_strings():
    panel = rate_panel()
    expected = default_rate_table(*rate_columns(panel), horizon=3)
    timestamps = panel["month"].dt.to_timestamp(how="end")
    for month in (timestamps, timestamps.dt.tz_localize("Asia/Shanghai")):
        table = default_rate_table(
            panel["loan"],
            month,
            panel["grade"],
            panel["defaulted"],
            horizon=3,
            starts=[
                "2024Q1",
                np.datetime64("2024-06-30"),
            ],
        )
        pd.testing.assert_frame_equal(table, expected)
    starts = pd.PeriodIndex(["2024-06", "2024-03"], freq="M")
    pd.testing.assert_frame_equal(
        default_rate_table(*rate_columns(panel), horizon=3, starts=starts), expected
    )


def test_default_rate_table_mean_probability():
    panel = rate_panel()
    probability = np.where(panel["defaulted"], np.nan, 0.01 * panel["loan"])
    table = default_rate_table(*rate_columns(panel), horizon=3, probability=probability)
    assert table["mean_probability"].round(10).tolist() == [0.015, 0.035, 0.05, 0.035]
    weighted = default_rate_table(
        *rate_columns(panel), horizon=3, probability=probability, weights=panel["loan"]
    )
    assert weighted["mean_probability"].iloc[0] == pytest.approx((0.01 + 0.04) / 3, rel=1e-12)


def test_default_rate_table_errors():
    panel = rate_panel()
    loan, month, grade, defaulted = rate_columns(panel)
    with pytest.raises(ValueError, match="^loan, month, grade and defaulted must have the same"):
        default_rate_table(loan.tolist()[:-1], month, grade, defaulted)
    with pytest.raises(ValueError, match="^loan, month, grade and defaulted must not be empty$"):
        default_rate_table([], [], [], [])
    with pytest.raises(ValueError, match="^loan and month must have identical indices$"):
        default_rate_table(
            loan, month.reset_index(drop=True).set_axis(month.index + 1), grade, defaulted
        )
    with pytest.raises(ValueError, match="^horizon must be a positive integer, got 0$"):
        default_rate_table(loan, month, grade, defaulted, horizon=0)
    with pytest.raises(ValueError, match="^horizon must be a positive integer, got True$"):
        default_rate_table(loan, month, grade, defaulted, horizon=True)
    with pytest.raises(ValueError, match="^loan must not contain missing values, got 5$"):
        default_rate_table(loan.where(loan != 5), month, grade, defaulted, horizon=3)
    with pytest.raises(
        ValueError, match="^month must hold monthly periods or datetimes, got dtype"
    ):
        default_rate_table(loan, month.astype(str), grade, defaulted, horizon=3)
    with pytest.raises(ValueError, match="^month must hold monthly periods or datetimes"):
        default_rate_table(loan, month.dt.asfreq("D"), grade, defaulted, horizon=3)
    with pytest.raises(ValueError, match="^month must not contain missing values, got 5$"):
        default_rate_table(loan, month.where(loan != 5), grade, defaulted, horizon=3)
    with pytest.raises(ValueError, match="^defaulted must contain only booleans or 0 and 1"):
        default_rate_table(loan, month, grade, defaulted.astype(int) * 2, horizon=3)
    with pytest.raises(
        ValueError, match="^each loan must have at most one row per month, got two rows for loan 5"
    ):
        default_rate_table(
            pd.concat([loan, loan.tail(1)]),
            pd.concat([month, month.tail(1)]),
            pd.concat([grade, grade.tail(1)]),
            pd.concat([defaulted, defaulted.tail(1)]),
            horizon=3,
        )


def test_default_rate_table_start_errors():
    columns = rate_columns(rate_panel())
    with pytest.raises(
        ValueError,
        match=re.escape(
            "the panel must cover a quarter end and the 12 months after it, got 2024-01 to "
            "2024-09; pass starts or a smaller horizon"
        ),
    ):
        default_rate_table(*columns)
    with pytest.raises(ValueError, match="^starts must be a list of months, got '2024-03'$"):
        default_rate_table(*columns, horizon=3, starts="2024-03")
    with pytest.raises(ValueError, match="^starts must not be empty$"):
        default_rate_table(*columns, horizon=3, starts=[])
    with pytest.raises(ValueError, match="^starts must be a date or a month, got 3$"):
        default_rate_table(*columns, horizon=3, starts=[3])
    with pytest.raises(ValueError, match="^starts must not repeat a month, got 2024-03 more than"):
        default_rate_table(*columns, horizon=3, starts=["2024-03", "2024Q1"])
    with pytest.raises(
        ValueError,
        match="^starts must not be before the first month of the panel, 2024-01, got 2023-12$",
    ):
        default_rate_table(*columns, horizon=3, starts=["2023-12"])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "starts must be followed by 1 month of the panel, which ends in 2024-09, got 2024-09"
        ),
    ):
        default_rate_table(*columns, horizon=1, starts=["2024-09"])


def test_default_rate_table_probability_and_weight_errors():
    columns = rate_columns(rate_panel())
    size = len(columns[0])
    with pytest.raises(ValueError, match="^probability must have one value per row of loan, got 2"):
        default_rate_table(*columns, horizon=3, probability=[0.1, 0.2])
    with pytest.raises(ValueError, match="^probability must be between 0 and 1, got 1.5$"):
        default_rate_table(*columns, horizon=3, probability=[1.5] * size)
    with pytest.raises(
        ValueError,
        match="^probability must not be missing at the start month of a cohort, got 7 missing",
    ):
        default_rate_table(*columns, horizon=3, probability=[np.nan] * size)
    with pytest.raises(ValueError, match="^weights must be finite and non-negative, got -1.0$"):
        default_rate_table(*columns, horizon=3, weights=[-1.0] * size)
