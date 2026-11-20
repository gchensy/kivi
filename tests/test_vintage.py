import numpy as np
import pandas as pd
import pytest

from kivi.datasets import make_loan_panel
from kivi.vintage import roll_rates, transition_matrix, vintage_table

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
    rates = roll_rates(["a", "a"], ["a", "a"])
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
        roll_rates([None, "a"], ["a", "a"])
