import itertools

import numpy as np
import pandas as pd
import pytest

from kivi.datasets import make_loan_panel
from kivi.lifetime import (
    assign_stages,
    convert_pd,
    expected_credit_loss,
    significant_increase,
    stage_table,
    survival_table,
    transition_pd,
)
from kivi.strategy import expected_loss
from kivi.vintage import transition_matrix, vintage_table

KINDS = ("cumulative", "marginal", "conditional")

# A cumulative curve and its marginal and conditional PDs, by hand.
CUMULATIVE = [0.02, 0.05, 0.08]
MARGINAL = [0.02, 0.03, 0.03]
CONDITIONAL = [0.02, 0.03 / 0.98, 0.03 / 0.95]

# Three states; D is the default state. Two-period PDs by hand:
# A: 0.90 * 0.02 + 0.08 * 0.20 + 0.02 * 1 = 0.054; B: 0.10 * 0.02 + 0.70 * 0.20 + 0.20 = 0.342.
STATES = ["A", "B", "D"]
MATRIX = pd.DataFrame(
    [[0.90, 0.08, 0.02], [0.10, 0.70, 0.20], [0.00, 0.00, 1.00]], index=STATES, columns=STATES
)

# Two accounts over three annual periods, discounted at 5% per period.
ECL_MARGINAL = [[0.02, 0.03, 0.03], [0.10, 0.08, 0.06]]
ECL_SCHEDULE = [[1000.0, 700.0, 400.0], [500.0, 250.0, 0.0]]


def random_curves(seed, shape=(6, 12)):
    """Cumulative PD curves that rise from 0 to at most 1."""
    rng = np.random.default_rng(seed)
    marginal = rng.dirichlet(np.ones(shape[1] + 1), size=shape[0])[:, :-1]
    return np.cumsum(marginal, axis=1)


@pytest.fixture(scope="module")
def panel():
    return make_loan_panel(n_loans=1500, n_months=18, rng=7)


@pytest.fixture(scope="module")
def monthly(panel):
    """Monthly transition matrix of the panel's delinquency buckets."""
    later = panel[["loan_id", "month_on_book", "status"]].assign(
        month_on_book=panel["month_on_book"] - 1
    )
    pairs = panel.merge(later, on=["loan_id", "month_on_book"], suffixes=("", "_next"))
    return transition_matrix(pairs["status"], pairs["status_next"], normalize=True)


def absorbing(matrix, default):
    result = np.array(matrix, dtype=float)
    result[default] = 0.0
    result[default, default] = 1.0
    return result


# convert_pd


def test_convert_pd_hand_example():
    for kind, values in (("marginal", MARGINAL), ("conditional", CONDITIONAL)):
        np.testing.assert_allclose(convert_pd(CUMULATIVE, kind="cumulative", to=kind), values)
        np.testing.assert_allclose(convert_pd(values, kind=kind, to="cumulative"), CUMULATIVE)
    np.testing.assert_allclose(convert_pd(MARGINAL, kind="marginal", to="conditional"), CONDITIONAL)
    np.testing.assert_allclose(convert_pd(CONDITIONAL, kind="conditional", to="marginal"), MARGINAL)


@pytest.mark.parametrize(("kind", "to"), list(itertools.permutations(KINDS, 2)))
def test_convert_pd_round_trips(kind, to):
    curves = convert_pd(random_curves(0), kind="cumulative", to=kind)
    there = convert_pd(curves, kind=kind, to=to)
    np.testing.assert_allclose(convert_pd(there, kind=to, to=kind), curves, rtol=1e-12, atol=1e-15)


def test_convert_pd_same_kind_returns_a_copy():
    curves = random_curves(1)
    for kind in KINDS:
        values = convert_pd(curves, kind="cumulative", to=kind)
        result = convert_pd(values, kind=kind, to=kind)
        np.testing.assert_array_equal(result, values)
        result[0, 0] = 0.5
        assert values[0, 0] != 0.5


def test_convert_pd_cumulative_rises_and_never_exceeds_one():
    rng = np.random.default_rng(2)
    conditional = rng.uniform(0, 0.5, size=(5, 60))
    cumulative = convert_pd(conditional, kind="conditional", to="cumulative")
    assert (np.diff(cumulative, axis=1) >= 0).all()
    assert (cumulative <= 1).all()
    # Marginal PDs that add up to 1 only up to rounding still give cumulative PDs of at most 1.
    marginal = convert_pd([0.1, 0.7, 1.0], kind="cumulative", to="marginal")
    cumulative = convert_pd(marginal, kind="marginal", to="cumulative")
    assert cumulative.max() <= 1
    np.testing.assert_allclose(cumulative, [0.1, 0.7, 1.0])


def test_convert_pd_matches_counts_of_a_vintage_panel(panel):
    bad = panel["status"] == "90+"
    table = vintage_table(panel["loan_id"], panel["cohort"], panel["month_on_book"], bad)
    conditional = convert_pd(table, kind="cumulative", to="conditional")
    # Brute force: loans first bad at k over the loans of the cohort not bad before k.
    first_bad = panel[bad].groupby("loan_id")["month_on_book"].min()
    loans = panel.groupby("loan_id")["cohort"].first()
    oldest = panel.groupby("cohort")["month_on_book"].max()
    for cohort in loans.unique()[::4]:
        members = loans.index[loans == cohort]
        first = first_bad.reindex(members)
        for k in table.columns:
            expected = np.nan
            if k <= oldest[cohort]:
                expected = (first == k).sum() / (~(first < k)).sum()
            np.testing.assert_allclose(conditional.loc[cohort, k], expected, rtol=1e-12)
    pd.testing.assert_frame_equal(
        convert_pd(conditional, kind="conditional", to="cumulative"), table, rtol=1e-12
    )


def test_convert_pd_keeps_months_not_reached_missing():
    table = pd.DataFrame(
        [[0.0, 0.5, 0.5], [0.5, 0.5, np.nan]],
        index=pd.Index(["2024-01", "2024-02"], name="cohort"),
        columns=pd.Index([1, 2, 3], name="month_on_book"),
    )
    expected = {
        "marginal": [[0.0, 0.5, 0.0], [0.5, 0.0, np.nan]],
        "conditional": [[0.0, 0.5, 0.0], [0.5, 0.0, np.nan]],
    }
    for kind, values in expected.items():
        result = convert_pd(table, kind="cumulative", to=kind)
        pd.testing.assert_frame_equal(result, pd.DataFrame(values, table.index, table.columns))
        pd.testing.assert_frame_equal(convert_pd(result, kind=kind, to="cumulative"), table)


def test_convert_pd_with_missing_values_before_and_inside_a_curve():
    # A panel that starts at month on book 2 leaves month on book 1 unknown.
    cumulative = [np.nan, 0.1, 0.3, np.nan, 0.4]
    marginal = convert_pd(cumulative, kind="cumulative", to="marginal")
    np.testing.assert_allclose(marginal, [np.nan, np.nan, 0.2, np.nan, np.nan])
    conditional = convert_pd(cumulative, kind="cumulative", to="conditional")
    np.testing.assert_allclose(conditional, [np.nan, np.nan, 0.2 / 0.9, np.nan, np.nan])
    # The cumulative PDs cannot be restored without the PD before the panel starts.
    assert np.isnan(convert_pd(marginal, kind="marginal", to="cumulative")).all()
    # A missing marginal or conditional PD makes every later cumulative PD missing.
    np.testing.assert_allclose(
        convert_pd([0.1, np.nan, 0.2], kind="marginal", to="cumulative"), [0.1, np.nan, np.nan]
    )


def test_convert_pd_after_everything_defaulted():
    cumulative = [0.5, 1.0, 1.0]
    conditional = convert_pd(cumulative, kind="cumulative", to="conditional")
    np.testing.assert_array_equal(conditional, [0.5, 1.0, 0.0])
    np.testing.assert_array_equal(
        convert_pd(cumulative, kind="cumulative", to="marginal"),
        [
            0.5,
            0.5,
            0.0,
        ],
    )
    np.testing.assert_array_equal(
        convert_pd(conditional, kind="conditional", to="cumulative"), cumulative
    )
    # A cohort that defaulted entirely and has not reached month on book 2.
    conditional = convert_pd([1.0, np.nan], kind="cumulative", to="conditional")
    np.testing.assert_array_equal(conditional, [1.0, np.nan])
    np.testing.assert_array_equal(
        convert_pd(conditional, kind="conditional", to="cumulative"), [1.0, np.nan]
    )


def test_convert_pd_keeps_the_shape_of_the_input():
    series = pd.Series(CUMULATIVE, index=pd.Index([1, 2, 3], name="month_on_book"), name="pd")
    result = convert_pd(series, kind="cumulative", to="marginal")
    pd.testing.assert_series_equal(result, pd.Series(MARGINAL, index=series.index, name="pd"))
    result = convert_pd([CUMULATIVE, CUMULATIVE], kind="cumulative", to="marginal")
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [MARGINAL, MARGINAL])
    frame = pd.DataFrame([CUMULATIVE], index=["a"], columns=[1, 2, 3])
    result = convert_pd(frame, kind="cumulative", to="marginal")
    pd.testing.assert_frame_equal(result, pd.DataFrame([MARGINAL], index=["a"], columns=[1, 2, 3]))


def test_convert_pd_accepts_booleans_and_nullable_values():
    np.testing.assert_array_equal(
        convert_pd([False, True, True], kind="cumulative", to="marginal"), [0.0, 1.0, 0.0]
    )
    values = pd.array([0.1, None, 0.3], dtype="Float64")
    np.testing.assert_allclose(
        convert_pd(values, kind="marginal", to="cumulative"), [0.1, np.nan, np.nan]
    )
    np.testing.assert_allclose(
        convert_pd([[0.1, None], [0.2, 0.3]], kind="cumulative", to="marginal"),
        [[0.1, np.nan], [0.2, 0.1]],
    )


def test_convert_pd_reads_columns_of_missing_values_as_numbers():
    # pandas reads a column of None only as objects.
    np.testing.assert_allclose(
        convert_pd([[0.1, None], [0.2, None]], kind="cumulative", to="marginal"),
        [[0.1, np.nan], [0.2, np.nan]],
    )
    assert np.isnan(convert_pd([None, None], kind="marginal", to="cumulative")).all()


def test_convert_pd_does_not_mutate_its_input():
    frame = pd.DataFrame(random_curves(3))
    original = frame.copy()
    for kind, to in itertools.permutations(KINDS, 2):
        values = convert_pd(frame, kind="cumulative", to=kind)
        convert_pd(values, kind=kind, to=to)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "hazard", "to": "marginal"}, "kind must be 'cumulative', 'marginal' or "),
        ({"kind": "cumulative", "to": None}, "to must be 'cumulative', 'marginal' or "),
        ({"kind": ["cumulative"], "to": "marginal"}, r"kind must be .* got \['cumulative'\]"),
    ],
)
def test_convert_pd_rejects_unknown_kinds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        convert_pd(CUMULATIVE, **kwargs)


@pytest.mark.parametrize(
    ("values", "kind", "message"),
    [
        (0.1, "cumulative", "probability must be one- or two-dimensional, got 0 dimensions"),
        (np.zeros((2, 2, 2)), "cumulative", "must be one- or two-dimensional, got 3 dimensions"),
        ([], "cumulative", "probability must have at least one period"),
        (np.zeros((2, 0)), "cumulative", "probability must have at least one period"),
        (["a", "b"], "cumulative", "probability must be real and numeric, got dtype"),
        ([0.1, 1.2], "conditional", r"probability must be between 0 and 1, got 1.2"),
        ([-0.1, 0.2], "marginal", r"probability must be between 0 and 1, got -0.1"),
        ([0.1, np.inf], "conditional", r"probability must be between 0 and 1, got inf"),
        (
            [0.1, 0.3, 0.2],
            "cumulative",
            r"must not decrease along a curve when kind is 'cumulative', got 0.2 after 0.3",
        ),
        (
            [0.3, np.nan, 0.2],
            "cumulative",
            r"must not decrease along a curve when kind is 'cumulative', got 0.2 after 0.3",
        ),
        (
            [[0.5, 0.4], [0.6, 0.5]],
            "marginal",
            r"must add up to at most 1 along a curve when kind is 'marginal', got 1.1",
        ),
    ],
)
def test_convert_pd_rejects_invalid_curves(values, kind, message):
    to = "marginal" if kind != "marginal" else "cumulative"
    with pytest.raises(ValueError, match=message):
        convert_pd(values, kind=kind, to=to)


# transition_pd


def test_transition_pd_hand_example():
    result = transition_pd(MATRIX, horizons=3)
    expected = pd.DataFrame(
        [[0.02, 0.054, 0.09596], [0.2, 0.342, 0.4448], [1.0, 1.0, 1.0]],
        index=pd.Index(STATES, name="state"),
        columns=pd.Index([1, 2, 3], name="horizon"),
    )
    pd.testing.assert_frame_equal(result, expected, rtol=1e-12)


def test_transition_pd_equals_the_default_column_of_matrix_powers(monthly):
    result = transition_pd(monthly, horizons=36)
    power = absorbing(monthly, 4)
    for h in (1, 2, 7, 36):
        expected = np.linalg.matrix_power(power, h)[:, 4]
        np.testing.assert_allclose(result[h], expected, rtol=1e-12, atol=1e-15)
    assert result.index.equals(pd.Index(monthly.index, name="state"))


@pytest.mark.parametrize(("a", "b"), [(1, 1), (3, 9), (12, 12), (5, 30)])
def test_transition_pd_chapman_kolmogorov(monthly, a, b):
    # P^(a+b) = P^a P^b: the PD at a + b is the PD at b averaged over the states after a.
    after_a = np.linalg.matrix_power(absorbing(monthly, 4), a)
    later = transition_pd(monthly, horizons=[b, a + b])
    np.testing.assert_allclose(later[a + b], after_a @ later[b], rtol=1e-12, atol=1e-15)


def test_transition_pd_rises_with_the_horizon_and_never_exceeds_one(monthly):
    result = transition_pd(monthly, horizons=600).to_numpy()
    assert (np.diff(result, axis=1) >= 0).all()
    assert (result >= 0).all() and (result <= 1).all()
    # Default is reachable from every state, so the PD keeps rising towards 1.
    assert (result[:-1, -1] > result[:-1, 299]).all()
    assert result[:, -1].min() > 0.95


def test_transition_pd_makes_the_default_state_absorbing():
    cures = MATRIX.copy()
    cures.loc["D"] = [0.3, 0.1, 0.6]
    pd.testing.assert_frame_equal(
        transition_pd(cures, horizons=5), transition_pd(MATRIX, horizons=5)
    )
    # The row of a default state without entities is missing, and is not used.
    cures.loc["D"] = np.nan
    pd.testing.assert_frame_equal(
        transition_pd(cures, horizons=5), transition_pd(MATRIX, horizons=5)
    )


def test_transition_pd_with_a_default_state_that_is_not_last():
    order = ["D", "B", "A"]
    reordered = MATRIX.loc[order, order]
    result = transition_pd(reordered, horizons=4, default_state="D")
    pd.testing.assert_frame_equal(result, transition_pd(MATRIX, horizons=4).loc[order])


def test_transition_pd_horizons_as_a_sequence():
    every = transition_pd(MATRIX, horizons=24)
    some = transition_pd(MATRIX, horizons=[1, 12, 24])
    pd.testing.assert_frame_equal(some, every[[1, 12, 24]])
    # A sequence is read like data, so whole numbers held as floats are accepted.
    pd.testing.assert_frame_equal(transition_pd(MATRIX, horizons=np.array([12.0])), every[[12]])


def test_transition_pd_at_very_long_horizons():
    result = transition_pd(MATRIX, horizons=[12, 10**15])
    np.testing.assert_allclose(result[10**15], [1.0, 1.0, 1.0])
    pd.testing.assert_series_equal(result[12], transition_pd(MATRIX, horizons=12)[12])
    for horizons in ([1e19], 2**63):
        with pytest.raises(ValueError, match="horizons must be a positive integer or a sequence"):
            transition_pd(MATRIX, horizons=horizons)


def test_transition_pd_with_states_of_two_levels():
    states = pd.MultiIndex.from_tuples([("retail", "A"), ("retail", "B"), ("retail", "D")])
    matrix = pd.DataFrame(MATRIX.to_numpy(), index=states, columns=states)
    result = transition_pd(matrix, horizons=3, default_state=("retail", "D"))
    assert result.index.equals(states)
    np.testing.assert_allclose(result.to_numpy(), transition_pd(MATRIX, horizons=3).to_numpy())


def test_transition_pd_messages_show_states_of_two_levels_as_plain_tuples():
    states = pd.MultiIndex.from_arrays([np.array([1, 1, 2]), np.array([10, 20, 30])])
    matrix = pd.DataFrame(
        [[0.5, 0.4, 0.0], [0.1, 0.7, 0.2], [0.0, 0.0, 1.0]], index=states, columns=states
    )
    with pytest.raises(ValueError, match=r"got 0\.9 for state \(1, 10\); divide each row"):
        transition_pd(matrix, horizons=2)
    repeated = pd.MultiIndex.from_arrays([np.array([1, 1]), np.array([10, 10])])
    with pytest.raises(ValueError, match=r"must not repeat a state, got \(1, 10\) more than once"):
        transition_pd(pd.DataFrame(np.eye(2), index=repeated, columns=repeated), horizons=2)


def test_transition_pd_horizons_as_integer_scalars():
    expected = transition_pd(MATRIX, horizons=2)
    pd.testing.assert_frame_equal(transition_pd(MATRIX, horizons=np.int64(2)), expected)
    # A number of periods is an integer, as for expected_credit_loss; 2.0 used to be accepted.
    for horizons in (2.0, np.float64(2.0), np.float32(3.0)):
        with pytest.raises(
            ValueError,
            match=rf"positive integer or a sequence of increasing positive integers, got "
            rf"{float(horizons)!r}$",
        ):
            transition_pd(MATRIX, horizons=horizons)
    with pytest.raises(
        ValueError, match=r"sequence of increasing positive integers, got \[2, 1\]$"
    ):
        transition_pd(MATRIX, horizons=pd.Series([2, 1]))


def test_transition_pd_default_state_must_be_a_label():
    states = pd.IntervalIndex.from_breaks([0, 30, 60, 90])
    matrix = pd.DataFrame(MATRIX.to_numpy(), index=states, columns=states)
    with pytest.raises(ValueError, match="default_state must be one of the states, got 75"):
        transition_pd(matrix, horizons=2, default_state=75)
    result = transition_pd(matrix, horizons=2, default_state=pd.Interval(60, 90))
    np.testing.assert_allclose(result[2], [0.054, 0.342, 1.0])


def test_transition_pd_of_an_array():
    result = transition_pd(MATRIX.to_numpy(), horizons=2)
    assert result.index.equals(pd.RangeIndex(3, name="state"))
    np.testing.assert_allclose(result.to_numpy(), transition_pd(MATRIX, horizons=2).to_numpy())
    result = transition_pd(MATRIX.to_numpy(), horizons=2, default_state=2)
    np.testing.assert_allclose(result[2], [0.054, 0.342, 1.0])


def test_transition_pd_gives_marginal_pds_through_convert_pd():
    cumulative = transition_pd(MATRIX, horizons=10)
    marginal = convert_pd(cumulative, kind="cumulative", to="marginal")
    np.testing.assert_allclose(marginal.sum(axis=1), cumulative[10])
    assert (marginal.to_numpy() >= 0).all()


def test_transition_pd_does_not_mutate_its_input():
    matrix = MATRIX.copy()
    matrix.loc["D"] = [0.3, 0.1, 0.6]
    original = matrix.copy()
    transition_pd(matrix, horizons=3)
    pd.testing.assert_frame_equal(matrix, original)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ([0.5, 0.5], "matrix must be two-dimensional, got 1 dimensions"),
        ([[0.5, 0.5]], r"matrix must be square and not empty, got shape \(1, 2\)"),
        (np.zeros((0, 0)), r"matrix must be square and not empty, got shape \(0, 0\)"),
        (
            pd.DataFrame(np.eye(2), index=["a", "b"], columns=["b", "a"]),
            r"same states in its index and its columns, got \['a', 'b'\] and \['b', 'a'\]",
        ),
        (
            pd.DataFrame(np.eye(2), index=["a", "a"], columns=["a", "a"]),
            "matrix must not repeat a state, got 'a' more than once",
        ),
        ([[True, False], [False, True]], "matrix must be real and numeric, got dtype bool"),
        (
            [[0.5, 0.4], [0.0, 1.0]],
            r"the rows of matrix must add up to 1, got 0.9 for state 0; divide each row by its sum",
        ),
        (
            [[1.5, -0.5], [0.0, 1.0]],
            r"matrix must hold probabilities between 0 and 1, got 1.5 in the row of state 0",
        ),
        (
            [[np.nan, np.nan, np.nan], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "missing values outside the row of the default state, got one in the row of state 0",
        ),
    ],
)
def test_transition_pd_rejects_invalid_matrices(matrix, message):
    with pytest.raises(ValueError, match=message):
        transition_pd(matrix, horizons=2)


@pytest.mark.parametrize(
    "horizons", [0, -1, True, 2.5, np.inf, "2", [], [2, 1], [1, 1], [0, 1], [1.5], [[1]]]
)
def test_transition_pd_rejects_invalid_horizons(horizons):
    with pytest.raises(ValueError, match="horizons must be a positive integer or a sequence"):
        transition_pd(MATRIX, horizons=horizons)


@pytest.mark.parametrize("default_state", ["E", ["D"], 2])
def test_transition_pd_rejects_unknown_default_states(default_state):
    with pytest.raises(ValueError, match="default_state must be one of the states, got"):
        transition_pd(MATRIX, horizons=2, default_state=default_state)


# survival_table

# Eight loans over three months (the docstring example): three default, two repay early, three
# are still on book at the end.
DURATION = [1, 1, 2, 2, 2, 3, 3, 3]
EVENT = [1, 0, 1, 0, 0, 1, 0, 0]
REPAID = [0, 1, 0, 1, 0, 0, 0, 0]


def survival_frame(columns, periods):
    return pd.DataFrame(columns, index=pd.Index(np.arange(1, periods + 1), name="period"))


def brute_force_survival(duration, event, competing=None, entry=None, weights=None):
    """survival_table account by account and period by period, straight from the formulas."""
    n = len(duration)
    competing = [0] * n if competing is None else competing
    entry = [0] * n if entry is None else entry
    weights = [1.0] * n if weights is None else weights
    rows = [i for i in range(n) if weights[i] > 0]
    periods = max(duration[i] for i in rows)
    table = {name: [] for name in ["at_risk", "event", "competing", "censored", "hazard"]}
    survival, cumulative, before, incidence = [], [], 1.0, 0.0
    for t in range(1, periods + 1):
        at_risk = sum(weights[i] for i in rows if entry[i] < t <= duration[i])
        stop = [i for i in rows if duration[i] == t]
        events = sum(weights[i] for i in stop if event[i])
        exits = sum(weights[i] for i in stop if competing[i])
        censored = sum(weights[i] for i in stop if not event[i] and not competing[i])
        hazard = events / at_risk if at_risk else np.nan
        incidence += before * hazard
        before *= 1 - (events + exits) / at_risk if at_risk else np.nan
        for name, value in zip(table, [at_risk, events, exits, censored, hazard], strict=True):
            table[name].append(value)
        survival.append(before)
        cumulative.append(incidence if any(competing) else 1 - before)
    table["survival"], table["cumulative_pd"] = survival, cumulative
    return survival_frame(table, periods)


def test_survival_table_hand_example_with_competing_exits():
    table = survival_table(DURATION, EVENT, competing=REPAID)
    expected = survival_frame(
        {
            "at_risk": [8, 6, 3],
            "event": [1, 1, 1],
            "competing": [1, 1, 0],
            "censored": [0, 1, 2],
            "hazard": [1 / 8, 1 / 6, 1 / 3],
            "survival": [6 / 8, 6 / 8 * 4 / 6, 6 / 8 * 4 / 6 * 2 / 3],
            # Aalen-Johansen: 1/8, then 1/8 + 3/4 * 1/6 = 1/4, then 1/4 + 1/2 * 1/3 = 5/12.
            "cumulative_pd": [1 / 8, 1 / 4, 5 / 12],
        },
        3,
    )
    pd.testing.assert_frame_equal(table, expected)


def test_survival_table_kaplan_meier_counts_early_exits_as_censored():
    table = survival_table(DURATION, EVENT)
    km = np.cumprod([7 / 8, 5 / 6, 2 / 3])
    expected = survival_frame(
        {
            "at_risk": [8, 6, 3],
            "event": [1, 1, 1],
            "censored": [1, 2, 2],
            "hazard": [1 / 8, 1 / 6, 1 / 3],
            "survival": km,
            "cumulative_pd": 1 - km,
        },
        3,
    )
    pd.testing.assert_frame_equal(table, expected)
    # Counting the early repayments as censored overstates the cumulative PD.
    competing = survival_table(DURATION, EVENT, competing=REPAID)["cumulative_pd"]
    assert table["cumulative_pd"].iloc[-1] > competing.iloc[-1]
    assert (table["cumulative_pd"] >= competing - 1e-15).all()


def test_survival_table_left_truncation_by_hand():
    # Loans c and d come under observation after period 1, and e and f after period 2.
    entry = [0, 0, 1, 1, 2, 2]
    duration = [2, 2, 2, 3, 4, 4]
    event = [1, 0, 0, 1, 0, 0]
    competing = [0, 1, 0, 0, 1, 0]
    table = survival_table(duration, event, competing=competing, entry=entry)
    expected = survival_frame(
        {
            "at_risk": [2, 4, 3, 2],
            "event": [0, 1, 1, 0],
            "competing": [0, 1, 0, 1],
            "censored": [0, 1, 0, 1],
            "hazard": [0.0, 1 / 4, 1 / 3, 0.0],
            "survival": [1.0, 1 / 2, 1 / 6 * 2, 1 / 6],
            # 0, then 1 * 1/4, then 1/4 + 1/2 * 1/3 = 5/12, and no default in period 4.
            "cumulative_pd": [0.0, 1 / 4, 5 / 12, 5 / 12],
        },
        4,
    )
    pd.testing.assert_frame_equal(table, expected)
    kaplan_meier = survival_table(duration, event, entry=entry)
    np.testing.assert_allclose(kaplan_meier["cumulative_pd"], [0.0, 1 / 4, 1 / 2, 1 / 2])
    assert kaplan_meier["at_risk"].tolist() == [2, 4, 3, 2]


def test_survival_table_censored_accounts_are_at_risk_in_their_period():
    # One default and two censorings tie in period 2: all three are at risk in it.
    table = survival_table([2, 2, 2, 3], [1, 0, 0, 0])
    assert table["at_risk"].tolist() == [4, 4, 1]
    assert table["censored"].tolist() == [0, 2, 1]
    np.testing.assert_allclose(table["hazard"], [0.0, 0.25, 0.0])
    np.testing.assert_allclose(table["cumulative_pd"], [0.0, 0.25, 0.25])


def test_survival_table_ties_of_defaults_and_exits():
    table = survival_table([1, 1, 1, 1, 2], [1, 1, 0, 0, 1], competing=[0, 0, 1, 0, 0])
    np.testing.assert_allclose(table["hazard"], [2 / 5, 1.0])
    np.testing.assert_allclose(table["survival"], [2 / 5, 0.0])
    np.testing.assert_allclose(table["cumulative_pd"], [2 / 5, 2 / 5 + 2 / 5])


def test_survival_table_with_every_account_censored():
    table = survival_table([3, 1, 2], [False, False, False], competing=[False, True, False])
    assert table["event"].tolist() == [0, 0, 0]
    assert table["hazard"].tolist() == [0.0, 0.0, 0.0]
    assert table["cumulative_pd"].tolist() == [0.0, 0.0, 0.0]
    np.testing.assert_allclose(table["survival"], [2 / 3, 2 / 3, 2 / 3])
    table = survival_table([3, 1, 2], [0, 0, 0])
    assert table["survival"].tolist() == [1.0, 1.0, 1.0]
    assert table["censored"].tolist() == [1, 1, 1]


def test_survival_table_when_every_account_defaults():
    table = survival_table([1, 2, 2], [1, 1, 1])
    np.testing.assert_allclose(table["hazard"], [1 / 3, 1.0])
    assert table["survival"].tolist()[-1] == 0.0
    assert table["cumulative_pd"].tolist()[-1] == 1.0
    competing = survival_table([1, 2, 2], [1, 1, 1], competing=[0, 0, 0])
    assert competing["cumulative_pd"].tolist()[-1] == 1.0


def test_survival_table_is_missing_from_an_empty_risk_set_on():
    # Loan b comes under observation after period 3, when loan a is no longer observed.
    table = survival_table([2, 5], [0, 1], entry=[0, 3])
    assert table["at_risk"].tolist() == [1, 1, 0, 1, 1]
    assert table["event"].tolist() == [0, 0, 0, 0, 1]
    np.testing.assert_array_equal(table["hazard"], [0.0, 0.0, np.nan, 0.0, 1.0])
    np.testing.assert_array_equal(table["survival"], [1.0, 1.0, np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(table["cumulative_pd"], [0.0, 0.0, np.nan, np.nan, np.nan])
    # The cumulative PDs still convert, and the hazards after the gap give the conditional PDs.
    conditional = convert_pd(table["cumulative_pd"], kind="cumulative", to="conditional")
    assert conditional.iloc[:2].tolist() == [0.0, 0.0] and conditional.iloc[2:].isna().all()
    later = convert_pd(table["hazard"].loc[4:], kind="conditional", to="cumulative")
    assert later.tolist() == [0.0, 1.0]


def test_survival_table_without_accounts_in_the_first_periods():
    table = survival_table([4, 3], [1, 0], entry=[2, 1])
    assert table["at_risk"].tolist() == [0, 1, 2, 1]
    np.testing.assert_array_equal(table["hazard"], [np.nan, 0.0, 0.0, 1.0])
    assert table["survival"].isna().all() and table["cumulative_pd"].isna().all()


def test_survival_table_competing_without_exits_equals_kaplan_meier():
    rng = np.random.default_rng(3)
    duration = rng.integers(1, 13, 300)
    event = rng.random(300) < 0.2
    kaplan_meier = survival_table(duration, event)
    incidence = survival_table(duration, event, competing=np.zeros(300, dtype=bool))
    assert incidence["competing"].sum() == 0
    pd.testing.assert_frame_equal(
        incidence.drop(columns="competing"), kaplan_meier, check_exact=False, rtol=1e-13
    )


@pytest.mark.parametrize("seed", range(6))
def test_survival_table_matches_the_formulas_account_by_account(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 80))
    duration = rng.integers(1, 30, n)
    entry = np.floor(rng.random(n) * duration).astype(int) * (rng.random(n) < 0.5)
    event = rng.random(n) < 0.2
    competing = ~event & (rng.random(n) < 0.3)
    weights = rng.random(n) * (rng.random(n) < 0.9)
    weights[0] = 1.0
    options = [{}, {"entry": entry}, {"weights": weights}, {"entry": entry, "weights": weights}]
    for kwargs in options:
        for exits in [None, competing]:
            table = survival_table(duration, event, competing=exits, **kwargs)
            expected = brute_force_survival(
                duration.tolist(),
                event.tolist(),
                None if exits is None else exits.tolist(),
                kwargs.get("entry", entry * 0).tolist(),
                kwargs.get("weights", np.ones(n)).tolist(),
            )
            if exits is None:
                expected = expected.drop(columns="competing")
            pd.testing.assert_frame_equal(
                table, expected, check_dtype=False, check_exact=False, rtol=1e-12, atol=0
            )


def test_survival_table_integer_weights_equal_repeated_accounts():
    rng = np.random.default_rng(4)
    duration = rng.integers(1, 10, 60)
    entry = np.floor(rng.random(60) * duration).astype(int)
    event = rng.random(60) < 0.25
    competing = ~event & (rng.random(60) < 0.3)
    weights = rng.integers(0, 4, 60)
    weighted = survival_table(duration, event, competing=competing, entry=entry, weights=weights)
    repeated = survival_table(
        *(np.repeat(values, weights) for values in (duration, event)),
        competing=np.repeat(competing, weights),
        entry=np.repeat(entry, weights),
    )
    pd.testing.assert_frame_equal(weighted, repeated, check_dtype=False)
    assert weighted["at_risk"].dtype == np.float64
    assert repeated["at_risk"].dtype == np.int64


def test_survival_table_leaves_out_accounts_with_weight_zero():
    table = survival_table([1, 2, 5], [1, 0, 1], weights=[1.0, 2.0, 0.0])
    pd.testing.assert_frame_equal(table, survival_table([1, 2, 2], [1, 0, 0]), check_dtype=False)
    assert table.index.tolist() == [1, 2]


def test_survival_table_with_fractional_weights_by_hand():
    table = survival_table([1, 2, 2], [1, 1, 0], competing=[0, 0, 1], weights=[0.5, 1.5, 2.0])
    np.testing.assert_allclose(table["at_risk"], [4.0, 3.5])
    np.testing.assert_allclose(table["hazard"], [0.125, 1.5 / 3.5])
    np.testing.assert_allclose(table["survival"], [0.875, 0.0])
    np.testing.assert_allclose(table["cumulative_pd"], [0.125, 0.125 + 0.875 * 1.5 / 3.5])


def test_survival_table_keeps_small_weights_next_to_a_large_one():
    # A difference of cumulative sums would give 1e20 + 1 - 1e20 = 0 accounts at risk in
    # period 3, where the account of weight 1 is at risk and the large one not yet.
    table = survival_table([5, 12], [1, 0], entry=[0, 10], weights=[1.0, 1e20])
    assert table["at_risk"].tolist() == [1.0] * 5 + [0.0] * 5 + [1e20] * 2
    assert table.loc[5, "hazard"] == 1.0


def test_survival_table_cumulative_pd_converts_to_marginal_and_conditional_pds():
    rng = np.random.default_rng(8)
    duration = rng.integers(1, 25, 500)
    event = rng.random(500) < 0.1
    competing = ~event & (rng.random(500) < 0.4)
    for exits in [None, competing]:
        table = survival_table(duration, event, competing=exits)
        marginal = convert_pd(table["cumulative_pd"], kind="cumulative", to="marginal")
        assert marginal.index.equals(table.index) and (marginal >= 0).all()
        np.testing.assert_allclose(marginal.sum(), table["cumulative_pd"].iloc[-1])
    # Without exits, the conditional PDs are the hazards.
    table = survival_table(duration, event)
    conditional = convert_pd(table["cumulative_pd"], kind="cumulative", to="conditional")
    np.testing.assert_allclose(conditional, table["hazard"], rtol=1e-12)


def test_survival_table_of_full_cohorts_equals_the_vintage_table():
    panel = make_loan_panel(n_loans=2000, n_months=18, prepayment=0.03, rng=2)
    bad = panel["status"] == "90+"
    last = panel.groupby("loan_id").tail(1).set_index("loan_id")
    first_bad = panel[bad].groupby("loan_id")["month_on_book"].min()
    defaulted = last.index.isin(first_bad.index)
    duration = first_bad.reindex(last.index).fillna(last["month_on_book"]).astype(int)
    # Early and scheduled repayments both end the risk of default.
    repaid = ~defaulted & (last["balance"] == 0).to_numpy()
    vintage = vintage_table(panel["loan_id"], panel["cohort"], panel["month_on_book"], bad)
    for cohort in vintage.index:
        rows = (last["cohort"] == cohort).to_numpy()
        table = survival_table(duration[rows], defaulted[rows], competing=repaid[rows])
        np.testing.assert_allclose(
            table["cumulative_pd"], vintage.loc[cohort, table.index], rtol=1e-12, atol=1e-15
        )
    # The early repayments are there; counted as censored, they overstate the PD.
    assert (repaid & (last["month_on_book"] < last["term"]).to_numpy()).sum() > 100
    pooled = survival_table(duration, defaulted, competing=repaid)
    censored = survival_table(duration, defaulted)
    assert (censored["cumulative_pd"] >= pooled["cumulative_pd"] - 1e-15).all()
    assert censored["cumulative_pd"].iloc[-1] > 1.2 * pooled["cumulative_pd"].iloc[-1]


def test_survival_table_accepts_series_booleans_and_whole_floats():
    index = list("abcdefgh")
    table = survival_table(
        pd.Series(np.array(DURATION, dtype=float), index=index),
        pd.Series(np.array(EVENT, dtype=bool), index=index),
        competing=pd.Series(REPAID, index=index, dtype="Int64"),
        entry=pd.Series(0, index=index),
        weights=pd.Series(1, index=index),
    )
    pd.testing.assert_frame_equal(
        table, survival_table(DURATION, EVENT, competing=REPAID), check_dtype=False
    )


def test_survival_table_does_not_mutate_its_inputs():
    duration = pd.Series([3, 1, 2, 2])
    event = np.array([1, 0, 1, 0])
    competing = [0, 1, 0, 0]
    entry = np.array([0, 0, 1, 0])
    weights = pd.Series([1.0, 2.0, 0.0, 1.0])
    copies = [values.copy() for values in (duration, event, competing, entry, weights)]
    survival_table(duration, event, competing=competing, entry=entry, weights=weights)
    pd.testing.assert_series_equal(duration, copies[0])
    np.testing.assert_array_equal(event, copies[1])
    assert competing == copies[2]
    np.testing.assert_array_equal(entry, copies[3])
    pd.testing.assert_series_equal(weights, copies[4])


def test_survival_table_options_are_keyword_only():
    with pytest.raises(TypeError, match="positional argument"):
        survival_table([1, 2], [1, 0], [0, 1])


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        (([], []), {}, "duration must not be empty"),
        (([[1, 2]], [0]), {}, "duration must be one-dimensional, got 2 dimensions"),
        (([1, 2], [1]), {}, "duration and event must have the same length, got 2 and 1"),
        (([0, 1], [0, 1]), {}, "duration must hold positive whole numbers of periods, got 0$"),
        (([1.5], [0]), {}, "duration must hold positive whole numbers of periods, got 1.5"),
        (([np.nan], [0]), {}, "duration must hold positive whole numbers of periods, got nan"),
        (([np.inf], [0]), {}, "duration must hold positive whole numbers of periods, got inf"),
        (([True], [0]), {}, "duration must be real and numeric, got dtype bool"),
        ((["1"], [0]), {}, "duration must be real and numeric, got dtype .* the value '1'"),
        (([1], [2]), {}, "event must contain only booleans or 0 and 1, got 2"),
        (([1], [np.nan]), {}, "event must contain only booleans or 0 and 1, got nan"),
        (([1, 2], [1, 1]), {"competing": [0, 1]}, "event and competing must not both be 1"),
        (([1, 2], [0, 0]), {"competing": [0]}, "duration and competing must have the same len"),
        (([1], [0]), {"competing": ["yes"]}, "competing must contain only booleans or 0 and 1"),
        (([2], [0]), {"entry": [2]}, "entry must be smaller than duration, .* entry of 2 with a"),
        (([2], [0]), {"entry": [-1]}, "entry must hold non-negative whole numbers .*, got -1"),
        (([2], [0]), {"entry": [0.5]}, "entry must hold non-negative whole numbers .*, got 0.5"),
        (([2], [0]), {"entry": [np.nan]}, "entry must hold non-negative whole numbers .* got nan"),
        (([2, 3], [0, 0]), {"entry": [0]}, "duration and entry must have the same length"),
        (([2], [0]), {"weights": [-1.0]}, "weights must be finite and non-negative"),
        (([2], [0]), {"weights": [0.0]}, "weights must have a finite, positive total"),
        (([2], [0]), {"weights": [1.0, 1.0]}, "weights must have one weight per value of duration"),
        (([2], [0]), {"weights": [True]}, "weights must be numeric, got dtype bool"),
        (
            (pd.Series([1, 2]), pd.Series([0, 1], index=[3, 4])),
            {},
            "duration and event must have identical indices",
        ),
        (
            (pd.Series([1, 2]), [0, 1]),
            {"entry": pd.Series([0, 0], index=[5, 6])},
            "duration and entry must have identical indices",
        ),
    ],
)
def test_survival_table_rejects_invalid_inputs(args, kwargs, message):
    with pytest.raises(ValueError, match=message):
        survival_table(*args, **kwargs)


# expected_credit_loss


def test_expected_credit_loss_hand_example():
    lifetime = expected_credit_loss(ECL_MARGINAL, 0.5, ECL_SCHEDULE, rate=0.05)
    first = 0.02 * 0.5 * 1000 / 1.05 + 0.03 * 0.5 * 700 / 1.05**2 + 0.03 * 0.5 * 400 / 1.05**3
    second = 0.10 * 0.5 * 500 / 1.05 + 0.08 * 0.5 * 250 / 1.05**2
    np.testing.assert_allclose(lifetime, [first, second], rtol=1e-12)
    twelve_month = expected_credit_loss(ECL_MARGINAL, 0.5, ECL_SCHEDULE, rate=0.05, horizon=1)
    np.testing.assert_allclose(twelve_month, [10 / 1.05, 25 / 1.05], rtol=1e-12)
    two = expected_credit_loss(ECL_MARGINAL, 0.5, ECL_SCHEDULE, rate=0.05, horizon=2)
    np.testing.assert_allclose(two, [10 / 1.05 + 10.5 / 1.05**2, 25 / 1.05 + 10 / 1.05**2])


def test_expected_credit_loss_without_discounting_is_a_sum_of_expected_losses():
    rng = np.random.default_rng(4)
    marginal = convert_pd(random_curves(4, (8, 5)), kind="cumulative", to="marginal")
    lgd = rng.uniform(0.2, 0.8, size=(8, 5))
    ead = rng.uniform(0, 1000, size=(8, 5))
    result = expected_credit_loss(marginal, lgd, ead, rate=0.0)
    np.testing.assert_allclose(result, expected_loss(marginal, lgd, ead).sum(axis=1))
    # One period without discounting is the expected loss of kivi.strategy.
    np.testing.assert_allclose(
        expected_credit_loss(marginal[:, :1], lgd[:, 0], ead[:, 0], rate=0),
        expected_loss(marginal[:, 0], lgd[:, 0], ead[:, 0]),
    )


def test_expected_credit_loss_broadcasts_constants_accounts_and_periods():
    marginal = np.array(ECL_MARGINAL)
    lgd = np.array([0.4, 0.6])
    ead = np.array([1000.0, 500.0])
    full = expected_credit_loss(
        marginal, np.repeat(lgd[:, None], 3, axis=1), np.repeat(ead[:, None], 3, axis=1), rate=0.05
    )
    np.testing.assert_allclose(expected_credit_loss(marginal, lgd, ead, rate=0.05), full)
    np.testing.assert_allclose(
        expected_credit_loss(marginal, lgd[:, None], ead[:, None], rate=[0.05, 0.05]), full
    )
    # One row for all accounts, and one value for all accounts and periods.
    schedule = np.array([[1000.0, 600.0, 200.0]])
    np.testing.assert_allclose(
        expected_credit_loss(marginal, 0.5, schedule, rate=0.0),
        (marginal * 0.5 * schedule).sum(axis=1),
    )
    np.testing.assert_allclose(
        expected_credit_loss(marginal, [0.5], 100.0, rate=0.0), marginal.sum(axis=1) * 50
    )


def test_expected_credit_loss_rate_per_account_and_period_length():
    marginal = [[0.0] * 11 + [0.05], [0.0] * 11 + [0.05]]
    annual = 0.08
    monthly = (1 + annual) ** (1 / 12) - 1
    result = expected_credit_loss(marginal, 1.0, 100.0, rate=[monthly, 0.0])
    np.testing.assert_allclose(result, [5 / 1.08, 5.0], rtol=1e-12)


def test_expected_credit_loss_twelve_months_equals_lifetime_after_maturity():
    marginal = convert_pd(random_curves(5, (4, 24)), kind="cumulative", to="marginal")
    schedule = np.tile(np.linspace(1000, 100, 24), (4, 1))
    schedule[:, 12:] = 0.0
    np.testing.assert_allclose(
        expected_credit_loss(marginal, 0.45, schedule, rate=0.01, horizon=12),
        expected_credit_loss(marginal, 0.45, schedule, rate=0.01),
    )
    # With exposure after maturity, the lifetime loss is larger.
    lifetime = expected_credit_loss(marginal, 0.45, 1000.0, rate=0.01)
    twelve_month = expected_credit_loss(marginal, 0.45, 1000.0, rate=0.01, horizon=12)
    assert (lifetime > twelve_month).all()


def test_expected_credit_loss_is_missing_where_an_added_period_is_missing():
    marginal = np.array(ECL_MARGINAL)
    expected = expected_credit_loss(marginal, 0.5, ECL_SCHEDULE, rate=0.05, horizon=2)
    # Account 1 misses its PD in period 3 and account 2 its EAD in period 2.
    gaps = marginal.copy()
    gaps[0, 2] = np.nan
    schedule = np.array(ECL_SCHEDULE)
    schedule[1, 1] = np.nan
    lifetime = expected_credit_loss(gaps, 0.5, schedule, rate=0.05)
    assert np.isnan(lifetime).all()
    # Periods after the horizon are not used: account 1 is complete in periods 1 and 2.
    two = expected_credit_loss(gaps, 0.5, schedule, rate=0.05, horizon=2)
    np.testing.assert_allclose(two[0], expected[0])
    assert np.isnan(two[1])
    np.testing.assert_allclose(
        expected_credit_loss(gaps, 0.5, schedule, rate=0.05, horizon=1),
        expected_credit_loss(marginal, 0.5, ECL_SCHEDULE, rate=0.05, horizon=1),
    )
    # A missing LGD or rate makes the account's loss missing.
    result = expected_credit_loss(marginal, [0.5, np.nan], ECL_SCHEDULE, rate=[np.nan, 0.05])
    assert np.isnan(result).all()


def test_expected_credit_loss_returns_a_series_for_pandas_inputs():
    index = pd.Index(["x", "y"], name="account")
    frame = pd.DataFrame(ECL_MARGINAL, index=index, columns=[1, 2, 3])
    result = expected_credit_loss(
        frame, 0.5, pd.DataFrame(ECL_SCHEDULE, index, [1, 2, 3]), rate=0.05
    )
    expected = expected_credit_loss(ECL_MARGINAL, 0.5, ECL_SCHEDULE, rate=0.05)
    pd.testing.assert_series_equal(
        result, pd.Series(expected, index=index, name="expected_credit_loss")
    )
    result = expected_credit_loss(ECL_MARGINAL, pd.Series([0.5, 0.5], index=index), 100.0, rate=0)
    assert result.index.equals(index) and result.name == "expected_credit_loss"


def test_expected_credit_loss_broadcasts_pandas_inputs_with_one_row_or_column():
    expected = expected_credit_loss(ECL_MARGINAL, 0.5, ECL_SCHEDULE, rate=0.05)
    # One row for all accounts: the labels are not compared, and the result is an array.
    for lgd in (pd.Series([0.5]), pd.DataFrame([[0.5, 0.5, 0.5]]), pd.DataFrame([[0.5]])):
        result = expected_credit_loss(ECL_MARGINAL, lgd, ECL_SCHEDULE, rate=pd.Series([0.05]))
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, expected)
    # One column for all periods keeps the index of the accounts.
    index = pd.Index(["x", "y"], name="account")
    frame = pd.DataFrame(ECL_MARGINAL, index=index, columns=[1, 2, 3])
    schedule = pd.DataFrame(ECL_SCHEDULE, index=index, columns=[1, 2, 3])
    one_row = pd.DataFrame([[0.5] * 3], columns=[1, 2, 3])
    for lgd in (pd.DataFrame({"lgd": [0.5, 0.5]}, index=index), one_row):
        result = expected_credit_loss(frame, lgd, schedule, rate=0.05)
        pd.testing.assert_series_equal(
            result, pd.Series(expected, index=index, name="expected_credit_loss")
        )
    with pytest.raises(ValueError, match="probability and lgd must have identical indices"):
        expected_credit_loss(frame, pd.DataFrame({"lgd": [0.5, 0.5]}), schedule, rate=0.05)
    # A row for all accounts still has one column per period, so its columns must match.
    with pytest.raises(ValueError, match="probability and lgd must have identical columns"):
        expected_credit_loss(frame, pd.DataFrame([[0.5] * 3]), schedule, rate=0.05)


def test_expected_credit_loss_with_one_account_or_one_period():
    # With one account, a single row is broadcast: only the labels of probability are used.
    frame = pd.DataFrame([[0.1, 0.2]], index=["x"], columns=[1, 2])
    result = expected_credit_loss(frame, pd.DataFrame([[0.5]]), 100.0, rate=0.0)
    pd.testing.assert_series_equal(
        result, pd.Series([15.0], index=["x"], name="expected_credit_loss")
    )
    result = expected_credit_loss(frame, pd.Series([0.5], index=["y"]), 100.0, rate=0.0)
    assert result.index.tolist() == ["x"]
    # With one period, a column per account is broadcast over the periods.
    index = pd.Index(["x", "y"])
    one_period = pd.DataFrame({"pd": [0.1, 0.2]}, index=index)
    lgd = pd.DataFrame({"lgd": [0.5, 0.4]}, index=index)
    result = expected_credit_loss(one_period, lgd, 100.0, rate=0.0)
    np.testing.assert_allclose(result, [5.0, 8.0])
    with pytest.raises(ValueError, match="probability and lgd must have identical indices"):
        expected_credit_loss(one_period, lgd.set_axis(["y", "x"]), 100.0, rate=0.0)


def test_expected_credit_loss_discounts_only_the_periods_it_adds_up():
    marginal = np.full((1, 400), 0.001)
    with pytest.raises(
        ValueError, match="rate must keep the discount factors finite over 400 periods, got -0.9"
    ):
        expected_credit_loss(marginal, 0.5, 100.0, rate=-0.9)
    # Over one period the factor is 10, and nothing overflows.
    np.testing.assert_allclose(
        expected_credit_loss(marginal, 0.5, 100.0, rate=-0.9, horizon=1), [0.5], rtol=1e-12
    )


def test_expected_credit_loss_accepts_boolean_probabilities_only():
    result = expected_credit_loss([[True, False]], 0.5, 100.0, rate=0.0)
    np.testing.assert_allclose(result, [50.0])
    with pytest.raises(ValueError, match="lgd must be real and numeric, got dtype bool"):
        expected_credit_loss([[0.1, 0.1]], True, 100.0, rate=0.0)
    with pytest.raises(ValueError, match="ead must be real and numeric, got dtype bool"):
        expected_credit_loss([[0.1, 0.1]], 0.5, [[True, False]], rate=0.0)
    with pytest.raises(ValueError, match="rate must be real and numeric, got dtype bool"):
        expected_credit_loss([[0.1, 0.1]], 0.5, 100.0, rate=False)


def test_expected_credit_loss_does_not_mutate_its_inputs():
    frame = pd.DataFrame(ECL_MARGINAL)
    schedule = np.array(ECL_SCHEDULE)
    originals = frame.copy(), schedule.copy()
    expected_credit_loss(frame, 0.5, schedule, rate=0.05, horizon=2)
    pd.testing.assert_frame_equal(frame, originals[0])
    np.testing.assert_array_equal(schedule, originals[1])


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        (([0.1, 0.2], 0.5, 100.0), {}, "probability must be two-dimensional, .* got 1 dimensions"),
        ((np.zeros((0, 3)), 0.5, 100.0), {}, "probability must have at least one account"),
        (
            ([[0.6, 0.5]], 0.5, 100.0),
            {},
            "probability must add up to at most 1 in each row, got 1.1",
        ),
        (([[0.1, 1.5]], 0.5, 100.0), {}, "probability must be between 0 and 1, got 1.5"),
        (([[0.1, 0.1]], 1.2, 100.0), {}, "lgd must be between 0 and 1, got 1.2"),
        (([[0.1, 0.1]], "0.5", 100.0), {}, "lgd must be real and numeric, got dtype"),
        (
            ([[0.1, 0.1]], [0.5, 0.5], 100.0),
            {},
            r"lgd must be one value, one value per account or one per account and period, got "
            r"shape \(2,\) for 1 accounts and 2 periods",
        ),
        (
            ([[0.1, 0.1]], 0.5, [[100.0, 100.0, 100.0]]),
            {},
            r"ead must be one value, .* got shape \(1, 3\) for 1 accounts and 2 periods",
        ),
        (([[0.1, 0.1]], 0.5, np.zeros((1, 1, 2))), {}, r"ead must be one value, .* got shape"),
        (([[0.1, 0.1]], 0.5, -1.0), {}, "ead must be finite and non-negative, got -1.0"),
        (([[0.1, 0.1]], 0.5, [[100.0, np.inf]]), {}, "ead must be finite and non-negative, got"),
        (([[0.1, 0.1]], 0.5, 100.0), {"rate": -1.0}, "rate must be finite and greater than -1"),
        (([[0.1, 0.1]], 0.5, 100.0), {"rate": np.inf}, "rate must be finite and greater than -1"),
        (
            ([[0.1, 0.1]], 0.5, 100.0),
            {"rate": [[0.01]]},
            "rate must be one value or one value per account, got 2 dimensions",
        ),
        (
            ([[0.1, 0.1]], 0.5, 100.0),
            {"rate": [0.01, 0.02]},
            "rate must be one value or one value per account, got 2 values for 1 accounts",
        ),
        (([[0.1, 0.1]], 0.5, 100.0), {"horizon": 0}, "horizon must be a positive integer, got 0"),
        (([[0.1, 0.1]], 0.5, 100.0), {"horizon": True}, "horizon must be a positive integer"),
        (([[0.1, 0.1]], 0.5, 100.0), {"horizon": 1.0}, "horizon must be a positive integer"),
        (
            ([[0.1, 0.1]], 0.5, 100.0),
            {"horizon": 12},
            "horizon must be at most the number of periods of probability, 2, got 12",
        ),
    ],
)
def test_expected_credit_loss_rejects_invalid_inputs(args, kwargs, message):
    kwargs = {"rate": 0.01, **kwargs}
    with pytest.raises(ValueError, match=message):
        expected_credit_loss(*args, **kwargs)


def test_expected_credit_loss_rejects_misaligned_pandas_inputs():
    frame = pd.DataFrame(ECL_MARGINAL, index=["x", "y"])
    with pytest.raises(ValueError, match="probability and ead must have identical indices"):
        expected_credit_loss(frame, 0.5, pd.Series([1.0, 2.0], index=["y", "x"]), rate=0.0)
    with pytest.raises(ValueError, match="probability and rate must have identical indices"):
        expected_credit_loss(frame, 0.5, 1.0, rate=pd.Series([0.0, 0.0]))
    schedule = pd.DataFrame(ECL_SCHEDULE, index=["x", "y"], columns=[1, 2, 3])
    with pytest.raises(ValueError, match="probability and ead must have identical columns"):
        expected_credit_loss(frame, 0.5, schedule, rate=0.0)


# significant_increase


def test_significant_increase_hand_example():
    probability = [0.004, 0.012, 0.03, 0.09]
    origination = [0.001, 0.010, 0.01, 0.05]
    relative = significant_increase(probability, origination, relative=2.0, absolute=None)
    np.testing.assert_array_equal(relative, [True, False, True, False])
    absolute = significant_increase(probability, origination, relative=None, absolute=0.01)
    np.testing.assert_array_equal(absolute, [False, False, True, True])
    both = significant_increase(probability, origination, relative=2.0, absolute=0.01)
    np.testing.assert_array_equal(both, relative & absolute)
    assert both.dtype == bool


def test_significant_increase_thresholds_are_strict():
    # Exactly doubled, and up by exactly 0.25.
    flags = significant_increase([0.2, 0.5], [0.1, 0.25], relative=2.0, absolute=None)
    np.testing.assert_array_equal(flags, [False, False])
    flags = significant_increase([0.5, 0.5], [0.25, 0.2], relative=None, absolute=0.25)
    np.testing.assert_array_equal(flags, [False, True])
    # A PD of 0 at origination: any positive PD is an infinite ratio.
    flags = significant_increase([0.0, 0.001], [0.0, 0.0], relative=3.0, absolute=None)
    np.testing.assert_array_equal(flags, [False, True])


def test_significant_increase_does_not_let_rounding_decide_ties():
    # In floating point 0.04 - 0.03 > 0.01 and 0.45 > 1.5 * 0.30; all three used to be flagged.
    flags = significant_increase(
        [0.04, 0.07, 0.45], [0.03, 0.06, 0.30], relative=None, absolute=0.01
    )
    np.testing.assert_array_equal(flags, [False, False, True])
    flags = significant_increase([0.45, 0.6], [0.30, 0.2], relative=1.5, absolute=None)
    np.testing.assert_array_equal(flags, [False, True])
    # Every pair of two-decimal PDs 0.01 apart is a tie; 71 of the 89 used to be flagged.
    origination = np.arange(1, 90) / 100
    probability = np.arange(2, 91) / 100
    assert not significant_increase(probability, origination, relative=None, absolute=0.01).any()
    assert significant_increase(probability + 1e-9, origination, relative=None, absolute=0.01).all()
    # So is every PD that is exactly 1.5 or 2 times a three-decimal PD.
    for ratio, last in ((1.5, 666), (2.0, 499)):
        origination = np.arange(1, last + 1) / 1000
        probability = np.arange(1, last + 1) * int(ratio * 2) / 2000
        assert not significant_increase(
            probability, origination, relative=ratio, absolute=None
        ).any()
        assert significant_increase(
            probability * (1 + 1e-12), origination, relative=ratio, absolute=None
        ).all()


def test_significant_increase_returns_a_series_for_series_inputs():
    index = pd.Index([10, 20], name="account")
    probability = pd.Series([0.05, 0.01], index=index, name="pd")
    result = significant_increase(probability, [0.01, 0.01], relative=2.0, absolute=None)
    pd.testing.assert_series_equal(
        result, pd.Series([True, False], index=index, name="significant_increase")
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"relative": None, "absolute": None}, "relative and absolute must not both be None"),
        ({"relative": 1.0, "absolute": None}, "relative must be None or a ratio greater than 1"),
        ({"relative": 0.5, "absolute": None}, r"such as 2 for a PD that doubled, got 0.5"),
        ({"relative": np.inf, "absolute": None}, "relative must be None or a ratio greater than"),
        ({"relative": True, "absolute": None}, "relative must be None or a ratio greater than"),
        ({"relative": "2", "absolute": None}, "relative must be None or a ratio greater than"),
        ({"relative": None, "absolute": 0.0}, "absolute must be None or a number between 0 and"),
        ({"relative": None, "absolute": 1.0}, "absolute must be None or a number between 0 and"),
        ({"relative": None, "absolute": np.nan}, "absolute must be None or a number between 0"),
    ],
)
def test_significant_increase_rejects_invalid_thresholds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        significant_increase([0.1], [0.05], **kwargs)


def test_significant_increase_messages_show_numpy_scalars_as_numbers():
    with pytest.raises(ValueError, match=r"doubled, got 0\.5$"):
        significant_increase([0.1], [0.05], relative=np.float64(0.5), absolute=None)
    with pytest.raises(ValueError, match=r"got 90 and 90$"):
        assign_stages([0], stage_2_days=np.int64(90), stage_3_days=np.int64(90))


@pytest.mark.parametrize(
    ("probability", "origination", "message"),
    [
        ([], [], "probability must not be empty"),
        ([0.1, 0.2], [0.1], "probability and origination_probability must have the same length"),
        ([0.1, np.nan], [0.1, 0.1], "probability must not have missing values, got 1"),
        ([0.1, 0.1], [0.1, 1.1], "origination_probability must be between 0 and 1, got 1.1"),
        (["a"], [0.1], "probability must be real and numeric, got dtype"),
        ([None], [0.1], "probability must not have missing values, got 1"),
        (
            pd.Series([0.1, 0.2]),
            pd.Series([0.1, 0.2], index=[1, 2]),
            "probability and origination_probability must have identical indices",
        ),
    ],
)
def test_significant_increase_rejects_invalid_probabilities(probability, origination, message):
    with pytest.raises(ValueError, match=message):
        significant_increase(probability, origination, relative=2.0, absolute=None)


# assign_stages


def test_assign_stages_hand_example():
    stage = assign_stages(
        [False, True, False, False, False],
        days_past_due=[0, 0, 45, 120, 0],
        defaulted=[False, False, False, False, True],
    )
    np.testing.assert_array_equal(stage, [1, 2, 2, 3, 3])
    assert stage.dtype == np.int64


def test_assign_stages_takes_the_loan_data_by_position():
    stage = assign_stages([False, True, False], [0, 0, 120], [False, False, True])
    np.testing.assert_array_equal(stage, [1, 2, 3])
    np.testing.assert_array_equal(assign_stages([False, False], [0, 45]), [1, 2])
    with pytest.raises(TypeError):
        assign_stages([False], [0], [False], 30)


def test_assign_stages_backstops_follow_ifrs_9():
    # More than 30 days past due is stage 2 (5.5.11); 90 days past due is already stage 3 (B5.5.37).
    days = [0, 30, 31, 89, 90, 91]
    np.testing.assert_array_equal(assign_stages([0] * 6, days_past_due=days), [1, 1, 2, 2, 3, 3])
    np.testing.assert_array_equal(
        assign_stages([0] * 6, days_past_due=days, stage_2_days=29, stage_3_days=91),
        [1, 2, 2, 2, 2, 3],
    )
    # Fractions of a day compare as numbers.
    np.testing.assert_array_equal(
        assign_stages([0] * 3, days_past_due=[30.5, 89.9, 90.0]), [2, 2, 3]
    )


def test_assign_stages_without_days_past_due():
    np.testing.assert_array_equal(assign_stages([0, 1, 0, 1]), [1, 2, 1, 2])
    np.testing.assert_array_equal(assign_stages([0, 1, 0], defaulted=[1, 1, 0]), [3, 3, 1])


def test_assign_stages_returns_a_series_for_series_inputs():
    index = pd.Index(["a", "b", "c"], name="account")
    days = pd.Series([0.0, 35.0, 95.0], index=index)
    stage = assign_stages([False, False, False], days_past_due=days)
    pd.testing.assert_series_equal(stage, pd.Series([1, 2, 3], index=index, name="stage"))


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        (([],), {}, "significant_increase must not be empty"),
        (([0, 2],), {}, "significant_increase must contain only booleans or 0 and 1, got 2"),
        (([0, 1],), {"defaulted": [0, np.nan]}, "defaulted must contain only booleans or 0 and 1"),
        (
            ([0, 1],),
            {"defaulted": [0]},
            "significant_increase and defaulted must have the same length, got 2 and 1",
        ),
        (
            ([0, 1],),
            {"days_past_due": [0]},
            "significant_increase and days_past_due must have the same length",
        ),
        (([0],), {"days_past_due": [-1]}, "days_past_due must be finite and non-negative, got -1"),
        (([0],), {"days_past_due": [np.nan]}, "days_past_due must be finite and non-negative"),
        (([0],), {"days_past_due": [True]}, "days_past_due must be real and numeric, got dtype"),
        (([0],), {"stage_2_days": -1}, "stage_2_days must be a finite, non-negative number"),
        (([0],), {"stage_3_days": np.inf}, "stage_3_days must be a finite, non-negative number"),
        (([0],), {"stage_2_days": True}, "stage_2_days must be a finite, non-negative number"),
        (
            ([0],),
            {"stage_2_days": 90, "stage_3_days": 90},
            "stage_3_days must be greater than stage_2_days, got 90 and 90",
        ),
        (
            (pd.Series([0, 1]),),
            {"days_past_due": pd.Series([0, 1], index=[1, 2])},
            "significant_increase and days_past_due must have identical indices",
        ),
    ],
)
def test_assign_stages_rejects_invalid_inputs(args, kwargs, message):
    with pytest.raises(ValueError, match=message):
        assign_stages(*args, **kwargs)


# stage_table


def test_stage_table_hand_example():
    table = stage_table(
        [1, 1, 1, 2, 3], [1000.0, 2000.0, 1000.0, 800.0, 200.0], [5.0, 8.0, 3.0, 40.0, 90.0]
    )
    expected = pd.DataFrame(
        {
            "total": [3, 1, 1],
            "exposure": [4000.0, 800.0, 200.0],
            "exposure_share": [0.8, 0.16, 0.04],
            "expected_credit_loss": [16.0, 40.0, 90.0],
            "coverage": [0.004, 0.05, 0.45],
        },
        index=pd.Index([1, 2, 3], name="stage"),
    )
    pd.testing.assert_frame_equal(table, expected)


def test_stage_table_keeps_empty_stages():
    table = stage_table(pd.Series([1.0, 1.0]), [100.0, 300.0], [1.0, 2.0])
    assert table.index.tolist() == [1, 2, 3]
    assert table["total"].tolist() == [2, 0, 0]
    assert table.loc[1, "coverage"] == 3 / 400
    assert table.loc[[2, 3], "coverage"].isna().all()
    np.testing.assert_allclose(table["exposure_share"], [1.0, 0.0, 0.0])


def test_stage_table_integer_weights_equal_repeated_rows():
    stage = np.array([1, 2, 1, 3, 2])
    exposure = np.array([100.0, 250.0, 50.0, 80.0, 0.0])
    loss = np.array([1.0, 12.0, 0.5, 40.0, 0.0])
    weights = np.array([2, 1, 3, 0, 4])
    weighted = stage_table(stage, exposure, loss, weights=weights)
    repeated = stage_table(*(np.repeat(values, weights) for values in (stage, exposure, loss)))
    pd.testing.assert_frame_equal(weighted, repeated, check_dtype=False)
    assert weighted.loc[3, "total"] == 0
    assert np.isnan(weighted.loc[3, "coverage"])


def test_stage_table_with_fractional_weights_by_hand():
    stage = [1, 1, 2, 3]
    exposure = [100.0, 300.0, 200.0, 50.0]
    allowance = [1.0, 2.0, 10.0, 20.0]
    table = stage_table(stage, exposure, allowance, weights=[0.5, 1.5, 2.0, 0.25])
    # Stage 1: 0.5 + 1.5 = 2 exposures, 50 + 450 = 500 exposure and 0.5 + 3 = 3.5 allowance.
    np.testing.assert_allclose(table["total"], [2.0, 2.0, 0.25])
    np.testing.assert_allclose(table["exposure"], [500.0, 400.0, 12.5])
    np.testing.assert_allclose(table["exposure_share"], [500 / 912.5, 400 / 912.5, 12.5 / 912.5])
    np.testing.assert_allclose(table["expected_credit_loss"], [3.5, 20.0, 5.0])
    np.testing.assert_allclose(table["coverage"], [0.007, 0.05, 0.4])


def test_stage_table_without_exposure():
    table = stage_table([1, 2], [0.0, 0.0], [0.0, 0.0])
    assert table["exposure_share"].isna().all() and table["coverage"].isna().all()


def test_stage_table_totals_add_up_to_the_portfolio():
    rng = np.random.default_rng(6)
    stage = rng.integers(1, 4, size=200)
    exposure = rng.uniform(0, 1000, size=200)
    loss = exposure * rng.uniform(0, 0.2, size=200)
    table = stage_table(stage, exposure, loss)
    assert table["total"].sum() == 200
    np.testing.assert_allclose(table["exposure"].sum(), exposure.sum())
    np.testing.assert_allclose(table["exposure_share"].sum(), 1.0)
    np.testing.assert_allclose(table["expected_credit_loss"].sum(), loss.sum())


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        (([], [], []), {}, "stage must not be empty"),
        (([1, 4], [1.0, 1.0], [0.0, 0.0]), {}, "stage must contain only 1, 2 and 3, got 4"),
        (([1, np.nan], [1.0, 1.0], [0.0, 0.0]), {}, "stage must contain only 1, 2 and 3, got nan"),
        (([True], [1.0], [0.0]), {}, "stage must be real and numeric, got dtype bool"),
        ((["1"], [1.0], [0.0]), {}, "stage must be real and numeric, got dtype"),
        (([1, 2], [1.0], [0.0, 0.0]), {}, "stage and exposure must have the same length"),
        (([1], [-1.0], [0.0]), {}, "exposure must be finite and non-negative, got -1.0"),
        (([1], [1.0], [np.nan]), {}, "allowance must be finite and non-negative, got nan"),
        (([1], [1.0], [True]), {}, "allowance must be real and numeric, got dtype bool"),
        (([1], [1.0], [0.0]), {"weights": [-1.0]}, "weights must be finite and non-negative"),
        (([1], [1.0], [0.0]), {"weights": [1.0, 1.0]}, "weights must have one weight per value"),
        (
            (pd.Series([1, 2]), pd.Series([1.0, 1.0], index=[5, 6]), [0.0, 0.0]),
            {},
            "stage and exposure must have identical indices",
        ),
    ],
)
def test_stage_table_rejects_invalid_inputs(args, kwargs, message):
    with pytest.raises(ValueError, match=message):
        stage_table(*args, **kwargs)


# Workflow


def test_lifetime_workflow_on_a_loan_panel(panel):
    """From a vintage table to stages and loss allowances for the loans on book."""
    bad = panel["status"] == "90+"
    table = vintage_table(panel["loan_id"], panel["cohort"], panel["month_on_book"], bad)
    hazard = convert_pd(table, kind="cumulative", to="conditional").mean()
    periods = 12
    # Extend the pooled hazard flat beyond the oldest month on book.
    hazard = np.concatenate([hazard.to_numpy(), np.full(periods, hazard.iloc[-6:].mean())])
    last = panel[panel["month"] == panel["month"].max()]
    on_book = last[last["status"] != "90+"]
    age = on_book["month_on_book"].to_numpy()
    rows = hazard[age[:, None] + np.arange(periods)]
    marginal = convert_pd(rows, kind="conditional", to="marginal")
    assert (marginal.sum(axis=1) <= 1).all()
    remaining = (on_book["term"] - on_book["month_on_book"]).to_numpy()
    ead = np.where(np.arange(1, periods + 1) <= remaining[:, None], on_book[["balance"]], 0.0)
    lifetime = expected_credit_loss(marginal, 0.45, ead, rate=0.01)
    six_month = expected_credit_loss(marginal, 0.45, ead, rate=0.01, horizon=6)
    assert (lifetime >= six_month).all()
    days = on_book["status"].cat.codes.to_numpy() * 30.0
    stage = assign_stages(np.zeros(len(on_book), dtype=bool), days_past_due=days)
    allowance = np.where(stage == 1, six_month, lifetime)
    summary = stage_table(stage, on_book["balance"].to_numpy(), allowance)
    assert summary["total"].sum() == len(on_book)
    np.testing.assert_allclose(summary["expected_credit_loss"].sum(), allowance.sum())
