import itertools

import numpy as np
import pandas as pd
import pytest

from kivi.rules import rule_candidates, rule_set_table, rule_table
from kivi.strategy import cutoff_table

# Row 0 is hit by high_debt, row 1 by high_debt and young, row 2 by young and inquiries,
# row 3 by inquiries and row 5 by high_debt; rows 4, 6 and 7 by no rule. Rows 0, 1 and 3
# are events.
RULES = pd.DataFrame(
    {
        "high_debt": [1, 1, 0, 0, 0, 1, 0, 0],
        "young": [0, 1, 1, 0, 0, 0, 0, 0],
        "inquiries": [0, 0, 1, 1, 0, 0, 0, 0],
    }
)
TARGET = [1, 1, 0, 1, 0, 0, 0, 0]
NAMES = pd.Index(["high_debt", "young", "inquiries"], name="rule")

HIT_COLUMNS = ["hits", "hit_rate", "unique_hits"]
OUTCOME_COLUMNS = ["known", "event", "event_rate", "lift", "recall", "unique_event"]
CUMULATIVE_COLUMNS = [
    "approved",
    "approval_rate",
    "approved_event",
    "approved_event_rate",
    "rejected_event_share",
    "rejected_non_event_share",
]


def random_rules(seed=0, n=300, rates=(0.1, 0.25, 0.05, 0.4)):
    """Rules of different hit rates that overlap, a target with unknown outcomes and weights."""
    rng = np.random.default_rng(seed)
    rules = pd.DataFrame(
        {f"rule_{k}": rng.random(n) < rate for k, rate in enumerate(rates)},
        index=pd.RangeIndex(100, 100 + n),
    )
    risk = rules.to_numpy().sum(axis=1)
    target = (rng.random(n) < 0.1 + 0.15 * risk).astype(float)
    target[rng.random(n) < 0.2] = np.nan
    weights = rng.uniform(0.1, 3.0, n)
    return rules, target, weights


def replay(rules, target=None, weights=None):
    """The waterfall of rule_set_table, rule by rule with masks."""
    n = len(rules)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    remaining = np.ones(n, dtype=bool)
    if target is not None:
        labels = pd.Series(target, dtype=float).to_numpy()
        known = ~np.isnan(labels)
        event = known & (labels == 1)
        non_event = known & (labels == 0)

    def share(numerator, denominator):
        return numerator / denominator if denominator > 0 else np.nan

    rows = []
    for name in rules.columns:
        hit = rules[name].to_numpy(dtype=bool)
        new = remaining & hit
        remaining = remaining & ~hit
        row = {"new_hits": w[new].sum()}
        if target is not None:
            row["new_event"] = w[new & event].sum()
        row["approved"] = w[remaining].sum()
        row["approval_rate"] = w[remaining].sum() / w.sum()
        if target is not None:
            row["approved_event"] = w[remaining & event].sum()
            row["approved_event_rate"] = share(
                w[remaining & event].sum(), w[remaining & known].sum()
            )
            row["rejected_event_share"] = share(w[~remaining & event].sum(), w[event].sum())
            row["rejected_non_event_share"] = share(
                w[~remaining & non_event].sum(), w[non_event].sum()
            )
        rows.append(row)
    return pd.DataFrame(rows, index=rules.columns.rename("rule"))


# rule_table


def test_rule_table_matches_hand_calculation():
    table = rule_table(RULES, TARGET)
    expected = pd.DataFrame(
        {
            "hits": [3, 2, 2],
            "hit_rate": [3 / 8, 2 / 8, 2 / 8],
            "unique_hits": [2, 0, 1],
            "known": [3, 2, 2],
            "event": [2, 1, 1],
            "event_rate": [2 / 3, 1 / 2, 1 / 2],
            "lift": [(2 / 3) / (3 / 8), (1 / 2) / (3 / 8), (1 / 2) / (3 / 8)],
            "recall": [2 / 3, 1 / 3, 1 / 3],
            "unique_event": [1, 0, 1],
        },
        index=NAMES,
    )
    pd.testing.assert_frame_equal(table, expected)


def test_rule_table_without_target_only_counts_hits():
    table = rule_table(RULES)
    assert list(table.columns) == HIT_COLUMNS
    assert table["hits"].dtype == np.int64 and table["unique_hits"].dtype == np.int64
    pd.testing.assert_frame_equal(table, rule_table(RULES, TARGET)[HIT_COLUMNS])


def test_rule_table_matches_a_groupby_by_rule():
    rules, target, _ = random_rules()
    complete = np.nan_to_num(target)
    table = rule_table(rules, complete)
    alone = rules.sum(axis=1) == 1
    for name in rules.columns:
        by_hit = pd.Series(complete, index=rules.index).groupby(rules[name]).agg(["size", "sum"])
        assert table.loc[name, "hits"] == by_hit.loc[True, "size"]
        assert table.loc[name, "event"] == by_hit.loc[True, "sum"]
        assert table.loc[name, "event_rate"] == pytest.approx(
            by_hit.loc[True, "sum"] / by_hit.loc[True, "size"], rel=1e-15, abs=0
        )
        unique = pd.Series(complete, index=rules.index).groupby(rules[name] & alone).agg("sum")
        assert table.loc[name, "unique_hits"] == (rules[name] & alone).sum()
        assert table.loc[name, "unique_event"] == unique.loc[True]
    assert table["known"].equals(table["hits"])
    np.testing.assert_allclose(table["recall"], table["event"] / complete.sum(), rtol=1e-15)
    np.testing.assert_allclose(
        table["lift"], table["event_rate"] / complete.mean(), rtol=1e-14, atol=0
    )


def test_rule_table_hits_are_the_diagonal_of_the_overlap_matrix():
    rules, _, _ = random_rules()
    counts = rules.astype(int)
    overlap = counts.T @ counts
    table = rule_table(rules)
    assert np.diag(overlap).tolist() == table["hits"].tolist()
    # A rule's hits that another rule shares, by inclusion-exclusion over the overlaps of each
    # pair: with only two rules, the unique hits are the hits minus their overlap.
    pair = rules[["rule_0", "rule_1"]]
    pair_overlap = (pair.astype(int).T @ pair.astype(int)).to_numpy()
    assert rule_table(pair)["unique_hits"].tolist() == [
        pair_overlap[0, 0] - pair_overlap[0, 1],
        pair_overlap[1, 1] - pair_overlap[0, 1],
    ]


def test_rule_table_unique_hits_are_rows_that_no_other_rule_hits():
    rules, _, _ = random_rules()
    table = rule_table(rules)
    for name in rules.columns:
        others = rules.drop(columns=name).any(axis=1)
        assert table.loc[name, "unique_hits"] == (rules[name] & ~others).sum()


def test_rule_table_leaves_unknown_outcomes_out_of_the_outcome_columns():
    rules, target, _ = random_rules()
    known = ~np.isnan(target)
    table = rule_table(rules, target)
    on_known = rule_table(rules[known], target[known])
    # Every row counts in the hits, only the known outcomes in the other columns.
    pd.testing.assert_frame_equal(table[HIT_COLUMNS], rule_table(rules))
    assert table["known"].tolist() == on_known["hits"].tolist()
    pd.testing.assert_frame_equal(table[OUTCOME_COLUMNS], on_known[OUTCOME_COLUMNS])
    assert (table["known"] < table["hits"]).all()


def test_rule_table_outcomes_of_a_rule_that_declined_every_hit_are_unknown():
    observed = [None, None, 0, 1, 0, None, 0, 0]
    table = rule_table(RULES, observed)
    assert table["hits"].tolist() == [3, 2, 2]
    assert table["known"].tolist() == [0, 1, 2]
    assert table["event"].tolist() == [0, 0, 1]
    assert np.isnan(table.loc["high_debt", ["event_rate", "lift"]]).all()
    assert table.loc["high_debt", "recall"] == 0
    # One known event out of five known outcomes.
    assert table.loc["inquiries", "lift"] == pytest.approx((1 / 2) / (1 / 5), rel=1e-15, abs=0)
    assert table.loc["inquiries", "recall"] == 1


@pytest.mark.parametrize("target", [[np.nan] * 8, [0] * 8])
def test_rule_table_ratios_are_nan_without_known_events(target):
    table = rule_table(RULES, target)
    assert table["lift"].isna().all() and table["recall"].isna().all()
    assert table["event"].tolist() == [0, 0, 0]


def test_rule_table_rule_without_hits():
    rules = RULES.assign(never=0)
    table = rule_table(rules, TARGET)
    row = table.loc["never"]
    assert row[["hits", "hit_rate", "unique_hits", "known", "event", "recall"]].tolist() == [0] * 6
    assert np.isnan(row["event_rate"]) and np.isnan(row["lift"])
    # The other rules keep their unique hits.
    pd.testing.assert_frame_equal(table.drop(index="never"), rule_table(RULES, TARGET))


def test_rule_table_integer_weights_equal_repeated_rows():
    rules, target, _ = random_rules()
    weights = np.random.default_rng(1).integers(0, 4, len(rules))
    repeat = np.repeat(np.arange(len(rules)), weights)
    weighted = rule_table(rules, target, weights=weights)
    repeated = rule_table(rules.iloc[repeat], target[repeat])
    assert weighted["hits"].dtype == np.float64
    pd.testing.assert_frame_equal(weighted, repeated, check_dtype=False, check_exact=True)


def test_rule_table_fractional_weights_match_brute_force():
    rules, target, weights = random_rules()
    table = rule_table(rules, target, weights=weights)
    known = ~np.isnan(target)
    event = known & (target == 1)
    alone = rules.to_numpy().sum(axis=1) == 1
    for name in rules.columns:
        hit = rules[name].to_numpy()
        expected = {
            "hits": weights[hit].sum(),
            "hit_rate": weights[hit].sum() / weights.sum(),
            "unique_hits": weights[hit & alone].sum(),
            "known": weights[hit & known].sum(),
            "event": weights[hit & event].sum(),
            "event_rate": weights[hit & event].sum() / weights[hit & known].sum(),
            "recall": weights[hit & event].sum() / weights[event].sum(),
            "unique_event": weights[hit & alone & event].sum(),
        }
        expected["lift"] = expected["event_rate"] / (weights[event].sum() / weights[known].sum())
        for column, value in expected.items():
            assert table.loc[name, column] == pytest.approx(value, rel=1e-12, abs=0)


@pytest.mark.parametrize(
    "rules",
    [
        RULES.astype(bool),
        RULES.astype(float),
        RULES.astype("boolean"),
        RULES.astype("Int64"),
        RULES.astype(object),
        RULES.astype(bool).astype(object),
        RULES.astype("category"),
    ],
)
def test_rule_tables_accept_booleans_and_zero_one_columns(rules):
    pd.testing.assert_frame_equal(rule_table(rules, TARGET), rule_table(RULES, TARGET))
    pd.testing.assert_frame_equal(rule_set_table(rules, TARGET), rule_set_table(RULES, TARGET))


def test_rule_tables_keep_the_column_labels_and_their_order():
    rules = RULES.set_axis([3, 1, 2], axis=1)
    for table in (rule_table(rules), rule_set_table(rules)):
        assert table.index.tolist() == [3, 1, 2]
        assert table.index.name == "rule"


@pytest.mark.parametrize("function", [rule_table, rule_set_table])
def test_rule_tables_match_series_by_index(function):
    index = pd.Index(list("abcdefgh"))
    rules = RULES.set_axis(index)
    target = pd.Series(TARGET, index=index)
    weights = pd.Series(np.arange(1.0, 9.0), index=index)
    by_position = function(RULES, TARGET, weights=np.arange(1.0, 9.0))
    pd.testing.assert_frame_equal(function(rules, target, weights=weights), by_position)
    # Arrays and lists are matched by position, whatever the index of rules.
    pd.testing.assert_frame_equal(function(rules, TARGET, weights=np.arange(1.0, 9.0)), by_position)


@pytest.mark.parametrize("function", [rule_table, rule_set_table])
def test_rule_tables_do_not_change_their_inputs(function):
    rules, target, weights = random_rules()
    copies = rules.copy(), target.copy(), weights.copy()
    function(rules, target, weights=weights)
    pd.testing.assert_frame_equal(rules, copies[0])
    np.testing.assert_array_equal(target, copies[1])
    np.testing.assert_array_equal(weights, copies[2])


# rule_set_table


def test_rule_set_table_matches_hand_calculation():
    table = rule_set_table(RULES, TARGET)
    expected = pd.DataFrame(
        {
            "new_hits": [3, 1, 1],
            "new_event": [2, 0, 1],
            "approved": [5, 4, 3],
            "approval_rate": [5 / 8, 4 / 8, 3 / 8],
            "approved_event": [1, 1, 0],
            "approved_event_rate": [1 / 5, 1 / 4, 0.0],
            "rejected_event_share": [2 / 3, 2 / 3, 1.0],
            "rejected_non_event_share": [1 / 5, 2 / 5, 2 / 5],
        },
        index=NAMES,
    )
    pd.testing.assert_frame_equal(table, expected)


def test_rule_set_table_without_target_only_counts_hits():
    table = rule_set_table(RULES)
    assert list(table.columns) == ["new_hits", "approved", "approval_rate"]
    assert table["new_hits"].dtype == np.int64 and table["approved"].dtype == np.int64
    pd.testing.assert_frame_equal(table, rule_set_table(RULES, TARGET)[table.columns])


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_rule_set_table_matches_a_replay_rule_by_rule(seed, weighted):
    rules, target, weights = random_rules(seed)
    weights = weights if weighted else None
    table = rule_set_table(rules, target, weights=weights)
    expected = replay(rules, target, weights)
    pd.testing.assert_frame_equal(table, expected, check_dtype=False, rtol=1e-12, atol=0)
    without_target = rule_set_table(rules, weights=weights)
    pd.testing.assert_frame_equal(
        without_target, replay(rules, None, weights), check_dtype=False, rtol=1e-12, atol=0
    )


def test_rule_set_table_counts_unweighted_rows_exactly():
    rules, target, _ = random_rules()
    pd.testing.assert_frame_equal(
        rule_set_table(rules, target), replay(rules, target), check_dtype=False, check_exact=True
    )


def test_rule_set_table_new_hits_add_up_to_the_rows_any_rule_hits():
    rules, target, _ = random_rules()
    table = rule_set_table(rules, target)
    any_hit = rules.any(axis=1).to_numpy()
    assert table["new_hits"].sum() == any_hit.sum()
    assert table["new_event"].sum() == np.nansum(target[any_hit])
    assert table["approved"].iloc[-1] == (~any_hit).sum()
    assert (np.diff(table["approved"]) <= 0).all()


def test_rule_set_table_first_row_is_the_first_rule_on_its_own():
    rules, target, weights = random_rules()
    table = rule_set_table(rules, target, weights=weights)
    single = rule_table(rules, target, weights=weights).iloc[0]
    assert table["new_hits"].iloc[0] == pytest.approx(single["hits"], rel=1e-12, abs=0)
    assert table["new_event"].iloc[0] == pytest.approx(single["event"], rel=1e-12, abs=0)
    assert table["approval_rate"].iloc[0] == pytest.approx(1 - single["hit_rate"], rel=1e-12)
    assert table["rejected_event_share"].iloc[0] == pytest.approx(single["recall"], rel=1e-12)


def test_rule_set_table_rows_match_cutoff_table_of_the_rules_so_far():
    rules, target, weights = random_rules()
    complete = np.nan_to_num(target)
    table = rule_set_table(rules, complete, weights=weights)
    for position in range(len(rules.columns)):
        declined = rules.iloc[:, : position + 1].any(axis=1).astype(int)
        summary = cutoff_table(declined, complete, cutoffs=[0], weights=weights).iloc[0]
        pd.testing.assert_series_equal(
            table[CUMULATIVE_COLUMNS].iloc[position],
            summary,
            check_names=False,
            rtol=1e-12,
            atol=0,
        )


def test_rule_set_table_last_row_equals_cutoff_table_of_all_rules_or_ed():
    rules, target, _ = random_rules()
    complete = np.nan_to_num(target)
    declined = rules.any(axis=1).astype(int)
    summary = cutoff_table(declined, complete, cutoffs=[0])
    last = rule_set_table(rules, complete)[CUMULATIVE_COLUMNS].iloc[[-1]]
    pd.testing.assert_frame_equal(
        last.reset_index(drop=True), summary.reset_index(drop=True), check_exact=True
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_rule_set_table_last_row_does_not_depend_on_the_order(weighted):
    rules, target, weights = random_rules()
    weights = weights if weighted else None
    last = rule_set_table(rules, target, weights=weights)[CUMULATIVE_COLUMNS].iloc[-1]
    new_hits = set()
    for order in itertools.permutations(rules.columns):
        table = rule_set_table(rules[list(order)], target, weights=weights)
        # Exactly: the last row sums the same rows in the same order, whatever the order.
        pd.testing.assert_series_equal(
            table[CUMULATIVE_COLUMNS].iloc[-1], last, check_names=False, check_exact=True
        )
        new_hits.add(tuple(table.loc[list(rules.columns), "new_hits"]))
    # The new hits of each rule do depend on the order.
    assert len(new_hits) > 1


def test_rule_set_table_counts_unknown_outcomes_in_the_approved_rows_only():
    observed = [None, None, 0, 1, 0, None, 0, 0]
    table = rule_set_table(RULES, observed)
    assert table["approved"].tolist() == [5, 4, 3]
    assert table["new_event"].tolist() == [0, 0, 1]
    assert table["approved_event"].tolist() == [1, 1, 0]
    # The approved rows all have known outcomes: rows 2, 3, 4, 6 and 7 after high_debt.
    assert table["approved_event_rate"].tolist() == [1 / 5, 1 / 4, 0.0]
    # One known event and four known non-events, of which row 2 is hit by young.
    assert table["rejected_event_share"].tolist() == [0.0, 0.0, 1.0]
    assert table["rejected_non_event_share"].tolist() == [0.0, 0.25, 0.25]


def test_rule_set_table_event_rate_is_nan_when_no_approved_outcome_is_known():
    observed = [1, 0, 0, 1, None, 0, None, None]
    table = rule_set_table(RULES, observed)
    assert np.isnan(table["approved_event_rate"].iloc[-1])
    assert table["approved_event"].iloc[-1] == 0


@pytest.mark.parametrize(
    ("target", "column"),
    [([0] * 8, "rejected_event_share"), ([1] * 8, "rejected_non_event_share")],
)
def test_rule_set_table_shares_are_nan_without_known_events_or_non_events(target, column):
    table = rule_set_table(RULES, target)
    assert table[column].isna().all()


def test_rule_set_table_integer_weights_equal_repeated_rows():
    rules, target, _ = random_rules()
    weights = np.random.default_rng(2).integers(0, 4, len(rules))
    repeat = np.repeat(np.arange(len(rules)), weights)
    weighted = rule_set_table(rules, target, weights=weights)
    repeated = rule_set_table(rules.iloc[repeat], target[repeat])
    assert weighted["new_hits"].dtype == np.float64
    pd.testing.assert_frame_equal(weighted, repeated, check_dtype=False, check_exact=True)


def test_rule_set_table_one_rule():
    table = rule_set_table(RULES[["young"]], TARGET)
    assert table.index.tolist() == ["young"]
    assert table.loc["young", ["new_hits", "new_event", "approved"]].tolist() == [2, 1, 6]


# Errors


@pytest.mark.parametrize("function", [rule_table, rule_set_table])
@pytest.mark.parametrize(
    ("rules", "message"),
    [
        (RULES["young"], "rules must be a DataFrame with one column per rule, got Series"),
        (RULES.to_numpy(), "rules must be a DataFrame with one column per rule, got ndarray"),
        (pd.DataFrame(index=range(3)), "rules must have at least one column"),
        (RULES.iloc[:0], "rules must have at least one row"),
        (
            RULES.set_axis(pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)]), axis=1),
            "rules must have one level of column names, got 2",
        ),
        (RULES.set_axis(["a", "b", "a"], axis=1), "rules must have unique column names, got 'a'"),
        (
            RULES.astype(float).assign(young=[0, 1, np.nan, 0, 0, 0, 0, 0]),
            "rules column 'young' must not have missing values, got 1",
        ),
        (
            RULES.astype("boolean").assign(young=pd.array([pd.NA] * 8, dtype="boolean")),
            "rules column 'young' must not have missing values, got 8",
        ),
        (
            RULES.astype(object).assign(young=[None, 1, 1, 0, 0, 0, 0, 0]),
            "rules column 'young' must not have missing values, got 1",
        ),
        (
            RULES.assign(inquiries=[0, 0, 2, 1, 0, 0, 0, 0]),
            r"rules column 'inquiries' must contain only booleans or 0 and 1, got 2",
        ),
        (
            RULES.assign(young=["no"] * 8),
            r"rules column 'young' must contain only booleans or 0 and 1, got 'no'",
        ),
        (
            RULES.assign(young=0.5),
            r"rules column 'young' must contain only booleans or 0 and 1, got 0.5",
        ),
    ],
)
def test_rule_tables_reject_invalid_rules(function, rules, message):
    with pytest.raises(ValueError, match=message):
        function(rules)
    with pytest.raises(ValueError, match=message):
        function(rules, TARGET)


@pytest.mark.parametrize("function", [rule_table, rule_set_table])
@pytest.mark.parametrize(
    ("target", "message"),
    [
        (TARGET[:5], "target must have one label per row of rules, got 5 for 8"),
        (np.ones((8, 1)), "target must be one-dimensional, got 2 dimensions"),
        ([2, 1, 0, 1, 0, 0, 0, 0], "target must contain only 0, 1 and missing values, got 2"),
        (["yes"] * 8, "target must contain only 0, 1 and missing values, got 'yes'"),
        (
            pd.Series(TARGET, index=range(1, 9)),
            "rules and target must have identical indices",
        ),
    ],
)
def test_rule_tables_reject_invalid_targets(function, target, message):
    with pytest.raises(ValueError, match=message):
        function(RULES, target)


@pytest.mark.parametrize("function", [rule_table, rule_set_table])
@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([1.0] * 5, "weights must have one weight per row of rules, got 5 for 8"),
        ([1.0] * 7 + [-1.0], "weights must be finite and non-negative, got -1.0"),
        ([1.0] * 7 + [np.nan], "weights must be finite and non-negative, got nan"),
        ([0.0] * 8, "weights must have a finite, positive total, got 0.0"),
        ([True] * 8, "weights must be numeric, got dtype bool"),
        (
            pd.Series([1.0] * 8, index=range(1, 9)),
            "rules and weights must have identical indices",
        ),
    ],
)
def test_rule_tables_reject_invalid_weights(function, weights, message):
    with pytest.raises(ValueError, match=message):
        function(RULES, TARGET, weights=weights)
    with pytest.raises(ValueError, match=message):
        function(RULES, weights=weights)


@pytest.mark.parametrize("function", [rule_table, rule_set_table])
def test_rule_tables_reject_a_target_and_weights_with_different_indices(function):
    rules = RULES.set_axis(list("abcdefgh"))
    target = pd.Series(TARGET, index=rules.index)
    weights = pd.Series([1.0] * 8, index=list("hgfedcba"))
    with pytest.raises(ValueError, match="rules and weights must have identical indices"):
        function(rules, target, weights=weights)


def candidate_data(seed=0, n=600):
    """Two numeric columns with a risky tail, a category, a boolean, special codes and gaps."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, n).astype(float)
    debt = rng.random(n)
    p = 0.05 + 0.3 * (age < 25) + 0.3 * (debt > 0.8)
    frame = pd.DataFrame(
        {
            "age": age,
            "debt": debt,
            "region": rng.choice(["a", "b"], n),
            "flag": rng.random(n) < 0.5,
        },
        index=pd.RangeIndex(50, 50 + n),
    )
    frame.loc[frame.index[:20], "debt"] = -1.0
    frame.loc[frame.index[20:40], "age"] = np.nan
    target = pd.Series((rng.random(n) < p).astype(float), index=frame.index)
    target.iloc[40:60] = np.nan
    return frame, target


COLUMNS = ["variable", "operator", "threshold", "hits", "hit_rate", "event", "event_rate", "lift"]


def column_of(frame, row):
    values = frame[row["variable"]]
    return values > row["threshold"] if row["operator"] == ">" else values <= row["threshold"]


@pytest.mark.parametrize("method", ["tree", "ks"])
@pytest.mark.parametrize("weighted", [False, True])
def test_rule_candidates_match_rule_table(method, weighted):
    frame, target = candidate_data()
    weights = pd.Series(np.random.default_rng(1).random(len(frame)) * 3, index=frame.index)
    table = rule_candidates(
        frame,
        target,
        method=method,
        special={"debt": [-1]},
        weights=weights if weighted else None,
        min_hit_rate=0.0,
        min_lift=0.0,
    )
    assert list(table.columns) == COLUMNS
    assert table.index.name == "rule"
    assert set(table["variable"]) == {"age", "debt"}
    assert len(table) > 2
    for name, row in table.iterrows():
        rule = column_of(frame, row)
        if row["variable"] == "debt":
            rule &= frame["debt"] != -1
        expected = rule_table(
            pd.DataFrame({"rule": rule.fillna(False)}),
            target,
            weights=weights if weighted else None,
        ).iloc[0]
        for column in ["hits", "hit_rate", "event", "event_rate", "lift"]:
            assert row[column] == pytest.approx(expected[column], rel=1e-12)
        assert name == f"{row['variable']} {row['operator']} {row['threshold']!r}"
    assert table["lift"].is_monotonic_decreasing


def test_rule_candidates_pick_the_riskier_side_and_filter():
    frame, target = candidate_data()
    everything = rule_candidates(frame, target, min_hit_rate=0, min_lift=0, special={"debt": [-1]})
    assert set(everything["operator"]) == {"<=", ">"}
    # The rules of a threshold hit one side each: the chosen side has the higher event rate.
    for _, row in everything.iterrows():
        values = frame[row["variable"]]
        usable = values.notna() & (values != -1 if row["variable"] == "debt" else True)
        other = values <= row["threshold"] if row["operator"] == ">" else values > row["threshold"]
        known = target.notna()
        rate = target[other & usable & known].mean()
        assert row["event_rate"] >= rate or np.isnan(rate)
    strict = rule_candidates(frame, target, min_hit_rate=0.1, min_lift=1.5, special={"debt": [-1]})
    assert (strict["hit_rate"] >= 0.1).all() and (strict["lift"] >= 1.5).all()
    assert set(strict.index) <= set(everything.index)
    assert strict.index.equals(everything.loc[everything.index.isin(strict.index)].index)


def test_rule_candidates_find_the_planted_thresholds():
    frame, target = candidate_data(n=4000)
    table = rule_candidates(frame, target, special={"debt": [-1]})
    top = table.drop_duplicates("variable").set_index("variable")
    assert top.loc["age", "operator"] == "<=" and 24 <= top.loc["age", "threshold"] <= 26
    assert top.loc["debt", "operator"] == ">" and top.loc["debt", "threshold"] >= 0.75


def test_rule_candidates_special_and_missing_values_hit_nothing():
    frame, target = candidate_data()
    table = rule_candidates(frame, target, special={"debt": [-1]}, min_hit_rate=0, min_lift=0)
    debt = table[table["variable"] == "debt"]
    assert (debt["threshold"] > -1).all()
    for _, row in table.iterrows():
        rule = column_of(frame, row)
        if row["variable"] == "debt":
            hit = (frame["debt"] != -1) & rule
        else:
            hit = rule
        assert row["hits"] == hit.sum()


def test_rule_candidates_integer_weights_equal_repeated_rows():
    frame, target = candidate_data(n=300)
    target = target.fillna(0)
    weights = pd.Series(np.random.default_rng(3).integers(1, 4, len(frame)), index=frame.index)
    repeated = frame.loc[frame.index.repeat(weights)].reset_index(drop=True)
    repeated_target = target.loc[target.index.repeat(weights)].reset_index(drop=True)
    kwargs = {"min_hit_rate": 0, "min_lift": 0}
    for method in ("tree", "ks"):
        weighted = rule_candidates(frame, target, method=method, weights=weights, **kwargs)
        plain = rule_candidates(repeated, repeated_target, method=method, **kwargs)
        pd.testing.assert_frame_equal(
            weighted.reset_index(drop=True), plain.reset_index(drop=True), check_dtype=False
        )


def test_rule_candidates_columns_and_skipped_kinds():
    frame, target = candidate_data()
    assert set(rule_candidates(frame, target, min_lift=0)["variable"]) <= {"age", "debt"}
    only = rule_candidates(frame, target, columns=["age"], min_lift=0)
    assert set(only["variable"]) == {"age"}
    assert rule_candidates(frame, target, columns=["age"], max_bins=2, min_lift=0).shape[0] == 1


def test_rule_candidates_empty_and_input_unchanged():
    frame, target = candidate_data()
    before = frame.copy(), target.copy()
    table = rule_candidates(frame, target, min_lift=100.0)
    assert table.empty and list(table.columns) == COLUMNS and table.index.name == "rule"
    pd.testing.assert_frame_equal(frame, before[0])
    pd.testing.assert_series_equal(target, before[1])
    constant = pd.DataFrame({"x": np.ones(50)})
    assert rule_candidates(constant, [0, 1] * 25).empty


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"columns": ["region"]}, "must be numeric"),
        ({"columns": ["flag"]}, "must be numeric"),
        ({"columns": ["nope"]}, "columns must be columns of frame"),
        ({"columns": "age"}, "columns must be a sequence"),
        ({"method": "chimerge"}, "method must be 'tree' or 'ks'"),
        ({"max_bins": 0}, "max_bins must be a positive integer"),
        ({"min_hit_rate": 1.5}, "min_hit_rate must be a number between 0 and 1"),
        ({"min_hit_rate": True}, "min_hit_rate must be a number between 0 and 1"),
        ({"min_lift": -1}, "min_lift must be a finite number of at least 0"),
        ({"min_lift": float("inf")}, "min_lift must be a finite number of at least 0"),
        ({"special": {"region": [1]}}, "special must name searched columns"),
        ({"special": {"age": [True]}}, "special codes of column 'age'"),
        ({"special": [1]}, "special must be a mapping"),
        ({"weights": [1.0, 2.0]}, "weights must have one weight per row"),
        ({"weights": -np.ones(600)}, "weights"),
    ],
)
def test_rule_candidates_reject_invalid_arguments(kwargs, message):
    frame, target = candidate_data()
    with pytest.raises(ValueError, match=message):
        rule_candidates(frame, target, **kwargs)


def test_rule_candidates_reject_invalid_frames_and_targets():
    frame, target = candidate_data()
    with pytest.raises(ValueError, match="frame must be a DataFrame"):
        rule_candidates(frame["age"], target)
    with pytest.raises(ValueError, match="at least one row"):
        rule_candidates(frame.iloc[:0], target.iloc[:0])
    with pytest.raises(ValueError, match="at least one numeric column"):
        rule_candidates(frame[["region", "flag"]], target)
    with pytest.raises(ValueError, match="identical indices"):
        rule_candidates(frame, target.reset_index(drop=True))
    with pytest.raises(ValueError, match="one label per row"):
        rule_candidates(frame, target.iloc[:-1].tolist())
    with pytest.raises(ValueError, match="0, 1 and missing"):
        rule_candidates(frame, target + 2)
    with pytest.raises(ValueError, match="both 0 and 1"):
        rule_candidates(frame, np.zeros(len(frame)))
    with pytest.raises(ValueError, match="infinite values"):
        rule_candidates(frame.assign(age=frame["age"].where(frame["age"] != 30, np.inf)), target)
