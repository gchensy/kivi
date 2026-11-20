import numpy as np
import pandas as pd
import pytest

from kivi.binning import bin_edges
from kivi.datasets import load_bank
from kivi.metrics import gains_table
from kivi.strategy import cutoff_table, expected_loss, expected_loss_table, swap_set

# Events score 0.2, 0.6, 0.8 and 0.9; non-events score 0.1, 0.2, 0.3 and 0.5.
SCORE = [0.1, 0.2, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9]
TARGET = [0, 0, 1, 0, 0, 1, 1, 1]

CUTOFF_COLUMNS = [
    "approved",
    "approval_rate",
    "approved_event",
    "approved_event_rate",
    "rejected_event_share",
    "rejected_non_event_share",
]

# The current score approves rows 0-3 at 0.4, the new score rows 0, 1, 2 and 4.
CURRENT = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
NEW = [0.1, 0.3, 0.2, 0.6, 0.4, 0.7, 0.5, 0.8]
SWAP_TARGET = [0, 0, 1, 0, 1, 1, 0, 1]
DECISIONS = [
    ("approve", "approve"),
    ("approve", "decline"),
    ("decline", "approve"),
    ("decline", "decline"),
]

EL_COLUMNS = [
    "total",
    "exposure",
    "exposure_weighted_probability",
    "expected_loss",
    "expected_loss_rate",
]


def brute_force_cutoffs(score, target, cutoffs, loss=None, weights=None):
    """Recompute every column of cutoff_table with one mask per cut-off."""
    score, target = np.asarray(score, dtype=float), np.asarray(target)
    weights = np.ones(len(score)) if weights is None else np.asarray(weights, dtype=float)
    rows = []
    for cutoff in cutoffs:
        approved = score <= cutoff
        events = weights * (target == 1)
        non_events = weights * (target == 0)
        row = {
            "approved": weights[approved].sum(),
            "approval_rate": weights[approved].sum() / weights.sum(),
            "approved_event": events[approved].sum(),
            "approved_event_rate": (
                events[approved].sum() / weights[approved].sum() if approved.any() else np.nan
            ),
            "rejected_event_share": events[~approved].sum() / events.sum(),
            "rejected_non_event_share": non_events[~approved].sum() / non_events.sum(),
        }
        if loss is not None:
            row["approved_loss"] = (weights * np.asarray(loss))[approved].sum()
        rows.append(row)
    return pd.DataFrame(rows, index=pd.Index(np.asarray(cutoffs, dtype=float), name="cutoff"))


def test_cutoff_table_matches_hand_calculation():
    table = cutoff_table(SCORE, TARGET, cutoffs=[0.2, 0.5, 1.0])
    assert table.columns.tolist() == CUTOFF_COLUMNS
    assert table.index.name == "cutoff"
    assert table.index.tolist() == [0.2, 0.5, 1.0]
    # At 0.2 the three lowest scores are approved, one of them an event.
    assert table["approved"].tolist() == [3, 5, 8]
    assert table["approved"].dtype == np.int64
    assert table["approved_event"].tolist() == [1, 1, 4]
    np.testing.assert_allclose(table["approval_rate"], [3 / 8, 5 / 8, 1.0])
    np.testing.assert_allclose(table["approved_event_rate"], [1 / 3, 1 / 5, 4 / 8])
    np.testing.assert_allclose(table["rejected_event_share"], [3 / 4, 3 / 4, 0.0])
    np.testing.assert_allclose(table["rejected_non_event_share"], [2 / 4, 0.0, 0.0])


def test_cutoff_table_approves_a_score_equal_to_the_cutoff():
    # Both tied scores of 0.2 are approved at 0.2 and both declined just below it.
    table = cutoff_table(SCORE, TARGET, cutoffs=[np.nextafter(0.2, 0), 0.2])
    assert table["approved"].tolist() == [1, 3]
    assert table["approved_event"].tolist() == [0, 1]


def test_cutoff_table_matches_brute_force_on_random_scores():
    rng = np.random.default_rng(7)
    score = rng.integers(0, 30, 500) / 10
    target = (rng.random(500) < 0.1 + 0.25 * (score > 1.5)).astype(int)
    loss = rng.gamma(2.0, 100.0, 500)
    cutoffs = [-1.0, 0.0, 0.55, 1.5, 2.2, 2.9, 10.0]
    table = cutoff_table(score, target, cutoffs=cutoffs, loss=loss)
    pd.testing.assert_frame_equal(
        table, brute_force_cutoffs(score, target, cutoffs, loss), check_dtype=False
    )


def test_cutoff_table_without_approvals_has_no_event_rate():
    table = cutoff_table(SCORE, TARGET, cutoffs=[-np.inf, 0.0, np.inf])
    assert table["approved"].tolist() == [0, 0, 8]
    assert np.isnan(table["approved_event_rate"].iloc[:2]).all()
    assert table["rejected_event_share"].tolist() == [1.0, 1.0, 0.0]


def test_cutoff_table_integer_uses_bin_edges_above_the_lowest():
    rng = np.random.default_rng(3)
    score = rng.normal(size=400)
    target = (rng.random(400) < 1 / (1 + np.exp(-score))).astype(int)
    table = cutoff_table(score, target, cutoffs=5)
    np.testing.assert_array_equal(table.index, bin_edges(score, bins=5)[1:])
    assert table.index[-1] == np.inf
    assert table["approved"].tolist() == [80, 160, 240, 320, 400]
    assert table["approval_rate"].iloc[-1] == 1.0
    assert table["rejected_event_share"].iloc[-1] == 0.0


def test_cutoff_table_is_the_cumulative_gains_table_on_bank_data():
    bank = load_bank()
    score, target = -bank["age"], bank["target"]
    edges = bin_edges(score, bins=6)
    table = cutoff_table(score, target, cutoffs=edges[1:])
    # gains_table runs from the highest band down; cumulate it from the lowest band up.
    gains = gains_table(score, target, bins=edges).iloc[::-1]
    np.testing.assert_array_equal(table["approved"], np.cumsum(gains["total"]))
    np.testing.assert_array_equal(table["approved_event"], np.cumsum(gains["event"]))
    np.testing.assert_allclose(
        table["rejected_event_share"].iloc[:-1], gains["cum_event_share"].iloc[1:]
    )
    assert table["approved"].iloc[-1] == len(bank)
    assert table["approved_event"].iloc[-1] == 521


def test_cutoff_table_warns_when_ties_leave_fewer_cutoffs():
    score = [1, 1, 1, 1, 1, 1, 2, 3]
    with pytest.warns(UserWarning, match="tied edges leave"):
        table = cutoff_table(score, TARGET, cutoffs=4)
    assert table["approved"].iloc[-1] == 8


def test_cutoff_table_loss_sums_the_approved_losses():
    loss = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    table = cutoff_table(SCORE, TARGET, cutoffs=[0.2, 0.5, 1.0], loss=loss)
    assert table.columns.tolist() == [*CUTOFF_COLUMNS, "approved_loss"]
    assert table["approved_loss"].tolist() == [60.0, 150.0, 360.0]


def test_cutoff_table_integer_weights_equal_repeated_rows():
    rng = np.random.default_rng(11)
    score = rng.integers(0, 12, 60) / 4
    target = (rng.random(60) < 0.3).astype(int)
    target[:2] = [0, 1]
    loss = rng.uniform(0, 50, 60)
    weights = rng.integers(0, 4, 60)
    weights[:2] = 1
    repeat = np.repeat(np.arange(60), weights)
    cutoffs = [0.5, 1.0, 1.75, 3.0]
    weighted = cutoff_table(score, target, cutoffs=cutoffs, loss=loss, weights=weights)
    repeated = cutoff_table(score[repeat], target[repeat], cutoffs=cutoffs, loss=loss[repeat])
    pd.testing.assert_frame_equal(weighted, repeated, check_dtype=False)
    pd.testing.assert_frame_equal(
        weighted, brute_force_cutoffs(score, target, cutoffs, loss, weights), check_dtype=False
    )


def test_cutoff_table_integer_with_weights_uses_weighted_quantiles():
    rng = np.random.default_rng(5)
    score = rng.normal(size=200)
    target = (rng.random(200) < 0.2).astype(int)
    weights = rng.uniform(0.5, 2.0, 200)
    table = cutoff_table(score, target, cutoffs=4, weights=weights)
    np.testing.assert_array_equal(table.index, bin_edges(score, bins=4, weights=weights)[1:])


def test_cutoff_table_matches_series_by_index_and_arrays_by_position():
    score = pd.Series(SCORE, index=list("abcdefgh"))
    target = pd.Series(TARGET, index=list("abcdefgh"))
    expected = cutoff_table(SCORE, TARGET, cutoffs=[0.3, 1.0])
    pd.testing.assert_frame_equal(cutoff_table(score, target, cutoffs=[0.3, 1.0]), expected)
    pd.testing.assert_frame_equal(cutoff_table(score, TARGET, cutoffs=[0.3, 1.0]), expected)
    with pytest.raises(ValueError, match="identical indices"):
        cutoff_table(score, target.reset_index(drop=True), cutoffs=[0.3])
    with pytest.raises(ValueError, match="loss must have the same index as score"):
        cutoff_table(score, target, cutoffs=[0.3], loss=pd.Series(np.ones(8)))


def test_cutoff_table_does_not_mutate_inputs():
    score = pd.Series(SCORE)
    target = pd.Series(TARGET)
    loss = np.arange(8.0)
    cutoff_table(score, target, cutoffs=[0.3], loss=loss)
    pd.testing.assert_series_equal(score, pd.Series(SCORE))
    pd.testing.assert_series_equal(target, pd.Series(TARGET))
    np.testing.assert_array_equal(loss, np.arange(8.0))


@pytest.mark.parametrize(
    "cutoffs",
    [0, -2, True, 0.5, [], [0.5, 0.2], [0.2, 0.2], [0.1, np.nan], ["0.1"], [[0.1, 0.2]], [True]],
)
def test_cutoff_table_rejects_invalid_cutoffs(cutoffs):
    with pytest.raises(ValueError, match="cutoffs must be a positive integer or a sequence"):
        cutoff_table(SCORE, TARGET, cutoffs=cutoffs)


@pytest.mark.parametrize(
    ("loss", "message"),
    [
        ([1.0] * 7, "loss must have one value per value of score, got 7 for 8"),
        ([1.0] * 7 + [-1.0], "loss must be finite and non-negative, got -1.0"),
        ([1.0] * 7 + [np.nan], "loss must be finite and non-negative, got nan"),
        ([1.0] * 7 + [np.inf], "loss must be finite and non-negative, got inf"),
        (["a"] * 8, "loss must be real and numeric, got dtype (object|str)"),
        ([True] * 8, "loss must be real and numeric, got dtype bool"),
        (np.ones((8, 1)), "loss must be one-dimensional"),
    ],
)
def test_cutoff_table_rejects_invalid_loss(loss, message):
    with pytest.raises(ValueError, match=message):
        cutoff_table(SCORE, TARGET, cutoffs=[0.5], loss=loss)


def test_cutoff_table_rejects_invalid_scores_and_targets():
    with pytest.raises(ValueError, match="score must be finite"):
        cutoff_table([0.1, np.nan, 0.3], [0, 1, 0], cutoffs=[0.2])
    with pytest.raises(ValueError, match="target must contain both 0 and 1"):
        cutoff_table([0.1, 0.2], [0, 0], cutoffs=[0.2])
    with pytest.raises(ValueError, match="target must contain only 0 and 1, got 2"):
        cutoff_table([0.1, 0.2], [0, 2], cutoffs=[0.2])
    with pytest.raises(ValueError, match="weights must give both target classes"):
        cutoff_table([0.1, 0.2], [0, 1], cutoffs=[0.2], weights=[1, 0])


def test_swap_set_matches_hand_calculation():
    table = swap_set(CURRENT, NEW, SWAP_TARGET, current_cutoff=0.4, new_cutoff=0.4)
    assert table.index.names == ["current", "new"]
    assert table.index.tolist() == DECISIONS
    assert table.columns.tolist() == ["total", "share", "known", "event", "event_rate"]
    # Both approve rows 0-2; the swap-out is row 3, the swap-in row 4.
    assert table["total"].tolist() == [3, 1, 1, 3]
    assert table["total"].dtype == np.int64
    assert table["known"].tolist() == [3, 1, 1, 3]
    assert table["event"].tolist() == [1, 0, 1, 2]
    np.testing.assert_allclose(table["share"], [3 / 8, 1 / 8, 1 / 8, 3 / 8])
    np.testing.assert_allclose(table["event_rate"], [1 / 3, 0.0, 1.0, 2 / 3])


def test_swap_set_without_target_only_counts():
    table = swap_set(CURRENT, NEW, current_cutoff=0.4, new_cutoff=0.4)
    assert table.columns.tolist() == ["total", "share"]
    assert table["total"].tolist() == [3, 1, 1, 3]


def test_swap_set_approves_scores_equal_to_the_cutoffs():
    table = swap_set([1, 2, 2, 3], [5, 5, 6, 6], current_cutoff=2, new_cutoff=5)
    assert table["total"].tolist() == [2, 1, 0, 1]
    table = swap_set([1, 2, 2, 3], [5, 5, 6, 6], current_cutoff=1.999, new_cutoff=4.999)
    assert table["total"].tolist() == [0, 1, 0, 3]


@pytest.mark.parametrize(
    "target",
    [
        [0, 0, 1, 0, np.nan, None, np.nan, np.nan],
        pd.Series([0, 0, 1, 0, None, None, None, None], dtype="Int64"),
        pd.Series([0, 0, 1, 0, None, None, None, None], dtype=object),
        pd.Series([False, False, True, False, None, None, None, None], dtype="boolean"),
    ],
)
def test_swap_set_leaves_missing_outcomes_out_of_the_event_rate(target):
    table = swap_set(CURRENT, NEW, target, current_cutoff=0.4, new_cutoff=0.4)
    assert table["total"].tolist() == [3, 1, 1, 3]
    assert table["known"].tolist() == [3, 1, 0, 0]
    assert table["event"].tolist() == [1, 0, 0, 0]
    np.testing.assert_allclose(table["event_rate"], [1 / 3, 0.0, np.nan, np.nan])


def test_swap_set_keeps_empty_rows():
    table = swap_set(CURRENT, NEW, SWAP_TARGET, current_cutoff=np.inf, new_cutoff=0.4)
    assert table["total"].tolist() == [4, 4, 0, 0]
    assert table["share"].tolist() == [0.5, 0.5, 0.0, 0.0]
    assert np.isnan(table["event_rate"].iloc[2:]).all()


def test_swap_set_shares_give_the_approval_rates_of_both_scores():
    rng = np.random.default_rng(2)
    current, new = rng.random(300), rng.random(300)
    table = swap_set(current, new, current_cutoff=0.7, new_cutoff=0.6)
    share = table["share"]
    assert share.sum() == pytest.approx(1.0)
    assert share.xs("approve", level="current").sum() == pytest.approx(np.mean(current <= 0.7))
    assert share.xs("approve", level="new").sum() == pytest.approx(np.mean(new <= 0.6))


def test_swap_set_integer_weights_equal_repeated_rows():
    rng = np.random.default_rng(4)
    current, new = rng.random(80), rng.random(80)
    target = np.where(rng.random(80) < 0.2, 1.0, 0.0)
    target[current > 0.7] = np.nan
    weights = rng.integers(0, 4, 80)
    repeat = np.repeat(np.arange(80), weights)
    options = dict(current_cutoff=0.7, new_cutoff=0.65)
    weighted = swap_set(current, new, target, weights=weights, **options)
    repeated = swap_set(current[repeat], new[repeat], target[repeat], **options)
    pd.testing.assert_frame_equal(weighted, repeated, check_dtype=False)


def test_swap_set_matches_series_by_index():
    index = list("abcdefgh")
    current = pd.Series(CURRENT, index=index)
    new = pd.Series(NEW, index=index)
    target = pd.Series(SWAP_TARGET, index=index)
    expected = swap_set(CURRENT, NEW, SWAP_TARGET, current_cutoff=0.4, new_cutoff=0.4)
    pd.testing.assert_frame_equal(
        swap_set(current, new, target, current_cutoff=0.4, new_cutoff=0.4), expected
    )
    with pytest.raises(ValueError, match="current and new must have identical indices"):
        swap_set(current, new.reset_index(drop=True), current_cutoff=0.4, new_cutoff=0.4)
    with pytest.raises(ValueError, match="current and target must have identical indices"):
        swap_set(current, NEW, target.reset_index(drop=True), current_cutoff=0.4, new_cutoff=0.4)


@pytest.mark.parametrize(
    ("current", "new", "target", "message"),
    [
        (CURRENT, NEW[:-1], None, "current and new must have the same length, got 8 and 7"),
        ([], [], None, "current and new must not be empty"),
        (CURRENT[:-1] + [np.nan], NEW, None, "current must be finite, got 1 missing"),
        (CURRENT, NEW[:-1] + [np.inf], None, "new must be finite, got 1 missing or infinite"),
        (["a"] * 8, NEW, None, "current must be real and numeric, got dtype (object|str)"),
        (CURRENT, NEW, SWAP_TARGET[:-1], "target must have one label per value of current"),
        (CURRENT, NEW, [0, 1, 2, 0, 1, 1, 0, 1], "target must contain only 0, 1 and missing"),
        (CURRENT, NEW, ["0", "1"] * 4, r"target must contain only 0, 1 and missing.*'0'"),
    ],
)
def test_swap_set_rejects_invalid_inputs(current, new, target, message):
    with pytest.raises(ValueError, match=message):
        swap_set(current, new, target, current_cutoff=0.4, new_cutoff=0.4)


@pytest.mark.parametrize("cutoff", [None, np.nan, True, "0.4", [0.4]])
def test_swap_set_rejects_invalid_cutoffs(cutoff):
    with pytest.raises(ValueError, match="current_cutoff must be a number"):
        swap_set(CURRENT, NEW, current_cutoff=cutoff, new_cutoff=0.4)
    with pytest.raises(ValueError, match="new_cutoff must be a number"):
        swap_set(CURRENT, NEW, current_cutoff=0.4, new_cutoff=cutoff)


def test_swap_set_rejects_invalid_weights():
    with pytest.raises(ValueError, match="weights must be finite and non-negative"):
        swap_set(CURRENT, NEW, current_cutoff=0.4, new_cutoff=0.4, weights=[-1] + [1] * 7)
    with pytest.raises(ValueError, match="weights must have one weight per value of current"):
        swap_set(CURRENT, NEW, current_cutoff=0.4, new_cutoff=0.4, weights=[1] * 7)


def test_expected_loss_is_the_product_of_pd_lgd_and_ead():
    np.testing.assert_allclose(
        expected_loss([0.02, 0.1, 0.25], 0.45, [1000, 500, 200]), [9.0, 22.5, 22.5]
    )
    result = expected_loss(0.02, 0.5, 1000)
    assert isinstance(result, float)
    assert result == pytest.approx(10.0)
    np.testing.assert_allclose(expected_loss(0.1, [0.2, 0.4], 100.0), [2.0, 4.0])


def test_expected_loss_keeps_the_series_index_and_missing_values():
    probability = pd.Series([0.1, np.nan, 0.3], index=[5, 6, 7], name="pd")
    result = expected_loss(probability, 0.5, pd.Series([10.0, 20.0, 30.0], index=[5, 6, 7]))
    assert isinstance(result, pd.Series)
    assert result.name == "pd"
    assert result.index.tolist() == [5, 6, 7]
    np.testing.assert_allclose(result, [0.5, np.nan, 4.5])
    from_ead = expected_loss([0.1, 0.2], 0.5, pd.Series([10.0, 20.0], index=["a", "b"]))
    assert from_ead.index.tolist() == ["a", "b"]
    np.testing.assert_allclose(from_ead, [0.5, 2.0])


def test_expected_loss_does_not_mutate_inputs():
    probability = np.array([0.1, 0.2])
    ead = pd.Series([10.0, 20.0])
    expected_loss(probability, 0.5, ead)
    np.testing.assert_array_equal(probability, [0.1, 0.2])
    pd.testing.assert_series_equal(ead, pd.Series([10.0, 20.0]))


@pytest.mark.parametrize(
    ("probability", "lgd", "ead", "message"),
    [
        ([0.1, 1.2], 0.5, 100, "probability must be between 0 and 1, got 1.2"),
        ([-0.1, 0.2], 0.5, 100, "probability must be between 0 and 1, got -0.1"),
        ([0.1, 0.2], [0.5, 1.5], 100, "lgd must be between 0 and 1, got 1.5"),
        ([0.1, 0.2], 0.5, [100, -1], "ead must be finite and non-negative, got -1.0"),
        ([0.1, 0.2], 0.5, [100, np.inf], "ead must be finite and non-negative, got inf"),
        (["0.1"], 0.5, 100, "probability must be real and numeric, got dtype <U3"),
        ([0.1], True, 100, "lgd must be real and numeric, got dtype bool"),
        ([0.1], 0.5, None, "ead must be real and numeric, got dtype object"),
        ([0.1, 0.2], [0.5, 0.5, 0.5], 100, "cannot be broadcast together"),
    ],
)
def test_expected_loss_rejects_invalid_inputs(probability, lgd, ead, message):
    with pytest.raises(ValueError, match=message):
        expected_loss(probability, lgd, ead)


def test_expected_loss_rejects_mismatched_series():
    with pytest.raises(ValueError, match="probability and ead must have identical indices"):
        expected_loss(pd.Series([0.1, 0.2]), 0.5, pd.Series([1.0, 2.0], index=[1, 2]))
    with pytest.raises(ValueError, match="broadcast to the length 2 of their Series index"):
        expected_loss(pd.Series([0.1, 0.2]), np.full((3, 2), 0.5), 1.0)


def test_expected_loss_table_matches_hand_calculation():
    grades = ["A", "A", "B", "B", "B"]
    probability = [0.01, 0.03, 0.1, 0.2, 0.2]
    ead = [1000.0, 3000.0, 500.0, 0.0, 1500.0]
    table = expected_loss_table(grades, probability, 0.5, ead)
    assert table.columns.tolist() == EL_COLUMNS
    assert table.index.tolist() == ["A", "B"]
    assert table.index.name == "bin"
    assert table["total"].tolist() == [2, 3]
    assert table["exposure"].tolist() == [4000.0, 2000.0]
    # A: (0.01 * 1000 + 0.03 * 3000) / 4000; B: (0.1 * 500 + 0.2 * 1500) / 2000.
    np.testing.assert_allclose(table["exposure_weighted_probability"], [0.025, 0.175])
    np.testing.assert_allclose(table["expected_loss"], [50.0, 175.0])
    np.testing.assert_allclose(table["expected_loss_rate"], [0.0125, 0.0875])


def test_expected_loss_table_sums_to_the_expected_loss_of_each_exposure():
    rng = np.random.default_rng(9)
    groups = rng.choice(list("ABCD"), 200)
    probability = rng.uniform(0, 0.3, 200)
    lgd = rng.uniform(0.2, 0.8, 200)
    ead = rng.gamma(2.0, 500.0, 200)
    table = expected_loss_table(groups, probability, lgd, ead)
    assert table["expected_loss"].sum() == pytest.approx(expected_loss(probability, lgd, ead).sum())
    assert table["exposure"].sum() == pytest.approx(ead.sum())
    assert table["total"].sum() == 200
    group_a = groups == "A"
    assert table.loc["A", "exposure_weighted_probability"] == pytest.approx(
        np.average(probability[group_a], weights=ead[group_a])
    )


def test_expected_loss_table_rate_is_lgd_times_weighted_pd_for_one_lgd():
    rng = np.random.default_rng(8)
    groups = rng.choice(["low", "high"], 100)
    probability = rng.uniform(0, 1, 100)
    table = expected_loss_table(groups, probability, 0.45, rng.uniform(0, 100, 100))
    np.testing.assert_allclose(
        table["expected_loss_rate"], 0.45 * table["exposure_weighted_probability"]
    )


def test_expected_loss_table_orders_groups_like_group_counts():
    labels = pd.Categorical(
        ["B", None, "A", "B", "A"], categories=["C", "A", "B", "D"], ordered=True
    )
    table = expected_loss_table(pd.Series(labels), [0.1] * 5, 1.0, [1.0, 2.0, 3.0, 4.0, 5.0])
    # Category order, unused categories left out and missing values last.
    assert table.index.tolist()[:2] == ["A", "B"]
    assert pd.isna(table.index[2])
    assert table["exposure"].tolist() == [8.0, 5.0, 2.0]
    table = expected_loss_table(["y", "x", None, "y"], [0.1] * 4, 1.0, 1.0)
    assert table.index.tolist()[:2] == ["y", "x"]
    assert table["total"].tolist() == [2, 1, 1]


def test_expected_loss_table_without_exposure_has_no_rates():
    table = expected_loss_table(["A", "B"], [0.1, 0.2], 0.5, [0.0, 10.0])
    assert table["total"].tolist() == [1, 1]
    assert np.isnan(table.loc["A", "expected_loss_rate"])
    assert np.isnan(table.loc["A", "exposure_weighted_probability"])
    assert table.loc["B", "expected_loss_rate"] == pytest.approx(0.1)


def test_expected_loss_table_integer_weights_equal_repeated_rows():
    rng = np.random.default_rng(10)
    groups = rng.choice(list("ABC"), 50)
    probability = rng.uniform(0, 0.5, 50)
    lgd = rng.uniform(0, 1, 50)
    ead = rng.gamma(2.0, 100.0, 50)
    weights = rng.integers(0, 4, 50)
    weights[groups == "C"] = 0
    repeat = np.repeat(np.arange(50), weights)
    weighted = expected_loss_table(groups, probability, lgd, ead, weights=weights)
    repeated = expected_loss_table(groups[repeat], probability[repeat], lgd[repeat], ead[repeat])
    # The group of zero weight is left out, as in group_counts.
    assert "C" not in weighted.index
    pd.testing.assert_frame_equal(weighted.sort_index(), repeated.sort_index(), check_dtype=False)


def test_expected_loss_table_matches_series_by_index():
    index = [10, 11, 12]
    groups = pd.Series(["A", "B", "A"], index=index)
    probability = pd.Series([0.1, 0.2, 0.3], index=index)
    table = expected_loss_table(groups, probability, 0.5, pd.Series([1.0, 2.0, 3.0], index=index))
    pd.testing.assert_frame_equal(
        table, expected_loss_table(["A", "B", "A"], [0.1, 0.2, 0.3], 0.5, [1.0, 2.0, 3.0])
    )
    with pytest.raises(ValueError, match="groups and probability must have identical indices"):
        expected_loss_table(groups, probability.reset_index(drop=True), 0.5, 1.0)
    with pytest.raises(ValueError, match="groups and weights must have identical indices"):
        expected_loss_table(groups, probability, 0.5, 1.0, weights=pd.Series([1, 1, 1]))


@pytest.mark.parametrize(
    ("groups", "probability", "lgd", "ead", "message"),
    [
        ([], [], 0.5, 1.0, "groups and probability must not be empty"),
        (["A", "B"], [0.1, np.nan], 0.5, 1.0, "probability must not have missing values, got 1"),
        (["A", "B"], [0.1, 0.2], [0.5, np.nan], 1.0, "lgd must not have missing values"),
        (["A", "B"], [0.1, 0.2], 0.5, [1.0, np.nan], "ead must not have missing values"),
        (["A", "B"], [0.1, 0.2], [0.5] * 3, 1.0, r"lgd must be one value or one per exposure"),
        (["A", "B"], [0.1, 0.2, 0.3], 0.5, 1.0, "probability must be one value or one per"),
        (["A", "B"], 0.1, 0.5, 1.0, "probability must be one-dimensional"),
        (["A", "B"], [0.1, 2.0], 0.5, 1.0, "probability must be between 0 and 1, got 2.0"),
        (["A", "B"], [0.1, 0.2], 0.5, [1.0, -3.0], "ead must be finite and non-negative"),
    ],
)
def test_expected_loss_table_rejects_invalid_inputs(groups, probability, lgd, ead, message):
    with pytest.raises(ValueError, match=message):
        expected_loss_table(groups, probability, lgd, ead)


def test_expected_loss_table_rejects_invalid_weights():
    with pytest.raises(ValueError, match="weights must have a finite, positive total"):
        expected_loss_table(["A", "B"], [0.1, 0.2], 0.5, 1.0, weights=[0, 0])
