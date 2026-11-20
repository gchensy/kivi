import math
import re

import numpy as np
import pandas as pd
import pytest

from kivi import _logit
from kivi.datasets import Dataset
from kivi.selection import correlation_filter, logistic_regression, stepwise, vif
from kivi.woe import woe_table, woe_transform

# Two groups of four: one event at x = 0 and three at x = 1, so the odds are 1/3 and 3.
BINARY_X = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
BINARY_Y = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=float)


def two_sided(z):
    return math.erfc(abs(z) / math.sqrt(2))


def with_intercept(*columns):
    return np.column_stack([np.ones(len(columns[0])), *columns])


@pytest.fixture(scope="module")
def bank_woe():
    """WOE of the categorical bank variables, fitted on the whole sample."""
    data = Dataset.bank_data()
    columns = ["poutcome", "contact", "housing", "loan", "marital", "default"]
    X = pd.DataFrame(
        {
            column: woe_transform(data[column], woe_table(data[column], data["target"]))
            for column in columns
        }
    )
    return X, data["target"]


@pytest.fixture(scope="module")
def previous_campaign():
    """Three correlated WOE variables; previous gets a negative coefficient next to the others."""
    development = Dataset.bank_data().iloc[:3000]
    groups = {
        "poutcome": development["poutcome"],
        "pdays": pd.cut(development["pdays"], [-np.inf, -1, 100, 200, 300, np.inf]),
        "previous": pd.cut(development["previous"], [-np.inf, 0, 1, 3, np.inf]),
    }
    X = pd.DataFrame(
        {
            column: woe_transform(values, woe_table(values, development["target"]))
            for column, values in groups.items()
        }
    )
    return X, development["target"]


# --- _logit -------------------------------------------------------------------------------


def test_binary_predictor_slope_is_log_odds_ratio():
    fit = _logit.fit(with_intercept(BINARY_X), BINARY_Y)
    assert fit.converged
    np.testing.assert_allclose(fit.coef, [math.log(1 / 3), math.log(9)], rtol=1e-12)
    # Woolf: the variance of a log odds is the sum of the reciprocal cell counts.
    np.testing.assert_allclose(fit.se, [math.sqrt(1 + 1 / 3), math.sqrt(1 + 1 / 3 + 1 / 3 + 1)])
    np.testing.assert_allclose(fit.cov[0, 1], -(1 + 1 / 3))
    np.testing.assert_allclose(fit.z, fit.coef / fit.se)
    np.testing.assert_allclose(fit.p_value, [two_sided(z) for z in fit.z])
    np.testing.assert_allclose(fit.p_value[1], 0.1785, atol=1e-4)  # z = 1.3455
    np.testing.assert_allclose(fit.log_likelihood, 2 * math.log(0.25) + 6 * math.log(0.75))


def test_intercept_only_model_is_the_log_odds():
    target = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
    fit = _logit.fit(np.ones((8, 1)), target)
    np.testing.assert_allclose(fit.coef, [math.log(3 / 5)])
    np.testing.assert_allclose(fit.se, [math.sqrt(1 / 3 + 1 / 5)])
    np.testing.assert_allclose(fit.log_likelihood, 3 * math.log(3 / 8) + 5 * math.log(5 / 8))


def test_frequency_weights_equal_repeated_rows():
    weights = np.array([3, 1, 1, 3], dtype=float)
    weighted = _logit.fit(
        with_intercept([0.0, 0.0, 1.0, 1.0]), np.array([0, 1, 0, 1.0]), weights=weights
    )
    repeated = _logit.fit(with_intercept(BINARY_X), BINARY_Y)
    np.testing.assert_allclose(weighted.coef, repeated.coef)
    np.testing.assert_allclose(weighted.se, repeated.se)
    np.testing.assert_allclose(weighted.log_likelihood, repeated.log_likelihood)


def test_warm_start_reaches_the_same_estimates():
    cold = _logit.fit(with_intercept(BINARY_X), BINARY_Y)
    warm = _logit.fit(with_intercept(BINARY_X), BINARY_Y, start=np.array([-1.0, 2.0]))
    np.testing.assert_allclose(warm.coef, cold.coef, rtol=1e-12)
    assert warm.n_iter < cold.n_iter


def test_non_convergence_warns():
    with pytest.warns(RuntimeWarning, match="did not converge after 2 iterations"):
        fit = _logit.fit(with_intercept(BINARY_X), BINARY_Y, max_iter=2)
    assert not fit.converged and fit.n_iter == 2


@pytest.mark.parametrize(
    "x,y",
    [
        ([0, 0, 1, 1], [0, 0, 1, 1]),  # complete separation
        ([0, 0, 1, 1, 1], [0, 0, 0, 1, 1]),  # quasi-complete: x = 0 has no events
        ([1, 2, 3, 4], [1, 1, 0, 0]),
    ],
)
def test_separation_raises(x, y):
    with pytest.raises(_logit.SeparationError, match="maximum likelihood estimates do not exist"):
        _logit.fit(with_intercept(np.array(x, dtype=float)), np.array(y, dtype=float))


def test_dependent_column():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert _logit.dependent_column(with_intercept(x, [1.0, 0.0, 1.0, 5.0])) is None
    assert _logit.dependent_column(with_intercept(x, 2 * x)) == 2
    assert _logit.dependent_column(with_intercept(x, np.full(4, 7.0))) == 2
    assert _logit.dependent_column(with_intercept(x, [1.0, 0.0, 1.0, 5.0], x + 1)) == 3
    assert _logit.dependent_column(np.ones((2, 3))) == 1
    assert _logit.dependent_column(with_intercept([1.0, 2.0], [3.0, 5.0])) == 2


# --- logistic_regression ------------------------------------------------------------------


def test_logistic_regression_table():
    result = logistic_regression(pd.DataFrame({"x": BINARY_X}), pd.Series(BINARY_Y))
    assert result.index.tolist() == ["intercept", "x"]
    assert result.columns.tolist() == ["coef", "se", "z", "p_value"]
    np.testing.assert_allclose(result["coef"], [math.log(1 / 3), math.log(9)])
    np.testing.assert_allclose(result["se"], [math.sqrt(4 / 3), math.sqrt(8 / 3)])
    np.testing.assert_allclose(result["z"], result["coef"] / result["se"])
    np.testing.assert_allclose(result["p_value"], [two_sided(z) for z in result["z"]])


def test_single_woe_variable_has_coefficient_one():
    # Every group has both classes, so no cell is smoothed and the fit is saturated.
    groups = pd.Series(["a", "a", "a", "b", "b", "b", "b", "c", "c", "c"])
    target = pd.Series([1, 1, 0, 1, 0, 0, 0, 1, 0, 0])
    woe = woe_transform(groups, woe_table(groups, target))
    result = logistic_regression(woe.to_frame("woe"), target)
    # The intercept is the log odds of the whole sample: 4 events and 6 non-events.
    np.testing.assert_allclose(result["coef"], [math.log(4 / 6), 1.0], atol=1e-12)


def test_logistic_regression_accepts_array_like():
    X = np.column_stack([BINARY_X, [1, 0, 2, 1, 0, 2, 1, 0]])
    result = logistic_regression(X.tolist(), BINARY_Y.astype(int).tolist())
    assert result.index.tolist() == ["intercept", 0, 1]
    frame = logistic_regression(pd.DataFrame(X, columns=["a", "b"]), pd.Series(BINARY_Y))
    np.testing.assert_allclose(result.to_numpy(), frame.to_numpy())


def test_logistic_regression_matches_by_index_and_position():
    X = pd.DataFrame({"x": BINARY_X}, index=range(10, 18))
    expected = logistic_regression(X, pd.Series(BINARY_Y, index=X.index))
    by_position = logistic_regression(X, BINARY_Y)
    pd.testing.assert_frame_equal(by_position, expected)
    with pytest.raises(ValueError, match="X and target must have identical indices"):
        logistic_regression(X, pd.Series(BINARY_Y))


def test_logistic_regression_does_not_mutate_inputs():
    X = pd.DataFrame({"x": BINARY_X, "w": [1, 0, 2, 1, 0, 2, 1, 0]})
    target = pd.Series(BINARY_Y)
    before = X.copy(), target.copy()
    logistic_regression(X, target)
    stepwise(X, target, p_enter=0.5, p_remove=0.5)
    pd.testing.assert_frame_equal(X, before[0])
    pd.testing.assert_series_equal(target, before[1])


@pytest.mark.parametrize(
    "X,message",
    [
        ([1.0, 2.0, 3.0], "X must be two-dimensional, got 1 dimensions"),
        (pd.DataFrame(index=range(8)), "X must have at least one column"),
        (
            pd.DataFrame([[1.0, 2.0]] * 8, columns=["a", "a"]),
            "X must have unique column labels, got 'a' more than once",
        ),
        (
            pd.DataFrame({"g": list("abababab")}),
            "X column 'g' must be numeric, such as WOE values from woe_transform, got dtype",
        ),
        (
            pd.DataFrame({"a": BINARY_X, "b": [0, 1, 2, np.nan, 4, 5, 6, 7]}),
            "X column 'b' must not contain missing or infinite values",
        ),
        (
            pd.DataFrame({"a": BINARY_X, "b": [0, 1, 2, np.inf, 4, 5, 6, 7]}),
            "X column 'b' must not contain missing or infinite values",
        ),
        (
            pd.DataFrame({"a": BINARY_X, "c": np.full(8, 0.5)}),
            "X column 'c' is constant; it cannot be told apart from the intercept",
        ),
        (
            pd.DataFrame({"a": BINARY_X, "b": np.arange(8.0), "c": BINARY_X - np.arange(8.0)}),
            "X column 'c' is a linear combination of the intercept and the columns before it",
        ),
        (
            pd.DataFrame({"intercept": BINARY_X}),
            "X must not have a column labelled 'intercept'",
        ),
    ],
)
def test_invalid_X(X, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        logistic_regression(X, BINARY_Y)


@pytest.mark.parametrize(
    "target,message",
    [
        (np.zeros(8), "target must contain both 0 and 1 and no missing labels"),
        ([0, 1, 2, 0, 1, 2, 0, 1], "target must contain both 0 and 1 and no missing labels"),
        ([0, 1, None, 0, 1, 1, 0, 1], "target must contain both 0 and 1 and no missing labels"),
        (np.zeros((8, 1)), "target must be one-dimensional, got 2 dimensions"),
        ([0, 1, 0], "X and target must have the same number of rows, got 8 and 3"),
    ],
)
def test_invalid_target(target, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        logistic_regression(pd.DataFrame({"x": BINARY_X}), target)


def test_logistic_regression_rejects_separated_classes():
    X = pd.DataFrame({"x1": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1], "x2": [1, 2, 3, 4, 5, 6, 1, 3, 2, 5]})
    target = [0, 1, 0, 0, 1, 0, 1, 1, 1, 1]  # every row with x1 = 1 is an event
    with pytest.raises(ValueError, match="the classes of target are separated by X"):
        logistic_regression(X, target)


# --- vif ----------------------------------------------------------------------------------


def test_vif_of_two_columns_is_one_over_one_minus_r_squared():
    # Centered: (-2, -1, 0, 1, 2) and (-2, -1, 0, 2, 1), so r = 9 / 10.
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [1.0, 2.0, 3.0, 5.0, 4.0]})
    result = vif(X)
    assert result.name == "vif" and result.index.tolist() == ["a", "b"]
    np.testing.assert_allclose(result, 1 / (1 - 0.9**2))


def test_vif_equals_diagonal_of_inverse_correlation():
    X = pd.DataFrame(
        {
            "a": [1.0, 4.0, 2.0, 8.0, 5.0, 7.0],
            "b": [2.0, 3.0, 1.0, 9.0, 4.0, 4.0],
            "c": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        }
    )
    np.testing.assert_allclose(vif(X), np.diagonal(np.linalg.inv(X.corr().to_numpy())))


def test_vif_of_orthogonal_and_dependent_columns():
    orthogonal = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]], dtype=float)
    result = vif(orthogonal)
    assert result.index.tolist() == [0, 1, 2]
    np.testing.assert_allclose(result, 1.0)
    dependent = pd.DataFrame({"a": [1.0, 2.0, 4.0, 3.0], "b": [1.0, 0.0, 1.0, 5.0]})
    dependent["c"] = 2 * dependent["a"] - dependent["b"]
    np.testing.assert_equal(vif(dependent).to_numpy(), np.inf)
    np.testing.assert_equal(vif(dependent[["a"]]).to_numpy(), [1.0])


def test_vif_rejects_constant_column():
    with pytest.raises(
        ValueError, match=re.escape("X column 'c' is constant; its VIF is undefined")
    ):
        vif(pd.DataFrame({"a": [1.0, 2.0, 3.0], "c": [1.0, 1.0, 1.0]}))


# --- correlation_filter -------------------------------------------------------------------

# a and c are uncorrelated with equal variance and b = a + c, so b has correlation 1/sqrt(2)
# with each of them.
CHAIN = pd.DataFrame(
    {"a": [1.0, -1.0, 1.0, -1.0], "b": [2.0, 0.0, 0.0, -2.0], "c": [1.0, 1.0, -1.0, -1.0]}
)


def test_correlation_filter_drops_only_for_kept_columns():
    np.testing.assert_allclose(
        CHAIN.corr().loc["a", ["b", "c"]], [1 / math.sqrt(2), 0.0], atol=1e-12
    )
    assert correlation_filter(CHAIN).tolist() == ["a", "c"]
    assert correlation_filter(CHAIN, threshold=0.71).tolist() == ["a", "b", "c"]
    assert correlation_filter(CHAIN, priority=[1, 2, 1]).tolist() == ["b"]


def test_correlation_filter_priority_by_label_and_position():
    by_label = pd.Series({"c": 0.3, "b": 0.2, "a": 0.1})
    assert correlation_filter(CHAIN, priority=by_label).tolist() == ["a", "c"]
    assert correlation_filter(
        CHAIN, priority=pd.Series({"b": 0.5, "a": 0.1, "c": 0.3})
    ).tolist() == ["b"]
    assert correlation_filter(CHAIN, priority=np.array([0.1, 0.5, 0.3])).tolist() == ["b"]
    # The result follows the column order of X, not the priority.
    assert correlation_filter(CHAIN[["c", "b", "a"]], priority=by_label).tolist() == ["c", "a"]


def test_correlation_filter_ties_keep_the_earlier_column():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.5]})
    assert correlation_filter(X, priority=[1.0, 1.0]).tolist() == ["x"]
    assert correlation_filter(X[["y", "x"]], priority=[1.0, 1.0]).tolist() == ["y"]


def test_correlation_filter_uses_absolute_correlation():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "minus": [-1.0, -2.0, -3.0, -4.5]})
    assert correlation_filter(X).tolist() == ["x"]
    assert correlation_filter(X[["x"]]).tolist() == ["x"]
    assert correlation_filter(X.to_numpy()).tolist() == [0]


@pytest.mark.parametrize("threshold", [-0.1, 1.5, True, "0.7", np.nan])
def test_correlation_filter_rejects_threshold(threshold):
    with pytest.raises(
        ValueError, match=re.escape(f"threshold must be between 0 and 1, got {threshold!r}")
    ):
        correlation_filter(CHAIN, threshold=threshold)


@pytest.mark.parametrize(
    "priority,message",
    [
        (pd.Series({"a": 1.0, "b": 2.0}), "priority has no value for X column 'c'"),
        (
            pd.Series([1.0, 2.0, 3.0], index=["a", "a", "b"]),
            "priority must have one value per column label",
        ),
        ([1.0, 2.0], "priority must have one value per column of X, got 2 values for 3 columns"),
        ([1.0, np.nan, 2.0], "priority must contain numbers only, without missing values"),
        (["x", "y", "z"], "priority must contain numbers only, without missing values"),
        (1.0, "priority must be one-dimensional, got 0 dimensions"),
    ],
)
def test_correlation_filter_rejects_priority(priority, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        correlation_filter(CHAIN, priority=priority)


def test_correlation_filter_rejects_constant_column():
    with pytest.raises(ValueError, match="X column 'k' is constant; its correlation is undefined"):
        correlation_filter(CHAIN.assign(k=1.0))


# --- stepwise -----------------------------------------------------------------------------


def assert_stepwise_fixpoint(X, target, selected, p_enter, p_remove):
    """Every selected variable stays and no other variable can enter."""
    model = logistic_regression(X[selected], target).drop("intercept")
    assert (model["p_value"] <= p_remove).all() and (model["coef"] > 0).all()
    for candidate in X.columns.difference(selected):
        extended = logistic_regression(X[[*selected, candidate]], target).drop("intercept")
        assert extended.loc[candidate, "p_value"] >= p_enter or (extended["coef"] <= 0).any()


@pytest.mark.parametrize("direction", ["both", "forward", "backward"])
def test_stepwise_on_bank_data(bank_woe, direction):
    X, target = bank_woe
    selected = stepwise(X, target, direction=direction)
    assert selected.tolist() == ["poutcome", "contact", "housing", "loan", "marital"]
    assert_stepwise_fixpoint(X, target, selected, 0.05, 0.10)
    # default is the only variable left out, and only for its p-value.
    full = logistic_regression(X, target)
    assert full.loc["default", "coef"] > 0 and full.loc["default", "p_value"] > 0.10


def test_positive_rules_out_negative_coefficients(previous_campaign):
    X, target = previous_campaign
    full = logistic_regression(X, target)
    np.testing.assert_allclose(
        full.loc["previous", ["coef", "p_value"]], [-0.0950, 0.6025], atol=5e-5
    )
    for direction in ["both", "forward", "backward"]:
        lenient = {"p_enter": 0.7, "p_remove": 0.7, "direction": direction}
        assert stepwise(X, target, **lenient, positive=False).tolist() == list(X.columns)
        assert stepwise(X, target, **lenient).tolist() == ["poutcome", "pdays"]
    # With the default levels, previous fails on its p-value alone.
    assert stepwise(X, target, positive=False).tolist() == ["poutcome", "pdays"]


def test_dependent_candidate_does_not_enter():
    X = pd.DataFrame(
        {"x": BINARY_X, "copy": BINARY_X, "other": [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]}
    )
    target = BINARY_Y
    lenient = {"p_enter": 0.5, "p_remove": 0.5}
    assert stepwise(X, target, **lenient, direction="forward").tolist() == ["x"]
    assert stepwise(X[["copy", "x"]], target, **lenient).tolist() == ["copy"]
    with pytest.raises(ValueError, match="X column 'copy' is a linear combination"):
        stepwise(X, target, direction="backward")


def test_separating_candidate_does_not_enter():
    X = pd.DataFrame({"x1": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1], "x2": [1, 2, 3, 4, 5, 6, 1, 3, 2, 5]})
    target = [0, 1, 0, 0, 1, 0, 1, 1, 1, 1]
    assert stepwise(X, target, p_enter=0.9, p_remove=0.9, positive=False).tolist() == ["x2"]


def test_stepwise_may_select_nothing():
    selected = stepwise(np.array([[1.0], [2.0], [1.0], [2.0]]), [0, 0, 1, 1])
    assert selected.tolist() == []
    assert (
        stepwise(
            np.array([[1.0], [2.0], [1.0], [2.0]]), [0, 0, 1, 1], direction="backward"
        ).tolist()
        == []
    )


def test_stepwise_returns_positions_for_arrays(bank_woe):
    X, target = bank_woe
    assert stepwise(X.to_numpy(), target.to_numpy()).tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "options,message",
    [
        ({"direction": "up"}, "direction must be 'forward', 'backward' or 'both', got 'up'"),
        ({"p_enter": 0}, "p_enter must be between 0 and 1, got 0"),
        ({"p_enter": 1.5}, "p_enter must be between 0 and 1, got 1.5"),
        ({"p_remove": True}, "p_remove must be between 0 and 1, got True"),
        ({"p_enter": 0.2}, "p_enter must not exceed p_remove, got p_enter=0.2 and p_remove=0.1"),
    ],
)
def test_stepwise_rejects_options(options, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        stepwise(pd.DataFrame({"x": BINARY_X}), BINARY_Y, **options)


def test_p_enter_may_exceed_p_remove_in_one_direction():
    X = pd.DataFrame({"x": BINARY_X})
    assert stepwise(X, BINARY_Y, p_enter=0.2, direction="forward").tolist() == ["x"]
    assert stepwise(X, BINARY_Y, p_enter=0.2, direction="backward").tolist() == []
