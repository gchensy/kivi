import functools
import math
import re

import numpy as np
import pandas as pd
import pytest

from kivi import _logit, selection
from kivi.datasets import Dataset
from kivi.selection import (
    SeparationError,
    correlation_filter,
    logistic_regression,
    stepwise,
    vif,
)
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


def test_non_convergence_is_reported():
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


def test_singular_information_names_the_column():
    # The rank check is bypassed, so the duplicate column makes the information singular.
    with pytest.raises(_logit.SingularError) as raised:
        _logit.fit(with_intercept(BINARY_X, BINARY_X), BINARY_Y)
    assert raised.value.column == 2


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
        (pd.DataFrame({"x": [1.0]}), "X must have at least 2 rows, got 1"),
        (np.empty((0, 2)), "X must have at least 2 rows, got 0"),
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


@pytest.mark.parametrize(
    "offset,scale", [(1e8, 1.0), (1.7e9, 1.0), (0.0, 1e-6), (0.0, 1e9), (-5e3, 1e3)]
)
def test_offset_and_scale_do_not_affect_the_fit(offset, scale):
    # x' = offset + scale * x turns (b0, b1) into (b0 - b1 * o, b1 / scale) with o = offset / scale
    result = logistic_regression(pd.DataFrame({"x": offset + scale * BINARY_X}), BINARY_Y)
    o = offset / scale
    np.testing.assert_allclose(result.loc["x", "coef"], math.log(9) / scale, rtol=1e-9)
    np.testing.assert_allclose(result.loc["x", "se"], math.sqrt(8 / 3) / scale, rtol=1e-9)
    np.testing.assert_allclose(
        result.loc["intercept", "coef"], math.log(1 / 3) - o * math.log(9), rtol=1e-7, atol=1e-9
    )
    variance = 4 / 3 + o**2 * 8 / 3 + 2 * o * 4 / 3  # cov(b0, b1) = -4/3
    np.testing.assert_allclose(result.loc["intercept", "se"], math.sqrt(variance), rtol=1e-7)
    assert result.attrs["converged"]


def test_late_event_is_not_mistaken_for_separation():
    day = np.arange(300) % 60
    target = ((np.arange(300) % 5 == 0) | ((day > 40) & (np.arange(300) % 2 == 0))).astype(int)
    day, target = np.r_[day, 3000], np.r_[target, 1]  # one event far beyond the others
    plain = logistic_regression(pd.DataFrame({"t": day}), target)
    shifted = logistic_regression(pd.DataFrame({"t": 1.7e9 + day}), target)
    assert plain.loc["intercept", "coef"] + 3000 * plain.loc["t", "coef"] > 23
    np.testing.assert_allclose(shifted.loc["t"], plain.loc["t"], rtol=1e-8)


def test_matches_reference_values(bank_woe):
    # statsmodels 0.15 Logit on the same columns with a constant, tol=1e-12.
    X, target = bank_woe
    result = logistic_regression(X, target)
    coef = [-2.0374414462, 0.8864965878, 0.6889510651, 0.6940720001, 0.8345383078,
            0.9916740814, 10.6841200682]  # fmt: skip
    se = [0.050620481, 0.0667198009, 0.116834871, 0.1529484439, 0.2211736644,
          0.2444114846, 11.6802532017]  # fmt: skip
    np.testing.assert_allclose(result["coef"], coef, rtol=1e-9)
    np.testing.assert_allclose(result["se"], se, rtol=1e-8)
    np.testing.assert_allclose(result.attrs["log_likelihood"], -1441.8533694068324, rtol=1e-12)
    raw = Dataset.bank_data()[["age", "balance", "campaign"]].astype(float)
    result = logistic_regression(raw, target)
    np.testing.assert_allclose(
        result["coef"],
        [-2.3391249238, 1.2606588132e-02, 1.1975021338e-05, -9.6634873332e-02],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        result["se"],
        [1.9304485977e-01, 4.2723759827e-03, 1.3859155487e-05, 2.3319717764e-02],
        rtol=1e-8,
    )


def test_labels_and_attrs():
    X = pd.DataFrame({"x": BINARY_X, "w": [1, 0, 2, 1, 0, 2, 1, 0]})
    X.columns.name = "variable"
    result = logistic_regression(X, BINARY_Y)
    assert result.index.name == "variable" and result.index.tolist() == ["intercept", "x", "w"]
    assert result.attrs["converged"] is True
    single = logistic_regression(X[["x"]], BINARY_Y)
    np.testing.assert_allclose(
        single.attrs["log_likelihood"], 2 * math.log(0.25) + 6 * math.log(0.75)
    )
    assert vif(X).index.name == "variable"
    assert correlation_filter(X).name == "variable"
    assert stepwise(X, BINARY_Y, p_enter=0.5, p_remove=0.5).name == "variable"


def test_non_convergence_warns_at_the_callers_line(monkeypatch):
    monkeypatch.setattr(_logit, "fit", functools.partial(_logit.fit, max_iter=2))
    X = pd.DataFrame({"x": BINARY_X})
    with pytest.warns(RuntimeWarning, match="did not converge after 2 iterations") as record:
        result = logistic_regression(X, BINARY_Y)
    assert record[0].filename == __file__ and result.attrs["converged"] is False
    with pytest.warns(RuntimeWarning, match="did not converge") as record:
        stepwise(X, BINARY_Y, direction="backward")
    assert record[0].filename == __file__
    # A candidate whose model does not converge never enters.
    assert stepwise(X, BINARY_Y, p_enter=0.2, direction="forward").tolist() == []


def test_nearly_singular_fit_names_the_column(monkeypatch):
    def singular(*args, **kwargs):
        raise _logit.SingularError(2)

    monkeypatch.setattr(_logit, "fit", singular)
    X = pd.DataFrame({"a": BINARY_X, "b": [1, 0, 2, 1, 0, 2, 1, 0]})
    with pytest.raises(ValueError, match="X column 'b' is so close to a linear combination"):
        logistic_regression(X, BINARY_Y)


def test_separation_error_is_public():
    assert issubclass(SeparationError, ValueError)
    assert SeparationError.__module__ == "kivi.selection" and "SeparationError" in selection.__all__
    with pytest.raises(SeparationError):
        logistic_regression(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}), [0, 0, 1, 1])


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
    np.testing.assert_allclose(vif(dependent[["a"]]).to_numpy(), [1.0])


def test_vif_rejects_constant_column():
    with pytest.raises(
        ValueError, match=re.escape("X column 'c' is constant; its VIF is undefined")
    ):
        vif(pd.DataFrame({"a": [1.0, 2.0, 3.0], "c": [1.0, 1.0, 1.0]}))


def test_vif_is_infinite_for_exact_combinations_at_any_scale():
    a = np.arange(10.0)
    b = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0])
    np.testing.assert_equal(
        vif(pd.DataFrame({"a": a, "b": b, "c": 1e3 * a + b})).to_numpy(), np.inf
    )
    three = np.array([[1.0, 2.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 3.0]])  # centred rank 2
    np.testing.assert_equal(vif(three).to_numpy(), np.inf)
    # d takes no part in the dependence among a, b and c, so its VIF stays finite.
    d = np.array([0.0, 2.0, 1.0, 4.0, 3.0, 3.0, 6.0, 5.0, 9.0, 7.0])
    X = pd.DataFrame({"a": a, "b": b, "c": a + b, "d": d})
    result = vif(X)
    np.testing.assert_equal(result[["a", "b", "c"]].to_numpy(), np.inf)
    np.testing.assert_allclose(result["d"], vif(X[["a", "b", "d"]])["d"], rtol=1e-10)


def test_vif_is_infinite_for_combinations_with_a_small_coefficient():
    # a = (c - b) / 1e-5, so all three are combinations; a used to get a VIF of 1.003.
    a, b = np.random.default_rng(0).normal(size=(2, 100))
    np.testing.assert_equal(
        vif(pd.DataFrame({"a": a, "b": b, "c": b + 1e-5 * a})).to_numpy(), np.inf
    )
    # d takes no part, and its VIF is that of the regression on a and b alone.
    d = np.random.default_rng(1).normal(size=100) + a
    result = vif(pd.DataFrame({"a": a, "b": b, "c": b + 1e-5 * a, "d": d}))
    np.testing.assert_equal(result[["a", "b", "c"]].to_numpy(), np.inf)
    expected = vif(pd.DataFrame({"a": a, "b": b, "d": d}))["d"]
    np.testing.assert_allclose(result["d"], expected, rtol=1e-10)


def test_vif_is_infinite_where_r_squared_rounds_to_one():
    # Two rows make two columns collinear; the VIFs used to be 5.6e29.
    two_rows = pd.DataFrame({"a": [-0.7, -0.6], "b": [-1.3, 0.0]})
    np.testing.assert_equal(vif(two_rows).to_numpy(), np.inf)
    # With fewer rows than columns, a column outside the span of the others stays finite:
    # c has a squared correlation of 1 / 28 with a, b and e, which are proportional.
    X = pd.DataFrame({"a": [1.0, 2, 4], "b": [2.0, 4, 8], "e": [3.0, 6, 12], "c": [0.0, 1, 0]})
    result = vif(X)
    np.testing.assert_equal(result[["a", "b", "e"]].to_numpy(), np.inf)
    assert result["c"] == pytest.approx(28 / 27, rel=1e-14)
    # A near combination whose R^2 is distinguishable from 1 keeps a finite VIF.
    a, b, noise = np.random.default_rng(2).normal(size=(3, 50))
    near = vif(pd.DataFrame({"a": a, "b": b, "c": a + b + 1e-6 * noise})).to_numpy()
    assert np.isfinite(near).all() and (near > 1e11).all()


@pytest.mark.parametrize("function", [vif, correlation_filter])
def test_vif_and_filter_need_two_rows(function):
    for rows in (0, 1):
        with pytest.raises(ValueError, match=f"X must have at least 2 rows, got {rows}"):
            function(pd.DataFrame({"a": np.zeros(rows), "b": np.ones(rows)}))


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


@pytest.mark.parametrize("direction", ["both", "forward", "backward"])
def test_stepwise_rejects_dependent_columns_in_every_direction(direction):
    X = pd.DataFrame({"x": BINARY_X, "copy": BINARY_X})
    with pytest.raises(ValueError, match="X column 'copy' is a linear combination"):
        stepwise(X, BINARY_Y, direction=direction)


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
        ({"p_enter": 0}, "p_enter must be greater than 0 and at most 1, got 0"),
        ({"p_enter": 1.5}, "p_enter must be greater than 0 and at most 1, got 1.5"),
        ({"p_remove": True}, "p_remove must be greater than 0 and at most 1, got True"),
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


# Found by search: in both directions with p_enter = p_remove = 0.7, v0 enters, v1 enters,
# v0 leaves, v2 enters, v1 and v2 leave, and v0 would enter again and repeat the cycle.
CYCLE_X = pd.DataFrame(
    {
        "v0": [-1.5, 6.0, -4.3, 7.7, 3.5, -2.2, 1.6, -7.8, 0.7, -7.7, -5.7, -3.6],
        "v1": [0.9, -4.3, 0.9, -7.4, -1.8, -4.2, 1.4, 5.0, 0.3, 7.8, 3.0, 2.5],
        "v2": [-2.4, -1.8, -2.5, 3.1, -2.4, -3.7, 1.0, -6.0, 2.5, -4.1, -2.2, -2.4],
    }
)
CYCLE_Y = [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]


def test_stepwise_blocks_reentry_into_a_visited_set(monkeypatch):
    trace = []
    enter, remove = selection._enter, selection._remove

    def traced_enter(design, labels, mass, selected, visited, *rest):
        entering = enter(design, labels, mass, selected, visited, *rest)
        unguarded = enter(design, labels, mass, selected, set(), *rest)
        trace.append(
            ("enter", tuple(selected), entering and entering[0], unguarded and unguarded[0])
        )
        return entering

    def traced_remove(design, labels, mass, columns, selected, *rest):
        before = tuple(selected)
        fitted = remove(design, labels, mass, columns, selected, *rest)
        if before != tuple(selected):
            trace.append(("remove", before, tuple(selected)))
        return fitted

    monkeypatch.setattr(selection, "_enter", traced_enter)
    monkeypatch.setattr(selection, "_remove", traced_remove)
    selected = stepwise(CYCLE_X, CYCLE_Y, p_enter=0.7, p_remove=0.7, positive=False)
    assert selected.tolist() == []
    assert trace == [
        ("enter", (), 0, 0),
        ("enter", (0,), 1, 1),
        ("remove", (0, 1), (1,)),
        ("enter", (1,), 2, 2),
        ("remove", (1, 2), ()),
        ("enter", (), None, 0),  # v0 is eligible again, but {v0} was visited
    ]


@pytest.mark.parametrize("positive", ["no", 1, None])
def test_stepwise_rejects_non_boolean_positive(positive):
    with pytest.raises(
        ValueError, match=re.escape(f"positive must be True or False, got {positive!r}")
    ):
        stepwise(pd.DataFrame({"x": BINARY_X}), BINARY_Y, positive=positive)
    assert stepwise(pd.DataFrame({"x": BINARY_X}), BINARY_Y, positive=np.True_).tolist() == []


# --- weights ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def repeated(bank_woe):
    """Integer weights, some zero, and the rows they repeat."""
    X, target = bank_woe
    weights = np.random.default_rng(0).integers(0, 4, len(X))
    rows = np.repeat(np.arange(len(X)), weights)
    return weights, X.iloc[rows], target.iloc[rows]


def test_integer_weights_equal_repeated_rows(bank_woe, repeated):
    X, target = bank_woe
    weights, X_repeated, target_repeated = repeated
    columns = ["poutcome", "contact", "housing", "loan", "marital"]
    weighted = logistic_regression(X[columns], target, weights=weights)
    expected = logistic_regression(X_repeated[columns], target_repeated)
    pd.testing.assert_frame_equal(weighted, expected, rtol=1e-10)
    assert weighted.attrs["log_likelihood"] == pytest.approx(
        expected.attrs["log_likelihood"], rel=1e-12
    )


@pytest.mark.parametrize("direction", ["both", "forward", "backward"])
@pytest.mark.parametrize("positive", [True, False])
def test_stepwise_integer_weights_equal_repeated_rows(bank_woe, repeated, direction, positive):
    X, target = bank_woe
    weights, X_repeated, target_repeated = repeated
    options = {"direction": direction, "positive": positive, "p_enter": 0.1, "p_remove": 0.2}
    selected = stepwise(X, target, weights=weights, **options)
    assert selected.equals(stepwise(X_repeated, target_repeated, **options))
    assert len(selected) >= 4


def test_fractional_weights_match_statsmodels(bank_woe):
    X, target = bank_woe
    weights = np.random.default_rng(1).random(len(X))
    # statsmodels GLM(target, add_constant(X), Binomial(), freq_weights=weights)
    # .fit(tol=1e-15, tol_criterion="params").
    fitted = logistic_regression(X[["poutcome", "contact", "housing"]], target, weights=weights)
    np.testing.assert_allclose(
        fitted["coef"],
        [-2.038247578689713, 0.8582512300427991, 0.7397672649198815, 0.7547436694071427],
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        fitted["se"],
        [0.07112065770921419, 0.09237337253191864, 0.16620454775086865, 0.21525695942682385],
        rtol=1e-10,
    )
    assert fitted.attrs["log_likelihood"] == pytest.approx(-724.5255300904082, rel=1e-12)


def test_scaling_the_weights_scales_the_variances():
    X = pd.DataFrame({"x": BINARY_X})
    unweighted = logistic_regression(X, BINARY_Y)
    doubled = logistic_regression(X, BINARY_Y, weights=np.full(8, 2.0))
    np.testing.assert_allclose(doubled["coef"], unweighted["coef"], rtol=1e-12)
    np.testing.assert_allclose(doubled["se"], unweighted["se"] / math.sqrt(2), rtol=1e-12)
    assert doubled.attrs["log_likelihood"] == pytest.approx(
        2 * unweighted.attrs["log_likelihood"], rel=1e-12
    )


def test_binary_predictor_with_cell_weights():
    # Non-events and events weigh 1.5 and 0.5 at x = 0 and 0.5 and 1.5 at x = 1, so the
    # odds are 1/3 and 3, and the Woolf variances are sums of reciprocal cell weights.
    X = pd.DataFrame({"x": [0.0, 0.0, 1.0, 1.0]})
    fitted = logistic_regression(X, [0, 1, 0, 1], weights=[1.5, 0.5, 0.5, 1.5])
    np.testing.assert_allclose(fitted["coef"], [math.log(1 / 3), math.log(9)], rtol=1e-12)
    np.testing.assert_allclose(
        fitted["se"], [math.sqrt(1 / 1.5 + 1 / 0.5), math.sqrt(2 / 1.5 + 2 / 0.5)], rtol=1e-12
    )


def test_zero_weights_equal_dropped_rows(bank_woe):
    X, target = bank_woe
    weights = np.where(np.arange(len(X)) % 3 == 0, 0.0, 1.0)
    kept = weights > 0
    columns = ["poutcome", "contact", "housing"]
    pd.testing.assert_frame_equal(
        logistic_regression(X[columns], target, weights=weights),
        logistic_regression(X.loc[kept, columns], target[kept]),
        rtol=1e-10,
    )
    assert stepwise(X, target, weights=weights).equals(stepwise(X[kept], target[kept]))


def test_columns_are_checked_on_the_rows_with_positive_weight():
    X = pd.DataFrame({"a": [1.0, 2.0, 2.0, 2.0, 2.0], "b": [0.0, 1.0, 0.0, 1.0, 1.0]})
    target = [0, 0, 1, 1, 0]
    message = "X column 'a' is constant on the rows with positive weight"
    with pytest.raises(ValueError, match=message):
        logistic_regression(X, target, weights=[0, 1, 1, 1, 1])
    with pytest.raises(ValueError, match=message):
        stepwise(X, target, weights=[0, 1, 1, 1, 1])
    X["c"] = [5.0, 1.0, 0.0, 1.0, 1.0]
    message = (
        "X column 'c' is a linear combination of the intercept and the columns before it "
        "on the rows with positive weight"
    )
    with pytest.raises(ValueError, match=message):
        logistic_regression(X[["b", "c"]], target, weights=[0, 1, 1, 1, 1])


def test_zero_weights_can_separate_the_classes():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    logistic_regression(X, [0, 1, 0, 1])
    with pytest.raises(SeparationError):
        logistic_regression(X, [0, 1, 0, 1], weights=[1, 0, 0, 1])


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([1, 1, 1], "weights must have one weight per row of X, got 3 for 8"),
        ([1, -1, 1, 1, 1, 1, 1, 1], "weights must be finite and non-negative"),
        ([1, np.nan, 1, 1, 1, 1, 1, 1], "weights must be finite and non-negative"),
        (["1"] * 8, "weights must be numeric, got dtype"),
        ([True] * 8, "weights must be numeric, got dtype bool"),
        ([0] * 8, r"weights must have a finite, positive total, got 0.0"),
        (
            [1, 1, 1, 0, 1, 0, 0, 0],
            "weights must give both target classes a positive total, got 4.0 for 0 and 0.0 for 1",
        ),
        ([[1] * 8], "weights must be one-dimensional, got 2 dimensions"),
    ],
)
def test_invalid_weights(weights, message):
    X = pd.DataFrame({"x": BINARY_X})
    for function in (logistic_regression, stepwise):
        with pytest.raises(ValueError, match=message):
            function(X, BINARY_Y, weights=weights)


def test_weights_must_have_the_index_of_X():
    X = pd.DataFrame({"x": BINARY_X})
    with pytest.raises(ValueError, match="X and weights must have identical indices"):
        logistic_regression(X, BINARY_Y, weights=pd.Series(1.0, index=range(1, 9)))
    unit = logistic_regression(X, BINARY_Y, weights=np.ones(8))
    pd.testing.assert_frame_equal(unit, logistic_regression(X, BINARY_Y))


def test_weighted_fit_does_not_mutate_inputs(bank_woe):
    X, target = bank_woe
    weights = pd.Series(np.random.default_rng(2).random(len(X)), index=X.index)
    copies = X.copy(), target.copy(), weights.copy()
    logistic_regression(X[["poutcome", "contact"]], target, weights=weights)
    stepwise(X, target, weights=weights)
    pd.testing.assert_frame_equal(X, copies[0])
    pd.testing.assert_series_equal(target, copies[1])
    pd.testing.assert_series_equal(weights, copies[2])
