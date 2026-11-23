import fractions
import math

import numpy as np
import pandas as pd
import pytest

from kivi.expert import ahp_consistency, ahp_weights, combine_weights

# Saaty's random index for orders 1 to 15.
RANDOM_INDEX = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.53, 1.56, 1.57, 1.59]

SALES_MARGIN_LEVERAGE = pd.DataFrame(
    [[1, 2, 4], [1 / 2, 1, 2], [1 / 4, 1 / 2, 1]],
    index=["sales", "margin", "leverage"],
    columns=["sales", "margin", "leverage"],
)

# A three-factor matrix with slightly inconsistent judgments: 1 against 3 is 3, 2 against 3 is 3,
# but 1 against 3 is 5 rather than 3 * 3 = 9.
THREE = np.array([[1, 3, 5], [1 / 3, 1, 3], [1 / 5, 1 / 3, 1]])

# The criteria of the school selection example, with the order learning, friends, school life,
# vocational training, college preparation, music classes. T. L. Saaty, "A scaling method for
# priorities in hierarchical structures", Journal of Mathematical Psychology 15 (1977), 234-281.
# Saaty gives the priorities .32, .14, .03, .13, .24 and .14. He reports a consistency ratio of
# 0.24 (a lambda_max of 7.49); the exact principal eigenvalue of this matrix is 7.42 (0.23). Both
# are far above the 0.1 rule of thumb, which is what the example is used to show.
SCHOOL = np.array(
    [
        [1, 4, 3, 1, 3, 4],
        [1 / 4, 1, 7, 3, 1 / 5, 1],
        [1 / 3, 1 / 7, 1, 1 / 5, 1 / 5, 1 / 6],
        [1, 1 / 3, 5, 1, 1, 1 / 3],
        [1 / 3, 5, 5, 1, 1, 3],
        [1 / 4, 1, 6, 3, 1 / 3, 1],
    ]
)
SCHOOL_PRIORITIES = [0.32, 0.14, 0.03, 0.13, 0.24, 0.14]

SAATY_SCALE = np.array(
    [1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4, 5, 6, 7, 8, 9]
)


def consistent(weights):
    """The matrix a_ij = w_i / w_j of exactly consistent judgments."""
    weights = np.asarray(weights, dtype=float)
    return weights[:, None] / weights[None, :]


def random_matrix(order, rng):
    """A random reciprocal matrix with entries from Saaty's scale."""
    upper = np.triu(rng.choice(SAATY_SCALE, size=(order, order)), k=1)
    return upper + np.diag(np.ones(order)) + np.tril(1 / np.where(upper.T == 0, 1, upper.T), k=-1)


def power_iteration(matrix, steps=2000):
    """The principal eigenvector of a positive matrix by repeated multiplication."""
    vector = np.full(len(matrix), 1 / len(matrix))
    for _ in range(steps):
        vector = matrix @ vector
        vector /= vector.sum()
    return vector


def test_weights_of_a_consistent_matrix_are_hand_calculated():
    # Sales is twice as important as margin and four times as important as leverage, so the
    # weights are 4, 2 and 1 out of 7.
    for method in ("eigenvector", "geometric"):
        weights = ahp_weights(SALES_MARGIN_LEVERAGE, method=method)
        np.testing.assert_allclose(weights, [4 / 7, 2 / 7, 1 / 7], rtol=1e-12)
        assert weights.index.tolist() == ["sales", "margin", "leverage"]
        assert weights.index.name == "factor" and weights.name == "weight"
        assert weights.sum() == pytest.approx(1, rel=1e-14)


def test_geometric_weights_are_the_normalized_row_geometric_means():
    # The row products of THREE are 15, 1 and 1 / 15.
    means = np.array([15 ** (1 / 3), 1.0, 15 ** (-1 / 3)])
    np.testing.assert_allclose(
        ahp_weights(THREE, method="geometric"), means / means.sum(), rtol=1e-14
    )
    np.testing.assert_allclose(
        ahp_weights(THREE, method="geometric"), [0.6370, 0.2583, 0.1047], atol=5e-5
    )


@pytest.mark.parametrize("order", [2, 3])
def test_the_two_methods_agree_for_up_to_three_factors(order):
    # For a reciprocal matrix of order 3 or less the row geometric means are the eigenvector.
    rng = np.random.default_rng(order)
    for _ in range(20):
        matrix = random_matrix(order, rng)
        np.testing.assert_allclose(
            ahp_weights(matrix, method="geometric"), ahp_weights(matrix), rtol=1e-12
        )


def test_the_two_methods_differ_slightly_for_an_inconsistent_matrix_of_four_factors():
    matrix = np.array(
        [[1, 3, 5, 7], [1 / 3, 1, 3, 5], [1 / 5, 1 / 3, 1, 3], [1 / 7, 1 / 5, 1 / 3, 1]]
    )
    eigenvector = ahp_weights(matrix).to_numpy()
    geometric = ahp_weights(matrix, method="geometric").to_numpy()
    np.testing.assert_allclose(eigenvector, [0.5650, 0.2622, 0.1175, 0.0553], atol=5e-5)
    np.testing.assert_allclose(geometric, [0.5638, 0.2634, 0.1178, 0.0550], atol=5e-5)
    assert not np.allclose(eigenvector, geometric, rtol=1e-6)


def test_eigenvector_weights_solve_the_eigenvalue_equation_of_a_three_factor_matrix():
    weights = ahp_weights(THREE).to_numpy()
    lambda_max = ahp_consistency(THREE)["lambda_max"]
    np.testing.assert_allclose(THREE @ weights, lambda_max * weights, rtol=1e-12)
    # lambda_max is the largest root of the characteristic polynomial, found another way.
    assert lambda_max == pytest.approx(max(np.roots(np.poly(THREE)).real), rel=1e-12)
    assert lambda_max == pytest.approx(3.0385, abs=5e-5)
    np.testing.assert_allclose(weights, [0.6370, 0.2583, 0.1047], atol=5e-5)


@pytest.mark.parametrize("order", [3, 5, 9, 15])
def test_eigenvector_weights_match_numpy_and_power_iteration(order):
    rng = np.random.default_rng(order)
    for _ in range(5):
        matrix = random_matrix(order, rng)
        values, vectors = np.linalg.eig(matrix)
        expected = vectors[:, np.argmax(values.real)].real
        expected /= expected.sum()
        weights = ahp_weights(matrix)
        np.testing.assert_allclose(weights, expected, rtol=1e-10)
        np.testing.assert_allclose(weights, power_iteration(matrix), rtol=1e-6)
        assert ahp_consistency(matrix)["lambda_max"] == pytest.approx(values.real.max(), rel=1e-12)


@pytest.mark.parametrize("order", range(1, 11))
def test_a_consistent_matrix_gives_its_weights_and_no_inconsistency(order):
    weights = np.linspace(1, 9, order) ** 1.5
    matrix = consistent(weights)
    for method in ("eigenvector", "geometric"):
        np.testing.assert_allclose(
            ahp_weights(matrix, method=method), weights / weights.sum(), rtol=1e-12
        )
    statistics = ahp_consistency(matrix)
    assert statistics["lambda_max"] == pytest.approx(order, rel=1e-12)
    assert statistics["ci"] == pytest.approx(0, abs=1e-12)
    assert statistics["cr"] == pytest.approx(0, abs=1e-12)
    assert statistics["ci"] >= 0


def test_the_consistency_index_is_never_negative():
    # Rounding can put the computed eigenvalue just below the order for a consistent matrix.
    rng = np.random.default_rng(3)
    for order in range(3, 16):
        statistics = ahp_consistency(consistent(rng.uniform(0.1, 10, size=order)))
        assert statistics["lambda_max"] >= order
        assert statistics["ci"] >= 0


def test_consistency_of_a_three_factor_matrix_is_hand_calculated():
    statistics = ahp_consistency(THREE)
    assert statistics.index.tolist() == ["lambda_max", "ci", "ri", "cr"]
    assert statistics.index.name == "statistic" and statistics.name == "value"
    lambda_max = statistics["lambda_max"]
    assert statistics["ci"] == pytest.approx((lambda_max - 3) / 2, rel=1e-14)
    assert statistics["ri"] == 0.58
    assert statistics["cr"] == pytest.approx(statistics["ci"] / 0.58, rel=1e-14)
    assert statistics["cr"] == pytest.approx(0.0332, abs=5e-5)  # below the 0.1 rule of thumb


def test_saaty_school_example_priorities():
    weights = ahp_weights(SCHOOL)
    np.testing.assert_allclose(weights.round(2), SCHOOL_PRIORITIES, atol=1e-12)
    assert weights.sum() == pytest.approx(1, rel=1e-14)
    # The geometric means give nearly the same priorities.
    np.testing.assert_allclose(ahp_weights(SCHOOL, method="geometric"), weights, atol=0.01)


def test_an_inconsistent_matrix_has_a_consistency_ratio_above_one_tenth():
    statistics = ahp_consistency(SCHOOL)
    assert statistics["lambda_max"] == pytest.approx(7.42, abs=5e-3)
    assert statistics["ri"] == 1.24
    assert statistics["ci"] == pytest.approx((statistics["lambda_max"] - 6) / 5, rel=1e-14)
    assert statistics["cr"] == pytest.approx(0.229, abs=5e-3)
    assert statistics["cr"] > 0.1


def test_matrices_of_one_and_two_factors_are_consistent():
    one = ahp_consistency([[1]])
    assert one.tolist() == [1.0, 0.0, 0.0, 0.0]
    np.testing.assert_array_equal(ahp_weights([[1]]), [1.0])
    two = ahp_consistency([[1, 7], [1 / 7, 1]])
    assert two.tolist() == [2.0, 0.0, 0.0, 0.0]
    for method in ("eigenvector", "geometric"):
        np.testing.assert_allclose(
            ahp_weights([[1, 7], [1 / 7, 1]], method=method), [7 / 8, 1 / 8], rtol=1e-14
        )


def test_the_random_index_follows_the_order_of_the_matrix():
    for order, expected in enumerate(RANDOM_INDEX, start=1):
        assert ahp_consistency(np.ones((order, order)))["ri"] == expected
    # More inconsistency is expected from larger random matrices.
    assert np.all(np.diff(RANDOM_INDEX[1:]) > 0)


def test_the_random_index_is_near_the_average_of_random_matrices():
    # Saaty's table comes from smaller simulations than this one; the means agree within 0.07.
    rng = np.random.default_rng(0)
    for order in range(3, 16):
        sizes = (4000, order, order)
        upper = np.triu(rng.choice(SAATY_SCALE, size=sizes), k=1)
        matrices = (
            upper
            + np.eye(order)
            + np.tril(
                1 / np.where(upper.transpose(0, 2, 1) == 0, 1, upper.transpose(0, 2, 1)), k=-1
            )
        )
        lambda_max = np.linalg.eigvals(matrices).real.max(axis=1)
        assert ((lambda_max - order) / (order - 1)).mean() == pytest.approx(
            RANDOM_INDEX[order - 1], abs=0.07
        )


def test_consistency_needs_a_tabulated_random_index():
    assert ahp_consistency(np.ones((15, 15)))["ri"] == 1.59
    with pytest.raises(ValueError, match="matrix must compare at most 15 factors.*got 16"):
        ahp_consistency(np.ones((16, 16)))
    # The weights of a larger matrix are still defined.
    np.testing.assert_allclose(ahp_weights(np.ones((16, 16))), np.full(16, 1 / 16))


def test_dataframe_and_array_inputs_give_the_same_weights():
    labels = list("abcdef")
    frame = pd.DataFrame(SCHOOL, index=labels, columns=labels)
    for method in ("eigenvector", "geometric"):
        from_frame = ahp_weights(frame, method=method)
        from_array = ahp_weights(SCHOOL, method=method)
        from_list = ahp_weights(SCHOOL.tolist(), method=method)
        np.testing.assert_array_equal(from_frame, from_array)
        np.testing.assert_array_equal(from_list, from_array)
        assert from_frame.index.tolist() == labels
        assert from_array.index.tolist() == [0, 1, 2, 3, 4, 5]
        assert from_array.index.name == from_frame.index.name == "factor"
    pd.testing.assert_series_equal(ahp_consistency(frame), ahp_consistency(SCHOOL))


def test_the_input_is_not_changed_and_its_index_keeps_its_name():
    frame = SALES_MARGIN_LEVERAGE.rename_axis(index="rows", columns="columns")
    before = frame.copy()
    ahp_weights(frame)
    ahp_consistency(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert frame.index.name == "rows"


def test_exact_fractions_and_integer_matrices_are_read():
    third = fractions.Fraction(1, 3)
    matrix = [[1, 3], [third, 1]]
    np.testing.assert_allclose(ahp_weights(matrix), [0.75, 0.25])
    np.testing.assert_allclose(ahp_weights(np.ones((3, 3), dtype=int)), [1 / 3] * 3)
    np.testing.assert_allclose(
        ahp_weights(np.array(SCHOOL, dtype=np.float32), method="geometric"),
        ahp_weights(SCHOOL, method="geometric"),
        rtol=1e-6,
    )


def test_reciprocal_entries_are_checked_to_a_relative_tolerance():
    close = [[1, 3], [0.33333333333, 1]]
    assert ahp_weights(close).tolist() == pytest.approx([0.75, 0.25], abs=1e-9)
    with pytest.raises(ValueError, match="matrix must be reciprocal"):
        ahp_weights([[1, 3], [0.33, 1]])
    assert ahp_weights([[1 + 1e-9, 3], [1 / 3, 1]]).sum() == pytest.approx(1)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (
            [[1, 3], [0.5, 1]],
            r"matrix must be reciprocal: the entries at \[0, 1\] and \[1, 0\] must multiply to 1, "
            r"got 3\.0 and 0\.5",
        ),
        ([[1, 2, 3], [0.5, 1, 1.5], [1 / 3, 2, 1]], r"at \[1, 2\] and \[2, 1\].*got 1\.5 and 2\.0"),
        ([[1, 2], [0.5, 2]], r"matrix must have 1 on its diagonal, got 2\.0 at \[1, 1\]"),
        ([[1, 0], [0, 1]], r"matrix must hold finite, positive numbers, got 0\.0 at \[0, 1\]"),
        ([[1, -2], [-0.5, 1]], r"positive numbers, got -2\.0 at \[0, 1\]"),
        ([[1, np.nan], [np.nan, 1]], r"positive numbers, got nan at \[0, 1\]"),
        ([[1, np.inf], [0, 1]], r"positive numbers, got inf at \[0, 1\]"),
        ([[1, 2, 3], [0.5, 1, 6]], r"matrix must be square, got 2 rows and 3 columns"),
        ([1, 2, 3], r"matrix must be two-dimensional, got 1 dimensions"),
        (3, r"matrix must be two-dimensional, got 0 dimensions"),
        (np.ones((2, 2, 2)), r"matrix must be two-dimensional, got 3 dimensions"),
        (np.empty((0, 0)), r"matrix must compare at least one factor"),
        ([[True, True], [True, True]], r"matrix must be real and numeric, got dtype bool"),
        ([["1", "2"], ["0.5", "1"]], r"matrix must be real and numeric, got dtype <U"),
        (pd.Series([1.0, 1.0]), r"matrix must be two-dimensional, got 1 dimensions"),
    ],
)
def test_invalid_matrices_raise_errors_that_show_the_value(matrix, message):
    with pytest.raises(ValueError, match=message):
        ahp_weights(matrix)
    with pytest.raises(ValueError, match=message):
        ahp_consistency(matrix)


def test_invalid_matrices_name_the_factors_of_a_dataframe():
    labels = ["sales", "margin"]
    frame = pd.DataFrame([[1, 3], [0.5, 1]], index=labels, columns=labels)
    with pytest.raises(ValueError, match=r"at \['sales', 'margin'\] and \['margin', 'sales'\]"):
        ahp_weights(frame)
    frame = pd.DataFrame([[1, 3], [1 / 3, 2]], index=labels, columns=labels)
    with pytest.raises(ValueError, match=r"got 2\.0 at \['margin', 'margin'\]"):
        ahp_weights(frame)


def test_the_labels_of_a_dataframe_must_agree_and_be_distinct():
    frame = pd.DataFrame([[1, 3], [1 / 3, 1]], index=["a", "b"], columns=["b", "a"])
    with pytest.raises(
        ValueError,
        match=r"same labels on its index and columns, got \['a', 'b'\] and \['b', 'a'\]",
    ):
        ahp_weights(frame)
    frame = pd.DataFrame(np.ones((2, 2)), index=["a", "a"], columns=["a", "a"])
    with pytest.raises(ValueError, match="matrix must name each factor once, got 'a' twice"):
        ahp_consistency(frame)
    with pytest.raises(ValueError, match="matrix must compare at least one factor"):
        ahp_weights(pd.DataFrame())


def test_the_method_must_be_one_of_the_two_names():
    with pytest.raises(ValueError, match="method must be 'eigenvector' or 'geometric', got 'mean'"):
        ahp_weights(THREE, method="mean")
    with pytest.raises(ValueError, match="got None"):
        ahp_weights(THREE, method=None)


def test_combined_weights_multiply_the_weight_of_the_factor_and_of_the_sub_factor():
    parent = pd.Series({"finance": 0.75, "market": 0.25}, name="factor weight")
    children = {
        "market": pd.Series({"position": 1.0}),
        "finance": pd.Series({"leverage": 0.6, "margin": 0.4}),
    }
    combined = combine_weights(parent, children)
    assert combined.index.names == ["factor", "subfactor"] and combined.name == "weight"
    # The order is that of the parent and then of each factor's sub-factors.
    assert combined.index.tolist() == [
        ("finance", "leverage"),
        ("finance", "margin"),
        ("market", "position"),
    ]
    np.testing.assert_allclose(combined, [0.45, 0.30, 0.25], rtol=1e-14)
    assert combined.sum() == pytest.approx(1, rel=1e-14)
    # The inputs are unchanged.
    assert parent.name == "factor weight" and list(children) == ["market", "finance"]


def test_combined_weights_of_ahp_results_sum_to_one_and_allow_repeated_sub_factor_names():
    parent = ahp_weights(THREE)
    children = {
        0: ahp_weights(SALES_MARGIN_LEVERAGE),
        1: ahp_weights([[1, 5], [1 / 5, 1]], method="geometric"),
        2: pd.Series({"margin": 1.0}),
    }
    combined = combine_weights(parent, children)
    assert combined.sum() == pytest.approx(1, rel=1e-12)
    assert combined.index.get_level_values("subfactor").tolist().count("margin") == 2
    assert combined[(0, "sales")] == pytest.approx(parent[0] * 4 / 7, rel=1e-12)
    # Dropping the factor level indexes a score frame by sub-factor.
    assert combined.droplevel("factor").index.tolist()[:3] == ["sales", "margin", "leverage"]


@pytest.mark.parametrize(
    ("parent", "children", "message"),
    [
        ([0.5, 0.5], {}, r"parent must be a Series of weights, got list"),
        (pd.Series([0.5, 0.6]), {}, r"parent must sum to 1, got 1\.1"),
        (pd.Series([1.5, -0.5]), {}, r"parent must be finite and non-negative, got -0\.5"),
        (pd.Series([np.nan, 1.0]), {}, r"parent must be finite and non-negative, got nan"),
        (pd.Series([], dtype=float), {}, r"parent must have at least one weight"),
        (
            pd.Series([0.5, 0.5], index=["a", "a"]),
            {},
            r"parent must name each label once, got 'a' twice",
        ),
        (pd.Series({"a": 1.0}), [1.0], r"children must be a mapping .* got list"),
        (
            pd.Series({"a": 0.5, "b": 0.5}),
            {"a": pd.Series([1.0])},
            r"children must have an entry for every factor of parent, missing \['b'\]",
        ),
        (
            pd.Series({"a": 1.0}),
            {"a": pd.Series([1.0]), "c": pd.Series([1.0]), "d": pd.Series([1.0])},
            r"children must only have entries for the factors of parent, got \['c', 'd'\]",
        ),
        (
            pd.Series({"a": 1.0}),
            {"a": [1.0]},
            r"children\['a'\] must be a Series of weights, got list",
        ),
        (
            pd.Series({"a": 1.0}),
            {"a": pd.Series([0.4, 0.4])},
            r"children\['a'\] must sum to 1, got 0\.8",
        ),
        (
            pd.Series({"a": 1.0}),
            {"a": pd.Series({"x": 1.2, "y": -0.2})},
            r"children\['a'\] must be finite and non-negative, got -0\.2",
        ),
    ],
)
def test_combine_weights_validates_its_arguments(parent, children, message):
    with pytest.raises(ValueError, match=message):
        combine_weights(parent, children)


def test_sums_are_checked_to_a_relative_tolerance():
    parent = pd.Series({"a": 0.5, "b": 0.5 + 1e-9})
    children = {"a": pd.Series([1.0 + 1e-9]), "b": pd.Series([1.0])}
    assert math.isclose(combine_weights(parent, children).sum(), 1, rel_tol=1e-8)
