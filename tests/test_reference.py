"""Cross-checks of kivi against scikit-learn, SciPy and statsmodels.

Most of kivi was re-implemented on NumPy and pandas alone and checked once against these
libraries. The tests here repeat those checks in CI, on live calls instead of pasted numbers:
a test is skipped when its reference library is not installed (``pip install "kivi[test]"``
installs all three) and never imports one at module level.

Each comparison states the definition it relies on. Results that are chosen from the data,
such as the splits of a tree, must agree exactly; computed floats agree to a tolerance, because
the last digit of a sum or a logarithm depends on the CPU the tests run on.
"""

import math

import numpy as np
import pandas as pd
import pytest

from kivi import _stats
from kivi.binning import chimerge_edges, chimerge_groups, tree_edges
from kivi.calibration import (
    binomial_test,
    fit_grade_pd,
    hosmer_lemeshow,
    isotonic_fit,
    isotonic_predict,
    jeffreys_test,
    platt_fit,
)
from kivi.datasets import load_bank
from kivi.metrics import brier_score, delong_test, ks, roc_auc, roc_auc_interval, roc_table
from kivi.selection import logistic_regression, vif


@pytest.fixture(scope="module")
def bank():
    return load_bank()


def simulated_scores(n=1500, seed=0):
    """A noisy score that rises with the risk, rounded so that scores are tied.

    Returns the score, a probability that is not the true one, binary labels and positive
    weights, as Series with a shared index that does not start at zero.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    target = (rng.random(n) < 1 / (1 + np.exp(1.5 - 1.2 * latent))).astype(int)
    score = np.round(latent + rng.normal(scale=0.8, size=n), 1)
    probability = 1 / (1 + np.exp(1.8 - 0.9 * score))
    weights = rng.random(n) * 2 + 0.25
    index = pd.RangeIndex(100, 100 + n)
    return (
        pd.Series(score, index=index),
        pd.Series(probability, index=index),
        pd.Series(target, index=index),
        pd.Series(weights, index=index),
    )


def glm_fit(exog, target, weights=None):
    """Fit a logistic regression with statsmodels' GLM, converged to a tight tolerance."""
    sm = pytest.importorskip("statsmodels.api")
    options = {} if weights is None else {"freq_weights": np.asarray(weights)}
    model = sm.GLM(target, sm.add_constant(exog), family=sm.families.Binomial(), **options)
    return model.fit(tol=1e-14, tol_criterion="params", maxiter=200)


# ----------------------------------------------------------------------------- scikit-learn


def split_edges(model, values):
    """The splits of a fitted scikit-learn tree, as the edges `tree_edges` returns.

    scikit-learn splits at the midpoint of two neighbouring values and kivi at the largest
    observed value at or below it.
    """
    thresholds = np.sort(model.tree_.threshold[model.tree_.children_left != -1])
    observed = np.unique(values)
    interior = observed[np.searchsorted(observed, thresholds, side="right") - 1]
    assert len(interior) >= 1
    return np.r_[-np.inf, interior, np.inf]


@pytest.mark.parametrize("column", ["age", "balance", "duration", "campaign", "day", "previous"])
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("max_bins,min_bin_share", [(5, 0.05), (8, 0.02), (3, 0.1)])
def test_tree_edges_match_scikit_learn_splits(bank, column, criterion, max_bins, min_bin_share):
    tree = pytest.importorskip("sklearn.tree")
    values, target = bank[column], bank["target"]
    model = tree.DecisionTreeClassifier(
        criterion=criterion,
        max_leaf_nodes=max_bins,
        min_samples_leaf=math.ceil(min_bin_share * len(values)),
        random_state=0,
    ).fit(values.to_frame(), target)
    edges = tree_edges(
        values, target, max_bins=max_bins, min_bin_share=min_bin_share, criterion=criterion
    )
    np.testing.assert_array_equal(edges, split_edges(model, values))


@pytest.mark.parametrize("column", ["age", "balance", "duration"])
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_tree_edges_with_weights_match_scikit_learn_sample_weights(bank, column, criterion):
    tree = pytest.importorskip("sklearn.tree")
    values, target = bank[column], bank["target"]
    weights = np.random.default_rng(7).random(len(values)) + 0.2
    # The share of the total weight that each bin keeps is scikit-learn's min_weight_fraction_leaf.
    model = tree.DecisionTreeClassifier(
        criterion=criterion, max_leaf_nodes=6, min_weight_fraction_leaf=0.04, random_state=0
    ).fit(values.to_frame(), target, sample_weight=weights)
    edges = tree_edges(
        values, target, max_bins=6, min_bin_share=0.04, criterion=criterion, weights=weights
    )
    np.testing.assert_array_equal(edges, split_edges(model, values))


def isotonic_case(name):
    """Score, labels and weights of a calibration sample."""
    score, probability, target, weights = simulated_scores(seed=11)
    if name == "simulated":
        return score, target, None
    if name == "simulated-weighted":
        return score, target, weights
    if name == "almost-monotone":
        return probability + 1e-3 * score, target, None
    bank = load_bank()
    if name == "bank-duration":
        return bank["duration"], bank["target"], None
    return bank["age"], bank["target"], bank["balance"].abs() + 1


@pytest.mark.parametrize("increasing", [True, False])
@pytest.mark.parametrize(
    "case", ["simulated", "simulated-weighted", "almost-monotone", "bank-duration", "bank-age"]
)
def test_isotonic_fit_matches_scikit_learn_at_the_fitted_scores(case, increasing):
    isotonic = pytest.importorskip("sklearn.isotonic")
    score, target, weights = isotonic_case(case)
    if not increasing:
        score = -score  # like scorecard points: the risk, and the fitted rate, falls as it rises
    edges, probability = isotonic_fit(score, target, weights=weights, increasing=increasing)
    reference = isotonic.IsotonicRegression(increasing=increasing, out_of_bounds="clip")
    reference.fit(score, target, sample_weight=weights)
    expected = reference.predict(score)
    np.testing.assert_allclose(isotonic_predict(score, edges, probability), expected, rtol=1e-12)
    # The same pooled event rates, and no two adjacent bands share one.
    np.testing.assert_allclose(np.sort(np.unique(expected)), np.sort(probability), rtol=1e-12)
    assert len(probability) == len(edges) - 1 >= 8


def test_isotonic_predict_clips_and_takes_the_upper_band_between_bands():
    isotonic = pytest.importorskip("sklearn.isotonic")
    score, _, target, weights = simulated_scores(seed=12)
    edges, probability = isotonic_fit(score, target, weights=weights)
    reference = isotonic.IsotonicRegression(out_of_bounds="clip").fit(
        score, target, sample_weight=weights
    )
    # Outside the fitted range both give the probability of the outermost band.
    outside = np.array([score.min() - 10, score.max() + 10])
    np.testing.assert_allclose(
        isotonic_predict(outside, edges, probability), reference.predict(outside), rtol=1e-12
    )
    # Between two fitted scores kivi gives the band of the next higher fitted score, where
    # scikit-learn interpolates between the two bands.
    fitted = np.unique(score)
    between = (fitted[:-1] + fitted[1:]) / 2
    np.testing.assert_allclose(
        isotonic_predict(between, edges, probability), reference.predict(fitted[1:]), rtol=1e-12
    )


@pytest.mark.parametrize("use_weights", [False, True])
def test_roc_auc_matches_scikit_learn(use_weights):
    metrics = pytest.importorskip("sklearn.metrics")
    score, _, target, weights = simulated_scores(seed=21)
    weights = weights if use_weights else None
    expected = metrics.roc_auc_score(target, score, sample_weight=weights)
    assert roc_auc(score, target, weights=weights) == pytest.approx(expected, rel=1e-12, abs=0)
    # The same on plain arrays, with both the score and the labels reversed.
    assert roc_auc(-score.to_numpy(), 1 - target.to_numpy(), weights=weights) == pytest.approx(
        expected, rel=1e-12, abs=0
    )


@pytest.mark.parametrize("use_weights", [False, True])
def test_roc_table_matches_scikit_learn_roc_curve(use_weights):
    metrics = pytest.importorskip("sklearn.metrics")
    score, _, target, weights = simulated_scores(seed=22)
    weights = weights if use_weights else None
    table = roc_table(score, target, weights=weights)
    fpr, tpr, thresholds = metrics.roc_curve(
        target, score, sample_weight=weights, drop_intermediate=False
    )
    # The first threshold, which flags nobody, is infinite since scikit-learn 1.3.
    np.testing.assert_array_equal(table.index.to_numpy(), thresholds)
    np.testing.assert_allclose(table["false_positive_rate"], fpr, rtol=1e-12)
    np.testing.assert_allclose(table["true_positive_rate"], tpr, rtol=1e-12)
    # The counts are the rates times the class totals.
    total = np.ones(len(score)) if weights is None else weights.to_numpy()
    event = target.to_numpy() == 1
    np.testing.assert_allclose(table["cum_event"], tpr * total[event].sum(), rtol=1e-12)
    np.testing.assert_allclose(table["cum_non_event"], fpr * total[~event].sum(), rtol=1e-12)
    # The trapezoids under the curve are the AUC.
    area = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
    assert area == pytest.approx(roc_auc(score, target, weights=weights), rel=1e-12, abs=0)


@pytest.mark.parametrize("use_weights", [False, True])
def test_brier_score_matches_scikit_learn(use_weights):
    metrics = pytest.importorskip("sklearn.metrics")
    _, probability, target, weights = simulated_scores(seed=23)
    weights = weights if use_weights else None
    expected = metrics.brier_score_loss(target, probability, sample_weight=weights)
    assert brier_score(probability, target, weights=weights) == pytest.approx(
        expected, rel=1e-12, abs=0
    )


# --------------------------------------------------------------------------------- SciPy


def test_ks_matches_the_two_sample_statistic_of_scipy():
    stats = pytest.importorskip("scipy.stats")
    score, _, target, _ = simulated_scores(seed=31)
    event = score[target == 1].to_numpy()
    non_event = score[target == 0].to_numpy()
    expected = stats.ks_2samp(event, non_event, method="asymp").statistic
    assert ks(score, target) == pytest.approx(expected, rel=1e-12, abs=0)
    # The gap is absolute, so a score that runs the other way gives the same statistic.
    assert ks(-score, target) == pytest.approx(expected, rel=1e-12, abs=0)


def test_ks_with_integer_weights_matches_scipy_on_repeated_rows():
    stats = pytest.importorskip("scipy.stats")
    score, _, target, _ = simulated_scores(seed=32)
    weights = pd.Series(np.random.default_rng(32).integers(0, 5, len(score)), index=score.index)
    repeated = score.loc[score.index.repeat(weights)]
    labels = target.loc[target.index.repeat(weights)]
    expected = stats.ks_2samp(
        repeated[labels == 1].to_numpy(), repeated[labels == 0].to_numpy(), method="asymp"
    ).statistic
    assert ks(score, target, weights=weights) == pytest.approx(expected, rel=1e-12, abs=0)


BETA_SHAPES = [0.5, 1.0, 2.0, 7.5, 30.0, 150.0, 1e3, 2e4, 1e5]


def test_incomplete_beta_matches_scipy_on_a_grid():
    special = pytest.importorskip("scipy.special")
    a, b = (grid.ravel() for grid in np.meshgrid(BETA_SHAPES, BETA_SHAPES))
    mean = a / (a + b)
    sd = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    # Points from far below to far above the mean, which reach the tails of every distribution.
    x = np.concatenate(
        [mean + z * sd for z in (-8, -4, -2, -1, -0.3, 0, 0.3, 1, 2, 4, 8)]
        + [np.full_like(mean, point) for point in (1e-6, 0.01, 0.2, 0.5, 0.8, 0.99, 1 - 1e-6)]
    )
    a, b = np.tile(a, len(x) // len(mean)), np.tile(b, len(x) // len(mean))
    inside = (x > 0) & (x < 1)
    assert inside.sum() > 0.8 * len(x)
    a, b, x = a[inside], b[inside], x[inside]
    actual, expected = _stats.betainc(a, b, x), special.betainc(a, b, x)
    # Far in the tail the front factor of the continued fraction loses digits to the logarithms
    # it adds up: values below 1e-100 agree to about 9 digits, all others to 11.
    tail = expected < 1e-100
    assert tail.any() and (~tail).any()
    np.testing.assert_allclose(actual[~tail], expected[~tail], rtol=1e-10)
    np.testing.assert_allclose(actual[tail], expected[tail], rtol=1e-8)


def test_incomplete_beta_matches_scipy_at_the_end_points_and_in_the_tails():
    special = pytest.importorskip("scipy.special")
    x = np.array([0.0, 1e-300, 1e-12, 1e-6, 0.5, 1 - 1e-6, 1 - 1e-12, 1.0])
    for a, b in [(0.5, 0.5), (2.0, 3.0), (40.0, 5.0), (3.0, 800.0)]:
        np.testing.assert_allclose(_stats.betainc(a, b, x), special.betainc(a, b, x), rtol=1e-10)


GAMMA_SHAPES = [0.5, 1.0, 2.5, 10.0, 50.0, 500.0, 5000.0]
GAMMA_QUANTILES = [1e-12, 1e-6, 0.01, 0.5, 0.99, 1 - 1e-6]


@pytest.mark.parametrize("shape", GAMMA_SHAPES)
def test_incomplete_gamma_matches_scipy_on_a_grid(shape):
    special = pytest.importorskip("scipy.special")
    x = special.gammainccinv(shape, GAMMA_QUANTILES)
    expected = special.gammaincc(shape, x)
    actual = [_stats.gammaincc(shape, value) for value in x]
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


@pytest.mark.parametrize("dof", [1, 2, 3, 4, 5, 9, 10, 30, 99, 250, 1000])
def test_chi2_sf_matches_scipy(dof):
    stats = pytest.importorskip("scipy.stats")
    statistic = stats.chi2.isf([1e-12, 1e-6, 0.001, 0.05, 0.5, 0.9, 0.999, 1 - 1e-9], dof)
    expected = stats.chi2.sf(statistic, dof)
    actual = [_stats.chi2_sf(value, dof) for value in statistic]
    np.testing.assert_allclose(actual, expected, rtol=1e-10)
    assert _stats.chi2_sf(0.0, dof) == 1.0


def test_normal_quantile_matches_scipy():
    special = pytest.importorskip("scipy.special")
    stats = pytest.importorskip("scipy.stats")
    probability = np.r_[
        np.logspace(-300, -3, 60),
        np.linspace(0.001, 0.999, 999),
        1 - np.logspace(-3, -12, 30),
        0.5,
    ]
    actual = np.array([_stats.normal_quantile(p) for p in probability])
    np.testing.assert_allclose(actual, special.ndtri(probability), rtol=1e-13)
    np.testing.assert_allclose(actual, stats.norm.ppf(probability), rtol=1e-13)
    # The quantile inverts the normal CDF, the tail of which is `math.erfc` in the Wald tests.
    lower = probability < 0.5
    np.testing.assert_allclose(special.ndtr(actual[lower]), probability[lower], rtol=1e-10)


@pytest.mark.parametrize("per_value", [15, 60, 250, 1000, 4000, 15000])
def test_chimerge_merges_a_pair_exactly_when_its_chi_square_p_value_is_not_below_alpha(per_value):
    contingency = pytest.importorskip("scipy.stats").contingency
    # Two values with event rates of 10 % and 20 %: from a p-value near 0.5 to one near 1e-130.
    values = np.repeat([0, 1], per_value)
    target = np.r_[np.arange(per_value) < per_value // 10, np.arange(per_value) < per_value // 5]
    table = [
        [per_value - per_value // 10, per_value // 10],
        [per_value - per_value // 5, per_value // 5],
    ]
    p_value = contingency.chi2_contingency(table, correction=False).pvalue
    assert 1e-200 < p_value < 0.9
    kept = chimerge_edges(values, target.astype(int), max_bins=2, alpha=p_value * (1 + 1e-6))
    merged = chimerge_edges(values, target.astype(int), max_bins=2, alpha=p_value * (1 - 1e-6))
    np.testing.assert_array_equal(kept, [-np.inf, 0, np.inf])
    np.testing.assert_array_equal(merged, [-np.inf, np.inf])


def fine_bin_counts(values, target, bins):
    """Interior edges and (non_event, event) counts of the quantile bins ChiMerge starts from."""
    edges = chimerge_edges(values, target, max_bins=bins, alpha=None, initial_bins=bins)
    codes = pd.cut(values, edges).cat.codes.to_numpy()
    counts = np.zeros((len(edges) - 1, 2))
    np.add.at(counts, (codes, target.to_numpy()), 1)
    return edges[1:-1], counts


def pair_p_values(counts):
    """P-values of the chi-square tests of the adjacent pairs of bins, from scipy.

    A pair in which a class is absent from both bins has no expected count to test against, and
    nothing separates its two bins, so its p-value is 1.
    """
    stats = pytest.importorskip("scipy.stats")
    return np.array(
        [
            stats.contingency.chi2_contingency(pair, correction=False).pvalue
            if pair.sum(axis=0).all()
            else 1.0
            for pair in (counts[i : i + 2] for i in range(len(counts) - 1))
        ]
    )


def reference_chimerge(counts, max_bins, alpha):
    """ChiMerge as `chimerge_edges` describes it, with the statistics and tests of scipy.

    Returns the merged groups, each a list of indices of the bins ChiMerge started from.
    """
    counts = counts.copy()
    groups = [[i] for i in range(len(counts))]
    while len(counts) > 1:
        p_values = pair_p_values(counts)
        pair = int(np.argmax(p_values))  # the smallest statistic has the largest p-value
        significant = alpha is None or p_values[pair] < alpha
        if len(counts) <= max_bins and significant:
            break
        counts[pair] += counts[pair + 1]
        counts = np.delete(counts, pair + 1, axis=0)
        groups[pair] += groups.pop(pair + 1)
    return groups


@pytest.mark.parametrize("column", ["age", "balance", "duration", "day"])
@pytest.mark.parametrize("max_bins,alpha", [(5, 0.05), (8, 0.01), (20, 0.05), (3, None), (20, 0.3)])
def test_chimerge_edges_match_a_scipy_implementation(bank, column, max_bins, alpha):
    pytest.importorskip("scipy.stats")
    values, target = bank[column], bank["target"]
    interior, counts = fine_bin_counts(values, target, 20)
    groups = reference_chimerge(counts, max_bins, alpha)
    expected = np.r_[-np.inf, [interior[group[-1]] for group in groups[:-1]], np.inf]
    actual = chimerge_edges(values, target, max_bins=max_bins, alpha=alpha, initial_bins=20)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("column", ["age", "balance", "duration", "day"])
def test_chimerge_stops_merging_at_the_p_value_of_the_least_different_pair(bank, column):
    values, target = bank[column], bank["target"]
    interior, counts = fine_bin_counts(values, target, 8)
    p_max = pair_p_values(counts).max()
    assert p_max < 0.999
    # Nothing is forced to merge, so the pair with the smallest statistic merges exactly when
    # its p-value is not below alpha.
    kept = chimerge_edges(values, target, max_bins=8, alpha=p_max * (1 + 1e-6), initial_bins=8)
    merged = chimerge_edges(values, target, max_bins=8, alpha=p_max * (1 - 1e-6), initial_bins=8)
    np.testing.assert_array_equal(kept[1:-1], interior)
    assert len(merged) < len(kept)


@pytest.mark.parametrize("column", ["job", "month", "education", "poutcome"])
@pytest.mark.parametrize("max_groups,alpha", [(5, 0.05), (3, None), (8, 0.3), (12, 0.01)])
def test_chimerge_groups_match_a_scipy_implementation(bank, column, max_groups, alpha):
    pytest.importorskip("scipy.stats")
    values, target = bank[column], bank["target"]
    table = pd.crosstab(values, target)
    assert (table.to_numpy() > 0).all()  # no zero cell, so the order needs no smoothing
    # The categories side by side in increasing order of their event rate, which is the order
    # of their WOE, are the bins ChiMerge starts from.
    order = np.argsort(table[1].to_numpy() / table.sum(axis=1).to_numpy(), kind="stable")
    names = table.index.to_numpy()[order]
    groups = reference_chimerge(table.to_numpy()[order].astype(float), max_groups, alpha)
    expected = {frozenset(names[i] for i in group) for group in groups}
    fitted = chimerge_groups(values, target, max_groups=max_groups, alpha=alpha, min_group_share=0)
    members = {}
    for category, group in fitted.items():
        members.setdefault(group, set()).add(category)
    assert {frozenset(group) for group in members.values()} == expected
    assert len(expected) >= 2


def test_hosmer_lemeshow_matches_scipy_chi_square_test():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(41)
    for groups, ddof in [(10, 2), (10, 0), (6, 2), (25, 1)]:
        total = rng.integers(40, 400, groups)
        probability = np.sort(rng.uniform(0.01, 0.4, groups))
        event = rng.binomial(total, probability * rng.uniform(0.7, 1.4, groups))
        statistic, p_value = hosmer_lemeshow(event, total, probability, ddof=ddof)
        # The statistic is Pearson's chi-square of the events and non-events of every group.
        # scipy counts the degrees of freedom as the cells minus one minus its ddof.
        reference = stats.chisquare(
            np.r_[event, total - event],
            np.r_[total * probability, total * (1 - probability)],
            ddof=groups - 1 + ddof,
        )
        assert statistic == pytest.approx(reference.statistic, rel=1e-12, abs=0)
        assert p_value == pytest.approx(reference.pvalue, rel=1e-10, abs=0)
        assert p_value == pytest.approx(stats.chi2.sf(statistic, groups - ddof), rel=1e-10, abs=0)


EVENT_GRID = [
    (n, p, f)
    for n in (25, 300, 6_000, 10**6)
    for p in (0.001, 0.02, 0.3)
    for f in (0, 0.4, 1, 2, 5)
]


def test_binomial_test_matches_scipy_one_sided_exact_test():
    stats = pytest.importorskip("scipy.stats")
    for total, probability, factor in EVENT_GRID:
        event = min(total, round(factor * total * probability))
        expected = stats.binomtest(event, total, probability, alternative="greater").pvalue
        assert binomial_test(event, total, probability) == pytest.approx(expected, rel=1e-10, abs=0)


def test_binomial_test_broadcasts_like_a_vectorized_scipy_tail():
    stats = pytest.importorskip("scipy.stats")
    total = np.array([[50], [4000]])
    event = np.array([0, 1, 3, 20, 49])
    probability = 0.04
    event = np.minimum(event, total)
    expected = np.where(event > 0, stats.binom.sf(event - 1, total, probability), 1.0)
    np.testing.assert_allclose(binomial_test(event, total, probability), expected, rtol=1e-10)


def test_jeffreys_test_matches_the_beta_posterior_of_scipy():
    stats = pytest.importorskip("scipy.stats")
    for total, probability, factor in EVENT_GRID:
        event = min(total, round(factor * total * probability))
        expected = stats.beta.cdf(probability, event + 0.5, total - event + 0.5)
        assert jeffreys_test(event, total, probability) == pytest.approx(expected, rel=1e-10, abs=0)


def delong_reference(scores, target):
    """AUCs and their covariance matrix by DeLong, DeLong and Clarke-Pearson (1988).

    Compares every event with every non-event, in O(m n) time, instead of the midranks that
    kivi uses. The structural components V10 (one per event) and V01 (one per non-event) are the
    mean of 1 for a higher score and 1/2 for a tie; their covariances give the covariance of the
    AUCs as S10 / m + S01 / n.
    """
    scores, event = np.asarray(scores, dtype=float), np.asarray(target) == 1
    x, y = scores[:, event, None], scores[:, None, ~event]
    placement = (x > y) + 0.5 * (x == y)
    v10, v01 = placement.mean(axis=2), placement.mean(axis=1)
    covariance = (
        np.atleast_2d(np.cov(v10)) / v10.shape[1] + np.atleast_2d(np.cov(v01)) / v01.shape[1]
    )
    return v10.mean(axis=1), covariance


def test_roc_auc_interval_matches_an_independent_delong_computation():
    stats = pytest.importorskip("scipy.stats")
    score, _, target, _ = simulated_scores(seed=51)
    auc, covariance = delong_reference([score], target)
    se, z = math.sqrt(covariance[0, 0]), stats.norm.ppf(0.95)
    interval = roc_auc_interval(score, target, level=0.9)
    assert interval.loc["auc", "estimate"] == pytest.approx(auc[0], rel=1e-12, abs=0)
    assert interval.loc["auc", "se"] == pytest.approx(se, rel=1e-10, abs=0)
    assert interval.loc["auc", "lower"] == pytest.approx(auc[0] - z * se, rel=1e-10, abs=0)
    assert interval.loc["auc", "upper"] == pytest.approx(auc[0] + z * se, rel=1e-10, abs=0)
    # The Gini coefficient is 2 AUC - 1, so its standard error doubles.
    assert interval.loc["gini", "se"] == pytest.approx(2 * se, rel=1e-10, abs=0)


@pytest.mark.parametrize("seed", [52, 53])
def test_delong_test_matches_an_independent_delong_computation(seed):
    stats = pytest.importorskip("scipy.stats")
    score, _, target, _ = simulated_scores(seed=seed)
    # A second score that is the first plus independent noise, rounded to have ties of its own.
    noise = np.random.default_rng(1000 + seed).normal(scale=0.35, size=len(score))
    other = pd.Series(np.round(score + noise, 1), index=score.index)
    z = stats.norm.ppf(0.975)
    for current, new in [(score, other), (other, score)]:
        auc, covariance = delong_reference([current, new], target)
        difference = auc[1] - auc[0]
        se = math.sqrt(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
        p_value = 2 * stats.norm.sf(abs(difference / se))
        result = delong_test(current, new, target)
        assert result["current_auc"] == pytest.approx(auc[0], rel=1e-12, abs=0)
        assert result["new_auc"] == pytest.approx(auc[1], rel=1e-12, abs=0)
        assert result["difference"] == pytest.approx(difference, rel=1e-10, abs=0)
        assert result["se"] == pytest.approx(se, rel=1e-10, abs=0)
        assert result["z"] == pytest.approx(difference / se, rel=1e-10, abs=0)
        assert result["p_value"] == pytest.approx(p_value, rel=1e-9, abs=0)
        assert result["lower"] == pytest.approx(difference - z * se, rel=1e-9, abs=0)
        assert result["upper"] == pytest.approx(difference + z * se, rel=1e-9, abs=0)
        # The scores differ modestly, so the p-value is neither vacuous nor 1.
        assert 1e-3 < p_value < 0.9


# ----------------------------------------------------------------------------- statsmodels


@pytest.mark.parametrize("column", ["duration", "age", "balance"])
def test_platt_fit_matches_statsmodels_logit(bank, column):
    sm = pytest.importorskip("statsmodels.api")
    score, target = bank[column].astype(float), bank["target"]
    slope, intercept = platt_fit(score, target)
    fitted = sm.Logit(target, sm.add_constant(score)).fit(disp=0, tol=1e-14, maxiter=200)
    assert slope == pytest.approx(fitted.params[column], rel=1e-8, abs=0)
    assert intercept == pytest.approx(fitted.params["const"], rel=1e-8, abs=0)


@pytest.mark.parametrize("use_weights", [False, True])
def test_platt_fit_matches_statsmodels_glm_on_a_simulated_score(use_weights):
    score, _, target, weights = simulated_scores(seed=61)
    weights = weights if use_weights else None
    slope, intercept = platt_fit(score, target, weights=weights)
    fitted = glm_fit(score, target, weights)
    assert slope == pytest.approx(fitted.params.iloc[1], rel=1e-8, abs=0)
    assert intercept == pytest.approx(fitted.params.iloc[0], rel=1e-8, abs=0)


def design(bank):
    """Bank columns with very different scales and correlations, and a 0/1 column."""
    return pd.DataFrame(
        {
            "age": bank["age"],
            "balance": bank["balance"] / 1000,
            "duration": bank["duration"],
            "campaign": bank["campaign"],
            "previous": bank["previous"],
            "housing": (bank["housing"] == "yes").astype(float),
        }
    ).astype(float)


@pytest.mark.parametrize("kind", ["none", "integer", "fractional"])
def test_logistic_regression_matches_statsmodels_glm(bank, kind):
    X, target = design(bank), bank["target"]
    rng = np.random.default_rng(71)
    weights = {
        "none": None,
        "integer": rng.integers(1, 4, len(X)),
        "fractional": rng.random(len(X)) + 0.1,
    }[kind]
    ours = logistic_regression(X, target, weights=weights)
    reference = glm_fit(X, target, weights)
    assert list(ours.index) == ["intercept", *X.columns]
    np.testing.assert_allclose(ours["coef"], reference.params, rtol=1e-8)
    np.testing.assert_allclose(ours["se"], reference.bse, rtol=1e-8)
    np.testing.assert_allclose(ours["z"], reference.tvalues, rtol=1e-8)
    np.testing.assert_allclose(ours["p_value"], reference.pvalues, rtol=1e-7)
    assert ours.attrs["log_likelihood"] == pytest.approx(reference.llf, rel=1e-10, abs=0)


def test_logistic_regression_on_woe_like_columns_matches_statsmodels():
    # Columns in a narrow range with moderate correlation, as the WOE of binned variables.
    rng = np.random.default_rng(72)
    common = rng.normal(size=1200)
    X = pd.DataFrame(
        {name: 0.6 * common + rng.normal(scale=0.5, size=1200) for name in ["a", "b", "c"]}
    )
    probability = 1 / (1 + np.exp(1.0 + X @ np.array([0.8, 0.5, 0.3])))
    target = pd.Series((rng.random(1200) < probability).astype(int))
    ours = logistic_regression(X, target)
    reference = glm_fit(X, target)
    np.testing.assert_allclose(ours["coef"], reference.params, rtol=1e-8)
    np.testing.assert_allclose(ours["se"], reference.bse, rtol=1e-8)
    np.testing.assert_allclose(ours["p_value"], reference.pvalues, rtol=1e-7)


def test_vif_matches_statsmodels_on_a_design_with_a_constant(bank):
    sm = pytest.importorskip("statsmodels.api")
    outliers = pytest.importorskip("statsmodels.stats.outliers_influence")
    X = design(bank)
    X["balance_plus_age"] = (
        X["balance"] * 3 + X["age"] + np.random.default_rng(81).normal(scale=0.5, size=len(X))
    )
    exog = sm.add_constant(X)
    expected = [
        outliers.variance_inflation_factor(exog.to_numpy(), i + 1) for i in range(X.shape[1])
    ]
    actual = vif(X)
    assert list(actual.index) == list(X.columns)
    assert actual.max() > 5
    np.testing.assert_allclose(actual, expected, rtol=1e-8)


@pytest.mark.parametrize(
    "event,total",
    [
        ([0, 3, 5, 18, 30], [400, 500, 300, 400, 200]),
        ([1, 2, 6, 9, 21, 40, 83], [900, 800, 700, 500, 450, 400, 380]),
        # Grades 2 and 5 have no observations: they stay out of the fit but still get a PD.
        ([0, 0, 4, 9, 0, 30, 61], [300, 0, 350, 280, 0, 260, 250]),
    ],
)
def test_fit_grade_pd_matches_statsmodels_weighted_least_squares(event, total):
    sm = pytest.importorskip("statsmodels.api")
    event, total = np.array(event, dtype=float), np.array(total, dtype=float)
    grade = np.arange(1, len(total) + 1, dtype=float)
    seen = total > 0

    def expected_pd(smoothing):
        # ln(rate) on the grade number, weighted with the totals, where a zero count is replaced
        # by `smoothing` defaults; the line is extended to the grades without observations.
        log_rate = np.log(np.where(event > 0, event, smoothing)[seen] / total[seen])
        fitted = sm.WLS(
            log_rate, np.column_stack([np.ones(seen.sum()), grade[seen]]), weights=total[seen]
        ).fit()
        return np.exp(fitted.predict(np.column_stack([np.ones(len(grade)), grade])))

    np.testing.assert_allclose(fit_grade_pd(event, total), expected_pd(0.5), rtol=1e-10)
    np.testing.assert_allclose(
        fit_grade_pd(event, total, smoothing=2.0), expected_pd(2.0), rtol=1e-10
    )
