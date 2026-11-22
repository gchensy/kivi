import math
import re

import numpy as np
import pandas as pd
import pytest

from kivi.binning import bin_edges
from kivi.datasets import make_applications
from kivi.reject_inference import fuzzy_augmentation, parceling, reweighting
from kivi.selection import logistic_regression
from kivi.woe import woe_table, woe_transform

# Accepted applicants in two bands of [0, 0.5, 1]: the bad rate is 1/4 in the low band and
# 3/4 in the high band. One reject falls in the low band and four in the high band.
SCORE = [0.1, 0.2, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9]
TARGET = [0, 0, 1, 0, 1, 0, 1, 1]
REJECT_SCORE = [0.25, 0.65, 0.75, 0.85, 0.95]
EDGES = [0, 0.5, 1]


@pytest.fixture(scope="module")
def applications():
    """WOE fitted on the accepted applicants, the target and the approval of every applicant."""
    data = make_applications(n_applicants=4000, rng=0)
    data = data[~data["policy_decline"]]
    approved = data["approved"]
    edges = {
        "age": [-np.inf, 30, 40, 50, np.inf],
        "debt_to_income": [-np.inf, 0.2, 0.3, 0.4, np.inf],
        "inquiries": [-np.inf, 0, 1, 2, np.inf],
    }
    groups = {name: pd.cut(data[name], column_edges) for name, column_edges in edges.items()}
    groups["housing"] = data["housing"]
    X = pd.DataFrame(
        {
            name: woe_transform(values, woe_table(values[approved], data["target"][approved]))
            for name, values in groups.items()
        }
    )
    return X, data["target"], approved.to_numpy()


def accepts_only(applications):
    X, target, approved = applications
    model = logistic_regression(X[approved], target[approved])
    return model, X[approved], target[approved], X[~approved]


def refit(accepts, target, rejects, inferred):
    return logistic_regression(
        pd.concat([accepts, rejects.loc[inferred.index]]),
        pd.concat([target, inferred["target"]]),
        weights=pd.concat([pd.Series(1.0, index=accepts.index), inferred["weight"]]),
    )


def probability_of(model, X):
    """PD of a model from `logistic_regression` on the columns of `X`."""
    coef = model["coef"]
    return 1 / (1 + np.exp(-(coef["intercept"] + X @ coef.drop("intercept"))))


# --- fuzzy_augmentation -------------------------------------------------------------------


def test_fuzzy_rows_weights_and_index():
    probability = pd.Series([0.1, 0.4, 0.7], index=pd.Index(["a", "b", "c"], name="id"))
    inferred = fuzzy_augmentation(probability)
    assert list(inferred.columns) == ["target", "weight"]
    assert inferred.index.equals(pd.Index(["a", "a", "b", "b", "c", "c"], name="id"))
    assert inferred["target"].tolist() == [1, 0, 1, 0, 1, 0]
    assert inferred["target"].dtype == np.int64
    np.testing.assert_allclose(inferred["weight"], [0.1, 0.9, 0.4, 0.6, 0.7, 0.3])


def test_fuzzy_factor_scales_and_clips_the_bad_weight():
    inferred = fuzzy_augmentation([0.1, 0.4, 0.7, 0.0, 1.0], factor=1.5)
    # 1.5 * 0.7 = 1.05 and 1.5 * 1.0 are clipped to 1, so their good rows get weight 0.
    np.testing.assert_allclose(
        inferred["weight"], [0.15, 0.85, 0.6, 0.4, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    )
    assert inferred.index.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    totals = inferred.groupby(level=0)["weight"].sum()
    np.testing.assert_allclose(totals, 1.0)
    bad_rate = np.average(inferred["target"], weights=inferred["weight"])
    assert bad_rate == pytest.approx(np.mean([0.15, 0.6, 1.0, 0.0, 1.0]))


def test_fuzzy_factor_below_one_lowers_the_bad_weight():
    inferred = fuzzy_augmentation(np.array([0.2, 0.8]), factor=0.5)
    np.testing.assert_allclose(inferred["weight"], [0.1, 0.9, 0.4, 0.6])


def test_fuzzy_with_factor_one_reproduces_the_accepts_only_model(applications):
    model, accepts, target, rejects = accepts_only(applications)
    inferred = fuzzy_augmentation(probability_of(model, rejects))
    refitted = refit(accepts, target, rejects, inferred)
    np.testing.assert_allclose(refitted["coef"], model["coef"], rtol=1e-9)
    # Only the standard errors shrink, because the inferred rows count as observations.
    assert (refitted["se"] < model["se"]).all()


def test_fuzzy_with_a_higher_factor_raises_the_predicted_bad_rate_of_rejects(applications):
    model, accepts, target, rejects = accepts_only(applications)
    before = probability_of(model, rejects)
    refitted = refit(accepts, target, rejects, fuzzy_augmentation(before, factor=1.5))
    after = probability_of(refitted, rejects)
    assert before.mean() < after.mean() < np.minimum(1.5 * before, 1).mean()


def test_parceling_and_reweighting_refit_on_applications(applications):
    model, accepts, target, rejects = accepts_only(applications)
    accept_pd, reject_pd = probability_of(model, accepts), probability_of(model, rejects)
    labels = parceling(accept_pd, target, reject_pd, factor=1.5, rng=0)
    assert labels.index.equals(rejects.index)
    expected = parceling(accept_pd, target, reject_pd, factor=1.5, expected=True)
    # The random labels give each band its expected number of bads, up to rounding.
    assert abs(labels["target"].sum() - expected["weight"][expected["target"] == 1].sum()) < 10
    for inferred in (labels, expected):
        refitted = refit(accepts, target, rejects, inferred)
        assert probability_of(refitted, rejects).mean() > reject_pd.mean()
    weight = reweighting(accept_pd, reject_pd)
    assert weight.index.equals(accepts.index)
    assert weight.sum() == pytest.approx(len(accepts) + len(rejects))
    reweighted = logistic_regression(accepts, target, weights=weight)
    assert list(reweighted.index) == list(model.index)


def test_fuzzy_leaves_its_input_alone():
    probability = pd.Series([0.2, 0.5])
    copy = probability.copy()
    fuzzy_augmentation(probability, factor=3)
    pd.testing.assert_series_equal(probability, copy)


@pytest.mark.parametrize(
    ("probability", "message"),
    [
        ([0.2, 1.2], "probability must be between 0 and 1, got 1.2"),
        ([-0.1], r"probability must be between 0 and 1, got -0.1"),
        ([0.2, np.nan], r"probability must be finite, got 1 missing or infinite value\(s\)"),
        (["0.2"], "probability must be real and numeric, got dtype"),
        ([], "probability must not be empty"),
        ([[0.2]], "probability must be one-dimensional, got 2 dimensions"),
    ],
)
def test_fuzzy_rejects_invalid_probabilities(probability, message):
    with pytest.raises(ValueError, match=message):
        fuzzy_augmentation(probability)


@pytest.mark.parametrize("factor", [0, -1.0, math.inf, math.nan, True, "2", None])
def test_factor_must_be_a_positive_finite_number(factor):
    message = rf"factor must be a positive finite number, got {re_escape(factor)}"
    with pytest.raises(ValueError, match=message):
        fuzzy_augmentation([0.5], factor=factor)
    with pytest.raises(ValueError, match=message):
        parceling(SCORE, TARGET, REJECT_SCORE, bins=EDGES, factor=factor)


def re_escape(value):
    return re.escape(repr(value))


# --- parceling ----------------------------------------------------------------------------


def test_parceling_labels_the_share_of_each_band():
    for seed in range(20):
        inferred = parceling(SCORE, TARGET, REJECT_SCORE, bins=EDGES, rng=seed)
        assert list(inferred.columns) == ["target", "weight"]
        assert inferred["target"].dtype == np.int64
        np.testing.assert_array_equal(inferred["weight"], 1.0)
        # 3/4 of the four rejects of the high band, for every seed.
        assert inferred["target"].iloc[1:].sum() == 3


def test_parceling_rounds_the_number_of_bads_at_random():
    # The low band's one reject is bad with probability 1/4.
    bad = [
        parceling(SCORE, TARGET, REJECT_SCORE, bins=EDGES, rng=seed)["target"].iloc[0]
        for seed in range(400)
    ]
    # Binomial(400, 1/4): mean 100, standard deviation 8.7.
    assert 70 < sum(bad) < 130


def test_parceling_counts_stay_within_one_of_the_expected_number():
    rng = np.random.default_rng(7)
    score = rng.random(300)
    target = (rng.random(300) < score).astype(int)
    reject_score = rng.random(130)
    edges = np.array([0, 0.25, 0.5, 0.75, 1])
    inferred = parceling(score, target, reject_score, bins=edges, factor=1.3, rng=1)
    accepted = np.searchsorted(edges, score, side="left") - 1
    rejected = np.searchsorted(edges, reject_score, side="left") - 1
    for band in range(4):
        rate = min(1.3 * target[accepted == band].mean(), 1)
        expected = rate * (rejected == band).sum()
        bads = inferred["target"].to_numpy()[rejected == band].sum()
        assert math.floor(expected) <= bads <= math.ceil(expected)


def test_parceling_expected_weights_are_the_band_rates():
    inferred = parceling(SCORE, TARGET, REJECT_SCORE, bins=EDGES, expected=True)
    assert inferred.index.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert inferred["target"].tolist() == [1, 0] * 5
    np.testing.assert_allclose(inferred["weight"], [0.25, 0.75] + [0.75, 0.25] * 4)


def test_parceling_factor_scales_and_clips_the_band_rates():
    inferred = parceling(SCORE, TARGET, REJECT_SCORE, bins=EDGES, factor=2, expected=True)
    # 2 * 1/4 = 1/2 in the low band; 2 * 3/4 is clipped to 1 in the high band.
    np.testing.assert_allclose(inferred["weight"], [0.5, 0.5] + [1.0, 0.0] * 4)
    labels = parceling(SCORE, TARGET, REJECT_SCORE, bins=EDGES, factor=2, rng=0)
    assert labels["target"].iloc[1:].tolist() == [1, 1, 1, 1]


def test_parceling_keeps_the_index_of_reject_score():
    reject_score = pd.Series(REJECT_SCORE, index=list("vwxyz"))
    labels = parceling(SCORE, TARGET, reject_score, bins=EDGES, rng=0)
    assert labels.index.tolist() == list("vwxyz")
    expected = parceling(SCORE, TARGET, reject_score, bins=EDGES, expected=True)
    assert expected.index.tolist() == list("vvwwxxyyzz")


def test_parceling_is_reproducible_and_leaves_global_state_alone():
    # Reads the legacy global state only to check that the function does not touch it.
    state = np.random.get_state()  # noqa: NPY002
    first = parceling(SCORE, TARGET, REJECT_SCORE, bins=EDGES, rng=3)
    after = np.random.get_state()  # noqa: NPY002
    assert all(np.array_equal(a, b) for a, b in zip(state, after, strict=True))
    pd.testing.assert_frame_equal(
        first, parceling(SCORE, TARGET, REJECT_SCORE, bins=EDGES, rng=np.random.default_rng(3))
    )
    generator = np.random.default_rng(3)
    draws = [
        parceling(SCORE, TARGET, REJECT_SCORE, bins=EDGES, rng=generator)["target"].iloc[0]
        for _ in range(40)
    ]
    assert 0 < sum(draws) < 40


def test_integer_bins_are_quantiles_of_accepted_and_rejected_scores_together():
    rng = np.random.default_rng(11)
    score = rng.normal(0, 1, 200)
    target = (rng.random(200) < 0.3).astype(int)
    reject_score = rng.normal(1, 1, 80)
    edges = bin_edges(np.r_[score, reject_score], bins=4)
    pd.testing.assert_frame_equal(
        parceling(score, target, reject_score, bins=4, expected=True),
        parceling(score, target, reject_score, bins=edges, expected=True),
    )
    np.testing.assert_array_equal(
        reweighting(score, reject_score, bins=4), reweighting(score, reject_score, bins=edges)
    )


def test_tied_scores_warn_at_the_callers_line():
    score = [1.0] * 8 + [2.0, 3.0]
    target = [0, 1] * 5
    message = (
        "tied scores leave 2 of the 5 requested bands; the score 1.0 makes up 75% of the "
        "scores, pass fewer bins or explicit edges"
    )
    for call in (
        lambda: parceling(score, target, [1.0, 3.0], bins=5, rng=0),
        lambda: reweighting(score, [1.0, 3.0], bins=5),
    ):
        with pytest.warns(UserWarning, match=re.escape(message)) as record:
            call()
        assert len(record) == 1
        assert record[0].filename == __file__


def test_parceling_raises_for_a_band_without_accepted_applicants():
    message = (
        r"the score band \(0.5, 1.0\] holds 4 rejected applicant\(s\) but no accepted one, "
        r"so its bad rate is unknown; use fewer bins or wider edges"
    )
    with pytest.raises(ValueError, match=message):
        parceling(SCORE[:4], TARGET[:4], REJECT_SCORE, bins=EDGES)


def test_parceling_ignores_bands_without_rejects():
    inferred = parceling(SCORE, TARGET, [0.25], bins=[0, 0.5, 0.75, 1], expected=True)
    np.testing.assert_allclose(inferred["weight"], [0.25, 0.75])


def test_scores_outside_the_edges_raise():
    message = r"reject_score has 1 value\(s\) outside the edges in bins, such as 1.5"
    with pytest.raises(ValueError, match=message):
        parceling(SCORE, TARGET, [0.2, 1.5], bins=EDGES)
    with pytest.raises(ValueError, match=r"score has 1 value\(s\) outside the edges"):
        reweighting([0.2, -1.0], [0.3], bins=EDGES)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target": [0, 0, 0, 0, 0, 0, 0, 0]}, "target must contain both 0 and 1, got only 0"),
        ({"target": [0, 2, 0, 1, 0, 1, 0, 1]}, "target must contain only 0 and 1, got 2"),
        ({"score": SCORE[:-1]}, "score and target must have the same length, got 7 and 8"),
        ({"score": [np.nan] + SCORE[1:]}, r"score must be finite, got 1 missing or infinite"),
        ({"reject_score": []}, "reject_score must not be empty"),
        ({"reject_score": ["a"]}, "reject_score must be real and numeric, got dtype"),
        ({"reject_score": [0.2, np.inf]}, r"reject_score must be finite, got 1 missing"),
        ({"bins": 0}, "bins must be a positive integer or at least two increasing edges, got 0"),
        ({"bins": [0, 1, 0.5]}, "bins must be a positive integer or at least two increasing"),
        ({"expected": "yes"}, "expected must be True or False, got 'yes'"),
        ({"rng": 1.5}, "rng must be None, an integer, a SeedSequence or a Generator, got 1.5"),
        ({"rng": "a"}, "rng must be None, an integer, a SeedSequence or a Generator, got 'a'"),
    ],
)
def test_parceling_rejects_invalid_arguments(kwargs, message):
    arguments = {"score": SCORE, "target": TARGET, "reject_score": REJECT_SCORE, "bins": EDGES}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        parceling(
            arguments.pop("score"),
            arguments.pop("target"),
            arguments.pop("reject_score"),
            **arguments,
        )


def test_parceling_requires_identical_indices_for_series():
    with pytest.raises(ValueError, match="score and target must have identical indices"):
        parceling(pd.Series(SCORE), pd.Series(TARGET, index=range(1, 9)), REJECT_SCORE)


def test_options_are_keyword_only():
    with pytest.raises(TypeError, match="positional argument"):
        fuzzy_augmentation([0.5], 2)
    with pytest.raises(TypeError, match="positional argument"):
        parceling(SCORE, TARGET, REJECT_SCORE, EDGES)
    with pytest.raises(TypeError, match="positional argument"):
        reweighting(SCORE, REJECT_SCORE, EDGES)


def test_parceling_leaves_its_inputs_alone():
    score, target = pd.Series(SCORE), pd.Series(TARGET)
    reject_score = pd.Series(REJECT_SCORE)
    copies = score.copy(), target.copy(), reject_score.copy()
    parceling(score, target, reject_score, bins=EDGES, factor=2, rng=0)
    for original, copy in zip((score, target, reject_score), copies, strict=True):
        pd.testing.assert_series_equal(original, copy)


# --- reweighting --------------------------------------------------------------------------


def test_reweighting_gives_the_inverse_approval_rate_of_each_band():
    score = pd.Series([0.1, 0.2, 0.2, 0.3, 0.6, 0.7], index=list("abcdef"), name="pd")
    weight = reweighting(score, REJECT_SCORE, bins=EDGES)
    # Low band: (4 + 1) / 4; high band: (2 + 4) / 2.
    pd.testing.assert_series_equal(
        weight,
        pd.Series([1.25, 1.25, 1.25, 1.25, 3.0, 3.0], index=list("abcdef"), name="weight"),
    )
    assert weight.sum() == len(score) + len(REJECT_SCORE)


def test_reweighting_returns_an_array_for_an_array():
    weight = reweighting(np.array(SCORE), REJECT_SCORE, bins=EDGES)
    assert isinstance(weight, np.ndarray)
    np.testing.assert_allclose(weight, [1.25] * 4 + [2.0] * 4)


def test_reweighting_weights_add_up_to_the_applicants_of_every_band():
    rng = np.random.default_rng(2)
    score = rng.normal(0, 1, 500)
    reject_score = rng.normal(1.5, 1, 200)
    weight = reweighting(score, reject_score)
    edges = bin_edges(np.r_[score, reject_score], bins=10)
    accepted = np.searchsorted(edges, score, side="left") - 1
    rejected = np.searchsorted(edges, reject_score, side="left") - 1
    for band in range(10):
        assert weight[accepted == band].sum() == pytest.approx(
            (accepted == band).sum() + (rejected == band).sum()
        )
    assert weight.min() >= 1


def test_reweighting_raises_beyond_a_hard_cut_off():
    # Approval below 0.5 on the score itself leaves the upper bands without accepts.
    message = (
        r"the score band \(0.5, 1.0\] holds 4 rejected applicant\(s\) but no accepted one, so no "
        r"accepted applicant can stand in for them; use fewer bins or wider edges"
    )
    with pytest.raises(ValueError, match=message):
        reweighting(SCORE[:4], REJECT_SCORE, bins=EDGES)


@pytest.mark.parametrize(
    ("score", "reject_score", "message"),
    [
        ([], [0.3], "score must not be empty"),
        ([0.2, np.nan], [0.3], r"score must be finite, got 1 missing or infinite"),
        ([[0.2]], [0.3], "score must be one-dimensional, got 2 dimensions"),
        ([0.2], ["x"], "reject_score must be real and numeric, got dtype"),
    ],
)
def test_reweighting_rejects_invalid_scores(score, reject_score, message):
    with pytest.raises(ValueError, match=message):
        reweighting(score, reject_score, bins=EDGES)
