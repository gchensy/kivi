"""Type checkers infer the result type of the functions whose result follows their input.

Each case is an expression and the type that its overloads declare. mypy checks all the cases at
once, with the project's settings, through ``typing.assert_type``; each case also runs, so that
the declared type is the type of the result and no overload claims a type the function does not
return. The calls in `ARRAY_LIKE` are only checked: arguments typed ``numpy.typing.ArrayLike``
get the catch-all overload with the implementation's union, and a ``Series[Any]`` raises no error.
"""

import pathlib

import numpy as np
import pytest

import kivi

ROOT = pathlib.Path(__file__).parents[1]

SETUP = """\
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd

from kivi import (
    binning,
    calibration,
    capital,
    lgd,
    lifetime,
    reject_inference,
    scorecard,
    strategy,
    woe,
)

SERIES = pd.Series([0.1, 0.2, 0.4], index=[3, 5, 7], name="probability")
LIST = [0.1, 0.2, 0.4]
ARRAY: npt.NDArray[np.float64] = np.array([0.1, 0.2, 0.4])
EDGES = [-np.inf, 0.15, np.inf]
TABLE = woe.woe_table(binning.cut(SERIES, EDGES), pd.Series([0, 1, 1], index=SERIES.index))
LABELS = pd.Series(["a", "b", "b"], name="grade")
LABEL_TABLE = woe.woe_table(LABELS, pd.Series([0, 1, 0]))
GROUPS = binning.category_groups(["a", "a", "a", "b", "c"], min_share=0.3)
CURVES = pd.DataFrame([[0.01, 0.03], [0.02, 0.05]], index=["x", "y"], columns=[1, 2])
SCALE = calibration.master_scale(pd.Series([0.01, 0.05, 0.2], index=["A", "B", "C"]))
"""

CASES = {
    "pd_to_score-series": (
        "scorecard.pd_to_score(SERIES, pdo=20, base_score=600, base_odds=50)",
        "pd.Series",
    ),
    "pd_to_score-float": (
        "scorecard.pd_to_score(0.1, pdo=20, base_score=600, base_odds=50)",
        "float",
    ),
    "pd_to_score-list": (
        "scorecard.pd_to_score(LIST, pdo=20, base_score=600, base_odds=50)",
        "npt.NDArray[np.float64]",
    ),
    "pd_to_score-array": (
        "scorecard.pd_to_score(ARRAY, pdo=20, base_score=600, base_odds=50)",
        "npt.NDArray[np.float64]",
    ),
    "score_to_pd-series": (
        "scorecard.score_to_pd(SERIES * 1000, pdo=20, base_score=600, base_odds=50)",
        "pd.Series",
    ),
    "score_to_pd-int": (
        "scorecard.score_to_pd(600, pdo=20, base_score=600, base_odds=50)",
        "float",
    ),
    "prior_correction-series": (
        "calibration.prior_correction(SERIES, sample_rate=0.5, population_rate=0.1)",
        "pd.Series",
    ),
    "prior_correction-tuple": (
        "calibration.prior_correction((0.1, 0.2), sample_rate=0.5, population_rate=0.1)",
        "npt.NDArray[np.float64]",
    ),
    "platt_predict-float": ("calibration.platt_predict(0.5, 1.5, -0.5)", "float"),
    "platt_predict-array": (
        "calibration.platt_predict(ARRAY, 1.5, -0.5)",
        "npt.NDArray[np.float64]",
    ),
    "isotonic_predict-series": (
        "calibration.isotonic_predict(SERIES, EDGES, [0.05, 0.3])",
        "pd.Series",
    ),
    "binomial_test-series": (
        "calibration.binomial_test(pd.Series([1, 4]), 100, [0.01, 0.02])",
        "pd.Series",
    ),
    "binomial_test-float": ("calibration.binomial_test(1, 100, 0.01)", "float"),
    "jeffreys_test-series": (
        "calibration.jeffreys_test([1, 4], [100, 100], pd.Series([0.01, 0.02]))",
        "pd.Series",
    ),
    "jeffreys_test-list": (
        "calibration.jeffreys_test([1, 4], 100, [0.01, 0.02])",
        "npt.NDArray[np.float64]",
    ),
    "fit_grade_pd-series": (
        "calibration.fit_grade_pd([2, 5, 20], pd.Series([400, 300, 200]))",
        "pd.Series",
    ),
    "fit_grade_pd-list": (
        "calibration.fit_grade_pd([2, 5, 20], [400, 300, 200])",
        "npt.NDArray[np.float64]",
    ),
    "assign_grades-series": ("calibration.assign_grades(SERIES, SCALE)", "pd.Series"),
    "assign_grades-list": ("calibration.assign_grades(LIST, SCALE)", "pd.Categorical"),
    "expected_loss-series": ("strategy.expected_loss(SERIES, 0.45, 1000.0)", "pd.Series"),
    "expected_loss-ead-series": (
        "strategy.expected_loss(LIST, 0.45, pd.Series([1000, 500, 200]))",
        "pd.Series",
    ),
    "expected_loss-float": ("strategy.expected_loss(0.02, 0.45, 1000)", "float"),
    "expected_loss-list": (
        "strategy.expected_loss(LIST, 0.45, [1000, 500, 200])",
        "npt.NDArray[np.float64]",
    ),
    "reweighting-series": (
        "reject_inference.reweighting(SERIES, [0.3, 0.5], bins=[0, 0.25, 1])",
        "pd.Series",
    ),
    "reweighting-list": (
        "reject_inference.reweighting(LIST, [0.3, 0.5], bins=[0, 0.25, 1])",
        "npt.NDArray[np.float64]",
    ),
    "woe-series": ("woe.woe([1, 3], pd.Series([4, 2], index=['a', 'b']))", "pd.Series"),
    "woe-list": ("woe.woe([1, 3], [4, 2])", "npt.NDArray[np.float64]"),
    "woe_transform-series": ("woe.woe_transform(LABELS, LABEL_TABLE)", "pd.Series"),
    "woe_transform-list": ("woe.woe_transform(['b', 'a'], LABEL_TABLE)", "npt.NDArray[np.float64]"),
    "woe_transform-categorical": (
        "woe.woe_transform(binning.cut(LIST, EDGES), TABLE)",
        "npt.NDArray[np.float64]",
    ),
    "cut-series": ("binning.cut(SERIES, EDGES)", "pd.Series"),
    "cut-list": ("binning.cut(LIST, EDGES, special=[-1])", "pd.Categorical"),
    "cut-array": ("binning.cut(ARRAY, EDGES)", "pd.Categorical"),
    "group_categories-series": (
        "binning.group_categories(pd.Series(['a', 'c']), GROUPS)",
        "pd.Series",
    ),
    "group_categories-list": ("binning.group_categories(['a', 'c'], GROUPS)", "pd.Categorical"),
    "convert_pd-frame": (
        "lifetime.convert_pd(CURVES, kind='cumulative', to='marginal')",
        "pd.DataFrame",
    ),
    "convert_pd-series": (
        "lifetime.convert_pd(SERIES, kind='marginal', to='cumulative')",
        "pd.Series",
    ),
    "convert_pd-nested": (
        "lifetime.convert_pd([[0.01, 0.03]], kind='cumulative', to='marginal')",
        "npt.NDArray[np.float64]",
    ),
    "expected_credit_loss-frame": (
        "lifetime.expected_credit_loss(CURVES, 0.45, 1000, rate=0.05)",
        "pd.Series",
    ),
    "expected_credit_loss-nested": (
        "lifetime.expected_credit_loss([[0.01, 0.02]], 0.45, [1000], rate=0.05)",
        "npt.NDArray[np.float64]",
    ),
    "significant_increase-series": (
        "lifetime.significant_increase(SERIES, [0.05, 0.05, 0.1], relative=2, absolute=None)",
        "pd.Series",
    ),
    "significant_increase-list": (
        "lifetime.significant_increase(LIST, [0.05, 0.05, 0.1], relative=2, absolute=None)",
        "npt.NDArray[np.bool_]",
    ),
    "assign_stages-defaulted-series": (
        "lifetime.assign_stages([False, True], None, pd.Series([False, False]))",
        "pd.Series",
    ),
    "assign_stages-list": (
        "lifetime.assign_stages([False, True], [0, 45])",
        "npt.NDArray[np.int64]",
    ),
    "asset_correlation-series": (
        "capital.asset_correlation(SERIES, asset_class='corporate')",
        "pd.Series",
    ),
    "asset_correlation-turnover-series": (
        "capital.asset_correlation(LIST, asset_class='corporate', turnover=SERIES * 100)",
        "pd.Series",
    ),
    "asset_correlation-float": (
        "capital.asset_correlation(0.01, asset_class='corporate', turnover=10.0)",
        "float",
    ),
    "asset_correlation-list": (
        "capital.asset_correlation(LIST, asset_class='other_retail')",
        "npt.NDArray[np.float64]",
    ),
    "asset_correlation-array": (
        "capital.asset_correlation(ARRAY, asset_class='corporate', turnover=[10.0, 20.0, 60.0])",
        "npt.NDArray[np.float64]",
    ),
    "capital_requirement-series": (
        "capital.capital_requirement(SERIES, 0.45, asset_class='corporate')",
        "pd.Series",
    ),
    "capital_requirement-lgd-series": (
        "capital.capital_requirement(LIST, pd.Series([0.45, 0.4, 0.3]), asset_class='mortgage')",
        "pd.Series",
    ),
    "capital_requirement-maturity-series": (
        "capital.capital_requirement(LIST, 0.45, asset_class='corporate', maturity=SERIES * 10)",
        "pd.Series",
    ),
    "capital_requirement-turnover-series": (
        "capital.capital_requirement(LIST, 0.45, asset_class='corporate', turnover=SERIES * 100)",
        "pd.Series",
    ),
    "capital_requirement-correlation-series": (
        "capital.capital_requirement(LIST, 0.45, asset_class='mortgage', correlation=SERIES / 2)",
        "pd.Series",
    ),
    "capital_requirement-float": (
        "capital.capital_requirement(0.01, 0.45, asset_class='corporate', maturity=2.5)",
        "float",
    ),
    "capital_requirement-list": (
        "capital.capital_requirement(LIST, [0.45, 0.4, 0.3], asset_class='revolving')",
        "npt.NDArray[np.float64]",
    ),
    "capital_requirement-array": (
        "capital.capital_requirement(ARRAY, 0.45, asset_class='hvcre', maturity=ARRAY * 10)",
        "npt.NDArray[np.float64]",
    ),
    "conditional_pd-series": (
        "capital.conditional_pd(SERIES, correlation=0.12, factor=-1.0)",
        "pd.Series",
    ),
    "conditional_pd-correlation-series": (
        "capital.conditional_pd(LIST, correlation=pd.Series([0.1, 0.12, 0.15]), level=0.99)",
        "pd.Series",
    ),
    "conditional_pd-factor-series": (
        "capital.conditional_pd(0.02, correlation=0.12, factor=pd.Series([-1.0, 0.0, 1.0]))",
        "pd.Series",
    ),
    "conditional_pd-level-series": (
        "capital.conditional_pd(0.02, correlation=0.12, level=pd.Series([0.9, 0.99, 0.999]))",
        "pd.Series",
    ),
    "conditional_pd-float": (
        "capital.conditional_pd(0.02, correlation=0.12, level=0.999)",
        "float",
    ),
    "conditional_pd-list": (
        "capital.conditional_pd(LIST, correlation=0.12, factor=[-1.0, 0.0, 1.0])",
        "npt.NDArray[np.float64]",
    ),
    "conditional_pd-array": (
        "capital.conditional_pd(ARRAY, correlation=0.12, factor=0.0)",
        "npt.NDArray[np.float64]",
    ),
    "systematic_factor-series": (
        "capital.systematic_factor(SERIES, 0.2, correlation=0.12)",
        "pd.Series",
    ),
    "systematic_factor-probability-series": (
        "capital.systematic_factor(LIST, pd.Series([0.2, 0.2, 0.2]), correlation=0.12)",
        "pd.Series",
    ),
    "systematic_factor-correlation-series": (
        "capital.systematic_factor(LIST, 0.2, correlation=pd.Series([0.1, 0.12, 0.15]))",
        "pd.Series",
    ),
    "systematic_factor-float": (
        "capital.systematic_factor(0.03, 0.02, correlation=0.12)",
        "float",
    ),
    "systematic_factor-list": (
        "capital.systematic_factor(LIST, 0.2, correlation=0.12)",
        "npt.NDArray[np.float64]",
    ),
    "systematic_factor-array": (
        "capital.systematic_factor(ARRAY, ARRAY, correlation=0.12)",
        "npt.NDArray[np.float64]",
    ),
    "conversion_factor-series": (
        "lgd.conversion_factor(SERIES * 100, SERIES * 50, 100.0)",
        "pd.Series",
    ),
    "conversion_factor-drawn-series": (
        "lgd.conversion_factor(LIST, SERIES / 10, 1.0)",
        "pd.Series",
    ),
    "conversion_factor-limit-series": (
        "lgd.conversion_factor(LIST, 0.0, SERIES + 1)",
        "pd.Series",
    ),
    "conversion_factor-float": (
        "lgd.conversion_factor(80.0, 50.0, 100.0)",
        "float",
    ),
    "conversion_factor-list": (
        "lgd.conversion_factor(LIST, 0.05, [1.0, 1.0, 1.0])",
        "npt.NDArray[np.float64]",
    ),
    "conversion_factor-array": (
        "lgd.conversion_factor(ARRAY, ARRAY / 2, 1.0)",
        "npt.NDArray[np.float64]",
    ),
    "most_prudent_pd-event-series": (
        "capital.most_prudent_pd(pd.Series([0, 1, 2]), [100, 200, 300])",
        "pd.Series",
    ),
    "most_prudent_pd-total-series": (
        "capital.most_prudent_pd([0, 1, 2], pd.Series([100, 200, 300]))",
        "pd.Series",
    ),
    "most_prudent_pd-list": (
        "capital.most_prudent_pd([0, 1, 2], [100, 200, 300], level=0.9)",
        "npt.NDArray[np.float64]",
    ),
    "most_prudent_pd-array": (
        "capital.most_prudent_pd(np.zeros(2), np.ones(2) * 10, correlation=0.1)",
        "npt.NDArray[np.float64]",
    ),
}

ARRAY_LIKE = """
def array_like(values: npt.ArrayLike, frame: pd.DataFrame) -> None:
    typing.assert_type(
        scorecard.pd_to_score(values, pdo=20, base_score=600, base_odds=50),
        float | pd.Series | npt.NDArray[np.float64],
    )
    typing.assert_type(woe.woe(values, values), pd.Series | npt.NDArray[np.float64])
    typing.assert_type(binning.cut(values, EDGES), pd.Series | pd.Categorical)
    typing.assert_type(
        lifetime.convert_pd(values, kind="cumulative", to="marginal"),
        pd.DataFrame | pd.Series | npt.NDArray[np.float64],
    )
    # A DataFrame column, like a Series that kivi returns, is a Series[Any]. mypy matches it to the
    # Series overload and to the catch-all and so infers Any, where pyright infers the Series; with
    # either checker, the methods of a Series raise no error.
    scorecard.pd_to_score(frame["p"], pdo=20, base_score=600, base_odds=50).round(2).index
    woe.woe_transform(binning.cut(SERIES, EDGES), TABLE).round(2).index
"""


def snippet():
    """The module that mypy checks: the setup, then one ``typing.assert_type`` per case."""
    checks = "".join(
        f"typing.assert_type({expression}, {declared})  # {case}\n"
        for case, (expression, declared) in CASES.items()
    )
    return f"{SETUP}\n{checks}{ARRAY_LIKE}"


@pytest.fixture(scope="module")
def namespace():
    names = {}
    exec(SETUP, names)
    return names


@pytest.mark.parametrize("case", list(CASES))
def test_result_has_the_declared_type(case, namespace):
    expression, declared = CASES[case]
    result = eval(expression, namespace)
    if declared.startswith("npt.NDArray["):
        scalar = eval(declared.removeprefix("npt.NDArray[").removesuffix("]"), namespace)
        assert isinstance(result, np.ndarray) and result.dtype == scalar
    else:
        assert type(result) is eval(declared, namespace)


def test_mypy_infers_the_declared_types(tmp_path, monkeypatch):
    api = pytest.importorskip("mypy.api")
    module = tmp_path / "overloads.py"
    module.write_text(snippet())
    source = pathlib.Path(kivi.__file__).parents[1]
    if source == ROOT / "src":
        # mypy does not read PYTHONPATH: check the kivi that the tests import, also in a worktree.
        monkeypatch.setenv("MYPYPATH", str(source))
    options = ["--config-file", str(ROOT / "pyproject.toml"), "--cache-dir", str(tmp_path)]
    stdout, stderr, status = api.run([*options, str(module)])
    assert status == 0, stdout + stderr
