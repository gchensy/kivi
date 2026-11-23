"""Charts of the tutorial in docs/tutorial.

The tutorial follows one project across its chapters, so these recipes rebuild the
project's state once, with the code of the chapters, and draw each chart with the
functions of _plotting.py. The pages call them from ``.. plot::`` directives as
``kt.figure_*()``; change a recipe together with the example above its chart.
"""

import functools
from types import SimpleNamespace

import _plotting as kp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kivi.binning import bin_edges, chimerge_edges, monotonic_edges, tree_edges
from kivi.calibration import assign_grades, calibration_table, master_scale
from kivi.datasets import make_applications, make_loan_panel
from kivi.metrics import gains_table, gini, roc_auc, roc_table
from kivi.monitoring import monitoring_table, stability_table
from kivi.reject_inference import fuzzy_augmentation
from kivi.selection import logistic_regression, stepwise
from kivi.spec import fit_spec
from kivi.univariate import profile_table
from kivi.vintage import vintage_table
from kivi.woe import woe_table

FEATURES = ["age", "income", "debt_to_income", "inquiries", "housing", "employment", "incumbent_pd"]


@functools.cache
def _project() -> SimpleNamespace:
    """The samples, the binning and the models of the chapters."""
    applications = make_applications(n_applicants=40000, rng=0)
    months = pd.period_range("2024-01", periods=12, freq="M")
    applications["month"] = months[np.arange(40000) * 12 // 40000]
    eligible = applications[~applications["policy_decline"]]
    accepts = eligible[eligible["approved"]]
    development = accepts[accepts["month"] <= "2024-09"]
    validation = accepts[accepts["month"] >= "2024-10"]
    rejects = eligible[~eligible["approved"] & (eligible["month"] <= "2024-09")]

    method = {column: "monotonic" for column in ["age", "income", "debt_to_income", "incumbent_pd"]}
    method["employment"] = "chimerge"
    bins = {"inquiries": [-np.inf, 0, 1, 2, np.inf], "incumbent_pd": 8}
    spec = fit_spec(development[FEATURES], development["target"], method=method, bins=bins)
    selected = stepwise(spec.woe_transform(development), development["target"])
    accepts_model = logistic_regression(
        spec.woe_transform(development)[selected], development["target"]
    )

    def predict(model, frame):
        slopes = model["coef"].drop("intercept")
        log_odds = model.loc["intercept", "coef"] + spec.woe_transform(frame)[slopes.index] @ slopes
        return 1 / (1 + np.exp(-log_odds))

    inferred = fuzzy_augmentation(predict(accepts_model, rejects), factor=1.5)
    frame = pd.concat([development[FEATURES], rejects.loc[inferred.index, FEATURES]])
    target = pd.concat([development["target"], inferred["target"]])
    weights = pd.concat([pd.Series(1.0, index=development.index), inferred["weight"]])
    model = logistic_regression(spec.woe_transform(frame)[selected], target, weights=weights)
    return SimpleNamespace(
        eligible=eligible,
        development=development,
        validation=validation,
        spec=spec,
        model=model,
        predict=predict,
        master=master_scale(np.geomspace(0.02, 0.32, 5), labels=list("ABCDE")),
    )


def figure_profile_iv():
    p = _project()
    profile = profile_table(p.development[FEATURES], p.development["target"])
    return kp.iv_ranking(profile["iv"], limit=0.35)


def figure_binning_methods():
    p = _project()
    x, y = p.development["debt_to_income"], p.development["target"]
    candidates = {
        "quantile": bin_edges(x, bins=5),
        "chimerge": chimerge_edges(x, y, max_bins=5),
        "tree": tree_edges(x, y, max_bins=5),
        "monotonic": monotonic_edges(x, y, max_bins=5),
    }
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.2), sharey=True)
    for ax, (name, edges) in zip(axes.flat, candidates.items(), strict=True):
        kp._woe_bars(ax, woe_table(pd.cut(x, edges), y), name)
    fig.tight_layout()
    return fig


def figure_incumbent_woe():
    return kp.woe_rate_panels(_project().spec.tables["incumbent_pd"], "incumbent_pd")


def figure_roc():
    p = _project()
    y = p.validation["target"]
    probability = p.predict(p.model, p.validation)
    incumbent = p.validation["incumbent_pd"]
    return kp.roc_curves(
        {
            f"incumbent_pd: roc_auc = {roc_auc(incumbent, y):.3f}": roc_table(incumbent, y),
            f"probability: roc_auc = {roc_auc(probability, y):.3f}": roc_table(probability, y),
        }
    )


def figure_gains():
    p = _project()
    edges = bin_edges(p.predict(p.model, p.development), bins=5).round(3)
    probability = p.predict(p.model, p.validation)
    return kp.lift_gains_panels(gains_table(probability, p.validation["target"], bins=edges))


def figure_calibration():
    p = _project()
    probability = p.predict(p.model, p.validation)
    quarter = p.eligible[p.eligible["month"] >= "2024-10"]
    quarter_pd = p.predict(p.model, quarter)
    return kp.reliability_chart(
        {
            "validation": calibration_table(
                assign_grades(probability, p.master), p.validation["target"], probability
            ),
            "quarter": calibration_table(
                assign_grades(quarter_pd, p.master), quarter["target"], quarter_pd
            ),
        }
    )


@functools.cache
def _production() -> SimpleNamespace:
    """The applications of 2025 as the monitoring chapter scores them."""
    p = _project()
    new = make_applications(n_applicants=24000, rng=1)
    new["month"] = pd.period_range("2025-01", periods=12, freq="M")[np.arange(24000) * 12 // 24000]
    rng = np.random.default_rng(2)
    lost = (new["month"] >= "2025-10") & (new["incumbent_pd"] < 0.07) & (rng.random(24000) < 0.7)
    stream = new[~lost & ~new["policy_decline"]]
    stream_pd = p.predict(p.model, stream)
    approved = stream_pd <= p.master.loc["D", "upper"]
    outcome = stream["target"].where(approved & (stream["month"] <= "2025-09"))
    reference = p.eligible[p.eligible["month"] <= "2024-09"]
    return SimpleNamespace(
        report=monitoring_table(
            stream_pd, stream["month"], outcome, reference_score=p.predict(p.model, reference)
        ),
        stability=stability_table(
            p.spec.cut(stream), stream["month"], reference_groups=p.spec.cut(reference)
        ),
    )


def figure_monitoring():
    p = _project()
    development_gini = gini(p.predict(p.model, p.development), p.development["target"])
    return kp.monitoring_panels(_production().report, development_gini)


def figure_stability():
    fig = kp.stability_heatmap(_production().stability)
    plt.setp(fig.axes[0].get_xticklabels(), rotation=25, ha="left", rotation_mode="anchor")
    fig.tight_layout()
    return fig


def figure_vintage():
    panel = make_loan_panel(rng=0)
    ever_90 = vintage_table(
        panel["loan_id"], panel["cohort"], panel["month_on_book"], panel["status"] == "90+"
    )
    return kp.vintage_curves(ever_90.iloc[::6])
