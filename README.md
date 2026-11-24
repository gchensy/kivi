<div align="center">

<img src="https://raw.githubusercontent.com/gchensy/kivi/master/docs/_static/kivi-mark.svg" width="88" alt="kivi logo">

# kivi

**Credit risk modeling with pandas**

A toolkit for the whole scorecard project:<br>
derive point-in-time features, screen and bin the variables, fit and scale the model, calibrate it, price the cut-offs, monitor the model and provision for the loans it approves.

[![PyPI](https://img.shields.io/pypi/v/kivi?logo=pypi&logoColor=white&color=2f5bd3)](https://pypi.org/project/kivi/)
[![Downloads](https://img.shields.io/pypi/dm/kivi?color=0b7a70)](https://pypi.org/project/kivi/)
[![Documentation](https://github.com/gchensy/kivi/actions/workflows/docs.yml/badge.svg)](https://github.com/gchensy/kivi/actions/workflows/docs.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pandas 2.0+](https://img.shields.io/badge/pandas-2.0%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Docs: English | 中文](https://img.shields.io/badge/docs-English%20%7C%20%E4%B8%AD%E6%96%87-2f5bd3)](https://gchensy.github.io/kivi/)
[![License: MIT](https://img.shields.io/badge/license-MIT-0b7a70)](https://github.com/gchensy/kivi/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/gchensy/kivi?style=social)](https://github.com/gchensy/kivi)

**English** · [简体中文](https://github.com/gchensy/kivi/blob/master/README.zh-CN.md)

[Documentation](https://gchensy.github.io/kivi/en/) ·
[Quickstart](https://gchensy.github.io/kivi/en/getting_started/quickstart.html) ·
[User guide](https://gchensy.github.io/kivi/en/user_guide/index.html) ·
[API](https://gchensy.github.io/kivi/en/api.html) ·
[Release notes](https://gchensy.github.io/kivi/en/release/index.html)

</div>

---

## ✨ Highlights

- 🏦 **Built for risk teams.** Covers a credit scorecard project end to end: features of transactions and repayment histories, variable profiles, classing, variable selection, reject inference, a logistic fit or expert weights from pairwise comparisons, scaling to points, evaluation, calibration on a master scale, cut-off strategy and expected loss, deployment and monitoring by period, plus delinquency panels built from repayment schedules, vintage and roll-rate analysis of the approved loans and their IFRS 9 expected credit loss.
- 📐 **Fit once, reuse everywhere.** Bin edges, category groupings, WOE tables, master scales and cut-offs are fitted on the development sample and applied unchanged to later samples. `bin_edges` learns quantile or uniform edges, `chimerge_edges`, `tree_edges`, `ks_edges` and `monotonic_edges` also use the target, and `chimerge_groups` merges categories with similar event rates. The outer edges are infinite, so every future value still lands in a bin.
- ⏱️ **Point in time by construction.** `window_features` aggregates transactions and other events over windows of calendar months or days that end just before each observation time, such as an application date, and `panel_features` reads only the month-end repayment statuses before its month, so no later event leaks into a feature.
- 🔍 **Transparent results.** Tables return raw counts, smoothed shares and a contribution per bin, grade or cut-off, not just one number.
- 📏 **Uncertainty in view.** `roc_auc_interval` gives the DeLong confidence interval of an AUC and a Gini, and `delong_test` tests whether two scores rank the same sample equally well.
- 💾 **Reproducible in production.** `FeatureSpec` saves the definitions of the window and repayment features, `fit_spec` bins a whole data frame, and `BinningSpec` saves the edges and WOE tables, all as versioned JSON, so production computes and groups new data exactly as development did.
- 🧩 **Nothing silently dropped.** Missing values, special codes such as `-1` and categories seen in only one sample get their own group, and a category that a fitted grouping or WOE table does not know raises an error unless you say where it belongs.
- ⚖️ **Sample weights throughout.** Binning, WOE, PSI, the metrics, calibration, the model fits, the strategy tables, the monitoring report, vintage tables and the stage table take sample weights, and integer weights count like repeated rows: the same counts, rates and coefficients. Only quantile edges differ, because weighted quantiles do not interpolate.
- 🧊 **Honest smoothing.** Empty cells are replaced by `smoothing` (default 0.5) before normalizing, and the reported counts are never altered.
- 📊 **Charts from the tables.** `kivi.plotting` draws WOE, ROC, gains, calibration, cut-off, PSI, monitoring and vintage charts straight from the tables kivi returns, with matplotlib.
- 🐼 **Just pandas and NumPy at runtime.** matplotlib is needed only for the optional charts (`pip install "kivi[plot]"`). No Spark, no database drivers and no global random state in the core.
- 🧪 **Checked against reference libraries.** CI compares the re-implemented algorithms, such as the decision-tree splits, the isotonic and Platt calibrations, the logistic fits and the DeLong variance, with scikit-learn, SciPy and statsmodels.
- 🌏 **Bilingual, tested docs.** Every example in the English and Chinese manuals runs in CI.

## 📊 Gallery

<table>
  <tr>
    <td colspan="2" valign="top">
      <a href="https://gchensy.github.io/kivi/en/user_guide/features.html#features-of-synthetic-transactions"><img src="https://raw.githubusercontent.com/gchensy/kivi/master/docs/_static/readme-features.svg" alt="Share and event rate per band of the number of cash withdrawals in the last three months and in the three months before"></a>
      <br><b>Time-window features</b>: share and event rate per band of the cash withdrawals in the last three months and in the three months before, derived from transactions with a <code>FeatureSpec</code>.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="https://gchensy.github.io/kivi/en/user_guide/woe_iv.html#charting-woe"><img src="https://raw.githubusercontent.com/gchensy/kivi/master/docs/_static/readme-woe.svg" alt="WOE chart for age"></a>
      <br><b>WOE chart</b>: share of customers, event rate and WOE per bin.
    </td>
    <td width="50%" valign="top">
      <a href="https://gchensy.github.io/kivi/en/user_guide/risk_workflow.html"><img src="https://raw.githubusercontent.com/gchensy/kivi/master/docs/_static/readme-iv.svg" alt="Variables ranked by information value"></a>
      <br><b>Variable screening</b>: IV ranking with the usual reading bands; <code>duration</code> is a leak.
    </td>
  </tr>
  <tr>
    <td colspan="2" valign="top">
      <a href="https://gchensy.github.io/kivi/en/user_guide/psi.html#monitoring-over-time"><img src="https://raw.githubusercontent.com/gchensy/kivi/master/docs/_static/readme-psi.svg" alt="PSI by month with thresholds"></a>
      <br><b>Monitoring</b>: PSI per period against the development sample, with the 0.1 and 0.25 thresholds and sample sizes.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="https://gchensy.github.io/kivi/en/user_guide/metrics.html#gains-table"><img src="https://raw.githubusercontent.com/gchensy/kivi/master/docs/_static/readme-gains.svg" alt="Event rate per score band and the gains curve"></a>
      <br><b>Gains</b>: event rate and lift per score band, and the share of events captured.
    </td>
    <td width="50%" valign="top">
      <a href="https://gchensy.github.io/kivi/en/user_guide/vintage.html#vintage-curves"><img src="https://raw.githubusercontent.com/gchensy/kivi/master/docs/_static/readme-vintage.svg" alt="Cumulative bad rate by month on book for several cohorts"></a>
      <br><b>Vintage curves</b>: cumulative 90+ bad rate of each origination cohort.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="https://gchensy.github.io/kivi/en/user_guide/strategy.html#swap-sets"><img src="https://raw.githubusercontent.com/gchensy/kivi/master/docs/_static/readme-strategy.svg" alt="Bad rate of the approved book against the approval rate for a current and a new score"></a>
      <br><b>Cut-off strategy</b>: the bad rate of the approved book at each approval rate, for a current and a new score.
    </td>
    <td width="50%" valign="top">
      <a href="https://gchensy.github.io/kivi/en/user_guide/reject_inference.html#sensitivity-to-the-factor"><img src="https://raw.githubusercontent.com/gchensy/kivi/master/docs/_static/readme-reject.svg" alt="Inferred bad rate and refitted PD of the rejected applicants for several factors"></a>
      <br><b>Reject inference</b>: the inferred bad rate and the refitted PD of the declined applicants for each <code>factor</code>.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="https://gchensy.github.io/kivi/en/user_guide/monitoring.html#a-monitoring-report"><img src="https://raw.githubusercontent.com/gchensy/kivi/master/docs/_static/readme-monitoring.svg" alt="Population, known outcomes, Gini and PSI of a score by month"></a>
      <br><b>Model monitoring</b>: population, known outcomes, Gini and PSI of the score in every month.
    </td>
    <td width="50%" valign="top">
      <a href="https://gchensy.github.io/kivi/en/user_guide/lifetime.html#term-structures-from-vintage-curves"><img src="https://raw.githubusercontent.com/gchensy/kivi/master/docs/_static/readme-lifetime.svg" alt="Cumulative PD by month on book for each cohort and pooled"></a>
      <br><b>Lifetime PD</b>: the cumulative PD of each cohort and the pooled term structure used for the expected credit loss.
    </td>
  </tr>
</table>

## 🧭 How it works

Every step is defined or fitted on the development sample and applied unchanged to later samples:

| Step | Modules | Main functions |
| --- | --- | --- |
| Derive features | `kivi.features` | `window_features`, `panel_features`, `FeatureSpec` |
| Profile the candidates | `kivi.univariate` | `profile_table` |
| Bin and encode | `kivi.binning`, `kivi.woe` | `bin_edges`, `monotonic_edges`, `chimerge_groups`, `woe_table`, `woe_transform` |
| Select and fit | `kivi.selection`, `kivi.reject_inference` | `stepwise`, `logistic_regression`, `fuzzy_augmentation` |
| Scale and evaluate | `kivi.scorecard`, `kivi.metrics` | `points_table`, `roc_auc`, `roc_auc_interval`, `gains_table` |
| Calibrate | `kivi.calibration` | `master_scale`, `assign_grades`, `jeffreys_test` |
| Decide | `kivi.strategy` | `cutoff_table`, `confusion_table`, `swap_set`, `expected_loss_table` |
| Deploy and monitor | `kivi.spec`, `kivi.monitoring`, `kivi.stability` | `fit_spec`, `monitoring_table`, `stability_table`, `csi_table` |
| Follow the loans | `kivi.vintage`, `kivi.lifetime` | `delinquency_panel`, `vintage_table`, `transition_pd`, `expected_credit_loss`, `assign_stages` |
| Draw the results | `kivi.plotting` | `woe_chart`, `roc_chart`, `gains_chart`, `monitoring_chart`, `vintage_chart` |

The first steps show the pattern: define the features once and compute them for every sample, fit the bin edges once and cut every sample with them, then measure.

<p align="center">
  <img src="https://raw.githubusercontent.com/gchensy/kivi/master/docs/_static/kivi-workflow.svg" width="820" alt="Define features once with FeatureSpec and compute them for every sample, fit edges once with bin_edges, cut every sample with pandas.cut, then measure with woe_table and psi_table">
</p>

## 📦 Installation

```bash
pip install kivi
```

Requires Python 3.10+, NumPy 1.24+ and pandas 2.0+. For the latest development version, use `pip install "git+https://github.com/gchensy/kivi"`.

The charts of `kivi.plotting` also need matplotlib 3.7+: install them with `pip install "kivi[plot]"`.

> **Upgrading from 0.2.x?** The API changed in 0.3. Read the [migration guide](https://gchensy.github.io/kivi/en/release/migration.html) before upgrading existing code.

## 🚀 Quickstart

```python
import pandas as pd
from kivi.binning import bin_edges
from kivi.datasets import load_bank
from kivi.stability import psi_table
from kivi.woe import woe_table

data = load_bank()                              # 4,521 bank marketing customers
reference, current = data.iloc[:3000], data.iloc[3000:]

edges = bin_edges(reference["age"], bins=5)     # 1. fit once on the reference
before = pd.cut(reference["age"], edges)        # 2. cut every sample
after = pd.cut(current["age"], edges)

evidence = woe_table(before, reference["target"])  # 3. measure
stability = psi_table(before, after)

print(evidence[["event", "non_event", "woe", "iv"]].round(3))
print(f"IV = {evidence['iv'].sum():.3f}, PSI = {stability['psi'].sum():.4f}")
```

```text
              event  non_event    woe     iv
bin
(-inf, 32.0]     90        612  0.114  0.003
(32.0, 36.0]     55        475 -0.125  0.003
(36.0, 42.0]     58        529 -0.180  0.006
(42.0, 51.0]     71        569 -0.050  0.001
(51.0, inf]      74        467  0.189  0.007
IV = 0.019, PSI = 0.0024
```

Categorical columns skip the edges; pass their labels as they are, or group them first with `category_groups` or `chimerge_groups`:

```python
jobs = woe_table(reference["job"], reference["target"])
print(f"IV(job) = {jobs['iv'].sum():.3f}")      # IV(job) = 0.137
```

## 🧰 API at a glance

| Function | What it does |
| --- | --- |
| [`kivi.features.window_features`](https://gchensy.github.io/kivi/en/api/features.html#kivi.features.window_features) | Aggregate transactions and other events over windows of months or days that end just before each observation time, point in time: counts, sums, means, extremes, distinct values, days since the last event and ratios |
| [`kivi.features.panel_features`](https://gchensy.github.io/kivi/en/api/features.html#kivi.features.panel_features) | Summarize monthly repayment statuses before an observation time: months covered, bad months, longest bad and good runs, worst status and months since the last bad month |
| [`kivi.features.FeatureSpec`](https://gchensy.github.io/kivi/en/api/features.html#kivi.features.FeatureSpec) | Definitions of window and repayment features, computed again on any sample and saved as versioned JSON |
| [`kivi.univariate.profile_table`](https://gchensy.github.io/kivi/en/api/univariate.html#kivi.univariate.profile_table) | One row per candidate: missing values, distribution and shape, event rates, the IV and chi-squared test of a quick binning, and the Gini and KS of numeric values |
| [`kivi.binning.bin_edges`](https://gchensy.github.io/kivi/en/api/binning.html#kivi.binning.bin_edges) | Fit quantile or uniform edges on a reference sample, with infinite outer edges |
| [`kivi.binning.cut`](https://gchensy.github.io/kivi/en/api/binning.html#kivi.binning.cut) | Assign values to bins, with a separate group for each special value such as `-1` |
| [`kivi.binning.chimerge_edges`](https://gchensy.github.io/kivi/en/api/binning.html#kivi.binning.chimerge_edges) | Merge fine quantile bins with ChiMerge until neighbours differ significantly |
| [`kivi.binning.tree_edges`](https://gchensy.github.io/kivi/en/api/binning.html#kivi.binning.tree_edges) | Decision-tree splits by entropy or Gini, with a minimum share per bin |
| [`kivi.binning.ks_edges`](https://gchensy.github.io/kivi/en/api/binning.html#kivi.binning.ks_edges) | Best-KS splits where the cumulative shares of events and non-events differ most, with a minimum share per bin |
| [`kivi.binning.monotonic_edges`](https://gchensy.github.io/kivi/en/api/binning.html#kivi.binning.monotonic_edges) | Merge fine quantile bins until the WOE is monotonic |
| [`kivi.binning.category_groups`](https://gchensy.github.io/kivi/en/api/binning.html#kivi.binning.category_groups), [`kivi.binning.chimerge_groups`](https://gchensy.github.io/kivi/en/api/binning.html#kivi.binning.chimerge_groups), [`kivi.binning.group_categories`](https://gchensy.github.io/kivi/en/api/binning.html#kivi.binning.group_categories) | Group the categories of a categorical variable: pool the rare ones or merge those with similar event rates, then apply the grouping to every sample |
| [`kivi.woe.woe_table`](https://gchensy.github.io/kivi/en/api/woe.html#kivi.woe.woe_table) | Counts, shares, WOE and IV for every group |
| [`kivi.woe.woe_transform`](https://gchensy.github.io/kivi/en/api/woe.html#kivi.woe.woe_transform) | Replace each group with its development WOE, with an explicit policy for unseen groups |
| [`kivi.woe.woe`](https://gchensy.github.io/kivi/en/api/woe.html#kivi.woe.woe), [`kivi.woe.iv`](https://gchensy.github.io/kivi/en/api/woe.html#kivi.woe.iv), [`kivi.stability.psi`](https://gchensy.github.io/kivi/en/api/stability.html#kivi.stability.psi) | WOE, IV and PSI from counts that are already aggregated |
| [`kivi.selection.correlation_filter`](https://gchensy.github.io/kivi/en/api/selection.html#kivi.selection.correlation_filter), [`kivi.selection.vif`](https://gchensy.github.io/kivi/en/api/selection.html#kivi.selection.vif) | Drop correlated variables by IV, with Pearson or Spearman correlation, and variance inflation factors |
| [`kivi.selection.logistic_regression`](https://gchensy.github.io/kivi/en/api/selection.html#kivi.selection.logistic_regression), [`kivi.selection.stepwise`](https://gchensy.github.io/kivi/en/api/selection.html#kivi.selection.stepwise) | NumPy logistic regression and stepwise selection that keeps every coefficient positive, with frequency weights |
| [`kivi.reject_inference.fuzzy_augmentation`](https://gchensy.github.io/kivi/en/api/reject_inference.html#kivi.reject_inference.fuzzy_augmentation), [`kivi.reject_inference.parceling`](https://gchensy.github.io/kivi/en/api/reject_inference.html#kivi.reject_inference.parceling), [`kivi.reject_inference.reweighting`](https://gchensy.github.io/kivi/en/api/reject_inference.html#kivi.reject_inference.reweighting) | Infer the outcomes of declined applicants, as weighted rows or as weights of the approved ones |
| [`kivi.scorecard.score_scaling`](https://gchensy.github.io/kivi/en/api/scorecard.html#kivi.scorecard.score_scaling), [`kivi.scorecard.pd_to_score`](https://gchensy.github.io/kivi/en/api/scorecard.html#kivi.scorecard.pd_to_score), [`kivi.scorecard.score_to_pd`](https://gchensy.github.io/kivi/en/api/scorecard.html#kivi.scorecard.score_to_pd) | PDO scaling factor and offset, and conversion between PD and score |
| [`kivi.scorecard.points_table`](https://gchensy.github.io/kivi/en/api/scorecard.html#kivi.scorecard.points_table), [`kivi.scorecard.woe_to_points`](https://gchensy.github.io/kivi/en/api/scorecard.html#kivi.scorecard.woe_to_points), [`kivi.scorecard.reason_codes`](https://gchensy.github.io/kivi/en/api/scorecard.html#kivi.scorecard.reason_codes) | Points per group with PDO scaling, scores for new data, and reason codes |
| [`kivi.expert.ahp_weights`](https://gchensy.github.io/kivi/en/api/expert.html#kivi.expert.ahp_weights), [`kivi.expert.ahp_consistency`](https://gchensy.github.io/kivi/en/api/expert.html#kivi.expert.ahp_consistency) | Factor weights of an expert scorecard from a pairwise comparison matrix, by the principal eigenvector or the geometric means of the rows, and the consistency ratio of the judgments |
| [`kivi.expert.combine_weights`](https://gchensy.github.io/kivi/en/api/expert.html#kivi.expert.combine_weights) | Global weights of sub-factors: the weight of each factor times the weights of its sub-factors |
| [`kivi.metrics.roc_auc`](https://gchensy.github.io/kivi/en/api/metrics.html#kivi.metrics.roc_auc), [`kivi.metrics.gini`](https://gchensy.github.io/kivi/en/api/metrics.html#kivi.metrics.gini), [`kivi.metrics.ks`](https://gchensy.github.io/kivi/en/api/metrics.html#kivi.metrics.ks), [`kivi.metrics.brier_score`](https://gchensy.github.io/kivi/en/api/metrics.html#kivi.metrics.brier_score) | AUC, Gini, KS and Brier score of a score or PD against a binary target |
| [`kivi.metrics.gains_table`](https://gchensy.github.io/kivi/en/api/metrics.html#kivi.metrics.gains_table) | Gains and lift by score band, with cumulative capture and KS |
| [`kivi.metrics.roc_table`](https://gchensy.github.io/kivi/en/api/metrics.html#kivi.metrics.roc_table), [`kivi.metrics.roc_auc_interval`](https://gchensy.github.io/kivi/en/api/metrics.html#kivi.metrics.roc_auc_interval), [`kivi.metrics.delong_test`](https://gchensy.github.io/kivi/en/api/metrics.html#kivi.metrics.delong_test) | The ROC curve with one row per threshold, the DeLong confidence interval of an AUC and a Gini, and the DeLong test of two scores |
| [`kivi.calibration.master_scale`](https://gchensy.github.io/kivi/en/api/calibration.html#kivi.calibration.master_scale), [`kivi.calibration.fit_grade_pd`](https://gchensy.github.io/kivi/en/api/calibration.html#kivi.calibration.fit_grade_pd), [`kivi.calibration.assign_grades`](https://gchensy.github.io/kivi/en/api/calibration.html#kivi.calibration.assign_grades) | Build a master scale from the PD of each grade or fit it to default rates, and give each PD its grade |
| [`kivi.calibration.calibration_table`](https://gchensy.github.io/kivi/en/api/calibration.html#kivi.calibration.calibration_table), [`kivi.calibration.binomial_test`](https://gchensy.github.io/kivi/en/api/calibration.html#kivi.calibration.binomial_test), [`kivi.calibration.jeffreys_test`](https://gchensy.github.io/kivi/en/api/calibration.html#kivi.calibration.jeffreys_test), [`kivi.calibration.hosmer_lemeshow`](https://gchensy.github.io/kivi/en/api/calibration.html#kivi.calibration.hosmer_lemeshow) | Event rate against predicted PD per grade, with binomial, Jeffreys and Hosmer-Lemeshow tests |
| [`kivi.calibration.prior_correction`](https://gchensy.github.io/kivi/en/api/calibration.html#kivi.calibration.prior_correction), [`kivi.calibration.corrected_intercept`](https://gchensy.github.io/kivi/en/api/calibration.html#kivi.calibration.corrected_intercept), [`kivi.calibration.herfindahl_index`](https://gchensy.github.io/kivi/en/api/calibration.html#kivi.calibration.herfindahl_index) | Correct PDs for an oversampled sample, and measure grade concentration |
| [`kivi.calibration.platt_fit`](https://gchensy.github.io/kivi/en/api/calibration.html#kivi.calibration.platt_fit), [`kivi.calibration.platt_predict`](https://gchensy.github.io/kivi/en/api/calibration.html#kivi.calibration.platt_predict), [`kivi.calibration.isotonic_fit`](https://gchensy.github.io/kivi/en/api/calibration.html#kivi.calibration.isotonic_fit), [`kivi.calibration.isotonic_predict`](https://gchensy.github.io/kivi/en/api/calibration.html#kivi.calibration.isotonic_predict) | Recalibrate a score to probabilities with Platt scaling or isotonic regression |
| [`kivi.strategy.cutoff_table`](https://gchensy.github.io/kivi/en/api/strategy.html#kivi.strategy.cutoff_table), [`kivi.strategy.swap_set`](https://gchensy.github.io/kivi/en/api/strategy.html#kivi.strategy.swap_set), [`kivi.strategy.confusion_table`](https://gchensy.github.io/kivi/en/api/strategy.html#kivi.strategy.confusion_table) | Approval and bad rates of each cut-off, its confusion matrix, precision, recall and F1, and the swap set between a current and a new score |
| [`kivi.strategy.expected_loss`](https://gchensy.github.io/kivi/en/api/strategy.html#kivi.strategy.expected_loss), [`kivi.strategy.expected_loss_table`](https://gchensy.github.io/kivi/en/api/strategy.html#kivi.strategy.expected_loss_table) | Expected loss, PD × LGD × EAD, of each exposure and by grade |
| [`kivi.spec.fit_spec`](https://gchensy.github.io/kivi/en/api/spec.html#kivi.spec.fit_spec), [`kivi.spec.BinningSpec`](https://gchensy.github.io/kivi/en/api/spec.html#kivi.spec.BinningSpec) | Bin and WOE-encode a whole data frame, and save the specification as versioned JSON |
| [`kivi.stability.psi_table`](https://gchensy.github.io/kivi/en/api/stability.html#kivi.stability.psi_table) | Per-group PSI contributions between two samples |
| [`kivi.stability.csi_table`](https://gchensy.github.io/kivi/en/api/stability.html#kivi.stability.csi_table) | Characteristic stability index: how much each variable's shift moves the average score |
| [`kivi.monitoring.monitoring_table`](https://gchensy.github.io/kivi/en/api/monitoring.html#kivi.monitoring.monitoring_table), [`kivi.monitoring.stability_table`](https://gchensy.github.io/kivi/en/api/monitoring.html#kivi.monitoring.stability_table) | One row per period with the population, the known outcomes, AUC, Gini, KS and the PSI of the score, and the PSI of every variable by period |
| [`kivi.vintage.delinquency_panel`](https://gchensy.github.io/kivi/en/api/vintage.html#kivi.vintage.delinquency_panel) | Monthly delinquency panel from instalment schedules and payments: days past due, status, and the overdue and outstanding amounts |
| [`kivi.vintage.vintage_table`](https://gchensy.github.io/kivi/en/api/vintage.html#kivi.vintage.vintage_table) | Cumulative bad rate of each origination cohort by month on book |
| [`kivi.vintage.transition_matrix`](https://gchensy.github.io/kivi/en/api/vintage.html#kivi.vintage.transition_matrix), [`kivi.vintage.roll_rates`](https://gchensy.github.io/kivi/en/api/vintage.html#kivi.vintage.roll_rates) | Movements between delinquency buckets and the forward roll rate of each bucket |
| [`kivi.lifetime.convert_pd`](https://gchensy.github.io/kivi/en/api/lifetime.html#kivi.lifetime.convert_pd), [`kivi.lifetime.transition_pd`](https://gchensy.github.io/kivi/en/api/lifetime.html#kivi.lifetime.transition_pd) | Convert PD term structures between cumulative, marginal and conditional PDs, and project them from a transition matrix |
| [`kivi.lifetime.expected_credit_loss`](https://gchensy.github.io/kivi/en/api/lifetime.html#kivi.lifetime.expected_credit_loss), [`kivi.lifetime.significant_increase`](https://gchensy.github.io/kivi/en/api/lifetime.html#kivi.lifetime.significant_increase), [`kivi.lifetime.assign_stages`](https://gchensy.github.io/kivi/en/api/lifetime.html#kivi.lifetime.assign_stages), [`kivi.lifetime.stage_table`](https://gchensy.github.io/kivi/en/api/lifetime.html#kivi.lifetime.stage_table) | IFRS 9: 12-month and lifetime expected credit loss, significant increase in credit risk, stages 1 to 3 and the loss allowance by stage |
| [`kivi.plotting.woe_chart`](https://gchensy.github.io/kivi/en/api/plotting.html#kivi.plotting.woe_chart), [`kivi.plotting.psi_chart`](https://gchensy.github.io/kivi/en/api/plotting.html#kivi.plotting.psi_chart), [`kivi.plotting.correlation_chart`](https://gchensy.github.io/kivi/en/api/plotting.html#kivi.plotting.correlation_chart) | Charts of the variables: WOE and event rate per bin, reference and current shares, and a correlation heatmap |
| [`kivi.plotting.roc_chart`](https://gchensy.github.io/kivi/en/api/plotting.html#kivi.plotting.roc_chart), [`kivi.plotting.gains_chart`](https://gchensy.github.io/kivi/en/api/plotting.html#kivi.plotting.gains_chart), [`kivi.plotting.calibration_chart`](https://gchensy.github.io/kivi/en/api/plotting.html#kivi.plotting.calibration_chart), [`kivi.plotting.cutoff_chart`](https://gchensy.github.io/kivi/en/api/plotting.html#kivi.plotting.cutoff_chart), [`kivi.plotting.monitoring_chart`](https://gchensy.github.io/kivi/en/api/plotting.html#kivi.plotting.monitoring_chart), [`kivi.plotting.vintage_chart`](https://gchensy.github.io/kivi/en/api/plotting.html#kivi.plotting.vintage_chart) | Charts of a score and a portfolio: ROC and gains curves with the KS, calibration, cut-offs, monitoring by period and vintage curves |
| [`kivi.datasets.load_bank`](https://gchensy.github.io/kivi/en/api/datasets.html#kivi.datasets.load_bank), [`kivi.datasets.make_loan_panel`](https://gchensy.github.io/kivi/en/api/datasets.html#kivi.datasets.make_loan_panel), [`kivi.datasets.make_applications`](https://gchensy.github.io/kivi/en/api/datasets.html#kivi.datasets.make_applications), [`kivi.datasets.make_transactions`](https://gchensy.github.io/kivi/en/api/datasets.html#kivi.datasets.make_transactions) | The bundled bank marketing data, and seeded synthetic loan panels, loan applications and the account transactions before each application |

## 📚 Documentation

| | English | 中文 |
| --- | --- | --- |
| Quickstart | [Read](https://gchensy.github.io/kivi/en/getting_started/quickstart.html) | [阅读](https://gchensy.github.io/kivi/zh/getting_started/quickstart.html) |
| Tutorial: a credit scorecard project in eleven chapters | [Read](https://gchensy.github.io/kivi/en/tutorial/index.html) | [阅读](https://gchensy.github.io/kivi/zh/tutorial/index.html) |
| Risk modeling workflow | [Read](https://gchensy.github.io/kivi/en/user_guide/risk_workflow.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/risk_workflow.html) |
| Time-window features | [Read](https://gchensy.github.io/kivi/en/user_guide/features.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/features.html) |
| Univariate profile | [Read](https://gchensy.github.io/kivi/en/user_guide/univariate.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/univariate.html) |
| Binning | [Read](https://gchensy.github.io/kivi/en/user_guide/binning.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/binning.html) |
| WOE and IV | [Read](https://gchensy.github.io/kivi/en/user_guide/woe_iv.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/woe_iv.html) |
| Variable selection | [Read](https://gchensy.github.io/kivi/en/user_guide/selection.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/selection.html) |
| Reject inference | [Read](https://gchensy.github.io/kivi/en/user_guide/reject_inference.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/reject_inference.html) |
| Scorecard | [Read](https://gchensy.github.io/kivi/en/user_guide/scorecard.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/scorecard.html) |
| Expert scorecards | [Read](https://gchensy.github.io/kivi/en/user_guide/expert.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/expert.html) |
| Model performance | [Read](https://gchensy.github.io/kivi/en/user_guide/metrics.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/metrics.html) |
| Calibration | [Read](https://gchensy.github.io/kivi/en/user_guide/calibration.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/calibration.html) |
| Credit strategy | [Read](https://gchensy.github.io/kivi/en/user_guide/strategy.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/strategy.html) |
| Binning specifications | [Read](https://gchensy.github.io/kivi/en/user_guide/spec.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/spec.html) |
| PSI | [Read](https://gchensy.github.io/kivi/en/user_guide/psi.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/psi.html) |
| Model monitoring | [Read](https://gchensy.github.io/kivi/en/user_guide/monitoring.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/monitoring.html) |
| Vintage and roll-rate analysis | [Read](https://gchensy.github.io/kivi/en/user_guide/vintage.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/vintage.html) |
| Lifetime PD and expected credit loss | [Read](https://gchensy.github.io/kivi/en/user_guide/lifetime.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/lifetime.html) |
| Charts | [Read](https://gchensy.github.io/kivi/en/user_guide/plotting.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/plotting.html) |
| Example data | [Read](https://gchensy.github.io/kivi/en/user_guide/datasets.html) | [阅读](https://gchensy.github.io/kivi/zh/user_guide/datasets.html) |
| Conventions | [Read](https://gchensy.github.io/kivi/en/api/conventions.html) | [阅读](https://gchensy.github.io/kivi/zh/api/conventions.html) |
| Compatibility | [Read](https://gchensy.github.io/kivi/en/api/compatibility.html) | [阅读](https://gchensy.github.io/kivi/zh/api/compatibility.html) |
| Glossary | [Read](https://gchensy.github.io/kivi/en/api/glossary.html) | [阅读](https://gchensy.github.io/kivi/zh/api/glossary.html) |
| Migration guide | [Read](https://gchensy.github.io/kivi/en/release/migration.html) | [阅读](https://gchensy.github.io/kivi/zh/release/migration.html) |

## 🗺️ Roadmap

- ✅ Quantile and uniform binning, WOE, IV and PSI on a pandas core
- ✅ English and Simplified Chinese documentation with tested examples
- ✅ Special values such as `-1` or `-999` in their own bins
- ✅ Charted risk examples: special values, coarse classing, IV screening, PSI monitoring
- ✅ Supervised binning: ChiMerge, decision tree and monotonic WOE
- ✅ WOE encoding of new data, with an explicit policy for unseen groups
- ✅ Sample weights in binning, WOE and PSI
- ✅ Scorecard workflow: variable selection, logistic regression, PDO scaling, reason codes, evaluation and calibration
- ✅ CSI, vintage and roll-rate analysis, batch binning with saved specifications
- ✅ Platt and isotonic calibration, and sample weights in the metrics
- ✅ Credit strategy: cut-off tables, swap sets and expected loss
- ✅ Reject inference with weighted model fits
- ✅ Master scales and univariate variable profiles
- ✅ Topic guides on one outline, a conventions page and a glossary
- ✅ Categorical grouping, ROC tables and DeLong tests of the AUC
- ✅ Model monitoring by period
- ✅ Lifetime PD term structures, IFRS 9 expected credit loss and stage allocation
- ✅ Type hints checked in CI, test coverage, and an API review: the 0.9 deprecations and the compatibility promise for 1.0
- ✅ 1.0: a stable API under semantic versioning, with the 0.9 deprecations removed
- ✅ Diagrams of how kivi computes its results, and a tutorial that follows one credit risk project
- ✅ 1.1: charts of every result table (`kivi[plot]`), the confusion matrix and F1 of a cut-off, shape statistics, KS and a chi-squared test in the variable profile, Spearman correlation filtering, and cross-checks against scikit-learn, SciPy and statsmodels in CI
- ✅ 1.2: Best-KS binning, delinquency panels from instalment schedules and payments, and expert scorecard weights by the analytic hierarchy process, with their consistency check
- ✅ 1.3: point-in-time window features of transactions and other events, repayment-behaviour features from monthly status panels, feature specifications saved as versioned JSON, and simulated transactions before each application

## 🤝 Contributing

```bash
git clone https://github.com/gchensy/kivi.git
cd kivi
python -m pip install -e '.[test,docs,lint]'              # the lint extra needs Python 3.11 or newer
pre-commit install                                        # ruff and mypy before every commit
python -m pytest --cov                                    # unit tests with branch coverage
python -m sphinx -W -b doctest docs docs/_build/doctest   # documentation examples
python docs/build.py                                      # English and Chinese HTML
```

After editing English pages or docstrings, run `python docs/build.py --update-po` and translate the new entries in `docs/locale/zh_CN/LC_MESSAGES/kivi.po`. See [Writing documentation](https://gchensy.github.io/kivi/en/development/documentation.html) for the translation steps and the conventions of the pages, and the [development guide](https://gchensy.github.io/kivi/en/development/index.html) for the checks that every pull request runs.

## 📄 License

The code is released under the [MIT License](https://github.com/gchensy/kivi/blob/master/LICENSE). The bundled Bank Marketing data by S. Moro, P. Rita and P. Cortez (2014) is licensed under CC BY 4.0; see [DATA-LICENSE.txt](https://github.com/gchensy/kivi/blob/master/DATA-LICENSE.txt).

<div align="center">

<sub>If kivi helps your models, a ⭐ helps others find it.</sub>

</div>
