Migration guide
===============

From 0.9 to 1.0
---------------

kivi 1.0 removes what 0.9 deprecated. Code that runs on 0.9 without a ``FutureWarning`` runs on 1.0 unchanged, except where it imports ``Dataset`` or uses the result of :func:`~kivi.woe.woe` for Series counts. To find the calls to change, run the code or its tests on 0.9 with ``FutureWarning`` turned into an error, as ``python -W error::FutureWarning`` or ``pytest -W error::FutureWarning`` do.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 0.9
     - 1.0
   * - ``smoothing=True`` in ``woe_table``, ``woe``, ``iv``, ``psi_table``, ``psi``, ``monotonic_edges``, ``fit_grade_pd``, ``monitoring_table`` and ``stability_table`` counts as 1 with a ``FutureWarning``
     - Raises a ``ValueError``, as in every other function. Pass ``smoothing=1`` for the result of 0.9, or leave ``smoothing`` out for the default 0.5
   * - ``woe``, ``iv`` and ``psi`` match two count Series with different indices by position, with a ``FutureWarning``
     - Raise a ``ValueError``. Align the counts first, as in ``reference, current = reference.align(current, fill_value=0)``, or pass arrays when both list the groups in the same order
   * - ``woe`` returns an array
     - Returns a Series named ``woe`` with the index of the counts when a count argument is a Series. Use ``.to_numpy()`` for the array and ``.iloc`` to select by position; lists and arrays still give an array
   * - ``Dataset.bank_data()`` warns with a ``FutureWarning``
     - Removed, together with the class ``Dataset``. ``kivi.datasets.load_bank()`` returns the same frame
   * - ``isotonic_predict(score, edges, probabilities=...)`` warns with a ``FutureWarning``
     - Removed. Pass ``probability=``, or the probabilities by position

kivi 1.0 still supports Python 3.10, NumPy 1.24 and pandas 2.0. From 1.0 on, kivi follows semantic versioning: a deprecation keeps working with a ``FutureWarning`` until the next major release. :doc:`/api/compatibility` describes what a release may change.

From 0.2.x
----------

The functions below replace the 0.2.x classes. The example data of ``Dataset.bank_data()``, which 1.0 removed, come from ``kivi.datasets.load_bank()``. The target describes deposit subscription.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 0.2.x
     - Replacement
   * - ``FeatureTools``
     - ``kivi.features``: ``window_features`` with ``ratios`` for the aggregations of each window and their ratios, and ``panel_features`` for the consecutive months with and without payments, from 1.3.0
   * - ``FeatureToolsV2``: ``one_batch_data``, ``features_agg``, ``gen_batch_features``
     - ``kivi.features.window_features``, with all the snapshot dates of a backfill as rows of the observations, from 1.3.0
   * - ``index/collect.py``: ``CollectIndex`` and ``IndexField``
     - ``kivi.features.FeatureSpec`` for the configuration, and ratios to window pairs for the changes from one period to the next, from 1.3.0
   * - ``DistanceBins``
     - ``bin_edges(values, method="uniform")``, then ``pandas.cut`` and ``woe_table``
   * - ``FrequencyBins``
     - ``bin_edges(values, method="quantile")``, then ``pandas.cut`` and ``woe_table``
   * - ``abnormal_vals=[...]``
     - ``special=[...]`` in ``bin_edges`` and ``kivi.binning.cut``
   * - ``ManuallyBins``
     - Explicit edges, such as ``[-inf, 20, 50, inf]``, with ``kivi.binning.cut`` and then ``woe_table``
   * - ``KSBins``
     - ``ks_edges(values, target)``, then ``kivi.binning.cut`` and ``woe_table``, from 1.2.0
   * - ``CategoryBins``
     - ``woe_table(labels, target)``, after ``category_groups`` or ``chimerge_groups`` and ``group_categories`` where the labels should be grouped, from 0.8.0
   * - ``WOEScore``
     - ``woe_transform(groups, table)`` on groups formed with the development edges, from 0.5.0
   * - ``PSI.fit()``
     - ``psi_table(before, after)`` on groups formed with the same edges
   * - ``BinaryMetrics``, ``ModelEvaluate``, ``ScoreEvaluate``
     - ``kivi.metrics``: ``roc_auc``, ``gini``, ``ks``, ``gains_table`` and ``brier_score``, from 0.5.0, and ``roc_table``, ``roc_auc_interval`` and ``delong_test``, from 0.8.0; ``kivi.strategy.confusion_table`` for the confusion matrix and F1, from 1.1.0
   * - ``ScoreEvaluate.score_psi``, ``add_psi``
     - ``kivi.monitoring``: ``monitoring_table`` and ``stability_table``, from 0.8.0
   * - ``VIF``, ``StepWise``, ``feature.Corr``
     - ``kivi.selection``: ``vif``, ``stepwise``, ``correlation_filter`` and ``logistic_regression``, from 0.5.0; ``correlation_filter(X, method="spearman")`` for ``Corr.spearmanr``, from 1.1.0
   * - ``PDToScore``, ``BaseScoreAndPDO``
     - ``kivi.scorecard``: ``pd_to_score``, ``score_to_pd``, ``score_scaling`` and ``points_table``, from 0.5.0
   * - ``AHP``
     - ``kivi.expert``: ``ahp_weights(matrix, method="geometric")`` for the legacy weights, ``ahp_consistency`` and ``combine_weights``, from 1.2.0
   * - ``get_omega``, ``BinomialTest``, ``herfindahl``
     - ``kivi.calibration``: ``prior_correction``, ``calibration_table``, ``binomial_test`` and ``herfindahl_index``, from 0.5.0
   * - ``MasterScale``, ``PDToRant``
     - ``kivi.calibration``: ``master_scale``, ``fit_grade_pd`` and ``assign_grades``, from 0.7.0
   * - ``FeatureEvaluate``
     - ``kivi.univariate.profile_table``, from 0.7.0
   * - ``WOEBatch``, ``ManualBinsTool``
     - ``kivi.spec``: ``fit_spec`` and ``BinningSpec``, from 0.6.0
   * - ``CSI``
     - ``kivi.stability.csi_table`` on groups formed with the same edges, from 0.6.0
   * - ``VintagePd``
     - ``kivi.vintage.vintage_table`` on a loan panel, from 0.6.0, which ``delinquency_panel`` builds from repayment schedules from 1.2.0
   * - ``PortfolioAnalysis.Vintage``
     - ``kivi.vintage.delinquency_panel`` on the repayment schedules and payments, then ``vintage_table``, from 1.2.0
   * - ``plot_bin``, ``DefaultAnalysis``, ``Calibration.CalibrationCurve``, ``BinaryMetrics.plot_*``, ``diagonal_corr``, ``structure_corr``
     - ``kivi.plotting``: ``woe_chart``, ``calibration_chart``, ``roc_chart``, ``gains_chart`` and ``correlation_chart``, from 1.1.0
   * - ``Dataset.bank_data()``
     - ``kivi.datasets.load_bank()``, from 0.6.0. ``Dataset.bank_data()`` warned with a ``FutureWarning`` in 0.9.0 and was removed in 1.0

Binning and WOE
~~~~~~~~~~~~~~~

DistanceBins and FrequencyBins previously calculated group assignments and statistics together. Use bin_edges with uniform or quantile, call pandas.cut, then pass the groups to woe_table. Codes listed in abnormal_vals become the special argument: pass them to bin_edges, and use kivi.binning.cut with the same codes in place of pandas.cut. CategoryBins users can pass the label series directly.

Version 0.5.0 adds woe_transform in place of WOEScore. WOEScore matched raw values against the min_bin and max_bin columns of a WOE table. Form the groups with the development edges instead, then pass them to woe_transform with the woe_table result. A value that matches no group raises an error, as WOEScore did with error="error"; pass unseen to choose its WOE instead.

Version 0.4.0 adds ``chimerge_edges`` and ``tree_edges`` in place of ``Chi2Bins`` and ``TreeBins``. They are new implementations rather than ports, so the edges can differ. ``Chi2Bins`` started from 10 quantile bins, tested at a 0.9 confidence level, added 0.5 to every expected count and kept at least three bins; ``chimerge_edges(values, target, alpha=0.1, initial_bins=10)`` comes closest. ``TreeBins`` reported the midpoints that scikit-learn chooses as thresholds, while ``tree_edges`` reports the largest observed value below each cut. Compare the edges on a fixed reference sample before switching.

Codes listed in ``abnormal_vals`` of ``Chi2Bins`` and ``TreeBins`` become the ``special`` argument of ``chimerge_edges`` and ``tree_edges``. As before, the codes and their labels are left out of the fit; pass the same codes to ``kivi.binning.cut`` to give each one its own group.

Population stability
~~~~~~~~~~~~~~~~~~~~

For PSI.fit(), form both sets of groups using the same reference edges, then call psi_table. It returns raw counts and a contribution per bin. Previously the psi column repeated the overall total. The new smoothing rule preserves counts and normalizes each distribution after replacing zeros. Recheck thresholds when migrating saved analyses.

Models, scorecards and calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These modules are new implementations, so compare their output with the legacy code on a fixed reference sample before switching.

* The legacy ``lift`` column was cumulative. In ``gains_table``, ``lift`` is the lift of each band and ``cum_lift`` the cumulative lift. The metrics take a score that rises with risk, such as a PD; pass ``-score`` for scorecard points. ``ModelEvaluate`` rescaled each sample by its own minimum and maximum, so its scores were not comparable across samples; the new functions use the scores as given.
* ``VIF`` called statsmodels without a constant, which measures uncentred R²; ``vif`` includes an intercept in every auxiliary regression, as in the textbook definition, so the values differ. ``stepwise`` replaces ``StepWise`` and by default keeps every coefficient positive, as WOE inputs require. ``logistic_regression`` fits the model in NumPy, so statsmodels is no longer needed.
* ``BaseScoreAndPDO`` took bad:good odds, for example ``odds=1/50``. ``base_odds`` is good:bad, so pass ``base_odds=50``; passing ``1/50`` silently gives an offset of 712.9. The legacy ``PDToScore`` divided by ln 2 only in one of its branches; ``pd_to_score`` and ``score_to_pd`` are exact inverses.
* ``get_omega`` becomes ``prior_correction`` for probabilities and ``corrected_intercept`` for the model intercept.

Portfolio analysis and production binning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Version 0.6.0 replaces three more legacy classes. They are new implementations as well, so compare their output with the legacy code on a fixed reference sample before switching.

* ``WOEBatch`` fitted the WOE of many columns at once, and ``ManualBinsTool`` merged bins by hand afterwards. ``fit_spec`` fits every column with the binning function chosen for it, and its ``BinningSpec`` saves the edges and WOE tables as versioned JSON. Nothing is re-binned automatically: to merge bins by hand, pass the merged edges for that column in ``bins``.
* ``CSI`` matched the points to the bins by position and lost the missing group. ``csi_table`` matches the points to the groups by label, keeps the missing group, and raises for an observed group without points unless ``unseen_points`` gives them. Its contributions sum to the change of the variable's average points.
* ``VintagePd`` built the vintage table from repayment schedules and observation dates. ``vintage_table`` takes a long panel with one row per loan and month on book and a bad flag that you define, for example ``status == "90+"``, and from 1.2.0 ``delinquency_panel`` builds that panel from the schedules and the payments. Pass ``weights=amount`` for the amount-weighted rates of ``VintagePd``. Months that a cohort has not reached are NaN rather than 0.

Master scales, variable profiles and strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Version 0.7.0 replaces the legacy master scale and variable profile. Compare their output with the legacy code on a fixed reference sample before switching.

* ``MasterScale.fit_grade_df`` and ``confirm_master_scale`` become ``fit_grade_pd`` and ``master_scale``. ``fit_grade_pd`` fits the logarithm of the default rate by least squares weighted with the number of observations of each grade, where the legacy fit gave every grade the same weight; the two agree when the grades are equally large. A grade without defaults enters the fit with half a default instead of a logarithm of zero. The bounds between grades are the geometric means of adjacent PDs; pass ``boundary="arithmetic"`` for the midpoints of the legacy scale.
* ``PDToRant(pd, ranking)`` becomes ``assign_grades(probability, scale)``. The legacy function put a PD that lies exactly on a bound in the grade above it; ``assign_grades`` puts it in the grade below, as the intervals of ``pandas.cut`` are closed on the right. It returns an ordered categorical of the grade labels instead of grade numbers.
* ``FeatureEvaluate(df, target_name).stats()`` becomes ``profile_table(frame.drop(columns=target_name), frame[target_name])``. The columns follow kivi's names, such as ``missing_share``, ``missing_event_rate`` and ``non_missing_lift``, and a target without events raises an error, where the legacy lifts were -9999. The signed Gini coefficient of the values replaces the AUC and KS of a univariate logistic fit. The table adds the IV of a quick binning, special codes, sample weights and the ``tied_bins`` column.
* Strategy analysis, reject inference and weighted model fits have no 0.2.x counterpart. ``kivi.strategy`` replaces approval and bad-rate tables built by hand from score bands, ``kivi.reject_inference`` infers the outcomes of declined applicants, and ``logistic_regression`` and ``stepwise`` take ``weights`` to refit on them.

Category grouping, validation and monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Version 0.8.0 groups categorical variables, adds the ROC table and the uncertainty of an AUC, and monitors a model period by period. These are new implementations as well.

* ``CategoryBins`` kept the labels of a categorical variable as they were. ``category_groups`` pools the rare categories and ``chimerge_groups`` merges the categories with similar event rates; both return a dictionary from category to group that is fitted on the reference sample, and ``group_categories`` applies it to every sample. A category that the grouping does not know raises an error unless ``unseen`` names its group. ``fit_spec`` groups a column with the methods ``"rare"`` and ``"chimerge"``, and a specification that groups a column is saved as format version 2, which 0.7.0 and earlier refuse to read.
* ``BinaryMetrics.roc_curve`` returned the points of scikit-learn's ``roc_curve`` with its default ``drop_intermediate=True``. ``roc_table`` has one row for every distinct score, as ``roc_curve`` with ``drop_intermediate=False``, with the cumulative counts, the two rates and the precision. Its ``precision`` and ``true_positive_rate`` columns are the precision and the recall at each threshold, so it also gives the precision-recall curve. ``roc_auc_interval`` adds the DeLong standard error and interval of the AUC and the Gini, and ``delong_test`` compares the AUCs of two scores on the same sample; the legacy code had neither.
* ``ScoreEvaluate.score_psi`` and ``add_psi`` compared the band totals of two lift tables. ``monitoring_table`` compares every period with one reference sample on score bands fitted once, adds the known outcomes, the AUC, the Gini and the KS of each period, and counts an observation without a known outcome in the population only. ``stability_table`` gives the PSI of every variable in every period.
* Lifetime PD term structures, the IFRS 9 expected credit loss and the allocation to stages have no 0.2.x counterpart; see ``kivi.lifetime``.

Charts, classification measures and variable statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Version 1.1.0 draws charts and adds the measures of a classifier and more statistics to the variable profile.

* ``plot_bin`` and ``DefaultAnalysis`` drew the bins of a variable from raw data, ``Calibration.CalibrationCurve`` the predicted against the observed rate, the ``plot_*`` methods of ``BinaryMetrics`` the ROC and KS curves, and ``diagonal_corr`` and ``structure_corr`` correlation heatmaps. The functions of ``kivi.plotting`` draw the tables that kivi returns instead: ``woe_chart`` a ``woe_table``, ``calibration_chart`` a ``calibration_table``, ``roc_chart`` and ``gains_chart`` a ``roc_table`` and a ``gains_table`` with the KS, and ``correlation_chart`` a correlation matrix such as ``X.corr()``. They return the matplotlib Axes rather than saving image files, and they need matplotlib, which ``pip install "kivi[plot]"`` installs.
* The F1 score and the confusion matrix of ``BinaryMetrics`` judged labels that the caller had predicted. ``confusion_table(score, target, cutoffs=...)`` takes a score and the cut-offs of ``cutoff_table`` instead: a score above the cut-off is declined, and a declined event is a true positive. Pass labels already predicted as the score with ``cutoffs=[0]`` to get the legacy counts. Where nothing is declined, the precision is NaN, where the legacy code gave 0 with a warning, and the table has no ``lift`` column; divide ``precision`` by the event rate for it.
* The skewness, kurtosis and coefficient of variation of ``FeatureEvaluate`` are the ``skewness``, ``kurtosis`` and ``cv`` columns of ``profile_table``. ``cv`` is NaN when the mean is 0, where the legacy code gave -9999. ``ks`` is the KS statistic of the values themselves, which the legacy ``ks_test`` computed with SciPy, without its p-value. ``chi2_p_value`` tests the groups of the quick binning against the target, while the legacy chi-squared test was scikit-learn's ``chi2`` on the raw values, so the two p-values differ.
* ``Corr.spearmanr`` gave the Spearman correlation of each pair of columns. ``correlation_filter(X, method="spearman")`` drops one variable of each pair whose Spearman correlation is too high; ``X.corr(method="spearman")`` gives the matrix itself.

Delinquency panels, KS binning and expert weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Version 1.2.0 builds delinquency panels from repayment schedules, adds Best-KS binning and weighs the factors of expert scorecards. These are new implementations as well, so compare their output with the legacy code on a fixed reference sample before switching.

* ``PortfolioAnalysis.Vintage`` and ``VintagePd`` derived the delinquency of each loan from its repayment schedule at month-end observation dates. ``delinquency_panel(loan, due_date, amount_due, payment_loan, payment_date, payment_amount, end=...)`` builds that panel from the schedule and the payments as they were made, and ``vintage_table`` turns it into vintage curves. The rules differ from the legacy code in the following ways:

  * The legacy schedule had one row per instalment with its own payment date and remaining amounts, so the source system had already allocated the payments. ``delinquency_panel`` applies the payments to the oldest unpaid instalment first.
  * The legacy code counted an instalment as unpaid only while its interest was unpaid, so unpaid principal alone did not make a loan delinquent. ``delinquency_panel`` counts the whole amount due, up to ``tolerance``.
  * For a loan with nothing unpaid, the legacy code took the days from the due date to the payment date of its latest instalment, so a loan that paid its latest instalment late kept those days after catching up. ``delinquency_panel`` gives such a loan 0 days.
  * ``VintagePd`` left out of an observation a loan whose first instalment was paid but a later one was not, while the Spark ``Vintage`` kept its earliest unpaid instalment. ``delinquency_panel`` keeps every loan from its first due month until it is repaid.
  * The legacy buckets were M0 = 0, M1 = 1–30, M2 = 31–60, M3 = 61–90 and M4+ over 90 days. ``status`` has ``current``, ``1-29``, ``30-59``, ``60-89`` and ``90+`` (90 days or more), so a loan exactly 30, 60 or 90 days past due falls one bucket later. Use ``days_past_due > 90`` for the legacy M4+.
  * The legacy cohort was the month of the origination date ``loan_date``. Pass the origination dates as ``origination`` for the same cohorts. ``month_on_book`` then counts the months from the origination month, 0 at its end, so it is one less than the legacy ``book_month``, which rounded the months from the first day of the origination month to the observation date; add 1 for the legacy numbering. Without ``origination``, ``cohort`` is the month before the first due date and ``month_on_book`` is 1 in the first due month, which counts from the origination month only when the first instalment falls due a month after origination.
  * The legacy remaining principal subtracted only the principal paid on instalments due by the observation date. ``balance`` counts every payment made by the month end, including payments made ahead.
  * The legacy code observed every loan at ``observe_period`` month ends from ``observe_date``. ``delinquency_panel`` observes each loan from the month of its first due date to ``end``, and drops it after the month end at which it is repaid.
  * ``VintagePd`` reported, at each observation date, the remaining principal of the M4+ loans as a share of the cohort's amount lent. That is a point-in-time rate, which falls when loans cure or are repaid. ``vintage_table`` counts the loans that were ever bad, by number or by a weight such as the amount lent. For the legacy rate, sum ``balance`` where ``days_past_due > 90`` by cohort and month on book, and divide by the amount lent.

* ``ks_edges`` replaces ``KSBins``. It is a new implementation, so the edges can differ. ``KSBins`` always split the largest bin, while ``ks_edges`` splits the bin whose cut ranks the most pairs of an event and a non-event apart, and stops when no bin has an allowed cut that separates the classes. ``KSBins`` raised an error instead when the largest bin had no allowed cut, for example when a special code such as ``pdays == -1`` filled it, or when it held one class only. ``min_sample_rate`` counted every row, missing values and ``abnormal_vals`` included, while ``min_bin_share`` counts only the fitted values, in weight; ``weight`` did not affect the cuts of ``KSBins``, while the cuts of ``ks_edges`` use sums of weights. Both put each edge at the largest value of the bin to its left and break ties towards the smallest value. Codes listed in ``abnormal_vals`` become the ``special`` argument, as for ``tree_edges``, and ``fit_spec`` fits a column with ``ks_edges`` when its ``method`` is ``"ks"``.
* The ``AHP`` class took a long frame with the columns ``group``, ``experimental``, ``control`` and ``value``, weighed each group by the geometric means of the rows of its comparison matrix and estimated the principal eigenvalue as the mean of :math:`(A w)_i / w_i` for those weights. ``ahp_weights(matrix, method="geometric")`` gives the same weights for one comparison matrix, and ``ahp_consistency`` the CI and CR. kivi's default is the exact principal eigenvector. For up to three factors it gives the same weights, eigenvalue and CR as the legacy estimate; from four factors on they differ slightly when the judgments are inconsistent.
* ``weight()`` returned one frame with the columns ``dst``, the factor, ``geo_mean``, the geometric mean of its row, and ``group_weight``, its weight within the group, with the groups one after the other in sorted order and no group column. ``group_ci`` and ``group_cr`` held the CI and CR of each group. Weigh each group on its own and join the results, as the example below shows; ``np.exp(np.log(matrix).mean(axis=1))`` gives ``geo_mean`` if it is needed, and ``combine_weights`` multiplies the weights of groups and of their factors into global weights.
* The legacy random indices are the same for 3 to 11, 13 and 15 factors, but 1.54 for 12 and 1.58 for 14 factors, where ``ahp_consistency`` uses the 1.53 and 1.57 of Tone's table, and the legacy table went on to 1.60 for 16 factors, for which ``ahp_consistency`` raises an error. For one factor the legacy CI was NaN with a ``RuntimeWarning``; for one or two factors ``ahp_consistency`` returns 0 for ``ci``, ``ri`` and ``cr``. A CR above 0.1 is no error in either: the legacy class logged it in red, and ``ahp_consistency`` returns it for the reader to judge.

A long frame of the legacy layout, with one group of three factors and one of two, becomes one matrix per group:

>>> import numpy as np
>>> import pandas as pd
>>> from kivi.expert import ahp_consistency, ahp_weights
>>> long = pd.DataFrame(
...     {
...         "group": ["finance"] * 9 + ["market"] * 4,
...         "experimental": np.repeat(["coverage", "leverage", "margin", "position", "share"], [3, 3, 3, 2, 2]),
...         "control": ["coverage", "leverage", "margin"] * 3 + ["position", "share"] * 2,
...         "value": [1, 1 / 2, 3, 2, 1, 4, 1 / 3, 1 / 4, 1, 1, 3, 1 / 3, 1],
...     }
... )
>>> weights, statistics = {}, {}
>>> for name, group in long.groupby("group"):
...     matrix = group.pivot(index="experimental", columns="control", values="value")
...     weights[name] = ahp_weights(matrix, method="geometric")
...     statistics[name] = ahp_consistency(matrix)
>>> pd.concat(weights, names=["group"]).round(4)
group    factor
finance  coverage    0.3196
         leverage    0.5584
         margin      0.1220
market   position    0.7500
         share       0.2500
Name: weight, dtype: float64
>>> pd.DataFrame(statistics).T[["ci", "cr"]].round(4)
statistic      ci      cr
finance    0.0091  0.0158
market     0.0000  0.0000

Time-window and repayment features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Version 1.3.0 computes features of event data and of monthly repayment statuses, the work of ``FeatureTools``, ``FeatureToolsV2`` and ``index/collect.py``. These are new implementations as well, so compare their output with the legacy features on a fixed sample before switching.

* The legacy tools ran on Spark DataFrames and wrote long tables of ``uuid``, ``index_name``, ``index_val`` and ``load_date`` into Hive partitions. :func:`~kivi.features.window_features` and :func:`~kivi.features.panel_features` take pandas DataFrames and return a wide DataFrame with one column per feature, on the index of the observations; ``DataFrame.melt`` gives a long table where one is needed. A backfill, which ``batch_derivative_index`` ran date by date, is one call with a row for every entity and snapshot date in the observations.
* The legacy windows included the snapshot date. ``one_batch_data`` kept the rows whose ``months_between(now_date, load_date)``, a fractional number of months, lay between 0 and ``month - 1``, and ``FeatureTools`` and ``CollectIndex`` took the month of the snapshot date as the last month of every window. kivi's windows are point in time: ``window_features`` takes :math:`[t - k, t)` by default, without the events at the observation time, and ``closed="right"`` gives :math:`(t - k, t]`, the closest equivalent to the legacy windows for records that are complete at the end of their day. Months are calendar months, counted as ``pandas.DateOffset`` counts them. ``panel_features`` uses only the months before the month of the observation time, because the status at the end of that month is not known yet.
* ``FeatureTools`` aggregated the amounts and counts of each month first, with its ``agg_type``, and then took the sum, mean, extremes and standard deviation of these monthly totals over each window, so its ``amt_max`` was the largest monthly total. ``window_features`` aggregates the events themselves; to aggregate monthly totals, pass them as events dated at the start of the next month, when they are complete. Its ``filter_rules`` become filtered event tables with their own ``prefix``, one group each in a ``FeatureSpec``. Its ratios to the window twice as long, to the window of twelve months and to the window before (``div2tr``, ``div1y`` and ``chain_rate``) are ``ratios``, such as ``("amount_sum_3m", "amount_sum_6m")``, ``("amount_sum_3m", "amount_sum_12m")`` and ``("amount_sum_3m", "amount_sum_6m_3m")`` with the window pair ``("6M", "3M")``. A ratio is NaN where its denominator is 0 or missing, where ``FeatureTools`` subtracted 1e-6 from the denominator and the Spark ratio operators returned codes such as -9999.
* ``FeatureTools`` also counted the months of a window without a payment and the longest runs of consecutive months with and without one. ``panel_features`` counts the same on a monthly panel with an ordered status: ``bad_months_{window}``, ``longest_bad_run_{window}`` and ``longest_good_run_{window}``. Give the panel a row for every month of every entity, with a bad status for a month without a payment, and pass that status as ``bad``; on a delinquency panel, ``bad="1-29"`` makes any arrears bad. A month without a row counts in no window and ends a run, where the legacy code counted it as a month without a payment.
* ``features_agg`` takes the aggregations that its docstring lists under their kivi names: ``cnt`` is ``count``, or the ``events`` column, which also counts the rows without a value, ``avg`` and ``mean`` are ``mean``, ``cnt_dist`` is ``nunique``, ``stddev`` and ``stddev_samp`` are ``std``, whose square gives ``variance`` and ``var_samp``, and ``sum``, ``min``, ``max``, ``first`` and ``last`` keep their names. ``sum_dist``, ``skewness``, ``kurtosis``, ``collect_set``, ``collect_list``, ``approx_count_distinct`` and the population statistics ``stddev_pop`` and ``var_pop`` are not ported. The features are named ``{prefix}{column}_{aggregation}_{window}``, such as ``amount_sum_3m``, rather than ``ent_{column}_last_{month}_{operator}``.
* An ``IndexField`` configured the columns, operators and months of an indicator, and ``CollectIndex`` computed it. A :class:`~kivi.features.FeatureSpec` holds the same as groups of ``aggregations`` and ``windows``, saved as versioned JSON, without the database, table and partition settings, because the caller reads and writes the tables. Its ``id_levels`` become one specification per level, each with its own ``id`` column. ``CollectIndex`` declared steps for the change from the previous period and from the same period a year earlier but left them empty; they are ratios to window pairs, such as ``("amount_sum_3m", "amount_sum_6m_3m")`` and ``("amount_sum_3m", "amount_sum_15m_12m")``.

Not ported
~~~~~~~~~~

Spark, SQL, model wrappers and application code are out of scope and remain in the historical revisions: the Spark sessions, partitions and Hive writers of ``spark``, ``index`` and ``FeatureToolsV2``, whose features :mod:`kivi.features` computes with pandas instead, the wrappers of ``Model``, and the Excel, Markdown, logging and date helpers of ``utils``. These 0.2.x tools have no replacement either:

* The binning classes ``KmeansBins``, which cut at the boundaries of k-means clusters, and ``PearsonBins``, which lowered the number of quantile bins until their rank correlation with the bad rate reached a threshold. ``monotonic_edges`` reaches the monotonic WOE that ``PearsonBins`` aimed at, and the supervised methods of :doc:`/user_guide/binning` replace the clustering of ``KmeansBins``.
* The sampling helpers ``sampling`` and ``upsampling``, the median and missing-value cross features of ``joint_feature`` in ``FeatureTools``, its trend flags and its counts of months above and below the average, its ``ratio`` of two columns of a frame, and the min-max and z-score scaling of ``Scale``. Use pandas. The ratios of window features are the ``ratios`` of ``window_features`` from 1.3.0.
* The industry percentile ranks of ``index_indu_pct`` in ``FeatureToolsV2`` and ``pct_rank``. Rank features within groups with pandas, as in ``features.groupby(industry).rank(pct=True)``.
* The grading by the Beta distribution in ``BetaCalibration``, ``BetaDistribution`` and ``MasterScale.cal_quantile``, and ``model_score_by_weight``. ``master_scale`` and ``woe_to_points`` cover the same steps without them.
* The univariate logistic fit of ``FeatureEvaluate`` and the p-value of its KS test. The ``gini`` and ``ks`` of ``profile_table`` measure how well the values rank the target directly.
* ``ODDS``, which computed a stability index of score bands, and ``StatsLogit``, a wrapper of statsmodels' logistic regression. Use ``psi_table`` and ``logistic_regression``.
* ``Dataset.crime_data()``, a table of crime rates of US states that has nothing to do with credit risk, and the synthetic data of ``MakeData``, a scikit-learn classification sample split for training, and of ``fake_data``, which drew each column from one range for the bad and another for the good customers. ``make_applications`` and ``make_loan_panel`` simulate loan applications and a loan portfolio instead, and from 1.3.0 ``make_transactions`` the account transactions before each application, which neither legacy generator produced.
