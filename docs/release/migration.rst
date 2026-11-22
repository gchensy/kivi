Migrating from 0.2.x
====================

The functions below replace the 0.2.x classes. Dataset.bank_data() still returns the same table, but warns with a FutureWarning from 0.9.0 and is removed in 1.0; use kivi.datasets.load_bank() instead. The target describes deposit subscription.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 0.2.x
     - Replacement
   * - ``DistanceBins``
     - ``bin_edges(values, method="uniform")``, then ``pandas.cut`` and ``woe_table``
   * - ``FrequencyBins``
     - ``bin_edges(values, method="quantile")``, then ``pandas.cut`` and ``woe_table``
   * - ``abnormal_vals=[...]``
     - ``special=[...]`` in ``bin_edges`` and ``kivi.binning.cut``
   * - ``CategoryBins``
     - ``woe_table(labels, target)``, after ``category_groups`` or ``chimerge_groups`` and ``group_categories`` where the labels should be grouped, from 0.8.0
   * - ``WOEScore``
     - ``woe_transform(groups, table)`` on groups formed with the development edges, from 0.5.0
   * - ``PSI.fit()``
     - ``psi_table(before, after)`` on groups formed with the same edges
   * - ``BinaryMetrics``, ``ModelEvaluate``, ``ScoreEvaluate``
     - ``kivi.metrics``: ``roc_auc``, ``gini``, ``ks``, ``gains_table`` and ``brier_score``, from 0.5.0, and ``roc_table``, ``roc_auc_interval`` and ``delong_test``, from 0.8.0
   * - ``ScoreEvaluate.score_psi``, ``add_psi``
     - ``kivi.monitoring``: ``monitoring_table`` and ``stability_table``, from 0.8.0
   * - ``VIF``, ``StepWise``, ``feature.Corr``
     - ``kivi.selection``: ``vif``, ``stepwise``, ``correlation_filter`` and ``logistic_regression``, from 0.5.0
   * - ``PDToScore``, ``BaseScoreAndPDO``
     - ``kivi.scorecard``: ``pd_to_score``, ``score_to_pd``, ``score_scaling`` and ``points_table``, from 0.5.0
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
     - ``kivi.vintage.vintage_table`` on a loan panel, from 0.6.0
   * - ``Dataset.bank_data()``
     - ``kivi.datasets.load_bank()``, from 0.6.0. ``Dataset.bank_data()`` returns the same data with a ``FutureWarning`` from 0.9.0 and is removed in 1.0

Binning and WOE
---------------

DistanceBins and FrequencyBins previously calculated group assignments and statistics together. Use bin_edges with uniform or quantile, call pandas.cut, then pass the groups to woe_table. Codes listed in abnormal_vals become the special argument: pass them to bin_edges, and use kivi.binning.cut with the same codes in place of pandas.cut. CategoryBins users can pass the label series directly.

Version 0.5.0 adds woe_transform in place of WOEScore. WOEScore matched raw values against the min_bin and max_bin columns of a WOE table. Form the groups with the development edges instead, then pass them to woe_transform with the woe_table result. A value that matches no group raises an error, as WOEScore did with error="error"; pass unseen to choose its WOE instead.

Version 0.4.0 adds ``chimerge_edges`` and ``tree_edges`` in place of ``Chi2Bins`` and ``TreeBins``. They are new implementations rather than ports, so the edges can differ. ``Chi2Bins`` started from 10 quantile bins, tested at a 0.9 confidence level, added 0.5 to every expected count and kept at least three bins; ``chimerge_edges(values, target, alpha=0.1, initial_bins=10)`` comes closest. ``TreeBins`` reported the midpoints that scikit-learn chooses as thresholds, while ``tree_edges`` reports the largest observed value below each cut. Compare the edges on a fixed reference sample before switching.

Codes listed in ``abnormal_vals`` of ``Chi2Bins`` and ``TreeBins`` become the ``special`` argument of ``chimerge_edges`` and ``tree_edges``. As before, the codes and their labels are left out of the fit; pass the same codes to ``kivi.binning.cut`` to give each one its own group.

Population stability
--------------------

For PSI.fit(), form both sets of groups using the same reference edges, then call psi_table. It returns raw counts and a contribution per bin. Previously the psi column repeated the overall total. The new smoothing rule preserves counts and normalizes each distribution after replacing zeros. Recheck thresholds when migrating saved analyses.

Models, scorecards and calibration
----------------------------------

These modules are new implementations, so compare their output with the legacy code on a fixed reference sample before switching.

* The legacy ``lift`` column was cumulative. In ``gains_table``, ``lift`` is the lift of each band and ``cum_lift`` the cumulative lift. The metrics take a score that rises with risk, such as a PD; pass ``-score`` for scorecard points. ``ModelEvaluate`` rescaled each sample by its own minimum and maximum, so its scores were not comparable across samples; the new functions use the scores as given.
* ``VIF`` called statsmodels without a constant, which measures uncentred R²; ``vif`` includes an intercept in every auxiliary regression, as in the textbook definition, so the values differ. ``stepwise`` replaces ``StepWise`` and by default keeps every coefficient positive, as WOE inputs require. ``logistic_regression`` fits the model in NumPy, so statsmodels is no longer needed.
* ``BaseScoreAndPDO`` took bad:good odds, for example ``odds=1/50``. ``base_odds`` is good:bad, so pass ``base_odds=50``; passing ``1/50`` silently gives an offset of 712.9. The legacy ``PDToScore`` divided by ln 2 only in one of its branches; ``pd_to_score`` and ``score_to_pd`` are exact inverses.
* ``get_omega`` becomes ``prior_correction`` for probabilities and ``corrected_intercept`` for the model intercept.

Portfolio analysis and production binning
-----------------------------------------

Version 0.6.0 replaces three more legacy classes. They are new implementations as well, so compare their output with the legacy code on a fixed reference sample before switching.

* ``WOEBatch`` fitted the WOE of many columns at once, and ``ManualBinsTool`` merged bins by hand afterwards. ``fit_spec`` fits every column with the binning function chosen for it, and its ``BinningSpec`` saves the edges and WOE tables as versioned JSON. Nothing is re-binned automatically: to merge bins by hand, pass the merged edges for that column in ``bins``.
* ``CSI`` matched the points to the bins by position and lost the missing group. ``csi_table`` matches the points to the groups by label, keeps the missing group, and raises for an observed group without points unless ``unseen_points`` gives them. Its contributions sum to the change of the variable's average points.
* ``VintagePd`` built the vintage table from repayment schedules and observation dates. ``vintage_table`` takes a long panel with one row per loan and month on book and a bad flag that you define, for example ``status == "90+"``. Pass ``weights=amount`` for the amount-weighted rates of ``VintagePd``. Months that a cohort has not reached are NaN rather than 0.

Master scales, variable profiles and strategy
---------------------------------------------

Version 0.7.0 replaces the legacy master scale and variable profile. Compare their output with the legacy code on a fixed reference sample before switching.

* ``MasterScale.fit_grade_df`` and ``confirm_master_scale`` become ``fit_grade_pd`` and ``master_scale``. ``fit_grade_pd`` fits the logarithm of the default rate by least squares weighted with the number of observations of each grade, where the legacy fit gave every grade the same weight; the two agree when the grades are equally large. A grade without defaults enters the fit with half a default instead of a logarithm of zero. The bounds between grades are the geometric means of adjacent PDs; pass ``boundary="arithmetic"`` for the midpoints of the legacy scale.
* ``PDToRant(pd, ranking)`` becomes ``assign_grades(probability, scale)``. The legacy function put a PD that lies exactly on a bound in the grade above it; ``assign_grades`` puts it in the grade below, as the intervals of ``pandas.cut`` are closed on the right. It returns an ordered categorical of the grade labels instead of grade numbers.
* ``FeatureEvaluate(df, target_name).stats()`` becomes ``profile_table(frame.drop(columns=target_name), frame[target_name])``. The columns follow kivi's names, such as ``missing_share``, ``missing_event_rate`` and ``non_missing_lift``, and a target without events raises an error, where the legacy lifts were -9999. The signed Gini coefficient of the values replaces the AUC and KS of a univariate logistic fit. The table adds the IV of a quick binning, special codes, sample weights and the ``tied_bins`` column.
* Strategy analysis, reject inference and weighted model fits have no 0.2.x counterpart. ``kivi.strategy`` replaces approval and bad-rate tables built by hand from score bands, ``kivi.reject_inference`` infers the outcomes of declined applicants, and ``logistic_regression`` and ``stepwise`` take ``weights`` to refit on them.

Category grouping, validation and monitoring
--------------------------------------------

Version 0.8.0 groups categorical variables, adds the ROC table and the uncertainty of an AUC, and monitors a model period by period. These are new implementations as well.

* ``CategoryBins`` kept the labels of a categorical variable as they were. ``category_groups`` pools the rare categories and ``chimerge_groups`` merges the categories with similar event rates; both return a dictionary from category to group that is fitted on the reference sample, and ``group_categories`` applies it to every sample. A category that the grouping does not know raises an error unless ``unseen`` names its group. ``fit_spec`` groups a column with the methods ``"rare"`` and ``"chimerge"``, and a specification that groups a column is saved as format version 2, which 0.7.0 and earlier refuse to read.
* ``BinaryMetrics.roc_curve`` returned the points of scikit-learn's ``roc_curve`` with its default ``drop_intermediate=True``. ``roc_table`` has one row for every distinct score, as ``roc_curve`` with ``drop_intermediate=False``, with the cumulative counts, the two rates and the precision. Its ``precision`` and ``true_positive_rate`` columns are the precision and the recall at each threshold, so it also gives the precision-recall curve. ``roc_auc_interval`` adds the DeLong standard error and interval of the AUC and the Gini, and ``delong_test`` compares the AUCs of two scores on the same sample; the legacy code had neither.
* ``ScoreEvaluate.score_psi`` and ``add_psi`` compared the band totals of two lift tables. ``monitoring_table`` compares every period with one reference sample on score bands fitted once, adds the known outcomes, the AUC, the Gini and the KS of each period, and counts an observation without a known outcome in the population only. ``stability_table`` gives the PSI of every variable in every period.
* Lifetime PD term structures, the IFRS 9 expected credit loss and the allocation to stages have no 0.2.x counterpart; see ``kivi.lifetime``.

Not ported
----------

Spark, SQL, AHP, model wrappers and application code are out of scope and remain in the historical revisions. These 0.2.x tools have no replacement either:

* The charts of ``DefaultAnalysis``, ``plot_bin``, ``Calibration.CalibrationCurve`` and the ``plot_*`` methods of ``BinaryMetrics``, and the correlation heatmaps ``diagonal_corr`` and ``structure_corr`` of ``evaluate/Heatmap.py``. kivi returns tables, which any plotting library can draw; for the heatmaps, draw ``X.corr()`` of the WOE-encoded candidates, which ``correlation_filter`` uses as well.
* The F1 score and the confusion matrix of ``BinaryMetrics``. They judge predicted labels at one threshold, which a scorecard sets with a cut-off: ``cutoff_table`` gives the events that each cut-off approves and the shares of events and non-events that it rejects, and each row of ``roc_table`` gives the counts of the flagged events and non-events at its threshold. Use scikit-learn's ``f1_score`` and ``confusion_matrix`` where they are needed.
* The sampling helpers ``sampling`` and ``upsampling``, and the ratio and cross features of ``FeatureTools``. Use pandas.
* The grading by the Beta distribution in ``BetaCalibration``, ``BetaDistribution`` and ``MasterScale.cal_quantile``, and ``model_score_by_weight``. ``master_scale`` and ``woe_to_points`` cover the same steps without them.
* The repayment-schedule calculations of ``PortfolioAnalysis.Vintage``. Derive the delinquency status with pandas and pass the panel to ``vintage_table``.
* The skewness, kurtosis, coefficient of variation, KS test, univariate logistic fit and chi-squared test of ``FeatureEvaluate``. Compute them with pandas or scipy where they are needed.
