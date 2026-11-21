Release notes
=============

Unreleased
----------

Changes merged since the last release.

.. release-fragments::

0.8.0 (2026-11-21)
------------------

New
^^^

* ``kivi.binning`` groups the categories of categorical variables in the same two steps as numeric binning. ``category_groups`` pools the categories below ``min_share`` or ``min_count`` into one group, and ``chimerge_groups`` sorts the categories by their smoothed WOE and merges neighbours by ChiMerge under ``max_groups``, ``alpha`` and ``min_group_share``. Both return a dict from category to group, take ``special`` and ``weights``, and leave missing values out. ``group_categories`` applies the dict to any sample; a category that it does not map raises an error unless ``unseen`` names the group it should join.
* ``fit_spec`` groups a categorical column that ``method`` names with ``"rare"`` or ``"chimerge"``; ``special`` and ``options`` then apply to it, and so does ``bins`` with ``"chimerge"``. ``BinningSpec`` keeps the groupings in a new ``groups`` field and applies them in ``cut`` and ``woe_transform``. Columns without a grouping are fitted as before.
* The unseen policy of a grouped column can name one of its groups as ``{"group": label}``: ``cut`` then assigns new categories to that group, and ``woe_transform`` gives them its WOE.
* A ``BinningSpec`` with a grouped column is saved with ``format_version`` 2, which adds the groupings and the unseen groups; every other specification is still saved as version 1, so earlier versions of kivi read it. ``from_json`` reads both versions.

* The new module ``kivi.lifetime`` covers lifetime PDs and the expected credit loss of IFRS 9. ``convert_pd`` converts a PD term structure, such as the cumulative bad rates of a vintage table, between cumulative, marginal and conditional PDs, and keeps the months on book that a cohort has not reached missing. ``transition_pd`` projects the cumulative PD of each state of a transition matrix over any number of periods, with the default state made absorbing.
* ``expected_credit_loss`` adds up the marginal PD, the LGD and the EAD of each period after the reporting date, discounted at the effective interest rate, into the 12-month or the lifetime expected credit loss of each account. The LGD and the EAD can be constants, one value per account or schedules per account and period.
* ``significant_increase`` flags a significant increase in credit risk from a relative and an absolute PD threshold, which have no defaults because they are the entity's policy. ``assign_stages`` gives stages 1 to 3 from these flags, the days past due backstops, by default more than 30 days for stage 2 and 90 days or more for stage 3 as presumed by IFRS 9, and defaulted flags, and ``stage_table`` sums the exposure and the loss allowance of each stage, with its coverage ratio.

* The new module ``kivi.monitoring`` tracks a score over time. ``monitoring_table`` gives one row per period, such as a month, with the number of observations, the known outcomes and events, the AUC, Gini and KS of the known outcomes, and the PSI of the score against a reference period or a reference sample, on bands fitted once on the reference. A missing target marks an outcome that is not known yet; it counts in ``total`` but not in the event columns, and a period without both classes has NaN discrimination instead of an error.
* ``stability_table`` gives the PSI of every grouped variable in every period against the same reference, for the variable part of a stability report. Both functions take ``weights``, and integer weights give the same table as repeating rows, for ``monitoring_table`` on the same band edges.

* ``kivi.metrics.roc_table`` lists the points of the ROC curve, one row per distinct score from the highest to the lowest, with the cumulative event and non-event counts, the true and false positive rates and the precision. The trapezoids under it add up to ``roc_auc``, and the largest gap between the rates is ``ks``.
* ``kivi.metrics.roc_auc_interval`` gives the AUC and the Gini coefficient a standard error and a confidence interval at a ``level``, using DeLong's method with the fast midrank algorithm of Sun and Xu.
* ``kivi.metrics.delong_test`` tests whether two scores of the same observations have the same AUC, and returns both AUCs, their difference with its standard error and interval, the z statistic and the two-sided p-value.
* The standard errors read ``weights`` as frequency weights, so integer weights give the same results as repeating rows. They reproduce the DeLong results that Stata's and SAS's manuals publish for the data of Hanley and McNeil (1982) and of DeLong et al. (1988).

Changed
^^^^^^^

* ``BinningSpec.format_version`` is now the version of the JSON format that each specification writes, 1 or 2, instead of a class attribute equal to 1. The new class constant ``BinningSpec.FORMAT_VERSION`` is the newest version that kivi reads, now 2.

* The topic guides follow one outline with the same section titles, such as ``Special values``, ``Sample weights`` and ``Errors and warnings``, and their openings link to the steps before and after them. The binning guide gathers its special values in one section, and the univariate profile shows special codes after the screening by IV. The first mention of a kivi function in each section links to its API entry.
* The API reference starts with two new pages: the conventions that every function follows, such as the order of the arguments, keyword-only options, inputs and results, sample weights and error messages, and a glossary of credit risk terms. Its module pages come in the order of the user guide, each with an introduction and a link to its guide, and ``:mod:`` links now reach them.
* The migration guide lists the ROC and precision-recall curves, precision, recall, F1 score and confusion matrix of ``BinaryMetrics`` and the correlation heatmaps of ``evaluate/Heatmap.py`` as not ported, with their replacements.

* The user guide gains the guides "Model monitoring" and "Lifetime PD and expected credit loss" in their places in a project. The glossary and the conventions page cover the terms and rules of the new functions. The risk modeling workflow reports the DeLong interval of the Gini and points on to monitoring and the expected credit loss. The guides have new charts for category grouping, the ROC curve, the monitoring report, variable stability by period and PD term structures, and the README shows the monitoring and lifetime PD charts.

Fixed
^^^^^

* ``kivi.metrics.roc_auc`` and ``kivi.metrics.gini`` could return a value just above 1 when fractional ``weights`` rounded the AUC past it. Both are now at most 1, as documented. Other values are unchanged.

* ``kivi.stability.psi_table``, ``kivi.stability.csi_table`` and ``kivi.monitoring.stability_table`` counted the groups of the current sample under the wrong labels when both samples were unordered categoricals with the same categories listed in another order, such as ``["a", "b"]`` and ``["b", "a"]``. The groups are now matched by label.
* ``kivi.stability.psi_table`` and ``kivi.stability.csi_table`` raised pandas' ``IntCastingNaNError`` for two samples of ``pandas.cut`` bins on other integer edges when one of them had missing values. Such bins are now joined as intervals of floats, which compare equal to the integer intervals.

* ``kivi.stability.psi_table`` and ``kivi.stability.csi_table`` no longer emit a ``RuntimeWarning`` ("invalid value encountered in cast") on NumPy older than 2.2 when one sample holds ``pandas.cut`` bins with an infinite edge and the other sample has other groups: bins cut on other edges or with unused categories removed, intervals that are not categorical, or other labels. The tables are unchanged.

* A ``weights`` Series was compared only with the index of the data, so when the score, the values or the groups were an array, it was matched by position to a ``target`` Series with another index. The functions of ``kivi.metrics``, ``kivi.woe.woe_table``, ``kivi.calibration.calibration_table``, ``platt_fit`` and ``isotonic_fit``, the supervised functions of ``kivi.binning``, and ``kivi.selection.logistic_regression`` and ``stepwise`` now raise in that case, for example "target and weights must have identical indices", as they do for any other two Series that describe the same observations.

0.7.0 (2026-11-20)
------------------

New
^^^

* ``kivi.datasets.make_applications`` simulates seeded loan applications with numeric and categorical features, the PD of an incumbent scorecard, policy declines, the approval decision and a true default flag for every applicant, to try reject inference on.

* ``kivi.calibration.master_scale`` builds a master scale from the PD of each grade, such as a geometric grid from ``numpy.geomspace``: one row per grade with its lower bound, PD and upper bound, covering 0 to 1. The bounds are the geometric means of adjacent PDs by default, or their arithmetic means with ``boundary="arithmetic"``, as in the 0.2 versions.
* ``fit_grade_pd`` fits the PD of each grade to observed default rates, with the logarithm of the rate a straight line in the grade number, by least squares weighted with the number of observations. A grade without defaults enters the fit with ``smoothing`` defaults, and a grade without observations gets a PD from the line.
* ``assign_grades`` gives each predicted PD the grade of a scale as an ordered categorical, which ``calibration_table`` counts directly, in the order of the scale. A PD on a bound goes to the lower grade, as with ``pandas.cut``.

* The new module ``kivi.reject_inference`` infers outcomes for rejected applicants, so that a scorecard developed on approved applicants can be refitted on all of them. ``fuzzy_augmentation`` splits each reject into a bad and a good row weighted by its PD, ``parceling`` labels the rejects of each score band at the band's bad rate with a seeded generator, or returns the expected weights, and ``reweighting`` weights the approved applicants of each band by the inverse of its approval rate. A ``factor`` above 1 raises the inferred bad rate, and bands without approved applicants raise an error.

* ``logistic_regression`` and ``stepwise`` take ``weights``, frequency weights like those of statsmodels' ``GLM(freq_weights=...)``. Integer weights give the same coefficients, standard errors and selection as repeating rows; weights that do not count observations, such as those of reject inference, make the standard errors and p-values too small.

* The new module ``kivi.strategy`` supports the credit decisions that follow a scorecard. ``cutoff_table`` shows, for each candidate cut-off of a PD or of negated points, the approval rate, the bad rate of the approved book, the shares of bads and goods it rejects and, with ``loss``, the expected loss it approves. Scores at or below the cut-off are approved, so tied scores always get the same decision.
* ``swap_set`` compares the decisions of a current and a new score at their cut-offs: the applicants both approve, the swap-ins and swap-outs, and those both decline, with event rates where outcomes are known. A missing target marks an unknown outcome, such as that of an applicant the current score declined.
* ``expected_loss`` multiplies PD, LGD and EAD, and ``expected_loss_table`` adds up the exposure and the expected loss of each grade, with the expected loss rate and the exposure-weighted PD.

* The new module ``kivi.univariate`` profiles the candidates of a model. ``profile_table`` summarizes every column of a data frame in one row: its kind, missing values, distinct and most frequent values and, for numeric columns, the range, quantiles, mean and standard deviation. With a target it adds the event rates and lifts of missing and non-missing rows, the IV of a quick binning and the Gini coefficient of numeric values. Tied quantiles are recorded in the ``tied_bins`` column instead of raising a warning for each column, and ``special`` codes and ``weights`` work as in ``fit_spec``.

Changed
^^^^^^^

* The risk modeling workflow screens the candidates with ``profile_table``, calibrates the PD on a master scale with ``master_scale`` and ``assign_grades``, and sets a cut-off with ``cutoff_table`` and ``expected_loss_table``. The user guide has new charts for cut-off curves, reject inference, grade PDs and missing values, and the README shows the cut-off and reject inference charts.

Fixed
^^^^^

* ``gains_table`` warned about tied scores by suggesting the ``special`` argument of ``bin_edges``, which ``gains_table`` does not take. It now suggests fewer bins or explicit edges, as ``cutoff_table`` does, and the warning points at the line that called it.

0.6.0 (2026-11-20)
------------------

New
^^^

* ``kivi.calibration.platt_fit`` and ``platt_predict`` calibrate a score with a logistic curve, and ``isotonic_fit`` and ``isotonic_predict`` with a step function that rises with the score. The fits return plain numbers, a slope and an intercept or band edges and their probabilities, that apply to the scores of any later sample.

* ``kivi.stability.csi_table`` computes the characteristic stability index of a scorecard variable: the change of each group's share from the reference to the current sample, multiplied by the group's points. Its sum is the change of the variable's average points, and over all variables the change of the average score. The shares are not smoothed, and a group without points raises an error unless ``unseen_points`` gives its points.

* ``kivi.datasets.load_bank`` loads the bank marketing data. ``Dataset.bank_data`` returns the same data and keeps working without a warning.
* ``kivi.datasets.make_loan_panel`` simulates a seeded monthly panel of instalment loans whose delinquency rolls forward and cures, to try the vintage and roll-rate functions on.

* ``reason_codes`` takes ``unseen_points``, the points of applicants in groups that ``woe_transform`` encoded with ``unseen``, one value or values by variable. Points that match no group but lie within ``atol`` of ``unseen_points`` cost nothing, so the variable is not one of those applicants' reasons. Every other mismatch still raises an error, which now suggests checking ``atol`` and the model and scaling first.

* The new module ``kivi.spec`` bins all the columns of a data frame at once. ``fit_spec`` fits each numeric column with ``bin_edges`` or a supervised method chosen by column, or with given edges, and measures the WOE of every group; other columns keep their labels. The binning is never changed automatically.
* The resulting ``BinningSpec`` groups a sample with ``cut``, encodes it with ``woe_transform`` using an unseen-group policy by column, and keeps the ``woe_table`` of every column for review. ``to_json`` and ``from_json`` save and load it without changing the groups or the WOE encoding, so production reproduces development exactly.

* The new module ``kivi.vintage`` analyses loan portfolios. ``vintage_table`` gives the cumulative bad rate of each origination cohort by month on book, by count or by amount lent, and leaves the months that a cohort has not reached as NaN. ``transition_matrix`` counts the movements of loans between delinquency buckets or of obligors between rating grades, as counts, row shares or sums of balances, and ``roll_rates`` gives the share of each bucket that rolls to the next worse one.

* ``roc_auc``, ``gini``, ``ks``, ``gains_table``, ``brier_score`` and ``calibration_table`` take ``weights``, like ``woe_table``. Counts become sums of weights, and integer weights give the same result as repeating rows, for ``gains_table`` on fixed edges.

Changed
^^^^^^^

* The risk modeling workflow saves the shortlist's binning with ``fit_spec`` and adds the characteristic stability index of every scorecard variable to its stability report. The user guide has new charts for gains, calibration and vintage curves, and the README shows the gains and vintage charts.

Fixed
^^^^^

* On pandas 2.0, ``calibration_table`` no longer fails when the target has a nullable integer dtype.

* ``cut``, ``bin_edges`` and the supervised binning functions match special codes after converting numeric strings, so a column read as text, where ``pdays`` holds ``"-1"``, gives the code ``-1`` its own group instead of putting it into an interval. Boolean special codes, which ``True == 1`` made match the value 1, raise an error.

Removed
^^^^^^^

* ``kivi.evaluate`` is removed; import ``psi_table`` and the other stability functions from ``kivi.stability``.
* ``bin_edges`` no longer accepts ``method`` by position; pass ``method="uniform"`` by keyword.

0.5.0 (2026-11-20)
------------------

New
^^^

* The new module ``kivi.calibration`` checks the calibration of PDs. ``prior_correction`` and ``corrected_intercept`` correct a model for a development sample with another event rate. ``calibration_table`` compares the observed and predicted default rates of master scale grades or other groups, which ``binomial_test``, ``jeffreys_test`` and ``hosmer_lemeshow`` then test. ``herfindahl_index`` measures the concentration of the grades.

* ``kivi.woe.woe``, ``kivi.woe.iv`` and ``kivi.stability.psi`` compute WOE, IV and PSI from counts that are already aggregated, including weighted counts.
* ``kivi.stability`` holds ``psi_table`` and ``psi``.

* ``kivi.metrics`` evaluates a score against a binary target. ``roc_auc``, ``gini`` and ``ks`` measure how well it ranks events above non-events, ``gains_table`` reports event rates, captured events, KS and lift by score band, and ``brier_score`` compares predicted probabilities with the outcomes. Like ``woe_table``, each takes the score first and the target second, the reverse of scikit-learn. A higher score must mean a higher event probability, so negate scorecard points before passing them in.

* ``bin_edges``, ``chimerge_edges``, ``tree_edges``, ``monotonic_edges`` and ``woe_table`` take ``weights``, and ``psi_table`` takes ``reference_weights`` and ``current_weights``. Counts become sums of weights, quantile edges become weighted quantiles by the inverted CDF, and observations or groups with zero weight are left out.

* The new module ``kivi.scorecard`` scales a logistic regression on WOE values into scorecard points. ``score_scaling``, ``pd_to_score`` and ``score_to_pd`` convert between probabilities and scores from the points to double the odds, a base score and base odds. ``points_table`` gives the points of every group, ``woe_to_points`` those of each applicant, and ``reason_codes`` names the variables that cost an applicant the most points.

* The new ``kivi.selection`` module selects the variables of a scorecard from WOE-encoded candidates. ``correlation_filter`` drops one variable of each strongly correlated pair, keeping the one with the higher priority, such as its IV. ``vif`` computes variance inflation factors.
* ``logistic_regression`` fits a logistic regression in NumPy and reports each coefficient with its standard error, Wald z statistic and p-value. It raises ``SeparationError`` when the variables separate the classes, so that the estimates do not exist.
* ``stepwise`` selects variables by their p-values in forward, backward or both directions. By default, it keeps every coefficient positive, as expected for kivi's WOE.

* ``woe_transform`` replaces each group with its WOE from a ``woe_table`` result, the encoding used by a scorecard model. Groups that are not in the table raise an error unless ``unseen`` gives their WOE.

Changed
^^^^^^^

* The API reference has a page for each module. Links to a function on the former single page now go to the page of its module.

* ``method`` of ``bin_edges`` is keyword-only. Passing it by position raises a ``FutureWarning`` and stops working in 0.6.0.
* ``kivi.evaluate`` is deprecated in favour of ``kivi.stability``. Importing ``psi_table`` from it raises a ``FutureWarning``, and the module is removed in 0.6.0.

* The risk modeling workflow runs end to end: variable selection, a fitted logistic model, scorecard points, evaluation, calibration and score stability.

* ``woe_table`` also reports ``total``, ``total_share`` and ``event_rate`` for each group, the columns of a classing report. They use the observed counts; only ``non_event_share`` and ``event_share`` are smoothed.

Fixed
^^^^^

* ``woe_table`` and ``psi_table`` no longer cause pandas to issue ``RuntimeWarning: invalid value encountered in cast`` on NumPy older than 2.2 when the groups come from ``pandas.cut`` with infinite edges.

* On pandas 2.0, ``cut`` and the supervised binning functions no longer fail when nullable integer or float values are combined with ``special``.

0.4.0 (2026-11-19)
------------------

New
^^^

* ``bin_edges`` takes a ``special`` argument. Codes such as ``-1`` or ``-999`` are left out when the edges are fitted.
* ``kivi.binning.cut`` assigns values to the fitted bins and gives each special value its own group. Without special values it returns the same result as ``pandas.cut``.

* ``chimerge_edges``, ``tree_edges`` and ``monotonic_edges`` fit edges from a binary target: ChiMerge merging of fine quantile bins, decision-tree splits by entropy or Gini with a minimum bin share, and merging until the WOE is monotonic. They accept the same array-like inputs as ``woe_table``, return infinite-ended edges like ``bin_edges`` and need only NumPy and pandas.
* The binning guide compares the three methods on the bank data, with a chart for each.

* ``chimerge_edges``, ``tree_edges`` and ``monotonic_edges`` take ``special`` like ``bin_edges``. The codes are left out of the fit together with their labels, and ``kivi.binning.cut`` with the same codes gives each one its own group.

Changed
^^^^^^^

* ``bin_edges``, ``woe_table`` and ``psi_table`` accept any one-dimensional array-like input, such as a list or a NumPy array. Group and target series passed to ``woe_table`` must still share an index; arrays and lists are matched by position.
* ``bin_edges`` issues a ``UserWarning`` when tied edges leave fewer bins than requested. The message names the most frequent value so that it can be passed in ``special``.
* Error messages name the argument and report the value received.
* ``kivi.__version__`` is ``"0+unknown"`` when kivi is imported from a source tree that is not installed, instead of raising an error.
* Group counts use ``numpy.bincount``, which is faster on large samples.
* The API reference documents errors, warnings, formulas and related functions.

* ``woe_table`` and ``psi_table`` list categorical groups, including the result of ``pandas.cut``, in category order with missing values last. Other labels keep the order in which they first appear. Previously all groups followed first appearance, and ``sort_index`` was needed to put intervals in order.

0.3.0 (2026-11-19)
------------------

A rewrite around a small pandas core. Read :doc:`migration` before upgrading code written for 0.2.x.

New
^^^

* ``bin_edges`` fits quantile or uniform edges on a reference sample. The outer edges are infinite, so the result can be passed to ``pandas.cut`` for any later sample.
* ``woe_table`` and ``psi_table`` return raw counts, smoothed shares and a contribution for each group, including missing values.
* Documentation in English and Simplified Chinese.

Changed
^^^^^^^

* NumPy and pandas are required dependencies. The 0.2.8 wheel placed them behind the ``all`` extra.
* The ``psi`` column holds the contribution of each bin. It previously repeated the overall total.
* Zero counts are replaced before each distribution is normalized, and reported counts are no longer modified.

Removed
^^^^^^^

* Spark, SQL, AHP, feature selection, model wrappers, weighted and supervised binning and scorecards. They remain in the repository history.

.. toctree::
   :hidden:

   migration
