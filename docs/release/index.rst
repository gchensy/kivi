Release notes
=============

Unreleased
----------

Changes merged since the last release.

.. release-fragments::

1.1.0 (2026-11-22)
------------------

New
^^^

* ``kivi.strategy.confusion_table`` counts, at each cut-off of a score, the declined events and non-events and the approved ones, with the precision, recall, F1 score and accuracy of the decision. A score above the cut-off is declined, as in ``cutoff_table``, and counts as the prediction of an event. The function takes ``cutoffs`` and ``weights`` as ``cutoff_table`` does, so the rows line up, and its ``recall`` is the ``rejected_event_share`` of that table. ``precision`` is NaN for a cut-off that declines nobody.

* The new module ``kivi.plotting`` draws the result tables of the other modules with matplotlib, an optional dependency installed with ``pip install "kivi[plot]"``. ``woe_chart`` draws the share and the event rate or WOE of each group of a WOE table, ``correlation_chart`` a correlation matrix, ``roc_chart`` the ROC curve with its KS, ``gains_chart`` the cumulative shares of a gains table, ``calibration_chart`` the observed against the predicted event rate of each group, ``cutoff_chart`` the event rate of the approved book against the approval rate, ``psi_chart`` the reference and current shares of a PSI table, ``monitoring_chart`` one column of a monitoring or stability table by period, as a line or as bars, and ``vintage_chart`` the curve of each cohort of a vintage table.
* Each chart draws on the matplotlib Axes passed as ``ax``, or on a new figure, and returns the Axes it drew on. It never shows the figure, never changes ``matplotlib.rcParams`` and never modifies the table, and the text inside it is limited to column names and data values.

* :func:`~kivi.univariate.profile_table` appends five columns. Numeric columns get ``skewness`` and ``kurtosis``, the bias-corrected sample skewness and excess kurtosis that ``pandas.Series.skew`` and ``pandas.Series.kurt`` compute, and ``cv``, the coefficient of variation, all from the values without missing values and special codes and weighted with ``weights``. With a target, numeric columns get ``ks``, the KS statistic of their values, and every column gets ``chi2_p_value``, the p-value of the chi-squared test of independence between the target and the groups of the quick binning that gives ``iv``.

* :func:`~kivi.selection.correlation_filter` takes ``method="spearman"`` for Spearman's rank correlation, as ``pandas.DataFrame.corr`` computes it, which suits raw variables with monotone but not linear relations and long tails. The default ``"pearson"`` keeps the previous results and remains the choice for WOE-encoded variables.

Changed
^^^^^^^

* Fourteen topic guides gain diagrams that show how a function computes its result, with the numbers of the guide's example: the rows of ``profile_table``, edges and ``pandas.cut``, ``woe_table`` and ``woe_transform``, ``correlation_filter`` and the steps of ``stepwise``, fuzzy augmentation and the bands of parceling and reweighting, the points of a scorecard, the ROC table, isotonic regression, the confusion matrix and the swap set of a cut-off, a binning specification from the fit to JSON, ``psi_table``, the monitoring table, the transition matrix, and the expected credit loss and the days-past-due backstops of the stages. The calibration guide gains a small example of isotonic regression to go with its diagram.

* The migration guide maps the charts, the confusion matrix and F1 score, the skewness, kurtosis and coefficient of variation, and the Spearman correlation of 0.2.x to their replacements in 1.1, and it lists the 0.2.x tools that have no replacement, such as ``KSBins``, ``KmeansBins``, ``PearsonBins`` and ``Dataset.crime_data()``.
* The installation page explains the ``plot`` extra, and the conventions and compatibility pages state the rules and the promises of the charts.

* The risk modeling workflow merges the twelve jobs into three groups with ``chimerge_groups`` before it measures their WOE, sends new jobs to the group of ``unknown`` with ``unseen``, and saves that grouping in its ``BinningSpec`` with ``method={"job": "chimerge"}``. Its model, scorecard, evaluation, calibration, cut-off and stability report are computed again on the grouped variable.

* The documentation gains a tutorial section: a credit scorecard project in ten chapters, from the applications of a year to the IFRS 9 loss allowance of the loans. It profiles and bins the candidates, selects the variables, infers the outcomes of the rejects, scales and evaluates the scorecard against the incumbent's PD, calibrates it on a master scale, chooses the cut-off, monitors a year of production with a shift in the applicants, and allocates the loan book to stages. Boxes marked "Simulation check" compare each estimate with the outcomes that only the simulated data reveal.

* Type checkers infer the precise result of the functions whose result follows the type of their input, which now declare it with overloads: a Series input gives a ``pandas.Series``, a number a ``float``, and a list or a one-dimensional NumPy array an array, or a ``pandas.Categorical`` for ``cut``, ``group_categories`` and ``assign_grades``. Code such as ``pd_to_score(series, ...).round(2).index`` no longer reports a false error in mypy or Pylance. The functions are ``pd_to_score`` and ``score_to_pd`` of ``kivi.scorecard``; ``prior_correction``, ``platt_predict``, ``isotonic_predict``, ``binomial_test``, ``jeffreys_test``, ``fit_grade_pd`` and ``assign_grades`` of ``kivi.calibration``; ``kivi.strategy.expected_loss``; ``convert_pd``, ``expected_credit_loss``, ``significant_increase`` and ``assign_stages`` of ``kivi.lifetime``; ``kivi.reject_inference.reweighting``; ``woe`` and ``woe_transform`` of ``kivi.woe``; and ``cut`` and ``group_categories`` of ``kivi.binning``. An argument typed ``numpy.typing.ArrayLike`` still gets the union of the possible results, mypy infers ``Any`` for a ``Series[Any]`` such as a DataFrame column, and nothing changes at run time.

1.0.0 (2026-11-22)
------------------

Changed
^^^^^^^

* ``kivi.woe.woe`` returns a Series named ``woe``, with the index of the counts, when a count argument is a Series, as other functions shape their results like their input. Lists and arrays still give an array. Use ``.to_numpy()`` for the array and ``.iloc`` to select by position.
* kivi 1.0 is the first stable release. From now on it follows semantic versioning, as :doc:`/api/compatibility` describes, and its package metadata says ``Development Status :: 5 - Production/Stable``.

Removed
^^^^^^^

* ``smoothing=True`` raises a ``ValueError`` in ``kivi.woe.woe_table``, ``kivi.woe.woe``, ``kivi.woe.iv``, ``kivi.stability.psi_table``, ``kivi.stability.psi``, ``kivi.binning.monotonic_edges``, ``kivi.calibration.fit_grade_pd``, ``kivi.monitoring.monitoring_table`` and ``kivi.monitoring.stability_table``, as it already did in ``kivi.spec.fit_spec`` and ``kivi.binning.chimerge_groups``. kivi 0.9 read it as 1 with a ``FutureWarning``; pass ``smoothing=1`` for that result.
* ``kivi.woe.woe``, ``kivi.woe.iv`` and ``kivi.stability.psi`` raise a ``ValueError`` when both counts are Series with different indices, which kivi 0.9 matched by position with a ``FutureWarning``. Align the counts first, for example with ``Series.align(other, fill_value=0)``, or pass arrays when both list the groups in the same order.
* ``kivi.datasets.Dataset`` and its method ``bank_data()`` are removed. ``kivi.datasets.load_bank()`` returns the same frame.
* The ``probabilities`` argument of ``kivi.calibration.isotonic_predict`` is removed; pass ``probability``.

0.9.0 (2026-11-22)
------------------

New
^^^

* A new page, :doc:`/api/compatibility`, states what kivi promises from 1.0: semantic versioning, with deprecations that warn with a ``FutureWarning`` for the rest of a major version, fixes of wrong results in any release, and Python 3.10, NumPy 1.24 and pandas 2.0 as the oldest supported versions, which rise only in a minor release announced by the release before. It also lists the public names, the index and the columns of every table, the exception and warning types, and what later releases read from a saved ``BinningSpec``.
* :doc:`/api/conventions` states more of the rules that every function follows: the order of the data of other observations, the names of frames and similar options, the default ``bins``, how Series results are named, the index names of tables, the fixed rows of some tables, missing values in the binning fits, and the functions without ``weights``.
* The tests keep specifications saved by kivi 0.6.0 and 0.8.0 and check that they still load and encode a sample as those releases did, and that they match a fresh fit up to rounding.

* kivi ships inline type hints and a ``py.typed`` marker, so type checkers such as mypy check code that calls it. Every function in the package is annotated and checked with mypy.

Changed
^^^^^^^

* ``kivi.woe.woe``, ``kivi.woe.iv`` and ``kivi.stability.psi`` warn with a ``FutureWarning`` when both counts are Series with different indices. They still match the counts by position; kivi 1.0 raises a ``ValueError``, as the other functions do for Series that describe the same groups. Pass Series with identical indices, or arrays to keep matching by position.
* ``smoothing=True`` warns with a ``FutureWarning`` in ``kivi.woe.woe_table``, ``kivi.woe.woe``, ``kivi.woe.iv``, ``kivi.stability.psi_table``, ``kivi.stability.psi``, ``kivi.binning.monotonic_edges``, ``kivi.calibration.fit_grade_pd``, ``kivi.monitoring.monitoring_table`` and ``kivi.monitoring.stability_table``. It still counts as 1, not as the default 0.5; kivi 1.0 raises a ``ValueError``, as ``kivi.spec.fit_spec`` and ``kivi.binning.chimerge_groups`` already do. Pass ``smoothing=1`` for the same result.
* ``kivi.datasets.Dataset.bank_data()`` warns with a ``FutureWarning`` and is removed in kivi 1.0. ``kivi.datasets.load_bank()`` returns the same frame.
* ``kivi.calibration.isotonic_predict`` takes the probability of each band as ``probability``, the name the other functions use, and ``kivi.calibration.isotonic_fit`` documents its second result under that name. ``probabilities=`` still works with a ``FutureWarning`` until kivi 1.0.
* ``kivi.strategy.expected_loss`` names a Series result ``expected_loss``. It took the name of the first Series among its inputs, such as ``pd`` or ``balance``.
* ``kivi.calibration.master_scale`` labels the grades by the index of a Series ``probability`` when ``labels`` is not given, so ``master_scale(fit_grade_pd(event, total))`` keeps the grades of a table. A list or an array still gives grades numbered from 1.
* ``kivi.lifetime.assign_stages`` also takes ``days_past_due`` and ``defaulted`` by position, as in ``assign_stages(significant_increase, days_past_due, defaulted)``. Calls with keywords work as before.
* The index of ``kivi.selection.logistic_regression`` and of ``kivi.selection.vif`` is named ``variable``, like that of ``kivi.scorecard.points_table``, whatever the name of the columns of ``X``. The index of ``kivi.metrics.roc_auc_interval`` is named ``statistic``. ``reset_index()`` therefore gives a column of that name, and ``kivi.scorecard.woe_to_points`` with coefficients taken from ``logistic_regression`` gives columns named ``variable``.

* Error messages show the value at fault in the same form everywhere. NumPy scalars appear as plain numbers, so ``bins=numpy.int64(0)`` gives ``got 0`` instead of ``got np.int64(0)``. Every function that takes a target says ``target must contain only 0 and 1, got 2`` or ``target must contain both 0 and 1, got only 0``; the weights messages show the first invalid weight or the total of each class; and a weights index that differs from the data says ``score and weights must have identical indices``, like the other arguments.
* The messages that NumPy and pandas gave for kivi's arguments now name those arguments: ``kivi.binning.cut`` says ``edges must be increasing numbers, got [3, 1]`` where pandas spoke of ``bins`` and of a ``duplicates`` keyword that ``cut`` does not have, and the binning functions say ``values must be numbers or special values, got 'a'``. ``kivi.binning.bin_edges`` and the supervised binning functions say ``values`` instead of ``reference values``.
* ``kivi.version`` and ``kivi.PackageNotFoundError``, which the package imported from ``importlib.metadata`` by accident, no longer exist; use ``kivi.__version__``. The public modules no longer bind the helpers they import, such as ``kivi.binning.as_series`` or ``kivi.metrics.normal_quantile``. None of these names was documented.

Fixed
^^^^^

* ``kivi.woe.woe_transform`` looked a number up in the interval of a ``pandas.cut`` table that contained it, so raw values and special codes that the table did not have, such as ``-1`` grouped by ``kivi.binning.cut`` with ``special=[-1]``, silently got the WOE of an interval. Labels are now matched by equality, as documented, and such groups are unseen: they raise an error unless ``unseen`` gives them a WOE.
* ``kivi.binning.cut`` now raises an error for an infinite value that is not a special value. ``pandas.cut`` left ``-inf`` in no interval, so that it was counted as a missing value, and put ``inf`` in the last interval. Pass infinite codes in ``special`` to give them their own group.
* ``kivi.spec.BinningSpec.woe_transform`` gave ``inf`` the WOE of the last interval, while ``-inf`` was unseen. Both are now unseen, like every infinite value that is not a special code of its column, so they raise an error unless the column has an unseen WOE. ``BinningSpec.cut`` leaves both missing.
* ``kivi.stability.csi_table`` raised an error for points of several variables only when the variables were in an index named ``variable``. A ``points_table`` result after ``reset_index()``, with the variables in a ``variable`` column, was read as one variable and mixed their groups. It now raises the same error.
* ``kivi.spec.fit_spec`` binned a column with a complex dtype by its real part, with NumPy's ``ComplexWarning``. It now raises an error, and so do ``BinningSpec.cut`` and ``BinningSpec.woe_transform`` for a complex column fitted as numeric.
* ``kivi.spec.BinningSpec`` and ``dataclasses.replace`` rejected a numeric table whose intervals ran between the exact edges whenever ``pandas.cut`` rounds them in its labels, such as an edge of 0.6428571428571428 shown as 0.643. Such intervals are now accepted and stored under the labels of ``kivi.binning.cut``, as ``BinningSpec.from_dict`` already did.
* ``kivi.woe.woe_transform`` accepted ``unseen=numpy.nan``, which gave unseen groups a missing WOE, and infinite values. ``unseen`` must now be ``"raise"`` or a finite number, as for ``kivi.spec.fit_spec`` and ``kivi.stability.csi_table``.

* ``kivi.calibration.prior_correction``, the ``probability`` of ``calibration_table``, ``binomial_test``, ``jeffreys_test``, ``hosmer_lemeshow`` and ``herfindahl_index`` read numbers given as strings, so that ``herfindahl_index(["50", "30"])`` gave 0.53125. They now raise an error that names the argument, such as "counts must be real and numeric, got dtype object with the value '50'", as the other functions of the module do.
* The counts of ``kivi.calibration`` read True and False as 1 and 0, so that ``binomial_test([True], [True], 0.5)`` gave 0.5. The ``event`` and ``total`` of ``binomial_test``, ``jeffreys_test``, ``hosmer_lemeshow`` and ``fit_grade_pd`` and the ``counts`` of ``herfindahl_index`` now reject booleans, as weights do. Probabilities and scores still count them as 1 and 0.
* ``kivi.calibration.fit_grade_pd`` raised "the fitted PDs must rise with the grade" for grades that were in order when ``smoothing`` was not smaller than the total of a grade without defaults, as with the counts of a ``calibration_table`` whose weights sum to 1. It now says what is wrong, such as "smoothing must be smaller than the total of each grade without defaults, got 0.5 for grade 1 with a total of 0.4", and the documentation says that weighted counts must be on the scale of observation counts.

* Invalid arguments that NumPy or pandas used to reject with a ``TypeError`` now raise a ``ValueError`` that names the argument, as the conventions promise: a ``smoothing`` that is a string, ``None`` or an array, an ``rng`` that ``numpy.random.default_rng`` does not accept in ``kivi.reject_inference.parceling``, ``kivi.datasets.make_applications`` and ``kivi.datasets.make_loan_panel``, ``edges`` of ``kivi.binning.cut`` that are not numbers, and a WOE table of ``kivi.woe.woe_transform`` or ``kivi.scorecard.points_table`` that holds a value that is not a number.

* ``kivi.lifetime.significant_increase`` let floating-point rounding decide ties: a PD of 0.04 against 0.03 counted as more than one percentage point higher, and 0.45 as more than 1.5 times 0.30. A PD within a few units in the last place of a threshold now meets it exactly and is not flagged, as the strict thresholds say. Of the 89 pairs of two-decimal PDs that are 0.01 apart, 71 were flagged before and none are now.
* ``kivi.lifetime.transition_pd`` accepted a float such as ``horizons=3.0``. A single number of periods must now be an integer, as the ``horizon`` of ``expected_credit_loss`` is; a sequence of horizons may still hold whole numbers as floats.

* A list or an object array of numbers with ``pd.NA``, such as ``[0.5, pd.NA]``, raised a ``TypeError`` or "must be real and numeric, got dtype object" in ``kivi.calibration``, ``kivi.scorecard``, ``kivi.strategy`` and ``kivi.lifetime``, and ``kivi.calibration.platt_predict([0.5, pd.NA], 1, 0)`` named the valid value 0.5 in its message. These functions now read ``pd.NA`` as a missing value, like None and NaN, and their messages show the first value that is not a number.

* ``kivi.scorecard.pd_to_score`` and ``score_to_pd`` read any input as floats: ``score_to_pd`` of a date gave a PD of 0, and ``pd_to_score(["0.1", "0.2"], ...)`` gave scores. Strings, dates and complex numbers now raise "must be real and numeric", as in ``kivi.metrics``, while booleans still count as 1 and 0.
* ``kivi.scorecard.reason_codes`` listed a variable twice, such as ``housing, housing, contact``, when a column of ``points`` was repeated. It now raises an error, as ``woe_to_points`` does.
* ``kivi.scorecard.reason_codes`` gave None between two reasons when the points of a group in ``table`` were missing, for example after editing the table by hand. Missing or infinite points of a variable of ``points`` now raise an error.
* ``kivi.scorecard.woe_to_points`` aligned a mapping of Series by their index labels, so that one row could combine the WOE values of different applicants. The Series of a mapping must now have identical indices, as the Series that describe the same observations must everywhere in kivi; other values are still matched by position.
* ``kivi.scorecard.score_scaling`` returned NumPy scalars for NumPy arguments, so that a ``float32`` ``pdo`` rounded the scores of ``pd_to_score`` and the points of ``points_table`` to single precision. It now returns Python floats.
* ``kivi.scorecard.points_table`` and ``woe_to_points`` said "tables has no entry for 1 variable(s): 'intercept'" for the ``coef`` column of ``kivi.selection.logistic_regression``, which includes the intercept. They now say "coefficients has an 'intercept' entry; drop it and pass it as intercept".

* ``kivi.selection.vif`` gave a finite VIF for some exact linear combinations: 1.003 for ``a`` when ``c`` is ``b + 1e-5 * a``, and 5.6e29 for the two columns of a frame with two rows. A column now has an infinite VIF when it is a linear combination of the others up to rounding, as documented; the VIFs of the other columns are unchanged.

* ``kivi.vintage.transition_matrix`` read any ``normalize`` that is true in Python as True, so that ``normalize="columns"``, as ``pandas.crosstab`` takes it, gave the shares of each row. It now raises unless ``normalize`` is True or False, as ``kivi.calibration.herfindahl_index`` does.
* ``kivi.vintage.transition_matrix`` and ``roll_rates`` gave a missing value in ``states`` a row and a column of its own, although missing values of ``before`` and ``after`` raise. A missing state now raises an error.
* ``kivi.vintage.vintage_table`` made a column of a negative ``month_on_book``. It now raises an error, because the months on book count from origination.

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
