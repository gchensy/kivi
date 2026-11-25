Conventions
===========

Every public function of kivi follows the rules on this page. The module pages and the guides take them for granted, and the docstrings mention only where a function departs from them or adds to them.

Imports
-------

kivi has no top-level imports. Import each function from its module, as in ``from kivi.woe import woe_table``; each module has its own page in this reference.

Only the names that a module lists in ``__all__`` are public. :doc:`compatibility` says which of them a release may change, and how it announces the change.

kivi needs only NumPy and pandas. :mod:`kivi.plotting` also needs matplotlib, which the ``plot`` extra installs (``pip install "kivi[plot]"``); it imports matplotlib only when it draws a chart, so importing kivi never requires it.

Arguments
---------

* The data come first and the target follows them, as in ``woe_table(groups, target)``, ``roc_auc(score, target)``, ``stepwise(X, target)`` and ``fit_spec(frame, target)``. A function with several data arguments takes all of them before the target, as ``swap_set(current, new, target)`` does, and a predicted probability comes after the target, as in ``calibration_table(groups, target, probability)``. The back-test of continuous estimates follows the same order, with the realized values in the place of the target and the estimates in that of the probability, as in ``backtest_table(groups, realized, estimated)``, and :func:`~kivi.metrics.generalized_auc` takes the estimated segments as its data, like a score, and the realized ones as its target, as in ``generalized_auc(estimated, realized)``. scikit-learn's metrics take the target first, so check the order when porting code from them.
* Data that describe other observations, such as the rejected applicants, follow the data and the target of the main sample, as in ``parceling(score, target, reject_score)`` and in ``reweighting(score, reject_score)``, which has no target. In the same way, the payments of :func:`~kivi.vintage.delinquency_panel` follow its schedule, as in ``delinquency_panel(loan, due_date, amount_due, payment_loan, payment_date, payment_amount)``.
* The functions of :mod:`kivi.features` take tables with named columns, as :func:`~kivi.spec.fit_spec` and :func:`~kivi.univariate.profile_table` do: first the events or the monthly panel, then ``observations``, with one row per entity and observation time, such as the applicants and their application dates, as in ``window_features(events, observations, id=..., time=..., observation_time=...)``. The names of the columns are keyword arguments. ``id`` names the entity in both tables, and its values are matched by equality, so ``1`` and ``"1"`` are different entities.
* ``X`` is a numeric model matrix, as in :func:`~kivi.selection.stepwise`, :func:`~kivi.selection.logistic_regression`, :func:`~kivi.selection.selection_path`, :func:`~kivi.selection.vif`, :func:`~kivi.selection.vif_filter` and :func:`~kivi.selection.correlation_filter`. ``frame`` holds raw columns, as in :func:`~kivi.spec.fit_spec`, :func:`~kivi.univariate.profile_table` and :meth:`BinningSpec.cut <kivi.spec.BinningSpec.cut>`, and ``woe`` is the matrix of WOE values that :func:`~kivi.scorecard.woe_to_points` turns into points.
* ``matrix`` is a square matrix of pairwise comparisons in :mod:`kivi.expert`: a DataFrame whose index and columns hold the same factor labels in the same order, or an array or a nested list, whose factors are numbered from 0. Its diagonal must be 1 and its entries reciprocal, :math:`a_{ij} a_{ji} = 1`, to the relative tolerance of the keyword ``rtol``, 1e-3 by default, which accepts reciprocals rounded to four decimals, such as 0.3333 for 1 / 3. The ``matrix`` of :func:`~kivi.lifetime.transition_pd` is a transition matrix instead, whose rows are probabilities that add up to 1.
* ``grades`` lists the grades of a rating system from the best to the worst, at least two, in :func:`~kivi.vintage.migration_bandwidth` and :func:`~kivi.vintage.migration_test`. They take the grade of each customer at the beginning of the period and its status at the end, as in ``migration_test(before, after, grades=grades)``, like :func:`~kivi.vintage.transition_matrix`. Every value of ``before`` must be a grade, while a status at the end that is not one, such as ``"default"`` or ``"closed"``, counts only among the customers of the grade at the beginning.
* ``realized`` and ``estimated`` are continuous values, such as the realized LGD of a closed default and its estimate, which may lie outside [0, 1]. For :func:`~kivi.metrics.generalized_auc` they are ordered segments: numbers in the order of their values or an ordered categorical in the order of its categories.
* A predicted probability of an event, such as a PD, is always called ``probability``, as in ``brier_score(probability, target)``, ``calibration_table(groups, target, probability)`` and ``expected_loss(probability, lgd, ead)``. What a ``score`` is depends on the module; see :ref:`conventions-scores`. A second probability is named after its meaning, as ``origination_probability`` in :func:`~kivi.lifetime.significant_increase`.
* An option with the same meaning has the same name in every module, such as ``bins``, ``special``, ``smoothing``, ``unseen`` and ``weights``. ``level`` is the confidence level of an interval or a bound, a number strictly between 0 and 1 that defaults to 0.95, as in :func:`~kivi.metrics.roc_auc_interval`, :func:`~kivi.metrics.generalized_auc` and :func:`~kivi.capital.most_prudent_pd`. The ``level`` of :func:`~kivi.capital.capital_requirement` is the confidence level of the one-factor model, 0.999 by default as the IRB approach sets it, and that of :func:`~kivi.capital.conditional_pd` the quantile of the economy, given instead of a ``factor``.
* ``bins`` defaults to 5 for the bins of a variable, as in :func:`~kivi.binning.bin_edges`, :func:`~kivi.spec.fit_spec` and :func:`~kivi.univariate.profile_table`, and to 10 for the bands of a score, as in :func:`~kivi.metrics.gains_table`, :func:`~kivi.monitoring.monitoring_table`, :func:`~kivi.reject_inference.parceling` and :func:`~kivi.reject_inference.reweighting`. The ``cutoffs`` of :func:`~kivi.strategy.cutoff_table` and :func:`~kivi.strategy.confusion_table` default to 10 bands as well.
* ``asset_class`` selects the risk-weight function of :mod:`kivi.capital`: ``"corporate"``, ``"hvcre"``, ``"mortgage"``, ``"revolving"`` or ``"other_retail"``. It is keyword-only and has no default, because the class decides the formula. ``correlation`` there is an asset correlation of at least 0 and below 1, and above 0 in :func:`~kivi.capital.systematic_factor`, and ``turnover`` and ``maturity`` are given in the units of the rules, such as millions of euros and years.
* ``event`` is a count of events per group where the rows are groups, as in :func:`~kivi.calibration.fit_grade_pd` and :func:`~kivi.capital.most_prudent_pd`, which take it with ``total``, and as the ``event`` column of the tables. In :func:`~kivi.lifetime.survival_table` it is a 0/1 flag per account instead, next to ``competing``, the flag of an exit for another reason, such as an early repayment; the two are never both 1 on a row. ``duration`` is the period in which an account defaulted, left or was last observed, a positive whole number, and ``entry`` the number of periods before it came under observation.
* ``special`` is a list of codes for a function of one variable, and a dict by column for :func:`~kivi.spec.fit_spec` and :func:`~kivi.univariate.profile_table`, which reject a list.
* Some options have similar names but different meanings. The ``min_share`` of :func:`~kivi.binning.category_groups` is the share below which a category counts as rare, while the ``min_bin_share`` of :func:`~kivi.binning.tree_edges` and :func:`~kivi.binning.ks_edges` and the ``min_group_share`` of :func:`~kivi.binning.chimerge_groups` set the smallest bin or group of the result. Only :func:`~kivi.binning.monotonic_edges` accepts ``max_bins=None``, which merges bins only until the WOE is monotonic. ``unseen`` says what to do with a group that a table lacks and defaults to ``"raise"``, while ``unseen_points`` is a number of points or None. ``method`` names the binning method in :func:`~kivi.binning.bin_edges` and :func:`~kivi.spec.fit_spec`, the correlation in :func:`~kivi.selection.correlation_filter` and the estimate of the weights in :func:`~kivi.expert.ahp_weights`.
* A reference that could be read in two ways has one argument for each. The functions of :mod:`kivi.monitoring` take either ``reference_period``, a label of the period argument, or a reference sample in ``reference_score`` or ``reference_groups``, and raise unless exactly one is given.
* The data, and the fitted results that a function applies, such as the edges of :func:`~kivi.binning.cut` or the table of :func:`~kivi.woe.woe_transform`, are positional arguments. Every option after them is keyword-only. The number of bins of :func:`~kivi.binning.bin_edges` is the one option that can also be passed by position:

>>> from kivi.binning import bin_edges
>>> bin_edges([1, 2, 3, 4], 2)
array([-inf,  2.5,  inf])
>>> bin_edges([1, 2, 3, 4], 2, method="uniform")
array([-inf,  2.5,  inf])
>>> bin_edges([1, 2, 3, 4], 2, "uniform")
Traceback (most recent call last):
    ...
TypeError: bin_edges() takes from 1 to 2 positional arguments but 3 were given

Inputs and results
------------------

* A one-dimensional input is array_like: a list, a NumPy array or a pandas Series. A function that works on several columns at once, such as :func:`~kivi.selection.stepwise` or :func:`~kivi.spec.fit_spec`, takes a DataFrame.
* The binning functions read their ``values`` with ``pandas.to_numeric``, so strings of numbers and booleans count as numbers there. Special codes must not be booleans, because ``True == 1``, and :func:`~kivi.spec.fit_spec` groups a boolean column as categorical instead of binning it.
* Dates, such as the due dates and payment dates of :func:`~kivi.vintage.delinquency_panel` and the application dates of :func:`~kivi.datasets.make_transactions`, are read with ``pandas.to_datetime``, as datetimes or as strings. Numbers raise an error instead of being read as nanoseconds since 1970, the time of day is ignored, and a date with a time zone keeps its local date. :func:`~kivi.datasets.make_transactions` also rejects periods, which ``.dt.to_timestamp()`` converts.
* The time columns of :mod:`kivi.features` are stricter: they must already hold datetimes, and strings raise an error that suggests ``pandas.to_datetime``, because parsing the strings of a large event table is slow and its format can be ambiguous. They keep the time of day. The times of the events and of the observations are both tz-naive or both in one time zone, compared by name, and any other combination raises; with a time zone, the bounds of a window are computed on the local calendar and the events compared as instants. The ``month`` of :func:`~kivi.features.panel_features` may hold monthly periods or datetimes, and months and observation times are each read on their own calendar, so tz-aware months and tz-aware observation times must share one time zone.
* A result with one value per input value is shaped like the input. A Series gives a Series with the same index and name, other array_like input gives a NumPy array, and a single number gives a float. :func:`~kivi.binning.cut` and :func:`~kivi.calibration.assign_grades` return groups, so for input other than a Series they return a ``pandas.Categorical``:

>>> import pandas as pd
>>> from kivi.calibration import prior_correction
>>> rates = dict(sample_rate=0.5, population_rate=0.02)
>>> prior_correction([0.5, 0.2], **rates).round(4)
array([0.02  , 0.0051])
>>> prior_correction(pd.Series([0.5, 0.2], index=["a", "b"], name="pd"), **rates).round(4)
a    0.0200
b    0.0051
Name: pd, dtype: float64
>>> round(prior_correction(0.5, **rates), 4)
0.02

* A Series result takes the name of the first data argument, as the results of :func:`~kivi.scorecard.pd_to_score`, :func:`~kivi.binning.cut` and :func:`~kivi.woe.woe_transform` do, so the scores of a Series named ``pd`` are named ``pd`` too. A result that is a new quantity is named after it: :func:`~kivi.woe.woe` names its result ``woe``, :func:`~kivi.strategy.expected_loss` ``expected_loss``, :func:`~kivi.lifetime.significant_increase` ``significant_increase``, :func:`~kivi.lifetime.assign_stages` ``stage``, :func:`~kivi.reject_inference.reweighting`, :func:`~kivi.expert.ahp_weights` and :func:`~kivi.expert.combine_weights` ``weight``, :func:`~kivi.capital.asset_correlation` ``correlation``, :func:`~kivi.capital.capital_requirement` ``capital_requirement``, :func:`~kivi.capital.conditional_pd` ``conditional_pd``, :func:`~kivi.capital.systematic_factor` ``factor`` and :func:`~kivi.capital.most_prudent_pd` ``most_prudent_pd``. The value-by-value functions of :mod:`kivi.capital` broadcast their inputs against each other, so one call covers a portfolio with a PD, an LGD and a maturity per exposure and one asset class.
* Some inputs must be one-dimensional even for one exposure. :func:`~kivi.lifetime.significant_increase` and :func:`~kivi.lifetime.assign_stages` raise for a single number, and :func:`~kivi.lifetime.expected_credit_loss` takes one row per account, so pass the single curve of :func:`~kivi.lifetime.convert_pd` as ``curve[None, :]``.
* Series that describe the same observations, such as the data, the target, the probabilities and the weights, must have identical indices; they are not aligned. Any other combination of these inputs is matched by position. Inputs that describe other observations, such as the two samples of :func:`~kivi.stability.psi_table`, the rejected applicants of :func:`~kivi.reject_inference.parceling` or the payments of :func:`~kivi.vintage.delinquency_panel`, can have any index. Counts per group, as in :func:`~kivi.woe.woe`, :func:`~kivi.woe.iv` and :func:`~kivi.stability.psi`, describe the same groups, so two Series of counts must have identical indices too.
* Tables are DataFrames. A table of the groups passed in, such as that of :func:`~kivi.woe.woe_table`, is indexed by the group labels, in an index named ``bin``. Other tables name the index after their rows, such as ``variable``, ``grade``, ``cutoff``, ``threshold``, ``statistic``, ``step``, ``period``, ``cohort``, ``state``, ``stage`` or ``before``, and name their columns when these are labels too, such as ``horizon``, ``after`` or ``month_on_book``. :doc:`compatibility` lists the index and the columns of every table. A total is the sum of a column, such as ``woe_table(groups, target)["iv"].sum()``.
* The features of :mod:`kivi.features` keep the index of ``observations``, in its order, so that they join back to the observations. Their names follow from the arguments: the column names and the ``prefix`` are kept as given, and only kivi's own parts of the names, the aggregations and the windows, are in lower case, so ``Amount`` and ``amount`` give ``Amount_sum_3m`` and ``amount_sum_3m``.
* Named statistics come back as a Series or a DataFrame with an index named ``statistic``, as from :func:`~kivi.metrics.roc_auc_interval`, :func:`~kivi.metrics.generalized_auc`, :func:`~kivi.calibration.brier_decomposition`, :func:`~kivi.calibration.concentration_test`, :func:`~kivi.vintage.migration_bandwidth` and :func:`~kivi.expert.ahp_consistency`, or as a Series indexed by the names alone, as from :func:`~kivi.metrics.delong_test`. Parameters that another function applies come back as a tuple, such as the edges and the probabilities that :func:`~kivi.calibration.isotonic_fit` returns for :func:`~kivi.calibration.isotonic_predict` and the long-run PD and the correlation that :func:`~kivi.capital.vasicek_fit` returns for :func:`~kivi.capital.conditional_pd`, and so does the statistic and p-value pair of :func:`~kivi.calibration.hosmer_lemeshow` and :func:`~kivi.calibration.spiegelhalter_test`.
* A chart of :mod:`kivi.plotting` takes the table that a kivi function returned first, and the Axes to draw on as the keyword-only ``ax``; without ``ax`` it draws on a new figure. It returns the Axes, never shows the figure and leaves matplotlib's global settings alone. The text it writes is only column names, argument names and values, so one image serves every language.
* Functions do not modify their inputs. Random draws come from a local generator seeded by ``rng``, so they never change NumPy's global state, and the same seed gives the same result.
* ``rng`` takes what ``numpy.random.default_rng`` takes, such as None, an integer, a ``SeedSequence`` or a ``Generator``. A ``Generator`` is used and advanced rather than copied, so two calls with the same ``Generator`` give different draws, while two calls with the same integer agree.

.. _conventions-scores:

Targets, scores and probabilities
---------------------------------

* A target is 1 for an event, such as a default, and 0 for a non-event. Any other label, and a missing label, raises an error. :func:`~kivi.strategy.swap_set` and :func:`~kivi.monitoring.monitoring_table` are the exceptions: they read a missing label as an outcome that is not known, such as that of a declined applicant or of a loan whose outcome window has not ended.
* The ``bad`` argument of :func:`~kivi.vintage.vintage_table` is the status of a loan in a month, such as being 90 days past due, rather than the target of an observation. It takes booleans or 0 and 1, without missing values. The ``bad`` of :func:`~kivi.features.panel_features` is the first bad category of an ordered status column instead, ``"30-59"`` by default: it and every worse status count as bad. The status column must be an ordered categorical, from the best status to the worst, so that labels such as ``current`` are never sorted alphabetically.
* In :mod:`kivi.metrics`, :mod:`kivi.monitoring` and :mod:`kivi.strategy`, a score rises with the event probability. Pass a PD as it is, and negate scorecard points, which fall as the risk rises. A cut-off of :mod:`kivi.strategy` approves the scores at or below it.
* In :mod:`kivi.scorecard`, a ``score`` is scorecard points, which fall as the risk rises: :func:`~kivi.scorecard.score_to_pd` gives a lower PD for a higher score.
* :func:`~kivi.calibration.platt_fit` accepts a score in either direction and fits a negative slope to one that falls as the risk rises. :func:`~kivi.calibration.isotonic_fit` assumes a rising score unless ``increasing=False``.
* True and False count as 1 and 0 in targets, scores and probabilities, and in the aggregations of :func:`~kivi.features.window_features`, where the ``sum`` of a flag counts the events that have it. Weights, LGDs, exposures, losses, the ``default_rate`` of :func:`~kivi.capital.systematic_factor` and :func:`~kivi.capital.vasicek_fit`, the amounts of :func:`~kivi.vintage.delinquency_panel`, the ``realized`` and ``estimated`` values of :func:`~kivi.calibration.backtest_table` and :func:`~kivi.metrics.generalized_auc` and the entries of a comparison matrix reject them, and so do most numeric options, such as ``bins``, ``max_bins``, ``factor`` and the cut-offs. Six inputs reject them because a boolean there is more likely a mistake: the ``reference_score`` of :func:`~kivi.monitoring.monitoring_table`, which would usually be a row mask, the ``matrix`` of :func:`~kivi.lifetime.transition_pd`, the ``stage`` of :func:`~kivi.lifetime.stage_table`, the ``days_past_due`` of :func:`~kivi.lifetime.assign_stages`, and the ``duration`` and ``entry`` of :func:`~kivi.lifetime.survival_table`:

>>> import numpy as np
>>> from kivi.metrics import roc_auc
>>> roc_auc([0.1, 0.9, 0.4], [False, True, True])
1.0
>>> from kivi.strategy import expected_loss
>>> expected_loss(np.array([0.1, 0.2]), np.array([True, False]), 100.0)
Traceback (most recent call last):
    ...
ValueError: lgd must be real and numeric, got dtype bool

Groups and missing values
-------------------------

The functions that count groups, such as :func:`~kivi.woe.woe_table`, :func:`~kivi.stability.psi_table` and :func:`~kivi.calibration.calibration_table`, take the labels as they are: the intervals of ``pandas.cut`` or :func:`kivi.binning.cut`, strings, numbers or categories.

* Missing values form a group of their own.
* Only observed groups appear. Unused categories are left out, and so are the groups whose weights sum to zero. Tables with a fixed set of rows are the exception and keep their empty rows: :func:`~kivi.lifetime.stage_table` always has stages 1, 2 and 3, :func:`~kivi.strategy.swap_set` always has its four rows, and the period tables of :mod:`kivi.monitoring` keep a row for every category of a categorical period, so that the rows of all reports line up.
* Categorical groups, including the intervals of ``pandas.cut``, follow the order of their categories, with the missing group last. Other labels follow the order in which they first appear.
* Two samples, as in :func:`~kivi.stability.psi_table`, follow the order of the categories only when both have the same categorical dtype. Otherwise, as for an ordered and an unordered categorical, their groups follow the order in which they first appear, in the reference sample and then in the current one.

>>> from kivi.woe import woe_table
>>> woe_table(["b", "a", None, "a"], [1, 0, 1, 0])[["non_event", "event"]]
     non_event  event
bin
b            0      1
a            2      0
NaN          0      1

:func:`~kivi.binning.cut` returns the ordered categorical of ``pandas.cut`` when there are no special codes, and an unordered one with ``special``, because the special codes come first and have no order with the intervals.

Categories are matched to a fitted table or grouping by equality, and a category that it does not know raises an error unless ``unseen`` says how to treat it. :func:`~kivi.woe.woe_transform`, :func:`~kivi.binning.group_categories` and :func:`~kivi.stability.csi_table` share this rule, the last through ``unseen_points``. :func:`~kivi.scorecard.reason_codes` matches points rather than categories, and its ``unseen_points`` exempts the points that ``woe_transform`` gave unseen groups.

The functions that fit edges or groupings, such as :func:`~kivi.binning.bin_edges`, :func:`~kivi.binning.chimerge_edges` and :func:`~kivi.binning.category_groups`, leave missing values out of the fit, and :func:`~kivi.binning.cut` and :func:`~kivi.binning.group_categories` keep them missing, so that they form a group of their own when counted.

The functions that count, rank or fit, such as :func:`~kivi.metrics.roc_auc`, :func:`~kivi.calibration.calibration_table`, :func:`~kivi.strategy.cutoff_table` and :func:`~kivi.calibration.platt_fit`, raise an error for a missing score or probability instead, so decide how missing values are scored before passing them in. The functions that convert each value on its own, such as :func:`~kivi.scorecard.pd_to_score`, :func:`~kivi.calibration.platt_predict` and :func:`~kivi.strategy.expected_loss`, return NaN for a missing value, and :func:`~kivi.calibration.assign_grades` returns a missing grade. The staging functions of :mod:`kivi.lifetime` work value by value but raise for a missing value, because a missing PD or number of days past due must not decide a stage, and the functions of :mod:`kivi.monitoring` raise for a missing period. The value-by-value functions of :mod:`kivi.capital` return NaN for a missing PD, LGD, default rate, factor or level, but raise for a missing ``turnover``, ``maturity`` or ``correlation``, which decides the formula. :func:`~kivi.lifetime.survival_table`, :func:`~kivi.capital.vasicek_fit` and :func:`~kivi.capital.most_prudent_pd` count or fit their inputs as a whole and raise for a missing value. :func:`~kivi.vintage.delinquency_panel` raises for a missing loan, date or amount, and the functions of :mod:`kivi.expert` for a missing comparison or weight. The validation tests raise for a missing value as well: :func:`~kivi.calibration.backtest_table` for a missing realized or estimated value, :func:`~kivi.metrics.generalized_auc` for a missing segment, :func:`~kivi.calibration.concentration_test` for a missing count, and :func:`~kivi.vintage.migration_bandwidth` and :func:`~kivi.vintage.migration_test` for a missing grade or status, so give the customers who left the grades a status of their own. The functions of :mod:`kivi.features` raise for a missing entity, time, month or status, and leave the missing values of an aggregated column out of its aggregations, while the ``events`` features count every event.

Sample weights
--------------

Most functions that count observations or fit a model take ``weights``: one finite, non-negative number per observation, matched to the data like the target. :func:`~kivi.stability.psi_table` and :func:`~kivi.stability.csi_table` take one set for each sample, ``reference_weights`` and ``current_weights``.

``weights`` belongs to the main data, and the weights of another sample are named after it, as the ``reference_weights`` of :func:`~kivi.monitoring.monitoring_table` and :func:`~kivi.monitoring.stability_table`. :func:`~kivi.selection.vif`, :func:`~kivi.selection.vif_filter`, :func:`~kivi.selection.correlation_filter`, :func:`~kivi.reject_inference.parceling` and :func:`~kivi.reject_inference.reweighting` take no weights and give every row the same weight, and so do the functions of :mod:`kivi.features`, which count each event once. :func:`~kivi.calibration.concentration_test` takes the counts or the exposures of the grades instead of rows, and so do :func:`~kivi.capital.most_prudent_pd`, which takes the obligors and defaults of each grade, and :func:`~kivi.capital.vasicek_fit`, which takes one default rate per period. The value-by-value functions of :mod:`kivi.capital` work on single exposures and take no weights. The validation sample of :func:`~kivi.selection.selection_path` counts each row once.

* Counts become sums of weights, and observations with zero weight are left out. Integer weights therefore give the same counts as repeating each row that many times.
* Weighted quantiles use the inverted CDF, so every interior quantile edge is an observed value, while unweighted quantiles interpolate between observations. Weighted edges therefore agree with repeated rows only where no interpolation is needed, and weights of 1 do not give the same edges as no weights.
* ``smoothing`` replaces an empty cell in units of weight and is not rescaled with the weights.
* :func:`~kivi.selection.logistic_regression`, :func:`~kivi.selection.stepwise`, :func:`~kivi.selection.selection_path`, :func:`~kivi.metrics.roc_auc_interval`, :func:`~kivi.metrics.delong_test`, :func:`~kivi.metrics.generalized_auc`, :func:`~kivi.calibration.spiegelhalter_test`, :func:`~kivi.calibration.backtest_table` and :func:`~kivi.vintage.migration_test` read the weights as frequency weights, so their standard errors count the total weight as the number of observations. Weights that correct for sampling, such as inverse sampling rates, make these standard errors too small.

Periods and thresholds
----------------------

* Periods are ordered as their categories when they are categorical and sorted otherwise. Period 1 of a term structure or an expected credit loss is the first period after its starting point, such as the reporting date, and :func:`~kivi.lifetime.expected_credit_loss` discounts a default in period :math:`t` over :math:`t` periods. In :func:`~kivi.lifetime.survival_table`, an account with the ``entry`` :math:`e` and the ``duration`` :math:`T` is at risk in the periods :math:`e < t \le T`, and an account censored in a period still counts as at risk in it, because its censoring comes after the defaults and exits of the period.
* A threshold is compared as the rule it implements is worded. A cut-off approves the scores at or below it. :func:`~kivi.lifetime.significant_increase` flags a PD increase only when it exceeds its thresholds, and :func:`~kivi.lifetime.assign_stages` moves a loan to stage 2 when it is more than ``stage_2_days`` past due and to stage 3 when it is at least ``stage_3_days`` past due, as IFRS 9 words its 30-day and 90-day presumptions.
* The ``end`` of :func:`~kivi.vintage.delinquency_panel` is the last month observed, given as the month, a date in it or a longer period that ends with it, and every month is observed on its last day, so a payment made on that day counts. An instalment is not past due on its due date, and the status ``90+`` starts at 90 days past due, as stage 3 does with the ``stage_3_days`` of :func:`~kivi.lifetime.assign_stages`.

Time windows
------------

* A feature of :mod:`kivi.features` uses only what is known at the observation time :math:`t`, so that it is *point in time*. A window of :func:`~kivi.features.window_features` is :math:`[t - k, t)` by default, so an event at :math:`t` or later counts in no window; ``closed="right"`` gives :math:`(t - k, t]` for events that are known at their own time, such as balances at the end of a day, which only the caller can judge. A window of :func:`~kivi.features.panel_features` holds the :math:`k` months before the month of :math:`t`, because the status at the end of that month is not known at :math:`t`.
* ``"3M"`` counts calendar months, as ``pandas.DateOffset(months=3)`` does, so one month before 31 March is 29 February in a leap year, and ``"90D"`` counts days. The bounds are written in upper case, and :func:`~kivi.features.panel_features` takes months only. In the column names the windows are in lower case, such as ``3m`` and ``90d``.
* ``windows`` is a list, also for one window, as in ``windows=["3M"]``. A pair of bounds with the same unit, the first further back, is the earlier period between them, as ``("6M", "3M")`` is :math:`[t - 6M, t - 3M)`, named ``6m_3m``. A pair is one window only as an item of the list, as in ``windows=[("6M", "3M")]``: ``windows=("6M", "3M")`` is the two windows ``6m`` and ``3m``.

Errors and warnings
-------------------

* Invalid input raises ``ValueError``, and so does a model that cannot be fitted, such as one with separated classes, which raises :exc:`kivi.selection.SeparationError`. The message names the argument and, where a single value is at fault, shows the value that was received:

>>> bin_edges([1, 2, 3], bins=0)
Traceback (most recent call last):
    ...
ValueError: bins must be a positive integer, got 0
>>> roc_auc([0.1, 0.9, 0.4], [0, 1, 2])
Traceback (most recent call last):
    ...
ValueError: target must contain only 0 and 1, got 2

* A ``UserWarning`` means that the result differs from the request, for example fewer bins than requested because of tied values, and says what to change. :func:`kivi.univariate.profile_table` records such ties in a column instead, so that profiling many columns raises no warnings.
* A ``RuntimeWarning`` means that a fit stopped before it converged.
* A ``FutureWarning`` announces a change of the public API. :doc:`compatibility` says when the change takes effect.
* The wording of the messages may change in any release, so a test that matches a message may need an update; the types of the errors and warnings stay.
