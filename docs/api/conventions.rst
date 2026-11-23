Conventions
===========

Every public function of kivi follows the rules on this page. The module pages and the guides take them for granted, and the docstrings mention only where a function departs from them or adds to them.

Imports
-------

kivi has no top-level imports. Import each function from its module, as in ``from kivi.woe import woe_table``; each module has its own page in this reference.

Only the names that a module lists in ``__all__`` are public. :doc:`compatibility` says which of them a release may change, and how it announces the change.

Arguments
---------

* The data come first and the target follows them, as in ``woe_table(groups, target)``, ``roc_auc(score, target)``, ``stepwise(X, target)`` and ``fit_spec(frame, target)``. A function with several data arguments takes all of them before the target, as ``swap_set(current, new, target)`` does, and a predicted probability comes after the target, as in ``calibration_table(groups, target, probability)``. scikit-learn's metrics take the target first, so check the order when porting code from them.
* Data that describe other observations, such as the rejected applicants, follow the data and the target of the main sample, as in ``parceling(score, target, reject_score)`` and in ``reweighting(score, reject_score)``, which has no target.
* ``X`` is a numeric model matrix, as in :func:`~kivi.selection.stepwise`, :func:`~kivi.selection.logistic_regression`, :func:`~kivi.selection.vif` and :func:`~kivi.selection.correlation_filter`. ``frame`` holds raw columns, as in :func:`~kivi.spec.fit_spec`, :func:`~kivi.univariate.profile_table` and :meth:`BinningSpec.cut <kivi.spec.BinningSpec.cut>`, and ``woe`` is the matrix of WOE values that :func:`~kivi.scorecard.woe_to_points` turns into points.
* A predicted probability of an event, such as a PD, is always called ``probability``, as in ``brier_score(probability, target)``, ``calibration_table(groups, target, probability)`` and ``expected_loss(probability, lgd, ead)``. What a ``score`` is depends on the module; see :ref:`conventions-scores`. A second probability is named after its meaning, as ``origination_probability`` in :func:`~kivi.lifetime.significant_increase`.
* An option with the same meaning has the same name in every module, such as ``bins``, ``special``, ``smoothing``, ``unseen`` and ``weights``. ``level`` is the confidence level of an interval, a number strictly between 0 and 1 that defaults to 0.95, as in :func:`~kivi.metrics.roc_auc_interval`.
* ``bins`` defaults to 5 for the bins of a variable, as in :func:`~kivi.binning.bin_edges`, :func:`~kivi.spec.fit_spec` and :func:`~kivi.univariate.profile_table`, and to 10 for the bands of a score, as in :func:`~kivi.metrics.gains_table`, :func:`~kivi.monitoring.monitoring_table`, :func:`~kivi.reject_inference.parceling` and :func:`~kivi.reject_inference.reweighting`. The ``cutoffs`` of :func:`~kivi.strategy.cutoff_table` default to 10 bands as well.
* ``special`` is a list of codes for a function of one variable, and a dict by column for :func:`~kivi.spec.fit_spec` and :func:`~kivi.univariate.profile_table`, which reject a list.
* Some options have similar names but different meanings. The ``min_share`` of :func:`~kivi.binning.category_groups` is the share below which a category counts as rare, while the ``min_bin_share`` of :func:`~kivi.binning.tree_edges` and the ``min_group_share`` of :func:`~kivi.binning.chimerge_groups` set the smallest bin or group of the result. Only :func:`~kivi.binning.monotonic_edges` accepts ``max_bins=None``, which merges bins only until the WOE is monotonic. ``unseen`` says what to do with a group that a table lacks and defaults to ``"raise"``, while ``unseen_points`` is a number of points or None.
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

* A Series result takes the name of the first data argument, as the results of :func:`~kivi.scorecard.pd_to_score`, :func:`~kivi.binning.cut` and :func:`~kivi.woe.woe_transform` do, so the scores of a Series named ``pd`` are named ``pd`` too. A result that is a new quantity is named after it: :func:`~kivi.woe.woe` names its result ``woe``, :func:`~kivi.strategy.expected_loss` ``expected_loss``, :func:`~kivi.lifetime.significant_increase` ``significant_increase``, :func:`~kivi.lifetime.assign_stages` ``stage`` and :func:`~kivi.reject_inference.reweighting` ``weight``.
* Some inputs must be one-dimensional even for one exposure. :func:`~kivi.lifetime.significant_increase` and :func:`~kivi.lifetime.assign_stages` raise for a single number, and :func:`~kivi.lifetime.expected_credit_loss` takes one row per account, so pass the single curve of :func:`~kivi.lifetime.convert_pd` as ``curve[None, :]``.
* Series that describe the same observations, such as the data, the target, the probabilities and the weights, must have identical indices; they are not aligned. Any other combination of these inputs is matched by position. Inputs that describe other observations, such as the two samples of :func:`~kivi.stability.psi_table` or the rejected applicants of :func:`~kivi.reject_inference.parceling`, can have any index. Counts per group, as in :func:`~kivi.woe.woe`, :func:`~kivi.woe.iv` and :func:`~kivi.stability.psi`, describe the same groups, so two Series of counts must have identical indices too.
* Tables are DataFrames. A table of the groups passed in, such as that of :func:`~kivi.woe.woe_table`, is indexed by the group labels, in an index named ``bin``. Other tables name the index after their rows, such as ``variable``, ``grade``, ``cutoff``, ``threshold``, ``statistic``, ``period``, ``cohort``, ``state``, ``stage`` or ``before``, and name their columns when these are labels too, such as ``horizon``, ``after`` or ``month_on_book``. :doc:`compatibility` lists the index and the columns of every table. A total is the sum of a column, such as ``woe_table(groups, target)["iv"].sum()``.
* Named statistics come back as a Series or a DataFrame, as from :func:`~kivi.metrics.delong_test` and :func:`~kivi.metrics.roc_auc_interval`. Parameters that another function applies come back as a tuple, such as the edges and the probabilities that :func:`~kivi.calibration.isotonic_fit` returns for :func:`~kivi.calibration.isotonic_predict`, and so does the statistic and p-value pair of :func:`~kivi.calibration.hosmer_lemeshow`.
* Functions do not modify their inputs. Random draws come from a local generator seeded by ``rng``, so they never change NumPy's global state, and the same seed gives the same result.
* ``rng`` takes what ``numpy.random.default_rng`` takes, such as None, an integer, a ``SeedSequence`` or a ``Generator``. A ``Generator`` is used and advanced rather than copied, so two calls with the same ``Generator`` give different draws, while two calls with the same integer agree.

.. _conventions-scores:

Targets, scores and probabilities
---------------------------------

* A target is 1 for an event, such as a default, and 0 for a non-event. Any other label, and a missing label, raises an error. :func:`~kivi.strategy.swap_set` and :func:`~kivi.monitoring.monitoring_table` are the exceptions: they read a missing label as an outcome that is not known, such as that of a declined applicant or of a loan whose outcome window has not ended.
* The ``bad`` argument of :func:`~kivi.vintage.vintage_table` is the status of a loan in a month, such as being 90 days past due, rather than the target of an observation. It takes booleans or 0 and 1, without missing values.
* In :mod:`kivi.metrics`, :mod:`kivi.monitoring` and :mod:`kivi.strategy`, a score rises with the event probability. Pass a PD as it is, and negate scorecard points, which fall as the risk rises. A cut-off of :mod:`kivi.strategy` approves the scores at or below it.
* In :mod:`kivi.scorecard`, a ``score`` is scorecard points, which fall as the risk rises: :func:`~kivi.scorecard.score_to_pd` gives a lower PD for a higher score.
* :func:`~kivi.calibration.platt_fit` accepts a score in either direction and fits a negative slope to one that falls as the risk rises. :func:`~kivi.calibration.isotonic_fit` assumes a rising score unless ``increasing=False``.
* True and False count as 1 and 0 in targets, scores and probabilities. Weights, LGDs, exposures and losses reject them, and so do most numeric options, such as ``bins``, ``max_bins``, ``factor`` and the cut-offs. Four inputs reject them because a boolean there is more likely a mistake: the ``reference_score`` of :func:`~kivi.monitoring.monitoring_table`, which would usually be a row mask, the ``matrix`` of :func:`~kivi.lifetime.transition_pd`, the ``stage`` of :func:`~kivi.lifetime.stage_table` and the ``days_past_due`` of :func:`~kivi.lifetime.assign_stages`:

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

The functions that count, rank or fit, such as :func:`~kivi.metrics.roc_auc`, :func:`~kivi.calibration.calibration_table`, :func:`~kivi.strategy.cutoff_table` and :func:`~kivi.calibration.platt_fit`, raise an error for a missing score or probability instead, so decide how missing values are scored before passing them in. The functions that convert each value on its own, such as :func:`~kivi.scorecard.pd_to_score`, :func:`~kivi.calibration.platt_predict` and :func:`~kivi.strategy.expected_loss`, return NaN for a missing value, and :func:`~kivi.calibration.assign_grades` returns a missing grade. The staging functions of :mod:`kivi.lifetime` work value by value but raise for a missing value, because a missing PD or number of days past due must not decide a stage, and the functions of :mod:`kivi.monitoring` raise for a missing period.

Sample weights
--------------

Most functions that count observations or fit a model take ``weights``: one finite, non-negative number per observation, matched to the data like the target. :func:`~kivi.stability.psi_table` and :func:`~kivi.stability.csi_table` take one set for each sample, ``reference_weights`` and ``current_weights``.

``weights`` belongs to the main data, and the weights of another sample are named after it, as the ``reference_weights`` of :func:`~kivi.monitoring.monitoring_table` and :func:`~kivi.monitoring.stability_table`. :func:`~kivi.selection.vif`, :func:`~kivi.selection.correlation_filter`, :func:`~kivi.reject_inference.parceling` and :func:`~kivi.reject_inference.reweighting` take no weights and give every row the same weight.

* Counts become sums of weights, and observations with zero weight are left out. Integer weights therefore give the same counts as repeating each row that many times.
* Weighted quantiles use the inverted CDF, so every interior quantile edge is an observed value, while unweighted quantiles interpolate between observations. Weighted edges therefore agree with repeated rows only where no interpolation is needed, and weights of 1 do not give the same edges as no weights.
* ``smoothing`` replaces an empty cell in units of weight and is not rescaled with the weights.
* :func:`~kivi.selection.logistic_regression`, :func:`~kivi.selection.stepwise`, :func:`~kivi.metrics.roc_auc_interval` and :func:`~kivi.metrics.delong_test` read the weights as frequency weights, so their standard errors count the total weight as the number of observations. Weights that correct for sampling, such as inverse sampling rates, make these standard errors too small.

Periods and thresholds
----------------------

* Periods are ordered as their categories when they are categorical and sorted otherwise. Period 1 of a term structure or an expected credit loss is the first period after its starting point, such as the reporting date, and :func:`~kivi.lifetime.expected_credit_loss` discounts a default in period :math:`t` over :math:`t` periods.
* A threshold is compared as the rule it implements is worded. A cut-off approves the scores at or below it. :func:`~kivi.lifetime.significant_increase` flags a PD increase only when it exceeds its thresholds, and :func:`~kivi.lifetime.assign_stages` moves a loan to stage 2 when it is more than ``stage_2_days`` past due and to stage 3 when it is at least ``stage_3_days`` past due, as IFRS 9 words its 30-day and 90-day presumptions.

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
