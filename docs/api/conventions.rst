Conventions
===========

Every public function of kivi follows the rules on this page. The module pages and the guides take them for granted, and the docstrings mention only where a function departs from them or adds to them.

Imports
-------

kivi has no top-level imports. Import each function from its module, as in ``from kivi.woe import woe_table``; each module has its own page in this reference.

Arguments
---------

* The data come first and the target second, as in ``woe_table(groups, target)``, ``roc_auc(score, target)``, ``stepwise(X, target)`` and ``fit_spec(frame, target)``. scikit-learn's metrics take the target first, so check the order when porting code from them.
* A predicted probability of an event, such as a PD, is always called ``probability``, as in ``brier_score(probability, target)``, ``calibration_table(groups, target, probability)`` and ``expected_loss(probability, lgd, ead)``. A ``score`` is any number that rises with the risk, and a PD is one of them.
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
* A result with one value per input value is shaped like the input. A Series gives a Series with the same index and name, other array_like input gives a NumPy array, and a single number gives a float:

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

* Two Series that are passed together, such as the groups and the target, must have identical indices; they are not aligned. Any other combination of inputs is matched by position.
* Tables are DataFrames. A table with one row per group is indexed by the group labels, in an index named ``bin``. A total is the sum of a column, such as ``woe_table(groups, target)["iv"].sum()``.
* Functions do not modify their inputs. Random draws come from a local generator seeded by ``rng``, so they never change NumPy's global state, and the same seed gives the same result.

Targets, scores and probabilities
---------------------------------

* A target is 1 for an event, such as a default, and 0 for a non-event. Any other label, and a missing label, raises an error. :func:`~kivi.strategy.swap_set` is the exception: it reads a missing label as an outcome that is not known.
* A score rises with the event probability. Pass a PD as it is, and negate scorecard points, which fall as the risk rises. A cut-off approves the scores at or below it.
* True and False count as 1 and 0 in targets, scores and probabilities. Weights, LGDs, exposures, losses and numeric options such as ``bins`` reject them:

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
* Only observed groups appear. Unused categories are left out, and so are the groups whose weights sum to zero.
* Categorical groups, including the intervals of ``pandas.cut``, follow the order of their categories, with the missing group last. Other labels follow the order in which they first appear.

>>> from kivi.woe import woe_table
>>> woe_table(["b", "a", None, "a"], [1, 0, 1, 0])[["non_event", "event"]]
     non_event  event
bin
b            0      1
a            2      0
NaN          0      1

A missing score or probability raises an error instead, so decide how missing values are scored before passing them in.

Sample weights
--------------

Most functions that count observations or fit a model take ``weights``: one finite, non-negative number per observation, matched to the data like the target. :func:`~kivi.stability.psi_table` and :func:`~kivi.stability.csi_table` take one set for each sample, ``reference_weights`` and ``current_weights``.

* Counts become sums of weights, and observations with zero weight are left out. Integer weights therefore give the same counts as repeating each row that many times.
* Weighted quantiles use the inverted CDF, so every interior quantile edge is an observed value, while unweighted quantiles interpolate between observations. Weighted edges therefore agree with repeated rows only where no interpolation is needed, and weights of 1 do not give the same edges as no weights.
* ``smoothing`` replaces an empty cell in units of weight and is not rescaled with the weights.
* :func:`~kivi.selection.logistic_regression` and :func:`~kivi.selection.stepwise` read the weights as frequency weights, so their standard errors count the total weight as the number of observations.

Errors and warnings
-------------------

* Invalid input raises ``ValueError``, and so does a model that cannot be fitted, such as one with separated classes, which raises :exc:`kivi.selection.SeparationError`. The message names the argument and shows the value that was received:

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
* A ``FutureWarning`` announces a change of the public API one minor version before it takes effect.
