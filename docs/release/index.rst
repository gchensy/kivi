Release notes
=============

Unreleased
----------

Changes merged since the last release.

.. release-fragments::

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
