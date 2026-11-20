Release notes
=============

Unreleased
----------

Changes merged since the last release.

.. release-fragments::

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
