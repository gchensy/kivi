Compatibility
=============

This page states what a release of kivi may change and what it keeps: the versions it supports, the public names, the layout of the tables it returns, its exception types and the files of :class:`~kivi.spec.BinningSpec` and :class:`~kivi.features.FeatureSpec`. The rules that every function follows, such as the order of the arguments, are on :doc:`conventions`, and they are part of the API too.

Releases and deprecations
-------------------------

From version 1.0, kivi follows semantic versioning:

* A major release, such as 2.0, may remove or change any part of the public API.
* A minor release, such as 1.3, adds functions, arguments, table columns or format versions, and may deprecate a part of the API.
* A patch release, such as 1.3.1, only fixes errors.
* A deprecated function, argument or behavior keeps working, with a ``FutureWarning`` that says what to use instead, for the rest of the major version. It changes in the next major release, so a deprecation of 1.4 takes effect in 2.0. The release notes list each deprecation under Changed.
* A fix of a wrong result can come in any release, minor or patch, even when code relies on the wrong result. The release notes list such fixes under Fixed.

Before 1.0 a shorter rule applied: what 0.9 deprecated changed in 1.0, as :doc:`/release/migration` lists. To find the calls that a release will change, run the code or its tests with ``FutureWarning`` turned into an error, as ``python -W error::FutureWarning`` or ``pytest -W error::FutureWarning`` do.

Supported versions
------------------

kivi supports Python 3.10 or newer, NumPy 1.24 or newer and pandas 2.0 or newer, and its tests run on these oldest versions and on the newest releases. The charts of :mod:`kivi.plotting` also need matplotlib 3.7 or newer, which ``pip install "kivi[plot]"`` installs; no other module imports it. From 1.0, a minimum version rises only in a minor release, and the notes of the release before it announce the change. A patch release never raises a minimum.

What is public
--------------

* The names that a public module lists in its ``__all__``: :mod:`kivi.features`, :mod:`kivi.binning`, :mod:`kivi.woe`, :mod:`kivi.stability`, :mod:`kivi.univariate`, :mod:`kivi.selection`, :mod:`kivi.scorecard`, :mod:`kivi.expert`, :mod:`kivi.metrics`, :mod:`kivi.calibration`, :mod:`kivi.strategy`, :mod:`kivi.reject_inference`, :mod:`kivi.spec`, :mod:`kivi.monitoring`, :mod:`kivi.vintage`, :mod:`kivi.lifetime`, :mod:`kivi.plotting` and :mod:`kivi.datasets`. Each has a page in this reference that documents every such name. ``kivi.__version__`` is public too.
* The documented members of :class:`~kivi.spec.BinningSpec`: its fields, ``FORMAT_VERSION``, ``format_version`` and ``columns``, its methods, and its equality, hashing, pickling and ``dataclasses.replace``. The same holds for :class:`~kivi.features.FeatureSpec`, which has ``FORMAT_VERSION`` and ``columns`` but no ``format_version``, because its format has one version so far. :exc:`~kivi.selection.SeparationError`, which :func:`~kivi.calibration.platt_fit` raises as well, stays in :mod:`kivi.selection`.
* The names of the arguments. A keyword is part of a call, so renaming an argument needs a deprecation, even one that is usually passed by position.
* Everything else is private: the modules and names that start with an underscore, and the modules that a public module imports, such as ``kivi.binning.np``. They can change in any release.

Results that are pairs of parameters, such as the ``(factor, offset)`` of :func:`~kivi.scorecard.score_scaling`, the ``(slope, intercept)`` of :func:`~kivi.calibration.platt_fit`, the ``(edges, probability)`` of :func:`~kivi.calibration.isotonic_fit` and the ``(statistic, p_value)`` of :func:`~kivi.calibration.hosmer_lemeshow` and :func:`~kivi.calibration.spiegelhalter_test`, are tuples. A minor release may turn them into named tuples, which unpack in the same way.

Tables
------

The index name, the column names and the order of the columns of every table are public. A minor release may append columns at the end of a table, but renames, reorders or removes a column only after a deprecation, so select columns by name rather than by position. The ``attrs`` of :func:`~kivi.selection.logistic_regression` are set on the returned frame, and pandas drops them in many operations.

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Function
     - Index
     - Columns
   * - :func:`~kivi.features.window_features`
     - The index of ``observations``, in its order
     - For each window in turn ``{prefix}events_{window}``, then ``{prefix}{column}_{aggregation}_{window}`` column by column and aggregation by aggregation, and the ratios last, ``{numerator}_over_{window}`` or ``{numerator}_over_{denominator}``
   * - :func:`~kivi.features.panel_features`
     - The index of ``observations``, in its order
     - For each window in turn ``{prefix}months_{window}``, ``{prefix}bad_months_{window}``, ``{prefix}longest_bad_run_{window}``, ``{prefix}longest_good_run_{window}``, ``{prefix}worst_status_{window}``, then ``{prefix}months_since_bad``
   * - :meth:`FeatureSpec.compute <kivi.features.FeatureSpec.compute>`
     - The index of ``observations``, in its order
     - The columns of each group of ``events``, then of each group of ``panel``, as the two functions above give them, and the ratios last; ``FeatureSpec.columns`` lists them
   * - :func:`~kivi.woe.woe_table`, ``BinningSpec.tables``
     - ``bin``
     - ``non_event``, ``event``, ``total``, ``non_event_share``, ``event_share``, ``total_share``, ``event_rate``, ``woe``, ``iv``
   * - :func:`~kivi.woe.woe` of Series counts
     - A Series named ``woe`` with the index of the counts
     -
   * - :func:`~kivi.stability.psi_table`
     - ``bin``
     - ``reference``, ``current``, ``reference_share``, ``current_share``, ``psi``
   * - :func:`~kivi.stability.csi_table`
     - ``bin``
     - ``reference``, ``current``, ``reference_share``, ``current_share``, ``points``, ``csi``
   * - :func:`~kivi.calibration.calibration_table`
     - ``bin``
     - ``total``, ``event``, ``event_rate``, ``mean_probability``
   * - :func:`~kivi.calibration.backtest_table`
     - ``bin``
     - ``total``, ``estimated``, ``realized``, ``difference``, ``variance``, ``statistic``, ``p_value``
   * - :func:`~kivi.calibration.brier_decomposition`
     - A Series with the index ``statistic``: ``brier``, ``reliability``, ``resolution``, ``uncertainty``, ``within_variance``, ``within_covariance``
     -
   * - :func:`~kivi.calibration.concentration_test`
     - A Series with the index ``statistic``: ``hi``, ``cv``, ``initial_hi``, ``initial_cv``, ``statistic``, ``p_value``
     -
   * - :func:`~kivi.strategy.expected_loss_table`
     - ``bin``
     - ``total``, ``exposure``, ``exposure_weighted_probability``, ``expected_loss``, ``expected_loss_rate``
   * - :func:`~kivi.metrics.gains_table`
     - ``bin``, the score bands as intervals
     - ``non_event``, ``event``, ``total``, ``event_rate``, ``lift``, ``cum_non_event_share``, ``cum_event_share``, ``cum_total_share``, ``ks``, ``cum_lift``
   * - :func:`~kivi.metrics.roc_table`
     - ``threshold``
     - ``cum_non_event``, ``cum_event``, ``false_positive_rate``, ``true_positive_rate``, ``precision``
   * - :func:`~kivi.metrics.roc_auc_interval`
     - ``statistic``: ``auc`` and ``gini``
     - ``estimate``, ``se``, ``lower``, ``upper``
   * - :func:`~kivi.metrics.delong_test`
     - A Series with the index ``current_auc``, ``new_auc``, ``difference``, ``se``, ``lower``, ``upper``, ``z``, ``p_value``
     -
   * - :func:`~kivi.metrics.generalized_auc`
     - ``statistic``: ``gauc`` and ``somers_d``
     - ``estimate``, ``se``, ``lower``, ``upper``
   * - :func:`~kivi.univariate.profile_table`
     - ``variable``
     - ``kind``, ``count``, ``missing``, ``missing_share``, ``distinct``, ``top``, ``top_share``, ``special``, ``special_share``, ``mean``, ``std``, ``min``, ``q01``, ``q50``, ``q99``, ``max``, with a target ``event_rate``, ``missing_event_rate``, ``non_missing_event_rate``, ``missing_lift``, ``non_missing_lift``, ``iv``, ``tied_bins``, ``gini``, then ``skewness``, ``kurtosis``, ``cv``, and with a target ``ks``, ``chi2_p_value``
   * - :func:`~kivi.selection.logistic_regression`
     - ``variable``: ``intercept``, then the columns of ``X``
     - ``coef``, ``se``, ``z``, ``p_value``
   * - :func:`~kivi.selection.vif`
     - A Series named ``vif`` with the index ``variable``
     -
   * - :func:`~kivi.selection.correlation_filter`, :func:`~kivi.selection.vif_filter`
     - A ``pandas.Index`` of the kept columns of ``X``, in their order, or of their positions when ``X`` is not a DataFrame
     -
   * - :func:`~kivi.selection.selection_path`
     - ``step``: 1 to the number of variables in ``order``
     - ``variable``, ``auc``, ``gini``, ``ks``, ``max_p_value``, ``min_coefficient``, and with ``validation`` ``validation_auc``, ``validation_gini``, ``validation_ks``
   * - :func:`~kivi.scorecard.points_table`
     - ``variable``
     - ``bin``, ``woe``, ``points``
   * - :func:`~kivi.scorecard.woe_to_points`
     - The index of ``woe``
     - One column per variable of ``coefficients``, with the name of its index
   * - :func:`~kivi.scorecard.reason_codes`
     - The index of ``points``
     - ``reason_1`` to ``reason_<top>``
   * - :func:`~kivi.expert.ahp_weights`
     - A Series named ``weight`` with the index ``factor``
     -
   * - :func:`~kivi.expert.ahp_consistency`
     - A Series with the index ``statistic``: ``lambda_max``, ``ci``, ``ri``, ``cr``
     -
   * - :func:`~kivi.expert.combine_weights`
     - A Series named ``weight`` with the index ``factor``, ``sub_factor``
     -
   * - :func:`~kivi.calibration.master_scale`
     - ``grade``
     - ``lower``, ``probability``, ``upper``
   * - :func:`~kivi.strategy.cutoff_table`
     - ``cutoff``
     - ``approved``, ``approval_rate``, ``approved_event``, ``approved_event_rate``, ``rejected_event_share``, ``rejected_non_event_share``, and with ``loss`` ``approved_loss``
   * - :func:`~kivi.strategy.confusion_table`
     - ``cutoff``
     - ``true_positive``, ``false_positive``, ``true_negative``, ``false_negative``, ``precision``, ``recall``, ``f1``, ``accuracy``
   * - :func:`~kivi.strategy.swap_set`
     - ``current`` and ``new``, four rows
     - ``total``, ``share``, and with a target ``known``, ``event``, ``event_rate``
   * - :func:`~kivi.reject_inference.fuzzy_augmentation`
     - The index of ``probability``, each reject twice
     - ``target``, ``weight``
   * - :func:`~kivi.reject_inference.parceling`
     - The index of ``reject_score``, each reject twice with ``expected=True``
     - ``target``, ``weight``
   * - :func:`~kivi.reject_inference.reweighting`
     - A Series named ``weight`` with the index of ``score``
     -
   * - :func:`~kivi.monitoring.monitoring_table`
     - ``period``
     - ``total``, ``known``, ``event``, ``event_rate``, ``auc``, ``gini``, ``ks``, ``psi``
   * - :func:`~kivi.monitoring.stability_table`
     - ``period``
     - One per column of ``groups``, in columns named ``variable``
   * - :func:`~kivi.vintage.delinquency_panel`
     - A range, one row per loan and month
     - ``loan_id``, ``cohort``, ``month``, ``month_on_book``, ``days_past_due``, ``overdue_amount``, ``outstanding``, with ``principal`` ``balance``, and ``status``
   * - :func:`~kivi.vintage.vintage_table`
     - ``cohort``
     - One per month on book, in columns named ``month_on_book``
   * - :func:`~kivi.vintage.transition_matrix`
     - ``before``
     - One per state, in columns named ``after``
   * - :func:`~kivi.vintage.roll_rates`
     - A Series named ``roll_rate`` with the index ``before``
     -
   * - :func:`~kivi.vintage.migration_bandwidth`
     - A Series with the index ``statistic``: ``upper``, ``lower``
     -
   * - :func:`~kivi.vintage.migration_test`
     - ``before`` and ``after``, one row per off-diagonal cell of ``grades``
     - ``total``, ``share``, ``statistic``, ``p_value``
   * - :func:`~kivi.lifetime.transition_pd`
     - ``state``
     - One per horizon, in columns named ``horizon``
   * - :func:`~kivi.lifetime.stage_table`
     - ``stage``: 1, 2 and 3
     - ``total``, ``exposure``, ``exposure_share``, ``expected_credit_loss``, ``coverage``
   * - :func:`~kivi.datasets.make_loan_panel`
     - A range
     - ``loan_id``, ``cohort``, ``month``, ``month_on_book``, ``term``, ``amount``, ``balance``, ``status``
   * - :func:`~kivi.datasets.make_applications`
     - A range
     - ``age``, ``income``, ``debt_to_income``, ``inquiries``, ``housing``, ``employment``, ``incumbent_pd``, ``policy_decline``, ``approved``, ``target``, and with ``include_risk=True`` ``risk``
   * - :func:`~kivi.datasets.make_transactions`
     - A range, sorted by applicant and time
     - ``applicant_id``, ``timestamp``, ``amount``, ``kind``, ``merchant_category``
   * - :func:`~kivi.datasets.load_bank`
     - A range
     - The 16 columns of the UCI data and ``target``

The feature names follow from the arguments. Column names and prefixes are kept as given, and kivi's own parts of the names, the aggregations, the windows and the names of the panel features, are in lower case, with windows written as ``3m``, ``90d`` or ``6m_3m``. The dtypes are public too: the ``events``, ``count`` and ``nunique`` features and the counts of :func:`~kivi.features.panel_features` are int64, ``worst_status`` has the dtype of the status column, and the other features are float64. A window without values gives 0 for these counts and for ``sum``, a missing status for ``worst_status``, and NaN for the other features.

:func:`~kivi.datasets.load_bank` always returns the same data. :func:`~kivi.datasets.make_loan_panel`, :func:`~kivi.datasets.make_applications` and :func:`~kivi.datasets.make_transactions` keep their columns and dtypes, including the categories of ``kind`` and ``merchant_category``, but the values that they simulate for a given ``rng`` stay the same only within a minor version, so that a minor release can improve the simulation.

:func:`~kivi.woe.woe_trend` returns one of six labels, ``"increasing"``, ``"decreasing"``, ``"u_shaped"``, ``"inverted_u"``, ``"irregular"`` and ``"flat"``, which are public like column names, and so is its rule that neighbouring bins whose WOE differ by at most a relative :math:`10^{-10}` neither rise nor fall.

The random indices of :func:`~kivi.expert.ahp_consistency`, the table of Tone (1986) for 3 to 15 factors, are constants. A release changes them only to correct a wrong value, as a fix that the release notes list under Fixed, so the consistency ratio of a matrix stays the same from release to release.

Charts
------

The functions of :mod:`kivi.plotting` keep the data that they draw from each table, which column goes on which axis, their arguments, the errors they raise and what they return: every function returns the ``matplotlib.axes.Axes`` it drew on, except :func:`~kivi.plotting.woe_chart`, the one chart with two y-axes, which returns the pair of Axes of its bars and its line. :func:`~kivi.plotting.distribution_chart` draws the shares of the raw ``non_event`` and ``event`` counts, not the smoothed share columns of a WOE table, and :func:`~kivi.plotting.precision_recall_chart` draws the rows of a ROC table that have a precision and a line at the precision of its last row, the event rate. Colors, line styles, figure sizes, the placement of legends and annotations, tick formats and the number of matplotlib artists that make up a chart may change in any release, so that a release can improve how the charts look. Code that styles a chart further should work on the returned Axes rather than on the artists inside it.

Errors and warnings
-------------------

* The exception types are public. Invalid input raises ``ValueError``, and a model that cannot be fitted raises :exc:`~kivi.selection.SeparationError`, a subclass of ``ValueError``. A call that Python itself rejects, such as one with a missing or an unknown argument, raises ``TypeError``.
* The warning categories are public: a ``UserWarning`` means that a result differs from the request, a ``RuntimeWarning`` that a fit did not converge, and a ``FutureWarning`` that a part of the API is deprecated.
* The wording of a message is not public and may change in any release. The rule that a message names the argument and the value received stays.

Saved specifications
--------------------

:meth:`BinningSpec.to_json <kivi.spec.BinningSpec.to_json>` and :meth:`FeatureSpec.to_json <kivi.features.FeatureSpec.to_json>` write files that later releases read; :doc:`/user_guide/spec` and :doc:`/user_guide/features` describe their formats.

* ``format`` is always ``"kivi.BinningSpec"`` or ``"kivi.FeatureSpec"``, and ``format_version`` is an integer. A binning specification has version 1 without category groupings and 2 when it groups a column. A feature specification has version 1, the format of kivi 1.3.0, whose keys after ``kivi_version`` are the arguments of :class:`~kivi.features.FeatureSpec` as it takes them: ``id``, ``observation_time``, ``events``, ``ratios`` and ``panel``, with every key of each group.
* Every 1.x release reads every format version of a binning specification written by kivi 0.6.0 or later, and of a feature specification written by kivi 1.3.0 or later.
* A specification is written in the lowest format version that can hold it, so a file that uses no newer feature stays readable by older releases. A new format version only adds content, and within a major version the meaning of an existing key never changes.
* A file with a newer format version raises ``ValueError`` and names the version, instead of being read in part. Unknown keys raise as well, so a new key always comes with a new format version. A feature specification also raises for a missing key and for a key repeated within one JSON object.
* ``kivi_version`` records the release that wrote the specification and never changes how a file is read. Equality and hashing include it, as they include every field, so a specification saved by an earlier release differs from the same fit today; :ref:`spec-saving` shows how to compare their content.
* Encoding with a binning specification reads only the ``bin`` and ``woe`` of each table row. The other columns describe the fit and are loaded as written.

The tests of kivi keep binning specifications written by 0.6.0, in format version 1, and by 0.8.0, in format version 2, and check that each loads and encodes a fixed sample exactly as the release that wrote it did, and that it matches a fresh fit apart from ``kivi_version`` and the rounding of the last digits. They keep a feature specification of 1.3.0 in the same way, with window and panel groups and ratios, and check that it loads, writes the same JSON again and computes the same features of a fixed sample.
