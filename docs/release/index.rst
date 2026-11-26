Release notes
=============

Unreleased
----------

Changes merged since the last release.

.. release-fragments::

1.8.0 (2026-11-25)
------------------

New
^^^

* ``kivi.vintage.default_rate_table`` gives the one-year default rate of each grade by the cohort approach of the EBA guidelines on PD estimation, from quarter-end start months by default. A loan in default at a start month belongs to no cohort, the grade is the one at the start month, a loan that defaults twice in a year counts once, and loans that leave the panel during the year without defaulting stay in the cohort and are counted in ``exited``. ``probability`` adds the mean PD of each cohort.
* ``kivi.calibration.central_tendency_shift`` returns the shift of the log odds that brings the mean PD of a calibration sample to a central tendency, such as the long-run average default rate, without changing the order of the PDs: the scaled likelihood ratio of Tasche (2013).
* ``kivi.calibration.quasi_moment_fit`` fits a logistic PD curve to a central tendency and an accuracy ratio, the quasi moment matching of Tasche (2009), and returns the slope and the intercept that ``kivi.calibration.platt_predict`` applies.

* ``kivi.binning.optimal_edges`` fits the edges with the largest information value over the merges of fine quantile bins, found exactly by dynamic programming, under a maximum number of bins, a minimum share (``min_bin_share``) and a minimum number of events and of non-events (``min_bin_events``) in every bin, and a ``trend`` of the WOE: ``"increasing"``, ``"decreasing"``, ``"u_shaped"``, ``"inverted_u"``, ``"any"``, or by default ``"auto"``, the better of the two monotonic trends. ``kivi.spec.fit_spec`` fits it with ``method="optimal"``.
* ``kivi.binning.chimerge_edges`` and ``kivi.binning.monotonic_edges`` take ``min_bin_share``: before the other merges, a bin smaller than that share of the fitted values is merged with the neighbour whose chi-square statistic with it is smaller. The default, 0, keeps the results of earlier versions.
* The guide to WOE and IV explains why the information value of supervised bins on their development sample is optimistic, and how to measure it on a validation sample or with a permutation test.

* :mod:`kivi.plotting` draws five more result tables: :func:`~kivi.plotting.survival_chart` (the steps of ``cumulative_pd`` or ``survival``, with both causes under competing risks for ``cumulative_pd``), :func:`~kivi.plotting.transition_chart` (a transition matrix as a heatmap with the diagonal outlined), :func:`~kivi.plotting.rule_set_chart` (the ``new_hits`` of each rule with the ``approval_rate``), :func:`~kivi.plotting.profit_chart` (``profit`` against ``approval_rate`` with the best cut-off marked) and :func:`~kivi.plotting.lgd_chart` (``mean``, ``zero_share`` and ``full_share`` of each group).

* :func:`kivi.rules.rule_candidates` searches the numeric columns of a frame for single-threshold policy rules: it fits cut points with :func:`kivi.binning.tree_edges` or :func:`kivi.binning.ks_edges`, keeps for each threshold the side with the higher event rate and lists the rules that meet ``min_hit_rate`` and ``min_lift`` from the highest lift down. Its counts equal those of :func:`kivi.rules.rule_table` for the same rule, and missing values and special codes hit no rule.

* ``kivi.binning.variance_edges`` fits the edges of a driver to a continuous response, such as a realized LGD or CCF: over the merges of fine quantile bins it finds exactly, by the dynamic programming of ``optimal_edges``, the binning with the smallest sum of squares of the response within the bins, under a maximum number of bins, a minimum share of the values in every bin and a ``trend`` of the bin means. The response may leave [0, 1] but must not be missing, and ``weights`` are frequency weights. Group the values with ``kivi.binning.cut`` and summarize them with ``kivi.lgd.lgd_table``; the guide to LGD and EAD fits the bands of the loan-to-value ratio of the secured loans in this way.

Changed
^^^^^^^

* The home page, the index of the user guide and both READMEs describe ``optimal_edges``, ``variance_edges`` and the ``min_bin_share`` of ChiMerge and monotonic binning, ``rule_candidates``, ``default_rate_table``, ``central_tendency_shift``, ``quasi_moment_fit`` and the five new charts. The guide to binning shows in a diagram how ``optimal_edges`` finds the best path through the candidate bins, the guide to vintage analysis draws the one-year default rates of the start months with their mean, the observed average default rate, and their pooled rate; the validation guide counts the N, D and PD of the Jeffreys test of each grade with ``default_rate_table``; and the risk modeling workflow points to the constraints of ``optimal_edges`` and to the optimistic information value of the development sample.
* The conventions page adds the ``response`` of ``variance_edges`` and ``fractional_regression`` next to the binary ``target``, the options and the missing values of the new functions, the start months of ``default_rate_table`` and the slope and intercept of ``quasi_moment_fit`` among the parameters that another function applies. The compatibility page lists the tables of ``rule_candidates`` and ``default_rate_table``, the pair of ``quasi_moment_fit``, the float of ``central_tendency_shift`` and what the five new charts draw, and states that the ``response`` argument and values such as the ``method="optimal"`` of ``fit_spec`` are public. The glossary defines the accuracy ratio, the calibration sample and the calibration segment, the central tendency, the cohort approach, the start month, the one-year, observed average and long-run average default rates, optimal binning, the permutation test, quasi moment matching and the scaled likelihood ratio, and the candidate rule now covers the thresholds that ``rule_candidates`` finds.

* kivi 1.9 will require Python 3.11 or newer, because Python 3.10 reached the end of its life in October 2026. NumPy 1.24 and pandas 2.0 stay the oldest supported versions.

1.7.0 (2026-11-25)
------------------

New
^^^

* ``kivi.strategy.cutoff_table`` takes ``revenue``, a non-negative revenue per observation such as the interest margin of a loan, and then adds the column ``approved_revenue``, the revenue of the approved book, and with ``loss`` also ``profit``, the revenue minus the losses. ``table["profit"].idxmax()`` gives the most profitable of the cut-offs. Without ``revenue`` the table is unchanged.
* ``kivi.metrics.expected_maximum_profit`` gives the expected maximum profit (EMP) of a credit score of Verbraken, Bravo, Weber and Baesens (2014): the profit per applicant of the best cut-off, as a share of the loan amount, averaged over a loss of default that is 0 with the probability ``no_loss_share``, total with the probability ``full_loss_share`` and evenly spread in between, together with the share of applicants that the best cut-off rejects. The defaults and the results are those of the R package EMP and the Python package empulse, and ``weights`` are frequency weights.

* The new module ``kivi.rules`` measures the policy rules of a credit strategy, passed as a DataFrame with one column of booleans, or of 0 and 1, per rule. ``rule_table`` gives the hits, the hit rate and the unique hits of each rule, those that no other rule of the set hits, and with a target the hits with a known outcome, the events, the event rate, the lift, the recall and the events among the unique hits. ``rule_set_table`` applies the rules in order, as a decision waterfall: each row gives the new hits of a rule and the events among them, and the book that the rules up to it leave approved, with the columns of ``kivi.strategy.cutoff_table``. Missing labels in the target are unknown outcomes, as in ``kivi.strategy.swap_set``, a missing value in a rule raises, and ``weights`` are frequency weights.
* The new guide to policy rules measures the two policy rules and two candidate rules of the simulated applications: their hits and event rates, the overlap matrix of the rules, a waterfall that reproduces the approval decisions with the score cut-off as its last rule, what the order of the rules changes, and what the outcomes of the approved applicants alone can and cannot show.

Changed
^^^^^^^

* The guide to policy rules comes after the credit strategy, which sets the cut-off that the rules complement, and before the binning specifications. The home page, the index of the user guide and both READMEs describe ``kivi.rules``, the revenue and profit of ``cutoff_table`` and ``expected_maximum_profit``, with a chart of the decision waterfall of the simulated applications and one of the revenue, expected loss and profit by cut-off. A diagram of the worked example shows how ``rule_set_table`` credits each decline to the first rule that hits it.
* The conventions page describes the ``rules`` argument, the ``loss`` and ``revenue`` of ``cutoff_table`` and the parameters of ``expected_maximum_profit``, and adds the rule tables to the functions that read a missing label as an unknown outcome. The compatibility page lists the tables of ``rule_table`` and ``rule_set_table``, the new columns of ``cutoff_table`` and the statistics of ``expected_maximum_profit``, and the glossary defines the policy rule, the candidate rule, the hit rate, the unique hits, the new hits, the decision waterfall, the overlap matrix, the test group, the revenue and profit, the expected maximum profit, the return on investment and the ROC convex hull.

1.6.0 (2026-11-25)
------------------

New
^^^

* The new module ``kivi.lgd`` measures and models the loss given default and the credit conversion factor of defaulted loans. ``workout_lgd`` discounts the recoveries and costs of each workout to the date of default at an annual rate, one for all loans or one per loan, with the days counted ACT/365, and gives the exposure, the present values of the recoveries and of the costs and the realized LGD of each loan, without clipping it to [0, 1]. Costs before the date of default are compounded to it, as paragraph 144 of the EBA's guidelines on LGD estimation asks.
* ``kivi.lgd.conversion_factor`` gives the realized CCF of each credit line from the amount drawn at default and the limit and the amount drawn at a reference date, NaN for lines that were fully drawn or overdrawn at that date, and ``kivi.lgd.lgd_table`` summarizes realized LGDs by group: the number of defaults, the average, the standard deviation, the range and the shares of defaults without loss and with a total loss, after flooring negative LGDs at 0 as the regulations ask for long-run averages.
* ``kivi.lgd.fractional_regression`` fits the fractional logit of Papke and Wooldridge to a response between 0 and 1, such as a realized LGD, and returns the coefficient table of ``logistic_regression`` with robust (HC0) standard errors. A response of only 0 and 1 gives the coefficients of ``logistic_regression``.
* The new guide to LGD and EAD lists the rules of the Basel framework, the EBA and China's 2023 capital rules for realized LGDs and CCFs side by side, with their paragraphs and articles: the discount rate, costs before default, drawings, write-offs, cures, open workouts, long-run averages, negative values, the LGD and EAD floors, the reference date, fully drawn lines and the CCFs of the foundation approach.

* ``kivi.datasets.make_defaults`` simulates defaulted loans for LGD and EAD models: secured loans, unsecured loans and revolving lines that defaulted from 2018 to 2024, with the exposure at default, the value of the collateral, the limit and drawn amount of the revolving lines 12 months before default, a discount rate fixed at the default date and the date the workout ended, missing for the workouts still open at the data cut-off of 30 June 2025. ``kivi.datasets.make_recoveries`` simulates the cash flows of their workouts, one row per recovery, cost or cure, so that the realized LGDs are bimodal: close to 0 for most secured and cured loans and close to 1 for the other unsecured loans and revolving lines, with a few negative LGDs, a few LGDs above 1 and a few legal costs before default. The realized CCFs of the revolving lines mostly lie from 0 to 1. The same ``rng`` seed always gives the same loans and cash flows.

Changed
^^^^^^^

* The guide to LGD and EAD comes after the guide to calibration and before the credit strategy, whose expected loss takes the LGDs and EADs. The home page, the index of the user guide and both READMEs describe ``kivi.lgd`` and the simulated defaults and recoveries, with charts of the realized LGDs by segment and of the mean LGD of secured loans by loan-to-value band.
* The conventions page describes the cash flows and the ``rate`` of ``workout_lgd``, the ``response`` of ``fractional_regression`` and how the functions of ``kivi.lgd`` treat dates, missing values, booleans and weights. The compatibility page lists the tables of ``workout_lgd``, ``lgd_table`` and ``fractional_regression`` and the columns of ``make_defaults`` and ``make_recoveries``, and the glossary defines the workout, the closed workout, the cure, the economic loss, the realized LGD, the long-run average LGD, the reference date of a CCF, the fractional logit and robust standard errors.

1.5.0 (2026-11-24)
------------------

New
^^^

* The new module ``kivi.capital`` computes the regulatory capital of the internal ratings-based approach. ``asset_correlation`` and ``capital_requirement`` evaluate the risk-weight functions of the Basel framework (CRE31) and of China's capital rules of 2023 for corporate exposures, high-volatility commercial real estate, residential mortgages, qualifying revolving and other retail exposures, with the SME adjustment of ``turnover`` over a ``turnover_range`` that takes either jurisdiction's bounds, the 1.25 multiplier of ``financial`` institutions and the maturity adjustment. They reproduce the 152 risk weights of Annex 5 of Basel II.
* ``kivi.capital.conditional_pd`` gives the PD of the one-factor model of Vasicek in a scenario of the economy, ``systematic_factor`` the factor that an observed default rate implies, and ``vasicek_fit`` estimates the long-run PD and the asset correlation from a series of default rates by the method of moments, which allows years without defaults.
* ``kivi.capital.most_prudent_pd`` gives the most prudent upper bounds of Pluto and Tasche for the PDs of a portfolio with few or no defaults, with independent or correlated defaults, and reproduces the tables of their paper.
* The new guide to regulatory capital lists the parameters of the Basel framework and of China's rules side by side, with their paragraphs and articles: the SME range and unit, the financial-institution multiplier, the PD and LGD floors, the foundation LGDs, the maturity, the scope of the advanced approach and the output floor.

* ``kivi.lifetime.survival_table`` estimates a cumulative PD term structure from the duration of each loan until it defaults, leaves for another reason or is last observed. It gives the loans at risk, the defaults, the other exits and the censored loans of each period, the hazard rate, the probability of being still on book and the cumulative PD: the Kaplan–Meier estimate, or with ``competing``, such as early repayments, the Aalen–Johansen estimate of the cumulative incidence of default, which does not overstate the PD as 1 − Kaplan–Meier does when repaid loans count as censored. ``entry`` handles left truncation, such as a panel that starts after the loans were originated, and ``weights`` are frequency weights. Its ``cumulative_pd`` converts with ``convert_pd`` like a vintage curve.
* ``kivi.datasets.make_loan_panel`` takes ``prepayment``, the monthly probability that a current loan repays its whole balance early. A loan that repays early has a balance of 0 in that month and no rows after it, so the columns stay the same. The default, 0, gives the same panel as before for every ``rng``.
* The lifetime guide shows how to derive durations, defaults and repayments from a monthly loan panel, why the cumulative incidence and not 1 − Kaplan–Meier is the lifetime PD when loans repay early, how ``entry`` corrects for a panel that starts late, and a discrete-time hazard model fitted with ``logistic_regression`` on a person-period table.

Changed
^^^^^^^

* The guide to regulatory capital comes after the guide to model validation. The home page, the index of the user guide and both READMEs describe ``kivi.capital`` and ``survival_table``, with charts of the risk weight of each asset class against the PD and of the cumulative PD with early repayments as a competing risk or as censored.
* The lifetime guide weights the expected credit loss over economic scenarios: ``conditional_pd`` shifts the PDs of each delinquency state to an upside, a base case and a downturn, and the probability-weighted ECL comes out above that of the base case.
* The conventions page describes ``asset_class``, the ``level`` of the capital functions, and ``event``, ``duration`` and ``entry`` of ``survival_table``. The compatibility page lists the table of ``survival_table``, the result of ``most_prudent_pd`` and the tuple of ``vasicek_fit``, and the glossary defines the IRB approach, the asset class and asset correlation, the capital requirement, the risk weight, the maturity and SME adjustments, HVCRE and QRRE, the one-factor model, the systematic factor, stressed and through-the-cycle PDs, most prudent estimates of low-default portfolios, the output floor, censoring, left truncation, the risk set, competing risks, early repayment, the cumulative incidence, the Kaplan–Meier and Aalen–Johansen estimators, discrete-time hazard models and economic scenarios.

1.4.0 (2026-11-24)
------------------

New
^^^

* ``kivi.plotting.distribution_chart`` draws the share of all non-events and the share of all events in each row of a gains table or a WOE table as bars side by side, from the raw ``non_event`` and ``event`` counts, in the order of the table and with its empty rows.
* ``kivi.plotting.precision_recall_chart`` draws the precision against the true positive rate, the recall, of every threshold of a ROC table as steps whose area is the average precision, with a dashed line at the event rate of the sample, the precision of a score that ranks at random.

* ``kivi.selection.vif_filter`` drops the variable with the largest VIF, recomputes the VIFs of the others and repeats until no VIF exceeds ``threshold`` (5 by default). Variables with infinite VIFs go first, and ``priority``, such as the information values, decides between tied VIFs. It returns the kept columns, like ``correlation_filter``.
* ``kivi.selection.selection_path`` fits the logistic regression on the first 1, 2, 3 and more variables of an order, such as the variables sorted by information value, and returns one row per step with the variable added, the AUC, Gini and KS of the model, its largest Wald p-value and its smallest coefficient. An optional validation sample adds the same metrics on that sample, scored with the models of the development sample. Each row equals what ``logistic_regression`` and ``kivi.metrics`` give on the same columns.

* ``kivi.vintage.migration_bandwidth`` and ``kivi.vintage.migration_test`` summarize and test the migrations of customers between rating grades, as sections 2.5.5.1 and 2.5.5.2 of the European Central Bank's instructions for reporting the validation results of internal models ask. The first gives the upper and lower matrix weighted bandwidth (MWB) of the migrations to worse and to better grades. The second gives, for every off-diagonal cell of the migration matrix, the z test of whether it holds more customers than its neighbour towards the diagonal. Both take the grades of each customer at the beginning and at the end of the period like ``transition_matrix``, with ``grades`` from the best to the worst, and count a customer whose status at the end is not a grade, such as ``"default"`` or ``"closed"``, among the customers of its grade at the beginning.
* ``kivi.calibration.concentration_test`` gives the HI and the coefficient of variation of the shares of the grades of a rating scale and tests whether they are more concentrated than at the initial validation, as section 2.5.5.3 of the ECB's instructions asks. The HI is ``1 + ln(HHI) / ln(K)`` for the Herfindahl-Hirschman index of ``herfindahl_index`` and ``K`` grades.
* ``kivi.calibration.backtest_table`` back-tests continuous estimates, such as LGDs, ELBEs and CCFs, against their realized values: for every group it gives the mean estimate and realized value, the variance of their difference and the t test of the ECB's instructions. Its p-value is one-sided, as sections 2.6.2.1, 2.8.2.1 and 2.9.3.1 ask for the LGD, the LGD in-default and the CCF, so small p-values mean that the estimates are too low; the ELBE test of section 2.7.2.1 is two-sided, with the p-value ``2 * min(p_value, 1 - p_value)``.
* ``kivi.calibration.spiegelhalter_test`` tests the probabilities of single observations with Spiegelhalter's z statistic, which compares their Brier score with the score they would have if they were the true probabilities. ``kivi.calibration.brier_decomposition`` splits the Brier score by groups into reliability, resolution and uncertainty, plus two terms for probabilities that vary within a group, which are 0 on a master scale whose grades each have one PD.
* ``kivi.metrics.generalized_auc`` gives the generalized AUC of the ECB's instructions (section 2.6.3.1 and annex 3.2) and Somers' D, with their standard errors and a confidence interval like ``roc_auc_interval``. It measures how well estimated segments, such as the facility grades of an LGD model, rank realized ones, such as the realized LGDs cut into segments.
* The new guide to model validation maps the quantitative tools of the ECB's validation report to kivi's functions, and shows the back-tests, the comparison of the current AUC and gAUC with those of the initial validation, and the twelve LGD segments of the ECB on simulated data.

* ``kivi.woe.woe_trend`` names the shape of the WOE over the bins of a numeric variable, from a ``woe_table`` result or a table of a ``BinningSpec``: ``"increasing"``, ``"decreasing"``, ``"u_shaped"``, ``"inverted_u"``, ``"irregular"`` or ``"flat"``. It reads the rows labeled with intervals in the order of the table, leaves out the rows of special codes and missing values, and counts neighbouring bins with equal WOE as neither rising nor falling.

Changed
^^^^^^^

* The vintage guide gains a diagram of how ``delinquency_panel`` turns the instalments and payments of one loan into its rows at each month end. The binning guide gains a diagram of the second step of ``ks_edges``, which compares the best cuts of the two bins of the first cut, and says why ``ks_edges`` is not in its comparison of the methods.
* The diagram of ``ahp_weights`` in the guide to expert scorecards shows the statistics of ``ahp_consistency`` as the unnamed Series that the function returns, without the column name ``value``.
* The glossary defines ChiMerge, Best-KS and monotonic binning, delinquency buckets, bad definitions, the overdue and outstanding amounts of ``delinquency_panel``, the tolerances of ``delinquency_panel`` and of the AHP functions and the global weights of sub-factors, and tells the events of ``kivi.features`` apart from the events of a target.

* The migration guide corrects its mappings of 0.2.x: ``StepWise`` compared the models of a fixed order and becomes ``selection_path``, while ``stepwise`` selects by p-values; ``VIF.LowVIFFeatures`` becomes ``vif_filter``, ``woe/utils.monotony`` becomes ``woe_trend``, and ``BinaryMetrics.plot_ks``, ``plot_score`` and ``plot_pr`` become ``gains_chart``, ``distribution_chart`` and ``precision_recall_chart``. It states how each differs from the legacy code, and gives pandas code for ``int_score`` and ``StepWise.top_n_features``, which are not ported.
* The conventions page describes the arguments ``realized``, ``estimated`` and ``grades``, the compatibility page lists the tables of ``selection_path`` and of the validation tests and the labels of ``woe_trend``, and the glossary defines back-tests, the Brier decomposition, the generalized AUC and Somers' D, the HHI and the HI, the migration matrix and its bandwidth, the initial validation, the ELBE and the CCF, the precision-recall curve, average precision and recall, the selection path, VIFs and their elimination, and the trend of the WOE.
* The guide to model validation comes after the guide to model monitoring, and the home page and the index of the user guide describe it. The guides gain diagrams of how ``generalized_auc``, ``migration_bandwidth`` and ``migration_test`` compute their results, and charts of a selection path, of the Brier decomposition, of the concentration of grades and of the back-test of LGD grades.

* The README gallery opens with a chart of time-window features: the share of applicants and the event rate per band of the cash withdrawals in the last three months and in the three months before, computed with a ``FeatureSpec`` from synthetic transactions. The features guide shows the same chart after its screening of these features.
* The workflow diagram of the README starts with defining the features once with a ``FeatureSpec`` and computing them for every sample, before the bin edges are fitted once and every sample is cut with them.

* The tutorial gains a chapter on transaction features, after the data and samples, so it has eleven chapters. The lender dates the applications, simulates the transactions on the applicants' accounts with ``make_transactions``, derives cash, repayment and purchase features as of each application date with a ``FeatureSpec``, checks that the transactions after the application count in none of them, profiles them with ``profile_table`` next to the application data, and saves the specification as JSON. The scorecard of the later chapters still uses the application data alone, and their results are unchanged.
* The chapter on the loan book under IFRS 9 reads the repayment history of each loan with ``panel_features`` and shows how a probation of six months for cured loans changes the stages and the loss allowance.

1.3.0 (2026-11-23)
------------------

New
^^^

* The new module ``kivi.features`` computes time-window features of event data, such as transactions, point in time. ``window_features`` aggregates the events of each entity over windows that end at the observation time of each row of an observation table, such as the application date, and never uses an event at or after that time. Windows count calendar months, as ``"3M"``, or days, as ``"90D"``, and a pair such as ``("6M", "3M")`` is the period before the most recent one; ``closed="right"`` includes the events at the observation time instead. The aggregations are ``count``, ``sum``, ``mean``, ``min``, ``max``, ``std``, ``nunique``, ``first``, ``last`` and ``days_since_last``, ``ratios`` divides two features, and ``prefix`` names the features of filtered events. The columns are named ``{prefix}{column}_{aggregation}_{window}`` next to ``{prefix}events_{window}``, the number of events of each window. The events are sorted once and the bounds of every window found by binary search, so counts, sums, means and first and last values take a time that grows with the number of rows, while ``min``, ``max``, ``std`` and ``nunique`` look at every event of every window.
* ``kivi.features.FeatureSpec`` keeps the definitions of one or more groups of window features and their ratios, computes them on any sample with ``compute`` and saves them as versioned JSON with ``to_json`` and ``from_json``, so that development and production compute the same features. Nothing in it is fitted to data.

* ``kivi.datasets.make_transactions`` simulates the account transactions that precede the applications of ``make_applications``: card purchases, cash withdrawals and instalment repayments over the twelve months (or ``n_months``) before each application date, as a long table with the applicant id, the timestamp, the amount, the kind and the merchant category. The transactions depend on the hidden risk factor and on the income, not on the outcome: applicants at risk build up stress, withdraw more cash as the application nears and repay less, so that features of time windows carry a clear but imperfect signal about ``target``, also within bands of ``incumbent_pd``. The dates of the applications, required, come from a column, as one date or as a Series. The same ``rng`` seed always gives the same transactions.
* ``kivi.datasets.make_applications`` takes ``include_risk=True`` to append the hidden risk factor as the last column ``risk``. The default output is unchanged.

* ``kivi.features.panel_features`` summarizes a monthly panel of repayment statuses, such as the result of ``kivi.vintage.delinquency_panel`` or ``kivi.datasets.make_loan_panel``, before each observation time. For every window of months it gives the number of months with a status, the bad months, the longest runs of consecutive bad and good months and the worst status, and it adds the months since the latest bad month. A status is bad from ``bad`` on, ``"30-59"`` by default, in the order of the ordered categorical status column. Only the month ends before the month of the observation time count, an entity with several loans takes the worst status of each month, and a month without a status counts in no window and ends a run.
* ``kivi.features.FeatureSpec`` holds groups of panel features in ``panel``, next to the groups of window features in ``events``, and ``compute`` takes the panels as ``panel``. Ratios may divide features of both kinds. Version 1 of its JSON format is final.

Changed
^^^^^^^

* The migration guide maps ``FeatureTools``, ``FeatureToolsV2`` and ``index/collect.py`` of 0.2.x to ``window_features``, ``panel_features`` and ``FeatureSpec`` of ``kivi.features``, and states how their windows, aggregations, ratios and runs of months differ from the legacy code. It no longer lists the ratios of ``FeatureTools`` among the tools without a replacement, and it names ``make_transactions`` next to the synthetic data of 0.2.x.
* The conventions page describes the tables, time columns, column names and windows of ``kivi.features``, the compatibility page lists the columns of its functions and of ``make_transactions`` and the JSON format of ``FeatureSpec``, and the glossary defines point in time, observations, windows and window pairs, and the bad months, runs and monthly status panels of repayment behaviour.
* The guide to time-window features comes first among the topic guides, before the univariate profile. It gains two diagrams, of the windows of ``window_features`` and of the months of ``panel_features``, and a section that computes features of the transactions of ``make_transactions`` with a ``FeatureSpec`` and screens them with ``profile_table``.

1.2.0 (2026-11-22)
------------------

New
^^^

* ``kivi.vintage.delinquency_panel`` builds the monthly loan panel that ``vintage_table``, ``transition_matrix`` and ``roll_rates`` take from the instalment schedules and the payments of the loans. It applies the payments of each loan to its oldest unpaid instalment first, so that an amount paid early counts towards the next instalment, and at every month end up to ``end`` counts only the payments made by then. Each row has the columns of ``make_loan_panel`` that a schedule gives, with the same names and in the same order: ``loan_id``, ``cohort``, ``month``, ``month_on_book``, with ``principal`` the unpaid principal as ``balance``, and ``status`` in the buckets of ``make_loan_panel``. It adds the days past due since the oldest unpaid instalment fell due and the amounts overdue and outstanding. The cohort is the month before the first due date, or with ``origination`` the origination month of each loan. ``tolerance`` lets an instalment count as paid when a small amount of it remains unpaid. A loan leaves the panel once it is repaid, and keeps rolling after its last due date while it is not.

* The new module ``kivi.expert`` weighs the factors of an expert scorecard, the kind of scorecard that credit analysts set by judgment where there are too few defaults to fit a model, with the analytic hierarchy process. ``ahp_weights`` turns a pairwise comparison matrix into weights that sum to 1, by Saaty's principal eigenvector or by the geometric mean of the rows, and ``ahp_consistency`` returns the principal eigenvalue, the consistency index, Saaty's random index and the consistency ratio, so that judgments with a ratio above the 0.1 rule of thumb can be reviewed. ``combine_weights`` multiplies the weights of factors and of their sub-factors into global weights. The matrix is a DataFrame whose index and columns name the factors, or an array, and it must have a diagonal of 1 and reciprocal entries, which the functions check. They replace the geometric-mean weights and the consistency check of the ``AHP`` class of the 0.2.x series, which is not part of kivi 1.

* :func:`~kivi.binning.ks_edges` fits edges by Best-KS binning. It cuts each bin where the cumulative shares of its events and of its non-events differ most, the KS of the bin, among the cuts that leave ``min_bin_share`` of the observations on each side, and splits the bins best first, by the pairs of an event and a non-event that a cut ranks apart: its KS times the numbers of events and non-events in the bin. It takes ``max_bins``, ``special`` and ``weights`` as ``tree_edges`` does and returns edges of the same form, with observed values as interior edges.
* :func:`~kivi.spec.fit_spec` fits a column with ``ks_edges`` when ``method`` is ``"ks"``. The saved specification stores the edges, so its format version does not change.

Changed
^^^^^^^

* The migration guide maps ``KSBins``, ``PortfolioAnalysis.Vintage`` and the ``AHP`` class of 0.2.x to ``ks_edges``, ``delinquency_panel`` and the functions of ``kivi.expert``, and states how each one differs from the legacy code. It no longer lists the analytic hierarchy process and the repayment-schedule calculations among the tools without a replacement.
* The conventions page describes the comparison matrices of ``kivi.expert`` and the dates of ``delinquency_panel``, the compatibility page lists the results of the new functions and keeps the random indices of ``ahp_consistency`` constant, and the glossary defines the terms of the analytic hierarchy process, the grace period and the repayment schedule.
* The binning guide gains a chart of the Best-KS cuts, and the guide to expert scorecards two diagrams: how the weights and the consistency ratio follow from a comparison matrix, and which judgments of an inconsistent matrix to discuss first.

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
