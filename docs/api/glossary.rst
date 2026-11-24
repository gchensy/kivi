Glossary
========

The terms of credit risk modeling as the guides and the docstrings use them. Each entry links to the guide that explains the term with an example.

.. glossary::

   Absorbing state
      A state of a transition matrix that a loan never leaves once it has entered it, such as default. The powers of the matrix then accumulate the probability of having reached it; see :doc:`/user_guide/lifetime`.

   Accepts and rejects
      The applicants that a previous policy approved and declined. The outcome is known only for the accepts, so a model developed on them alone may misjudge the rejects; see :doc:`/user_guide/reject_inference`.

   Analytic hierarchy process (AHP)
      Saaty's method of weighing the factors of a decision, such as those of an expert scorecard, from comparisons of the factors two at a time. The weights are the principal eigenvector, or the geometric means of the rows, of the pairwise comparison matrix, and the consistency ratio checks that the comparisons agree with each other; see :doc:`/user_guide/expert`.

   Area under the ROC curve (AUC)
      The probability that a randomly chosen event scores higher than a randomly chosen non-event, with ties counted as one half. A score unrelated to the target has an AUC of about 0.5; see :doc:`/user_guide/metrics`.

   Bad definition
      The condition that makes a loan bad at a month end, usually a delinquency bucket or worse, such as 90 days past due or more. A vintage table counts the loans that have met it at any month on book so far, so an earlier definition gives an earlier signal; see :doc:`/user_guide/vintage`.

   Bad month
      A month whose status at its end is ``bad`` or worse in the order of an ordered status column, such as 30 days past due or more with the default ``bad="30-59"`` of :func:`~kivi.features.panel_features`; see :doc:`/user_guide/features`.

   Best-KS binning
      Supervised binning that cuts each bin where the KS between its events and non-events is largest, and splits first the bin whose cut ranks the most pairs of an event and a non-event apart. Its first cut keeps the KS of the raw values; see :doc:`/user_guide/binning`.

   Binning
      Dividing the values of a numeric variable into intervals, the bins, whose edges are fitted on a reference sample and then applied to every other sample. Scorecard development calls it classing: fine classing forms many small bins, and coarse classing merges them into a few that are large enough to be stable. A categorical variable is binned by grouping its categories instead; see :doc:`/user_guide/binning`.

   Binning specification
      The edges, category groupings, special codes and WOE tables of every column of a model, fitted together and saved as versioned JSON, so that production data are grouped and encoded exactly as the development sample; see :doc:`/user_guide/spec`.

   Category grouping
      A mapping from each category of a categorical variable to a group, fitted on a reference sample by pooling rare categories or by merging categories with similar event rates, and applied unchanged to every other sample; see :ref:`category-groups`.

   Characteristic stability index (CSI)
      The change of a variable's average scorecard points from a reference sample to a current one: the change of each group's share times the group's points, summed over the groups. Over all the variables of a scorecard, the CSI values add up to the change of the average score; see :ref:`characteristic-stability`.

   ChiMerge
      Supervised binning that starts from many small bins and repeatedly merges the adjacent pair with the smallest chi-square statistic, the pair whose event rates are most alike, until no more than a given number of bins remain and every adjacent pair differs significantly. Categories are merged in the same way once sorted by their WOE; see :doc:`/user_guide/binning`.

   Cohort
      The loans originated in the same period, usually a month, followed together as they age in a vintage table; see :doc:`/user_guide/vintage`.

   Conditional PD
      The probability of a default in a period given that there was none before it, also called the hazard rate; see :doc:`/user_guide/lifetime`.

   Consistency index (CI)
      How far the judgments of a pairwise comparison matrix of :math:`n` factors contradict each other: :math:`(\lambda_{\max} - n) / (n - 1)`, where :math:`\lambda_{\max}` is the principal eigenvalue of the matrix. It is 0 when the judgments are consistent; see :doc:`/user_guide/expert`.

   Consistency ratio (CR)
      The consistency index of a pairwise comparison matrix divided by the random index for the same number of factors. Saaty's rule of thumb asks for at most 0.1 before the weights are used; see :doc:`/user_guide/expert`.

   Coverage ratio
      The loss allowance of a group of loans divided by their exposure; see :doc:`/user_guide/lifetime`.

   Credit-impaired
      IFRS 9 calls a loan credit-impaired once events with a detrimental effect on its expected cash flows have occurred, such as significant financial difficulty of the borrower or a breach of contract like a default or a past-due event. Lenders usually align it with their definition of default, and such loans are in stage 3; see :doc:`/user_guide/lifetime`.

   Cumulative PD
      The probability of a default in any of the periods from the first up to a given one, which never falls as the horizon grows; see :doc:`/user_guide/lifetime`.

   Cut-off
      The score that separates approval from decline. kivi approves the scores at or below the cut-off, because its scores rise with the risk; see :doc:`/user_guide/strategy`.

   Days past due (DPD)
      The number of days for which a contractual payment has been overdue. IFRS 9 rebuttably presumes a significant increase in credit risk when a payment is more than 30 days past due, and default no later than when it is 90 days past due; see :doc:`/user_guide/lifetime`, and :doc:`/user_guide/vintage` for how they are counted from a repayment schedule.

   Delinquency bucket
      A range of days past due that the status of a loan falls into at a month end, such as ``current``, ``1-29``, ``30-59``, ``60-89`` and ``90+``. Roll rates and transition matrices count how loans move between buckets from one month end to the next; see :doc:`/user_guide/vintage`.

   DeLong test
      A test of whether two scores have the same AUC on the same sample. It uses the variances and the covariance of the two AUCs that DeLong, DeLong and Clarke-Pearson (1988) derived from the placement values; see :doc:`/user_guide/metrics`.

   Deprecation
      A part of the public API that still works but warns with a ``FutureWarning``, because a later release removes or changes it. The warning says what to use instead; see :doc:`compatibility`.

   Effective interest rate
      The rate that discounts the estimated future cash flows of a loan over its expected life to its gross carrying amount at initial recognition. IFRS 9 discounts expected credit losses at this rate; see :doc:`/user_guide/lifetime`.

   Entity
      The customer, account or loan that events, monthly statuses and observations belong to, named by the ``id`` column of :mod:`kivi.features`; see :doc:`/user_guide/features`.

   Event and non-event
      The two outcomes of a binary target: an event has target 1, such as a default or another bad outcome, and a non-event has target 0. The guides read the subscription of the bundled bank data as the event; see :doc:`/user_guide/woe_iv`. The events that :mod:`kivi.features` aggregates are of another kind: dated records of an entity, such as its transactions; see :doc:`/user_guide/features`.

   Event rate
      The share of events among the observations of a group, called the bad rate when the event is a default. It is computed from the raw counts, without smoothing; see :doc:`/user_guide/woe_iv`.

   Ever bad
      A loan is ever bad by a month on book if it was bad at any month on book up to then, even if it has cured since. The cumulative bad rates of a vintage table count the loans that are ever bad; see :doc:`/user_guide/vintage`.

   Expected credit loss (ECL)
      Under IFRS 9, the present value of the credit losses of an exposure, weighted with the probabilities of default. The 12-month ECL counts the losses from defaults in the next 12 months, and the lifetime ECL those over the remaining life of the exposure; see :doc:`/user_guide/lifetime`.

   Expected loss (EL)
      The loss that an exposure is expected to cause: its probability of default times its loss given default times its exposure at default; see :doc:`/user_guide/strategy`.

   Expert scorecard
      A scorecard whose weights are set by the judgment of credit analysts rather than fitted to outcomes, for portfolios with too few defaults to fit a model; see :doc:`/user_guide/expert`.

   Exposure at default (EAD)
      The amount owed when a borrower defaults, such as the outstanding balance of a loan; see :doc:`/user_guide/strategy`.

   Feature specification
      The definitions of the window and repayment features of a model: the columns of the entities and of the observation times, groups of features with their tables, windows and aggregations, and ratios. Saved as versioned JSON, it lets production compute the same features as development. Unlike a binning specification, it is not fitted to data; see :doc:`/user_guide/features`.

   Format version
      The version of the JSON format of a binning or feature specification, stored in each file. A release reads every format version up to its own and rejects a newer one; see :doc:`compatibility`.

   Gains table
      A table of score bands from the riskiest to the safest, with the event rate and lift of each band and the cumulative shares of events and non-events; see :doc:`/user_guide/metrics`.

   Gini coefficient
      Twice the AUC minus 1: 0 for a score unrelated to the target and 1 for a score that ranks every event above every non-event. Scorecard development reports it more often than the AUC itself; see :doc:`/user_guide/metrics`.

   Global weight
      The weight of a sub-factor in a whole expert scorecard: the weight of its factor times its weight among the sub-factors of that factor. The global weights of all sub-factors sum to 1; see :doc:`/user_guide/expert`.

   Grace period
      The months after the origination of a loan in which nothing falls due, so that its first instalment falls due later than a month after origination. :func:`~kivi.vintage.delinquency_panel` puts a loan in the cohort of the month before its first due date unless ``origination`` gives the origination dates, which keep the loans with a grace period in the cohort of their origination month; see :doc:`/user_guide/vintage`.

   Grade
      One class of a rating system or master scale, with a PD of its own and bounds that decide which predicted PDs fall into it; see :doc:`/user_guide/calibration`.

   Information value (IV)
      The sum of the differences between the event share and the non-event share of each group, each times the group's WOE. It measures how strongly a grouped variable separates events from non-events on its own; see :doc:`/user_guide/woe_iv`.

   Kolmogorov-Smirnov statistic (KS)
      The largest absolute gap between the distribution functions of the scores of events and of non-events; see :doc:`/user_guide/metrics`.

   Lift
      The event rate of a group divided by the event rate of the whole sample. A lift above 1 marks a group that is riskier than average; see :doc:`/user_guide/metrics`.

   Loss allowance
      The provision that IFRS 9 requires for the expected credit losses of a loan: the 12-month ECL in stage 1 and the lifetime ECL in stages 2 and 3; see :doc:`/user_guide/lifetime`.

   Loss given default (LGD)
      The share of the exposure that is lost when a borrower defaults, after recoveries and collateral; see :doc:`/user_guide/strategy`.

   Marginal PD
      The probability, seen from the starting point, of a default in one given period. The marginal PDs of the periods up to a horizon add up to the cumulative PD; see :doc:`/user_guide/lifetime`.

   Master scale
      A fixed set of grades that divides the PD range from 0 to 1, so that ratings from different models share one meaning; see :doc:`/user_guide/calibration`.

   Monitoring report
      A table with one row per period that tracks a model in use: its population, the known outcomes, its discrimination and the stability of its score against a reference sample; see :doc:`/user_guide/monitoring`.

   Monotonic binning
      Supervised binning whose WOE rises or falls steadily from the lowest bin to the highest, so that the bins can be explained. kivi starts from many small bins and merges the adjacent bins that break the trend, pooling adjacent violators; see :doc:`/user_guide/binning`.

   Month on book
      The number of months since a loan was originated. A vintage table compares cohorts at the same month on book; see :doc:`/user_guide/vintage`.

   Monthly status panel
      A table with one row per loan and month and the status of the loan at the month end, such as its delinquency bucket. :func:`~kivi.vintage.delinquency_panel` builds one from repayment schedules and payments, and :func:`~kivi.features.panel_features` summarizes it before each observation time; see :doc:`/user_guide/vintage`.

   Observation
      In :mod:`kivi.features`, an entity and an observation time, such as an applicant and the date of the application: one row of the table ``observations``, for which every window is computed. An entity may have several observations, such as one per month end; see :doc:`/user_guide/features`.

   Observation month
      The calendar month of an observation time. Its month-end status is not known at the observation time, so :func:`~kivi.features.panel_features` uses only the months before it; see :doc:`/user_guide/features`.

   Observation time
      The time at which a feature is computed, such as the date of an application. Events at or after it are not used, unless ``closed="right"`` admits those at that very time; see :doc:`/user_guide/features`.

   Outcome window
      The time after which the outcome of an observation is known, such as twelve months after a loan was granted. Until it has passed, the target is missing and the outcome counts as unknown; see :doc:`/user_guide/monitoring`.

   Outstanding amount
      The unpaid amount of all the instalments of a loan at a month end, whether they have fallen due or not: the ``outstanding`` column of :func:`~kivi.vintage.delinquency_panel`. It includes interest, while the ``balance`` column, given the principal of each instalment, counts only the unpaid principal; see :doc:`/user_guide/vintage`.

   Overdue amount
      The unpaid amount of the instalments of a loan that fell due before a month end: the ``overdue_amount`` column of :func:`~kivi.vintage.delinquency_panel`. An instalment due on the month end itself is not overdue yet; see :doc:`/user_guide/vintage`.

   Pairwise comparison matrix
      A square matrix whose entry :math:`a_{ij}` says how many times more important factor :math:`i` is than factor :math:`j`, on the Saaty scale. Its diagonal is 1 and :math:`a_{ji} = 1 / a_{ij}`; see :doc:`/user_guide/expert`.

   PD term structure
      The PD of each period after a starting point, such as the origination of a loan or the reporting date, given as cumulative, marginal or conditional PDs; see :doc:`/user_guide/lifetime`.

   Placement value
      For an event, the share of non-events that score below it, and for a non-event, the share of events that score above it, with ties counted as one half. The average over either class is the AUC, and their variances give its DeLong standard error; see :doc:`/user_guide/metrics`.

   Point in time
      A feature is point in time when it uses only what was known at its observation time, so that it is computed in development as it would have been at the time of the decision, and no later event leaks into the model; see :doc:`/user_guide/features`.

   Points to double the odds (PDO)
      The number of scorecard points by which the score rises when the good:bad odds double. Together with a base score and the odds at that score, it sets the scale of a scorecard; see :doc:`/user_guide/scorecard`.

   Population stability index (PSI)
      A measure of how much the distribution of a variable or a score has shifted between a reference sample and a current one, summed over the groups from the change of each group's share; see :doc:`/user_guide/psi`.

   Precision
      The share of events among the observations that a score flags at a threshold, that is, the event rate of the flagged observations; see :doc:`/user_guide/metrics`.

   Probability of default (PD)
      The probability that a borrower defaults within a given horizon, usually twelve months. kivi calls a predicted PD, or any other predicted event probability, ``probability``; see :doc:`/user_guide/calibration`.

   Random index (RI)
      The average consistency index of pairwise comparison matrices filled with random judgments on the Saaty scale, which depends only on the number of factors. kivi uses the table of Tone (1986) for 3 to 15 factors; see :doc:`/user_guide/expert`.

   Reason codes
      The variables that lowered an applicant's score most, compared with the best group of each variable, as given to a declined applicant; see :doc:`/user_guide/scorecard`.

   Reference sample
      The sample that later samples are compared with, usually the development sample or a period right after it. Bin edges, score bands and groups are fitted on it; see :doc:`/user_guide/monitoring`.

   Reject inference
      Methods that supply outcomes for rejected applicants, as rows with weights or as weights for the accepts, so that a model can be refitted on all applicants; see :doc:`/user_guide/reject_inference`.

   Repayment schedule
      The instalments that a loan owes, each with its due date and its amount. Compared with the payments made, it gives the days past due of the loan at each month end; see :doc:`/user_guide/vintage`.

   Reporting date
      The date at which a loss allowance is measured. Period 1 of an expected credit loss is the first period after it; see :doc:`/user_guide/lifetime`.

   ROC curve
      The true positive rate, the share of events flagged, against the false positive rate, the share of non-events flagged, as the threshold moves through the scores from the riskiest down. The area under it is the AUC, and its largest vertical distance from the diagonal is the KS; see :doc:`/user_guide/metrics`.

   Roll rate
      The share of the loans in a delinquency bucket at one month end that are in the next worse bucket at the next month end; see :doc:`/user_guide/vintage`.

   Run
      A stretch of consecutive months with bad statuses, or with good ones. A month without a status ends a run, and a window counts only the months of a run that lie inside it; see :doc:`/user_guide/features`.

   Saaty scale
      The scale of pairwise comparisons from 1, equally important, through 3, 5 and 7 to 9, extremely more important, with 2, 4, 6 and 8 in between and the reciprocals for the reverse comparisons; see :doc:`/user_guide/expert`.

   Significant increase in credit risk (SICR)
      The IFRS 9 criterion that moves a loan from stage 1 to stage 2: its risk of default has risen significantly since initial recognition. Each lender sets the test in its own policy, for example as relative and absolute increases of the PD; see :doc:`/user_guide/lifetime`.

   Smoothing
      The value that replaces an empty count before shares are computed, 0.5 by default, so that the WOE, the IV and the PSI of a group stay finite. Counts are reported as observed, and only the shares are smoothed; see :doc:`/user_guide/woe_iv`.

   Special codes
      Values that stand for a state rather than a measurement, such as -1 for "never contacted", 999 for "no record" or the label "unknown" of a categorical variable. Passed in ``special``, they are left out of the fit of the edges or of the grouping and each gets a group of its own; see :ref:`special-values`.

   Stage
      One of the three IFRS 9 buckets of a loan: stage 1 for loans without a significant increase in credit risk, with a 12-month ECL; stage 2 for loans with one, with a lifetime ECL; and stage 3 for credit-impaired loans, also with a lifetime ECL; see :doc:`/user_guide/lifetime`.

   Standard error and confidence interval
      The standard error measures how much an estimate, such as an AUC, would vary from sample to sample. A confidence interval at a level such as 95% is a range around the estimate that would cover the true value in that share of samples; see :doc:`/user_guide/metrics`.

   Swap set
      The four groups of applicants that two scores, each with its cut-off, approve or decline. The swap-outs are approved only by the current score and the swap-ins only by the new one; see :doc:`/user_guide/strategy`.

   Tolerance
      How far a value may miss a rule and still count as meeting it. :func:`~kivi.vintage.delinquency_panel` counts an instalment as paid once at most ``tolerance``, an amount, remains unpaid of it and the older instalments together, so that a rounding difference does not make a loan past due; see :doc:`/user_guide/vintage`. :func:`~kivi.expert.ahp_weights` and :func:`~kivi.expert.ahp_consistency` accept a pairwise comparison matrix whose diagonal is 1 and whose entries are reciprocal within the relative tolerance ``rtol``, so that 0.3333 passes for 1 / 3; see :doc:`/user_guide/expert`.

   Total
      The number of observations in a row of a table, missing values included, or their sum of weights. The ``count`` of :func:`~kivi.univariate.profile_table` is the number of values that are not missing instead, as in ``pandas.DataFrame.describe``; see :doc:`/user_guide/univariate`.

   Transition matrix
      A table that counts the loans or obligors that moved from each state to each other state between two dates, such as delinquency buckets or rating grades; see :doc:`/user_guide/vintage`.

   Unseen category
      A category of a later sample that did not occur in the reference sample, so that its grouping or WOE table has no row for it. kivi raises an error for such a category unless an ``unseen`` policy says how to treat it; see :doc:`/user_guide/woe_iv`.

   Vintage table
      The cumulative ever-bad rate of each cohort by month on book; see :doc:`/user_guide/vintage`.

   Weight of evidence (WOE)
      The logarithm of a group's share of all events over its share of all non-events. A positive WOE marks a group that is relatively more common among events; see :doc:`/user_guide/woe_iv`.

   Window
      The period before an observation time :math:`t` over which a feature aggregates the events: :math:`[t - k, t)` for :math:`k` calendar months or days, or :math:`(t - k, t]` with ``closed="right"``. For monthly statuses, the :math:`k` months before the observation month; see :doc:`/user_guide/features`.

   Window pair
      Two bounds of the same unit, such as ``("6M", "3M")``, that make one window from the first to the second, here :math:`[t - 6M, t - 3M)`, the period before the most recent one. Its features are named with both bounds, as ``6m_3m``; see :doc:`/user_guide/features`.
