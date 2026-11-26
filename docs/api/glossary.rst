Glossary
========

The terms of credit risk modeling as the guides and the docstrings use them. Each entry links to the guide that explains the term with an example.

.. glossary::

   Aalen–Johansen estimator
      The estimate of the cumulative incidence of default when other exits, such as early repayments, compete with it: the sum over the periods of the share of the accounts still on book at the start of a period times the hazard rate of default in it. Without other exits it equals 1 − Kaplan–Meier; see :doc:`/user_guide/lifetime`.

   Absorbing state
      A state of a transition matrix that a loan never leaves once it has entered it, such as default. The powers of the matrix then accumulate the probability of having reached it; see :doc:`/user_guide/lifetime`.

   Accepts and rejects
      The applicants that a previous policy approved and declined. The outcome is known only for the accepts, so a model developed on them alone may misjudge the rejects; see :doc:`/user_guide/reject_inference`.

   Accuracy ratio (AR)
      The name of the Gini coefficient in rating validation and PD calibration: twice the AUC minus 1, with tied scores counted as half. Quasi moment matching fits a PD curve to a target accuracy ratio; see :doc:`/user_guide/calibration`.

   Analytic hierarchy process (AHP)
      Saaty's method of weighing the factors of a decision, such as those of an expert scorecard, from comparisons of the factors two at a time. The weights are the principal eigenvector, or the geometric means of the rows, of the pairwise comparison matrix, and the consistency ratio checks that the comparisons agree with each other; see :doc:`/user_guide/expert`.

   Area under the ROC curve (AUC)
      The probability that a randomly chosen event scores higher than a randomly chosen non-event, with ties counted as one half. A score unrelated to the target has an AUC of about 0.5; see :doc:`/user_guide/metrics`.

   Asset class
      The class of an exposure that fixes its asset correlation in the IRB risk-weight functions and whether the maturity adjustment applies: corporate, which includes sovereigns and banks, high-volatility commercial real estate, residential mortgages, qualifying revolving retail and other retail; see :doc:`/user_guide/capital`.

   Asset correlation
      In the one-factor model, the share of the variance of a borrower's assets that the systematic factor explains. The higher it is, the more the defaults of a portfolio cluster in bad years, and the higher the capital requirement; see :doc:`/user_guide/capital`.

   Average precision
      The area under the precision-recall curve drawn as steps: the sum over the thresholds of the gain in recall times the precision at the threshold, :math:`\sum_n (R_n - R_{n-1}) P_n`. A score that ranks at random has an average precision close to the event rate; see :doc:`/user_guide/plotting`.

   Back-test
      A comparison of the estimates of a model, such as the PDs of its grades or the LGDs of its facilities, with the values realized later. kivi tests PDs with the binomial and Jeffreys tests and continuous estimates with the t-tests of the European Central Bank; see :doc:`/user_guide/validation`.

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

   Brier decomposition
      The split of the Brier score of probabilities, by groups such as grades, into the reliability, how far the mean probability of each group lies from its event rate, the resolution, how far the event rates of the groups lie from the overall event rate, and the uncertainty, the variance of the outcomes. The Brier score is the reliability minus the resolution plus the uncertainty, plus two terms for probabilities that vary within a group; see :doc:`/user_guide/calibration`.

   Calibration sample
      The sample on which the PDs of a model are brought to their central tendency, such as the current portfolio. The EBA guidelines ask that it be comparable to the portfolio that the model rates; see :doc:`/user_guide/calibration`.

   Calibration segment
      The part of a portfolio whose PDs are calibrated to one long-run average default rate, such as the exposures of one rating system; the calibration of each grade within it is tested as well; see :doc:`/user_guide/calibration`.

   Candidate rule
      A policy rule that is considered but not yet in force, such as one that an analyst proposes or a threshold that :func:`~kivi.rules.rule_candidates` finds in the data. Its event rate can be measured on the applicants that the current policy approved, which shows what adding it would change, and a threshold found on the development sample is checked again on a validation sample; see :doc:`/user_guide/rules`.

   Capital requirement (K)
      The unexpected loss of an exposure as a share of its EAD: the loss in the economy that is worse only once in a thousand years, minus the expected loss. Its risk weight is 12.5 K, and its risk-weighted assets are 12.5 K times the EAD; see :doc:`/user_guide/capital`.

   Category grouping
      A mapping from each category of a categorical variable to a group, fitted on a reference sample by pooling rare categories or by merging categories with similar event rates, and applied unchanged to every other sample; see :ref:`category-groups`.

   Censoring
      Losing sight of an account before its outcome is known, as when the data end while a loan is still on book. A censored account counts among the accounts at risk up to its last period, and its later outcome stays unknown; see :doc:`/user_guide/lifetime`.

   Central tendency
      The mean PD to which a model is calibrated, such as the long-run average default rate of its calibration segment, or the event rate of a new sample; see :doc:`/user_guide/calibration`.

   Characteristic stability index (CSI)
      The change of a variable's average scorecard points from a reference sample to a current one: the change of each group's share times the group's points, summed over the groups. Over all the variables of a scorecard, the CSI values add up to the change of the average score; see :ref:`characteristic-stability`.

   ChiMerge
      Supervised binning that starts from many small bins and repeatedly merges the adjacent pair with the smallest chi-square statistic, the pair whose event rates are most alike, until no more than a given number of bins remain and every adjacent pair differs significantly. Categories are merged in the same way once sorted by their WOE; see :doc:`/user_guide/binning`.

   Closed workout
      A workout whose recovery process has ended, by the bank's definition of its end, with a cure, the sale of the collateral or the write-off of what was not recovered. Only a closed workout has a realized LGD, and back-tests use closed workouts only; see :doc:`/user_guide/lgd`.

   Coefficient of variation (CV)
      The standard deviation of values divided by their mean, as in the ``cv`` of :func:`~kivi.univariate.profile_table`. For the shares :math:`R_i` of :math:`K` grades it is :math:`\sqrt{K \sum_i (R_i - 1/K)^2}`, 0 for equally full grades and :math:`\sqrt{K - 1}` when one grade holds everybody, and the European Central Bank tests whether it has risen since the initial validation; see :doc:`/user_guide/calibration`.

   Cohort
      The loans originated in the same period, usually a month, followed together as they age in a vintage table. The cohort approach to one-year default rates groups loans by a start month instead; see :doc:`/user_guide/vintage`.

   Cohort approach
      The count of the one-year default rate behind the PDs of an IRB model: the obligors not in default at a start month, by their grade at that month, and of them those that default at least once in the following year, including those that leave the portfolio during it; see :doc:`/user_guide/vintage`.

   Competing risk
      An exit that ends the risk of default without being a default, such as an early repayment. Counting it as censored, as 1 − Kaplan–Meier does, overstates the share of the accounts that default; see :doc:`/user_guide/lifetime`.

   Conditional PD
      The probability of a default in a period given that there was none before it, also called the hazard rate; see :doc:`/user_guide/lifetime`. With competing exits, the ``hazard`` of :func:`~kivi.lifetime.survival_table` is conditional on the account being still on book. In the one-factor model, :func:`~kivi.capital.conditional_pd` gives the PD conditional on the systematic factor instead; see :doc:`/user_guide/capital`.

   Consistency index (CI)
      How far the judgments of a pairwise comparison matrix of :math:`n` factors contradict each other: :math:`(\lambda_{\max} - n) / (n - 1)`, where :math:`\lambda_{\max}` is the principal eigenvalue of the matrix. It is 0 when the judgments are consistent; see :doc:`/user_guide/expert`.

   Consistency ratio (CR)
      The consistency index of a pairwise comparison matrix divided by the random index for the same number of factors. Saaty's rule of thumb asks for at most 0.1 before the weights are used; see :doc:`/user_guide/expert`.

   Coverage ratio
      The loss allowance of a group of loans divided by their exposure; see :doc:`/user_guide/lifetime`.

   Credit conversion factor (CCF)
      The share of the undrawn amount of a credit line that a borrower draws by the time of default, so that the exposure at default is the drawn amount plus the CCF times the undrawn amount. The realized CCF of a defaulted line compares the amount drawn at default with the limit and the drawn amount at the reference date; see :doc:`/user_guide/lgd`.

   Credit-impaired
      IFRS 9 calls a loan credit-impaired once events with a detrimental effect on its expected cash flows have occurred, such as significant financial difficulty of the borrower or a breach of contract like a default or a past-due event. Lenders usually align it with their definition of default, and such loans are in stage 3; see :doc:`/user_guide/lifetime`.

   Cumulative incidence
      The probability of a default by the end of a period when other exits compete with it: the share of all accounts that default by then, estimated by Aalen–Johansen. For a cohort observed from origination to a common end, it is the cumulative bad rate of its vintage curve; see :doc:`/user_guide/lifetime`.

   Cumulative PD
      The probability of a default in any of the periods from the first up to a given one, which never falls as the horizon grows; see :doc:`/user_guide/lifetime`.

   Cure
      The return of a defaulted loan to performing status. The EBA's guidelines count it as a recovery of the amount outstanding on the date of the cure; see :doc:`/user_guide/lgd`.

   Cut-off
      The score that separates approval from decline. kivi approves the scores at or below the cut-off, because its scores rise with the risk; see :doc:`/user_guide/strategy`.

   Days past due (DPD)
      The number of days for which a contractual payment has been overdue. IFRS 9 rebuttably presumes a significant increase in credit risk when a payment is more than 30 days past due, and default no later than when it is 90 days past due; see :doc:`/user_guide/lifetime`, and :doc:`/user_guide/vintage` for how they are counted from a repayment schedule.

   Decision waterfall
      A set of policy rules applied in order, each declining the applicants it hits among those that the rules before it left approved, so that every decline counts once, under the first rule that hits it. The last row of :func:`~kivi.rules.rule_set_table` describes the book that the whole set approves, whatever the order; see :doc:`/user_guide/rules`.

   Delinquency bucket
      A range of days past due that the status of a loan falls into at a month end, such as ``current``, ``1-29``, ``30-59``, ``60-89`` and ``90+``. Roll rates and transition matrices count how loans move between buckets from one month end to the next; see :doc:`/user_guide/vintage`.

   DeLong test
      A test of whether two scores have the same AUC on the same sample. It uses the variances and the covariance of the two AUCs that DeLong, DeLong and Clarke-Pearson (1988) derived from the placement values; see :doc:`/user_guide/metrics`.

   Deprecation
      A part of the public API that still works but warns with a ``FutureWarning``, because a later release removes or changes it. The warning says what to use instead; see :doc:`compatibility`.

   Discrete-time hazard model
      A logistic regression of the hazard rate on the period and on the characteristics of each account, fitted on a person-period table, which has one row for every period in which an account is at risk and a target of 1 in the period of its default; see :doc:`/user_guide/lifetime`.

   Early repayment
      The repayment of the whole balance of a loan before its last instalment is due, also called prepayment. It ends the risk of default, so a survival table counts it as a competing risk; ``prepayment`` gives its monthly probability in :func:`~kivi.datasets.make_loan_panel`; see :doc:`/user_guide/lifetime`.

   Economic loss
      The loss of a workout as the regulations define it: the exposure at default plus the material direct and indirect costs, minus the recoveries, all discounted to the date of default, rather than the accounting loss; see :doc:`/user_guide/lgd`.

   Economic scenario
      A forecast of the economy, such as an upside, a base case or a downturn, with its probability. IFRS 9 weights the ECL of each scenario with its probability, and the one-factor model moves the PDs to a scenario through the systematic factor; see :doc:`/user_guide/lifetime`.

   Effective interest rate
      The rate that discounts the estimated future cash flows of a loan over its expected life to its gross carrying amount at initial recognition. IFRS 9 discounts expected credit losses at this rate; see :doc:`/user_guide/lifetime`.

   Entity
      The customer, account or loan that events, monthly statuses and observations belong to, named by the ``id`` column of :mod:`kivi.features`; see :doc:`/user_guide/features`.

   Estimated and realized values
      The estimates of a model, such as the LGD of a facility, and the values observed later, such as the loss realized when its recovery process closed. kivi takes them as ``estimated`` and ``realized``: continuous values in :func:`~kivi.calibration.backtest_table` and ordered segments in :func:`~kivi.metrics.generalized_auc`; see :doc:`/user_guide/validation`.

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

   Expected loss best estimate (ELBE)
      The best estimate of the loss of an exposure that is already in default, given the economic conditions at the time. The European Central Bank back-tests it against the realized losses with a two-sided t-test; see :doc:`/user_guide/validation`.

   Expected maximum profit (EMP)
      The profit per applicant, as a share of the loan amount, of the best cut-off of a credit score, compared with approving everyone and averaged over the uncertain loss of a default: none with one probability, all of the loan with another, and evenly spread in between. Proposed by Verbraken, Bravo, Weber and Baesens; see :doc:`/user_guide/metrics`.

   Expert scorecard
      A scorecard whose weights are set by the judgment of credit analysts rather than fitted to outcomes, for portfolios with too few defaults to fit a model; see :doc:`/user_guide/expert`.

   Exposure at default (EAD)
      The amount owed when a borrower defaults, such as the outstanding balance of a loan; see :doc:`/user_guide/lgd`.

   Feature specification
      The definitions of the window and repayment features of a model: the columns of the entities and of the observation times, groups of features with their tables, windows and aggregations, and ratios. Saved as versioned JSON, it lets production compute the same features as development. Unlike a binning specification, it is not fitted to data; see :doc:`/user_guide/features`.

   Format version
      The version of the JSON format of a binning or feature specification, stored in each file. A release reads every format version up to its own and rejects a newer one; see :doc:`compatibility`.

   Fractional logit
      A logistic model of the mean of a response between 0 and 1, such as an LGD, fitted by the quasi-likelihood of Papke and Wooldridge, with robust standard errors; see :doc:`/user_guide/lgd`.

   Gains table
      A table of score bands from the riskiest to the safest, with the event rate and lift of each band and the cumulative shares of events and non-events; see :doc:`/user_guide/metrics`.

   Generalized AUC (gAUC)
      The AUC extended to ordered outcomes that are not binary, such as realized LGDs cut into segments: of two observations in different estimated segments, the probability that the one estimated higher also realized more, with ties counting as one half. It is Somers' D plus one, halved; see :doc:`/user_guide/metrics`.

   Gini coefficient
      Twice the AUC minus 1: 0 for a score unrelated to the target and 1 for a score that ranks every event above every non-event. Scorecard development reports it more often than the AUC itself, and rating validation calls it the accuracy ratio; see :doc:`/user_guide/metrics`.

   Global weight
      The weight of a sub-factor in a whole expert scorecard: the weight of its factor times its weight among the sub-factors of that factor. The global weights of all sub-factors sum to 1; see :doc:`/user_guide/expert`.

   Grace period
      The months after the origination of a loan in which nothing falls due, so that its first instalment falls due later than a month after origination. :func:`~kivi.vintage.delinquency_panel` puts a loan in the cohort of the month before its first due date unless ``origination`` gives the origination dates, which keep the loans with a grace period in the cohort of their origination month; see :doc:`/user_guide/vintage`.

   Grade
      One class of a rating system or master scale, with a PD of its own and bounds that decide which predicted PDs fall into it. The functions that follow migrations take the grades of a rating system from the best to the worst as ``grades``; see :doc:`/user_guide/calibration`.

   Herfindahl-Hirschman index (HHI)
      The sum of the squared shares of the grades of a rating scale: :math:`1/K` when :math:`K` grades are equally full and 1 when one grade holds everybody. The European Central Bank measures the concentration with the HI, :math:`1 + \ln \mathrm{HHI} / \ln K`, which runs from 0 to 1 instead; see :doc:`/user_guide/calibration`.

   High-volatility commercial real estate (HVCRE)
      Commercial real estate lending whose repayment depends on the property and whose losses vary strongly with the economy. Its asset correlation falls from 0.30 to 0.12 as the PD rises, with 0.30 in place of the 0.24 of other corporate exposures; see :doc:`/user_guide/capital`.

   Hit rate
      The share of applicants that a policy rule hits, that is declines or flags; see :doc:`/user_guide/rules`.

   Information value (IV)
      The sum of the differences between the event share and the non-event share of each group, each times the group's WOE. It measures how strongly a grouped variable separates events from non-events on its own; see :doc:`/user_guide/woe_iv`.

   Initial validation
      The validation of a rating model before its first use. Later validations compare their AUC, gAUC and concentration of grades with those of the initial validation, which they take as known; see :doc:`/user_guide/validation`.

   Internal ratings-based approach (IRB)
      The approach of the Basel framework and of China's capital rules under which a bank sets its capital for credit risk from its own estimates of the PD, and for retail exposures and in the advanced approach also of the LGD and the EAD, through the risk-weight functions of the rules; see :doc:`/user_guide/capital`.

   Kaplan–Meier estimator
      The estimate of the probability of getting through a series of periods as the product of the shares of the accounts at risk that get through each one. One minus it is the cumulative PD when every exit other than a default counts as censored; see :doc:`/user_guide/lifetime`.

   Kolmogorov-Smirnov statistic (KS)
      The largest absolute gap between the distribution functions of the scores of events and of non-events; see :doc:`/user_guide/metrics`.

   Left truncation
      Observing accounts only from some age on, as in data that start after the loans were originated, so that those that defaulted or left before are missing. The ``entry`` of :func:`~kivi.lifetime.survival_table` keeps each account out of the risk sets of the periods before it was observed; see :doc:`/user_guide/lifetime`.

   Lift
      The event rate of a group, such as a score band or the applicants that a policy rule hits, divided by the event rate of the whole sample. A lift above 1 marks a group that is riskier than average; see :doc:`/user_guide/metrics`.

   Long-run average default rate
      The average of the one-year default rates over a historical observation period of at least five years that reflects the likely range of default rates, with good and bad years. The PDs of an IRB model are calibrated to it, and the lender adjusts it upwards where bad years are missing; see :doc:`/user_guide/vintage`.

   Long-run average LGD
      The default-weighted average LGD of all the defaults of a long period, each default counted once and negative LGDs counted as 0, with the open workouts included through an adjustment for their future recoveries. An LGD estimate may not fall below it; see :doc:`/user_guide/lgd`.

   Loss allowance
      The provision that IFRS 9 requires for the expected credit losses of a loan: the 12-month ECL in stage 1 and the lifetime ECL in stages 2 and 3; see :doc:`/user_guide/lifetime`.

   Loss given default (LGD)
      The share of the exposure that is lost when a borrower defaults, after recoveries and collateral; see :doc:`/user_guide/lgd`.

   Low-default portfolio
      A portfolio with so few defaults, such as banks, sovereigns or large corporates, that its default rates cannot calibrate a PD. The most prudent estimate bounds its PDs instead; see :doc:`/user_guide/capital`.

   Marginal PD
      The probability, seen from the starting point, of a default in one given period. The marginal PDs of the periods up to a horizon add up to the cumulative PD; see :doc:`/user_guide/lifetime`.

   Master scale
      A fixed set of grades that divides the PD range from 0 to 1, so that ratings from different models share one meaning; see :doc:`/user_guide/calibration`.

   Matrix weighted bandwidth (MWB)
      A summary of how far customers move between the grades of a rating system in a period: the distances of the moves to worse grades, for the upper MWB, or to better grades, for the lower MWB, relative to the furthest move possible from each customer's grade. Both run from 0 to 1, and smaller values mean shorter migrations; see :doc:`/user_guide/vintage`.

   Maturity adjustment
      The factor :math:`(1 + (M - 2.5)\,b) / (1 - 1.5\,b)` of the corporate risk-weight function, with the effective maturity :math:`M` in years and :math:`b` a function of the PD, which raises the capital of longer exposures. Retail exposures have none; see :doc:`/user_guide/capital`.

   Migration matrix
      The transition matrix of the grades of a rating system over a period, usually a year: the share of the customers of each grade at the beginning that are in each grade at the end. Customers who defaulted or left count among the customers of their grade at the beginning, so the shares of a row may add up to less than 1; see :doc:`/user_guide/vintage`.

   Monitoring report
      A table with one row per period that tracks a model in use: its population, the known outcomes, its discrimination and the stability of its score against a reference sample; see :doc:`/user_guide/monitoring`.

   Monotonic binning
      Supervised binning whose WOE rises or falls steadily from the lowest bin to the highest, so that the bins can be explained. kivi starts from many small bins and merges the adjacent bins that break the trend, pooling adjacent violators; see :doc:`/user_guide/binning`.

   Month on book
      The number of months since a loan was originated. A vintage table compares cohorts at the same month on book; see :doc:`/user_guide/vintage`.

   Monthly status panel
      A table with one row per loan and month and the status of the loan at the month end, such as its delinquency bucket. :func:`~kivi.vintage.delinquency_panel` builds one from repayment schedules and payments, and :func:`~kivi.features.panel_features` summarizes it before each observation time; see :doc:`/user_guide/vintage`.

   Most prudent estimate
      Pluto and Tasche's upper bound for the PD of a grade: the largest PD that the defaults of the grade and of all riskier grades do not reject at a confidence level, assuming that the PD does not fall from one grade to the next riskier one; see :doc:`/user_guide/capital`.

   New hits
      The applicants that a rule of a decision waterfall hits and no earlier rule did, the declines that the waterfall credits to it. They range from its unique hits, when the rule comes last, to all its hits, when it comes first; see :doc:`/user_guide/rules`.

   Observation
      In :mod:`kivi.features`, an entity and an observation time, such as an applicant and the date of the application: one row of the table ``observations``, for which every window is computed. An entity may have several observations, such as one per month end; see :doc:`/user_guide/features`.

   Observation month
      The calendar month of an observation time. Its month-end status is not known at the observation time, so :func:`~kivi.features.panel_features` uses only the months before it; see :doc:`/user_guide/features`.

   Observation time
      The time at which a feature is computed, such as the date of an application. Events at or after it are not used, unless ``closed="right"`` admits those at that very time; see :doc:`/user_guide/features`.

   Observed average default rate
      The arithmetic mean of the one-year default rates of the start months of a historical observation period, for a grade or for a calibration segment, each start month counted once however many obligors it holds; see :doc:`/user_guide/vintage`.

   One-factor model
      Vasicek's model of portfolio credit risk, behind the IRB risk-weight functions: a borrower defaults when its assets, a mix of a systematic factor that all borrowers share and a factor of its own, fall below a threshold that its PD sets. Given the factor, defaults are independent; see :doc:`/user_guide/capital`.

   One-year default rate
      The share of the obligors not in default at a start month that default at least once in the twelve months after it, counted by their grade at the start month; see :doc:`/user_guide/vintage`.

   Optimal binning
      Supervised binning that returns, of all the ways to merge fine bins, the one with the largest information value that meets constraints at once, such as a minimum share and a minimum number of events in every bin and a shape of the WOE. The IV is a sum over the bins, so dynamic programming finds it exactly; see :doc:`/user_guide/binning`.

   Outcome window
      The time after which the outcome of an observation is known, such as twelve months after a loan was granted. Until it has passed, the target is missing and the outcome counts as unknown; see :doc:`/user_guide/monitoring`.

   Output floor
      The rule that the risk-weighted assets of the IRB approach may not fall below 72.5% of those of the standardized approach. It applies to the total risk-weighted assets of a bank and is phased in over several years, and kivi leaves it out; see :doc:`/user_guide/capital`.

   Outstanding amount
      The unpaid amount of all the instalments of a loan at a month end, whether they have fallen due or not: the ``outstanding`` column of :func:`~kivi.vintage.delinquency_panel`. It includes interest, while the ``balance`` column, given the principal of each instalment, counts only the unpaid principal; see :doc:`/user_guide/vintage`.

   Overdue amount
      The unpaid amount of the instalments of a loan that fell due before a month end: the ``overdue_amount`` column of :func:`~kivi.vintage.delinquency_panel`. An instalment due on the month end itself is not overdue yet; see :doc:`/user_guide/vintage`.

   Overlap matrix
      The number of applicants that both rules of each pair hit, with the hits of each rule on the diagonal; see :doc:`/user_guide/rules`.

   Pairwise comparison matrix
      A square matrix whose entry :math:`a_{ij}` says how many times more important factor :math:`i` is than factor :math:`j`, on the Saaty scale. Its diagonal is 1 and :math:`a_{ji} = 1 / a_{ij}`; see :doc:`/user_guide/expert`.

   PD term structure
      The PD of each period after a starting point, such as the origination of a loan or the reporting date, given as cumulative, marginal or conditional PDs; see :doc:`/user_guide/lifetime`.

   Permutation test
      A test that computes a statistic again on many shuffles of the target, which break any link between the target and the variable, refitting every step that used the target, such as supervised bins. The statistic of the real target is significant when few shuffles reach it; see :doc:`/user_guide/woe_iv`.

   Placement value
      For an event, the share of non-events that score below it, and for a non-event, the share of events that score above it, with ties counted as one half. The average over either class is the AUC, and their variances give its DeLong standard error; see :doc:`/user_guide/metrics`.

   Point in time
      A feature is point in time when it uses only what was known at its observation time, so that it is computed in development as it would have been at the time of the decision, and no later event leaks into the model; see :doc:`/user_guide/features`.

   Points to double the odds (PDO)
      The number of scorecard points by which the score rises when the good:bad odds double. Together with a base score and the odds at that score, it sets the scale of a scorecard; see :doc:`/user_guide/scorecard`.

   Policy rule
      A condition on the data of an applicant, such as a debt-to-income ratio above a cap or an age below a minimum, that declines or refers the applicant whatever the score; see :doc:`/user_guide/rules`.

   Population stability index (PSI)
      A measure of how much the distribution of a variable or a score has shifted between a reference sample and a current one, summed over the groups from the change of each group's share; see :doc:`/user_guide/psi`.

   Precision
      The share of events among the observations that a score flags at a threshold, that is, the event rate of the flagged observations; see :doc:`/user_guide/metrics`.

   Precision-recall curve
      The precision against the recall at every threshold of a score, drawn as steps from one threshold to the next. It shows how the share of events among the flagged observations falls as more of the events are found, which matters most when events are rare; see :doc:`/user_guide/plotting`.

   Probability of default (PD)
      The probability that a borrower defaults within a given horizon, usually twelve months. kivi calls a predicted PD, or any other predicted event probability, ``probability``; see :doc:`/user_guide/calibration`.

   Qualifying revolving retail exposure (QRRE)
      Revolving, unsecured retail credit such as credit cards and overdrafts, within limits on the size of each exposure, which the IRB approach weights with a fixed asset correlation of 0.04; ``asset_class="revolving"`` in :mod:`kivi.capital`; see :doc:`/user_guide/capital`.

   Quasi moment matching
      Tasche's calibration of a logistic PD curve of a score to two targets: the mean PD equals a central tendency, and the accuracy ratio that the curve implies equals a target accuracy ratio, such as that of the development sample; see :doc:`/user_guide/calibration`.

   Random index (RI)
      The average consistency index of pairwise comparison matrices filled with random judgments on the Saaty scale, which depends only on the number of factors. kivi uses the table of Tone (1986) for 3 to 15 factors; see :doc:`/user_guide/expert`.

   Realized LGD
      The economic loss of a closed workout as a share of its exposure at default, with the cash flows discounted at the rate of its date of default. It is negative when the recoveries exceed the exposure and the costs, and above 1 when the costs exceed the recoveries; see :doc:`/user_guide/lgd`.

   Reason codes
      The variables that lowered an applicant's score most, compared with the best group of each variable, as given to a declined applicant; see :doc:`/user_guide/scorecard`.

   Recall
      The share of all events that a score flags at a threshold, or that a policy rule hits: the ``true_positive_rate`` of :func:`~kivi.metrics.roc_table` and the ``recall`` of :func:`~kivi.strategy.confusion_table` and :func:`~kivi.rules.rule_table`; see :doc:`/user_guide/metrics`.

   Reference date
      The date at which the limit and the drawn amount of a credit line are taken for its realized CCF, 12 months before default in the fixed-horizon approach of the regulations; see :doc:`/user_guide/lgd`.

   Reference sample
      The sample that later samples are compared with, usually the development sample or a period right after it. Bin edges, score bands and groups are fitted on it; see :doc:`/user_guide/monitoring`.

   Reject inference
      Methods that supply outcomes for rejected applicants, as rows with weights or as weights for the accepts, so that a model can be refitted on all applicants; see :doc:`/user_guide/reject_inference`.

   Repayment schedule
      The instalments that a loan owes, each with its due date and its amount. Compared with the payments made, it gives the days past due of the loan at each month end; see :doc:`/user_guide/vintage`.

   Reporting date
      The date at which a loss allowance is measured. Period 1 of an expected credit loss is the first period after it; see :doc:`/user_guide/lifetime`.

   Return on investment (ROI)
      In the expected maximum profit, the return that a good loan earns over its term as a share of the amount lent, which a cut-off gives up for each good applicant that it rejects; see :doc:`/user_guide/metrics`.

   Revenue and profit
      The revenue of an approved loan is what it earns, such as its interest margin and fees, and the profit of a cut-off is the revenue of the book that it approves less the losses of that book; see :doc:`/user_guide/strategy`.

   Risk set
      The accounts at risk of default in a period: already observed and with neither a default nor another exit before it. The accounts censored in the period still belong to it; see :doc:`/user_guide/lifetime`.

   Risk weight
      The capital requirement times 12.5, the share of an exposure that counts as risk-weighted assets (RWA), so that the minimum capital ratio of 8% of the RWA gives back the capital requirement; see :doc:`/user_guide/capital`.

   Robust standard error
      A standard error from the sandwich estimator, the inverse information times the sum of the outer products of the scores times the inverse information, which holds whatever the variance of the response, as for a fractional response; see :doc:`/user_guide/lgd`.

   ROC convex hull
      The upper convex hull of the ROC curve, the smallest concave curve on or above it. Whatever the loss of a default and the return of a good loan, the most profitable cut-off of a score lies on a vertex of the hull; see :doc:`/user_guide/metrics`.

   ROC curve
      The true positive rate, the share of events flagged, against the false positive rate, the share of non-events flagged, as the threshold moves through the scores from the riskiest down. The area under it is the AUC, and its largest vertical distance from the diagonal is the KS; see :doc:`/user_guide/metrics`.

   Roll rate
      The share of the loans in a delinquency bucket at one month end that are in the next worse bucket at the next month end; see :doc:`/user_guide/vintage`.

   Run
      A stretch of consecutive months with bad statuses, or with good ones. A month without a status ends a run, and a window counts only the months of a run that lie inside it; see :doc:`/user_guide/features`.

   Saaty scale
      The scale of pairwise comparisons from 1, equally important, through 3, 5 and 7 to 9, extremely more important, with 2, 4, 6 and 8 in between and the reciprocals for the reverse comparisons; see :doc:`/user_guide/expert`.

   Scaled likelihood ratio
      Tasche's name for the calibration that multiplies the odds of every PD by one constant, the same shift of every log odds, chosen so that the mean PD of the calibration sample equals the central tendency. It keeps the order of the PDs and never leaves [0, 1]; see :doc:`/user_guide/calibration`.

   Selection path
      The models fitted on the first 1, 2, 3 and more variables of a fixed order, such as the variables sorted by IV, with the measures of each model on the development sample and on a validation sample, which show where adding variables stops paying off; see :doc:`/user_guide/selection`.

   Significant increase in credit risk (SICR)
      The IFRS 9 criterion that moves a loan from stage 1 to stage 2: its risk of default has risen significantly since initial recognition. Each lender sets the test in its own policy, for example as relative and absolute increases of the PD; see :doc:`/user_guide/lifetime`.

   SME adjustment
      The reduction of the corporate asset correlation for small and medium-sized enterprises by up to 0.04, according to their annual sales within a range: 5 to 50 million euros in the Basel framework and 30 to 300 million yuan in China's rules; see :doc:`/user_guide/capital`.

   Smoothing
      The value that replaces an empty count before shares are computed, 0.5 by default, so that the WOE, the IV and the PSI of a group stay finite. Counts are reported as observed, and only the shares are smoothed; see :doc:`/user_guide/woe_iv`.

   Somers' D
      A rank correlation of an ordered outcome with ordered estimates: of the pairs of observations in different estimated segments, the share that the outcome orders the same way minus the share that it orders the other way. It runs from -1 to 1, and the generalized AUC is Somers' D plus one, halved; see :doc:`/user_guide/metrics`.

   Special codes
      Values that stand for a state rather than a measurement, such as -1 for "never contacted", 999 for "no record" or the label "unknown" of a categorical variable. Passed in ``special``, they are left out of the fit of the edges or of the grouping and each gets a group of its own; see :ref:`special-values`.

   Spiegelhalter test
      A test of the calibration of single probabilities, without groups: it standardizes the difference between the Brier score of the probabilities and the score that they would have if they were the true ones. For PDs below 1/2, a positive statistic means more defaults than the PDs predict and a negative one fewer; see :doc:`/user_guide/calibration`.

   Stage
      One of the three IFRS 9 buckets of a loan: stage 1 for loans without a significant increase in credit risk, with a 12-month ECL; stage 2 for loans with one, with a lifetime ECL; and stage 3 for credit-impaired loans, also with a lifetime ECL; see :doc:`/user_guide/lifetime`.

   Standard error and confidence interval
      The standard error measures how much an estimate, such as an AUC, would vary from sample to sample. A confidence interval at a level such as 95% is a range around the estimate that would cover the true value in that share of samples; see :doc:`/user_guide/metrics`.

   Start month
      The month at whose end a cohort of the cohort approach is formed; the defaults of the twelve months after it make its one-year default rates. The EBA guidelines ask for start months at least every quarter; see :doc:`/user_guide/vintage`.

   Stressed PD
      The PD of the one-factor model in a bad economy, such as the PD at the 0.1% quantile of the systematic factor, a ``level`` of 0.999, that the IRB capital requirement uses, or the PD of a downturn scenario; :func:`~kivi.capital.conditional_pd` gives it; see :doc:`/user_guide/capital`.

   Swap set
      The four groups of applicants that two scores, each with its cut-off, approve or decline. The swap-outs are approved only by the current score and the swap-ins only by the new one; see :doc:`/user_guide/strategy`.

   Systematic factor
      The state of the economy in the one-factor model, a standard normal variable that all borrowers share, positive in years better than the median and negative in worse ones. A default rate implies its value given the PD and the asset correlation; see :doc:`/user_guide/capital`.

   Test group
      A random share of the applicants that is approved despite a policy rule or the cut-off, so that the outcomes of the applicants the policy declines become known; see :doc:`/user_guide/rules`.

   Through-the-cycle PD (TTC)
      A PD that averages over the states of the economy, such as the long-run PD of the one-factor model, the mean of the yearly default rates. A point-in-time PD follows the economy of the moment instead, the PD given the systematic factor; this sense differs from the point-in-time features of :mod:`kivi.features`; see :doc:`/user_guide/capital`.

   Tolerance
      How far a value may miss a rule and still count as meeting it. :func:`~kivi.vintage.delinquency_panel` counts an instalment as paid once at most ``tolerance``, an amount, remains unpaid of it and the older instalments together, so that a rounding difference does not make a loan past due; see :doc:`/user_guide/vintage`. :func:`~kivi.expert.ahp_weights` and :func:`~kivi.expert.ahp_consistency` accept a pairwise comparison matrix whose diagonal is 1 and whose entries are reciprocal within the relative tolerance ``rtol``, so that 0.3333 passes for 1 / 3; see :doc:`/user_guide/expert`.

   Total
      The number of observations in a row of a table, missing values included, or their sum of weights. The ``count`` of :func:`~kivi.univariate.profile_table` is the number of values that are not missing instead, as in ``pandas.DataFrame.describe``; see :doc:`/user_guide/univariate`.

   Transition matrix
      A table that counts the loans or obligors that moved from each state to each other state between two dates, such as delinquency buckets or rating grades; see :doc:`/user_guide/vintage`.

   Unique hits
      The applicants that a policy rule hits and no other rule of the set does: the declines that rest on this rule alone, which removing it from the set would approve; see :doc:`/user_guide/rules`.

   Unseen category
      A category of a later sample that did not occur in the reference sample, so that its grouping or WOE table has no row for it. kivi raises an error for such a category unless an ``unseen`` policy says how to treat it; see :doc:`/user_guide/woe_iv`.

   Validation sample
      A sample left out of the development of a model, such as later applicants, on which the fitted model is measured, so that its measures do not reward a fit to the chance features of the development sample; see :doc:`/user_guide/metrics`.

   Variance inflation factor (VIF)
      How much the variance of a coefficient grows because its variable is correlated with the other variables: :math:`1 / (1 - R^2)` for the :math:`R^2` of the regression of the variable on the others, with an intercept. It is 1 without collinearity, and values above 5 or 10 are usually taken as too high; see :doc:`/user_guide/selection`.

   VIF elimination
      Dropping the variable with the largest VIF, computing the VIFs of the others again and repeating until no VIF exceeds a threshold, so that one collinear variable at a time is removed; see :doc:`/user_guide/selection`.

   Vintage table
      The cumulative ever-bad rate of each cohort by month on book; see :doc:`/user_guide/vintage`.

   Weight of evidence (WOE)
      The logarithm of a group's share of all events over its share of all non-events. A positive WOE marks a group that is relatively more common among events; see :doc:`/user_guide/woe_iv`.

   Window
      The period before an observation time :math:`t` over which a feature aggregates the events: :math:`[t - k, t)` for :math:`k` calendar months or days, or :math:`(t - k, t]` with ``closed="right"``. For monthly statuses, the :math:`k` months before the observation month; see :doc:`/user_guide/features`.

   Window pair
      Two bounds of the same unit, such as ``("6M", "3M")``, that make one window from the first to the second, here :math:`[t - 6M, t - 3M)`, the period before the most recent one. Its features are named with both bounds, as ``6m_3m``; see :doc:`/user_guide/features`.

   WOE trend
      The shape of the WOE over the bins of a numeric variable, from its lowest values to its highest: increasing, decreasing, U-shaped, an inverted U, irregular when it changes direction more than once, or flat. Monotonic binning makes it increasing or decreasing; see :doc:`/user_guide/woe_iv`.

   Workout
      The recovery process of a defaulted loan, from the default to the cure, the sale of the collateral or the write-off, whose recoveries and costs give the realized LGD; see :doc:`/user_guide/lgd`.
