User guide
==========

The user guide follows the steps of a credit risk model in the order of a project: compute features of the applicants from their transactions and repayment histories, profile and bin the candidate variables, encode them with WOE, select the variables and fit the model, infer the outcomes of rejected applicants, scale the model to a scorecard, or weigh the factors of an expert scorecard where there are too few defaults to fit one, evaluate and calibrate it, estimate the loss given default and the exposure at default of its loans, set the cut-offs, save the binning for production, monitor the model period by period, validate it as supervisors ask of internal rating models, compute the regulatory capital that its PDs require, follow the loans that it approves from their repayment schedules, and estimate their lifetime PDs and expected credit loss under IFRS 9. :doc:`plotting` draws the tables of every step. :doc:`risk_workflow` runs the steps from the profile to the stability report on the bundled data, leaving out the features of event data and reject inference, and the :doc:`/tutorial/index` follows a whole project from the first applications to the loss allowance.

kivi keeps fitting and applying apart. Bin edges, category groupings, WOE tables, master scales and cut-offs are fitted on the development sample and then applied unchanged to every later sample, so that counts and rates stay comparable. The tables take the groups as they are, so category labels can be passed directly.

Every function follows the rules of :doc:`/api/conventions`, such as the order of its arguments and the meaning of ``weights``, and the :doc:`/api/glossary` defines the terms that these guides use.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Risk modeling workflow
      :link: risk_workflow
      :link-type: doc

      An end-to-end example: profile and screen the variables, bin and encode the shortlist, fit and scale a scorecard, evaluate and calibrate it, set a cut-off and report stability.

   .. grid-item-card:: Time-window features
      :link: features
      :link-type: doc

      Point-in-time features of transactions and other events over windows of months or days, repayment behaviour from monthly status panels, and feature specifications saved as versioned JSON.

   .. grid-item-card:: Univariate profile
      :link: univariate
      :link-type: doc

      One row per candidate: missing values, distribution and shape, event rates, the IV and chi-squared test of a quick binning, and the Gini and KS of numeric values.

   .. grid-item-card:: Binning
      :link: binning
      :link-type: doc

      Quantile, uniform and custom edges, special values, merging fine bins, supervised binning by ChiMerge, decision trees, Best-KS and monotonic WOE, grouping the categories of categorical variables, and missing values.

   .. grid-item-card:: Weight of evidence and information value
      :link: woe_iv
      :link-type: doc

      Definitions, the smoothing rule for empty cells, how to read the result, the trend of the WOE over the bins and how to encode new data.

   .. grid-item-card:: Variable selection
      :link: selection
      :link-type: doc

      Correlation filtering by Pearson or Spearman correlation, variance inflation factors and their elimination, logistic regression, stepwise selection with positive coefficients, and the selection path of nested models on a validation sample.

   .. grid-item-card:: Reject inference
      :link: reject_inference
      :link-type: doc

      Fuzzy augmentation, parceling and reweighting, to refit a model developed on approved applicants on all applicants.

   .. grid-item-card:: Scorecard
      :link: scorecard
      :link-type: doc

      PDO scaling, points per group, scores for new data and reason codes.

   .. grid-item-card:: Expert scorecards
      :link: expert
      :link-type: doc

      Weigh the factors of a judgment-based scorecard with the analytic hierarchy process, check the consistency of the comparisons and combine factor and sub-factor weights.

   .. grid-item-card:: Model performance
      :link: metrics
      :link-type: doc

      AUC, Gini, KS, gains and ROC tables, with the score oriented so that higher means riskier, the confidence interval and DeLong test of an AUC, and the generalized AUC of ordered outcomes such as LGDs.

   .. grid-item-card:: Calibration
      :link: calibration
      :link-type: doc

      Prior correction, master scales built from PDs or fitted to default rates, tests of each grade's PD and of single PDs, the Brier decomposition, grade concentration, back-tests of LGDs and CCFs, and Platt and isotonic recalibration.

   .. grid-item-card:: LGD and EAD
      :link: lgd
      :link-type: doc

      Realized LGDs from the cash flows of workouts with per-loan discount rates, realized CCFs of credit lines, LGD tables by segment, the fractional logit, and the rules of Basel, the EBA and China side by side.

   .. grid-item-card:: Credit strategy
      :link: strategy
      :link-type: doc

      Approval and bad rates of each cut-off, the confusion matrix, precision, recall and F1 of a cut-off, swap sets between two scores, and expected loss by grade.

   .. grid-item-card:: Binning specifications
      :link: spec
      :link-type: doc

      Bin a whole data frame at once, including category groupings, and save the edges and WOE tables as versioned JSON for production scoring.

   .. grid-item-card:: Population stability index
      :link: psi
      :link-type: doc

      Compare the distribution of two samples, including categories seen in only one of them, and measure each variable's effect on the score with CSI.

   .. grid-item-card:: Model monitoring
      :link: monitoring
      :link-type: doc

      A per-period report of the population, the known outcomes, AUC, Gini, KS and PSI of a score, and the PSI of every variable by period.

   .. grid-item-card:: Model validation
      :link: validation
      :link-type: doc

      The tests of the ECB's validation report for PD, LGD and CCF models: back-tests, the current against the initial AUC and gAUC, migrations between grades and their concentration.

   .. grid-item-card:: Regulatory capital
      :link: capital
      :link-type: doc

      The IRB risk-weight functions of the Basel framework and of China's 2023 rules side by side, stressed PDs and the systematic factor of the Vasicek model, and most prudent PDs for portfolios with few defaults.

   .. grid-item-card:: Vintage and roll-rate analysis
      :link: vintage
      :link-type: doc

      Monthly delinquency panels from repayment schedules, cumulative bad rates by cohort and month on book, roll rates and transition matrices between delinquency buckets, and the bandwidth and tests of rating migrations.

   .. grid-item-card:: Lifetime PD and expected credit loss
      :link: lifetime
      :link-type: doc

      PD term structures from vintage curves, transition matrices and survival tables with early repayments as a competing risk, 12-month and lifetime expected credit loss over economic scenarios, and IFRS 9 stages.

   .. grid-item-card:: Charts
      :link: plotting
      :link-type: doc

      Draw WOE tables, ROC, precision-recall and gains curves, the distributions of events and non-events, calibration, cut-offs, PSI, monitoring reports and vintage curves with matplotlib.

   .. grid-item-card:: Example data
      :link: datasets
      :link-type: doc

      The bundled bank marketing data, a synthetic loan panel for portfolio analysis, synthetic applications for reject inference and the account transactions before them, and synthetic defaulted loans with the cash flows of their workouts.

.. toctree::
   :hidden:
   :glob:

   risk_workflow
   features
   univariate
   binning
   woe_iv
   selection
   reject_inference
   scorecard
   expert
   metrics
   calibration
   lgd
   strategy
   spec
   psi
   monitoring
   validation
   capital
   vintage
   lifetime
   plotting
   *
