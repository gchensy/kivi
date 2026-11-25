:html_theme.sidebar_secondary.remove: true

.. rst-class:: kivi-home

Credit risk modeling with pandas
================================

.. container:: kivi-lead

   kivi is a pandas toolkit for credit risk and financial modeling. It computes point-in-time features from transactions and repayment histories, profiles the candidate variables, bins them and groups their categories, measures how well each bin separates a binary outcome such as default, selects the variables and fits a logistic scorecard or weighs the factors of an expert scorecard, infers the outcomes of rejected applicants, scales the model to points, evaluates it and calibrates it on a master scale, estimates the loss given default and the exposure at default from workouts and credit lines, and prices the cut-offs and the expected loss. Once the model is live, it monitors the score period by period, validates the model with the tests that supervisors ask of internal rating models, computes the regulatory capital of its PDs, follows how the approved loans repay and perform, and estimates their lifetime PDs and expected credit loss under IFRS 9. Tables come back as pandas DataFrames with one row per bin, grade, cut-off, period or variable, and :mod:`kivi.plotting` draws them as charts.

.. container:: kivi-actions

   .. button-ref:: getting_started/quickstart
      :ref-type: doc
      :color: primary

   .. button-ref:: user_guide/risk_workflow
      :ref-type: doc
      :color: primary
      :outline:

   .. button-ref:: api
      :ref-type: doc
      :color: primary
      :outline:

.. code-block:: bash

   python -m pip install kivi

At a glance
-----------

.. container:: kivi-showcase

   .. testcode::

      import pandas as pd
      from kivi.binning import bin_edges
      from kivi.datasets import load_bank
      from kivi.stability import psi_table
      from kivi.woe import woe_table

      data = load_bank()
      reference, current = data.iloc[:3000], data.iloc[3000:]

      edges = bin_edges(reference["age"], bins=5)
      before = pd.cut(reference["age"], edges)
      after = pd.cut(current["age"], edges)

      evidence = woe_table(before, reference["target"])
      stability = psi_table(before, after)

.. plot::
   :alt: WOE chart for age with the share of customers and the event rate per bin
   :caption: The WOE chart for ``age``, drawn from ``evidence``.

   kp.figure_quickstart_woe()

Built for risk teams
--------------------

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Time-window features
      :link: user_guide/features
      :link-type: doc

      Transaction windows and repayment behaviour for every application, computed point in time from event tables and monthly status panels, with the definitions saved as versioned JSON.

   .. grid-item-card:: Univariate profile
      :link: user_guide/univariate
      :link-type: doc

      Missing values, distributions and their shape, event rates, a first IV ranking and a chi-squared test of every candidate in one table.

   .. grid-item-card:: Binning
      :link: user_guide/binning
      :link-type: doc

      Fine and coarse classing, special values, ChiMerge, decision-tree, Best-KS and monotonic binning, and grouped categories for logistic regression scorecards.

   .. grid-item-card:: WOE and IV
      :link: user_guide/woe_iv
      :link-type: doc

      The WOE and IV of every bin, with smoothing that leaves the counts alone, the trend of the WOE over the bins and the WOE encoding of new data.

   .. grid-item-card:: Variable selection
      :link: user_guide/selection
      :link-type: doc

      Rank candidates by IV, drop variables with a high Pearson or Spearman correlation or VIF, select a logistic model whose coefficients all point the right way and follow the AUC of nested models on a validation sample.

   .. grid-item-card:: Reject inference
      :link: user_guide/reject_inference
      :link-type: doc

      Infer the outcomes of declined applicants by fuzzy augmentation, parceling or reweighting, and refit the model on all applicants.

   .. grid-item-card:: Scorecard
      :link: user_guide/scorecard
      :link-type: doc

      Turn the model into points with PDO scaling, score new applications and explain each score with reason codes.

   .. grid-item-card:: Expert scorecards
      :link: user_guide/expert
      :link-type: doc

      Weigh the factors of a scorecard set by judgment with the analytic hierarchy process, and check that the pairwise comparisons agree.

   .. grid-item-card:: Model performance
      :link: user_guide/metrics
      :link-type: doc

      AUC, Gini, KS, gains and ROC tables on development and validation samples, with the confidence interval of an AUC, the DeLong test of two scores and the generalized AUC of LGD segments.

   .. grid-item-card:: Calibration
      :link: user_guide/calibration
      :link-type: doc

      Prior correction, master scales, binomial, Jeffreys and Hosmer-Lemeshow tests per grade, Spiegelhalter's test, the Brier decomposition, and Platt and isotonic recalibration.

   .. grid-item-card:: LGD and EAD
      :link: user_guide/lgd
      :link-type: doc

      Realized LGDs from the cash flows of workouts, realized CCFs of credit lines, LGD tables and the fractional logit, with the rules of Basel, the EBA and China.

   .. grid-item-card:: Credit strategy
      :link: user_guide/strategy
      :link-type: doc

      Approval and bad rates of every cut-off, its confusion matrix and F1, swap sets between a current and a new score, and expected loss by grade.

   .. grid-item-card:: Binning specifications
      :link: user_guide/spec
      :link-type: doc

      Save the binning and WOE of every variable as versioned JSON, so production groups new data exactly as development did.

   .. grid-item-card:: Population stability
      :link: user_guide/psi
      :link-type: doc

      Compare live data with the development sample bin by bin, track PSI over time and measure each variable's effect on the score with CSI.

   .. grid-item-card:: Model monitoring
      :link: user_guide/monitoring
      :link-type: doc

      One row per period with the population, the known outcomes, AUC, Gini, KS and the PSI of the score, and the PSI of every variable by period.

   .. grid-item-card:: Model validation
      :link: user_guide/validation
      :link-type: doc

      The tests of the ECB's validation report for PD, LGD and CCF models: back-tests, the current against the initial AUC and gAUC, rating migrations and the concentration of grades.

   .. grid-item-card:: Regulatory capital
      :link: user_guide/capital
      :link-type: doc

      The IRB capital requirement under the Basel framework and China's 2023 rules, stressed PDs of the Vasicek model and most prudent PDs of low-default portfolios.

   .. grid-item-card:: Vintage and roll-rate analysis
      :link: user_guide/vintage
      :link-type: doc

      Monthly delinquency panels from repayment schedules, vintage curves by origination cohort, roll rates and transition matrices between delinquency buckets, and the migrations between rating grades.

   .. grid-item-card:: Lifetime PD and expected credit loss
      :link: user_guide/lifetime
      :link-type: doc

      PD term structures from vintage curves, transition matrices and survival tables, 12-month and lifetime ECL over economic scenarios, and the three IFRS 9 stages.

   .. grid-item-card:: Charts
      :link: user_guide/plotting
      :link-type: doc

      WOE, ROC, precision-recall, gains, distribution, calibration, cut-off, PSI, monitoring and vintage charts drawn straight from kivi's tables with matplotlib.

How it works
------------

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: kivi-steps

   .. grid-item-card:: Derive, profile and bin

      ``window_features`` and ``panel_features`` turn transactions and monthly repayment statuses into candidates, using only what was known at each application date. ``profile_table`` gives a first look at every candidate. ``bin_edges``, or a supervised method such as ``monotonic_edges``, fits edges on the development sample once, and ``chimerge_groups`` groups the categories of a categorical variable. Every later sample is cut with the same edges and groups, and the outer edges are infinite, so later values always fall into a bin.

   .. grid-item-card:: Encode and fit

      ``woe_table`` measures each bin against the target, ``woe_trend`` names the shape of its WOE and ``woe_transform`` encodes it. ``vif_filter`` drops collinear variables, ``stepwise`` and ``logistic_regression`` select the variables and fit the model, with ``weights`` for inferred rejects, and ``selection_path`` shows on a validation sample where more variables stop paying off.

   .. grid-item-card:: Scale and evaluate

      ``points_table`` turns the model into points, and ``gains_table``, ``roc_auc`` and ``ks`` measure how well the score ranks applicants on a holdout sample. ``roc_auc_interval`` shows how precisely a sample measures the AUC.

   .. grid-item-card:: Calibrate and decide

      ``master_scale`` and ``assign_grades`` group the PDs into grades, and the binomial and Jeffreys tests check the PD of each grade. ``workout_lgd`` and ``conversion_factor`` give the realized LGDs and CCFs of defaulted loans, and ``lgd_table`` and ``fractional_regression`` model them. ``cutoff_table`` shows what each cut-off approves, ``confusion_table`` how many events it declines, and ``expected_loss_table`` adds up the expected loss of each grade. ``capital_requirement`` turns the PDs into the IRB capital requirement, and ``most_prudent_pd`` bounds the PDs of grades with too few defaults.

   .. grid-item-card:: Deploy, monitor and validate

      ``FeatureSpec`` and ``BinningSpec`` save the features and the binning as JSON for production. ``monitoring_table`` and ``stability_table`` compare every period with the development sample, and ``csi_table`` measures each variable's effect on the score. Once a year, ``backtest_table``, ``generalized_auc``, ``migration_test`` and ``concentration_test`` give the tests of the ECB's validation report.

   .. grid-item-card:: Follow the loans

      ``delinquency_panel`` builds the monthly delinquency of the approved loans from their repayment schedules and payments, and ``vintage_table`` and ``transition_matrix`` follow it. ``convert_pd``, ``transition_pd`` and ``survival_table`` turn the vintage curves, the transition matrices and the durations of the loans into PD term structures, ``conditional_pd`` shifts them to economic scenarios, and ``expected_credit_loss`` and ``assign_stages`` give the IFRS 9 loss allowance.

Documentation
-------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: getting_started/index
      :link-type: doc

      Install kivi and walk through a complete analysis of the bundled bank marketing data.

   .. grid-item-card:: Tutorial
      :link: tutorial/index
      :link-type: doc

      A credit scorecard project from the applications to the IFRS 9 loss allowance of the loans, in eleven chapters that use nearly every function of kivi.

   .. grid-item-card:: User guide
      :link: user_guide/index
      :link-type: doc

      One page per step of a risk model, from the features of the applicants to the expected credit loss of the approved loans, with definitions, worked examples and how to read each result.

   .. grid-item-card:: API reference
      :link: api
      :link-type: doc

      Parameters, return values and examples for every public function, the conventions they share, what a release may change and a glossary of the terms.

   .. grid-item-card:: Release notes
      :link: release/index
      :link-type: doc

      What changed in each release, and how to migrate code written for 0.9 or 0.2.x.

.. toctree::
   :hidden:

   getting_started/index
   tutorial/index
   user_guide/index
   api
   release/index
   development/index
