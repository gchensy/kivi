Tutorial
========

This tutorial follows one credit risk project with kivi from the first look at the data to the loss allowance of the loan book. A lender has decided its personal loan applications with a scorecard bought from a vendor. After a year, it builds a scorecard of its own, sets its cut-off, puts it into production, monitors it and measures the expected credit loss of its loans under IFRS 9. Each chapter is one step of the project, and together they use nearly every function of kivi.

The chapters build on each other, but each page runs on its own: a collapsed block at its top repeats the code of the earlier chapters that the page needs. The :doc:`/user_guide/index` explains each step in depth; the tutorial shows how the steps fit together and which decisions the project takes on the way.

The project
-----------

In 2024 the lender received 40,000 applications. Two policy rules declined some of them whatever their risk: a debt-to-income ratio above 0.6 or an age below 21. The vendor's scorecard decided the others and approved an application when its PD, ``incumbent_pd``, was below 15%. Whether a borrower defaults is known only for the approved applications.

The vendor's scorecard sees credit bureau data that the application form does not hold, but it ignores some of the application data, and the lender cannot explain its decisions to applicants. The project therefore builds a scorecard from the application data that keeps the vendor's PD as one of its characteristics, as lenders do with a credit bureau score. The new scorecard must rank applicants better than the vendor's PD alone, give a PD for every applicant, approved or not, and state the reasons for a decline. The applicants also hold current accounts with the lender, whose transactions the project turns into features for the next version of the scorecard.

The data
--------

The applications are simulated by :func:`~kivi.datasets.make_applications`, the transactions on the applicants' accounts by :func:`~kivi.datasets.make_transactions`, and the loans that the lender booked by :func:`~kivi.datasets.make_loan_panel`; :doc:`/user_guide/datasets` describes them. Because the data are simulated, the outcome of every applicant is known, also of the declined ones, and so is the risk factor from which the transactions are drawn. A real project never sees either. The tutorial uses them only in boxes like the one below, to show how close each estimate comes to the truth:

.. admonition:: Simulation check
   :class: kivi-check

   A box with this title uses information that a real project does not have. The steps of the project never depend on it.

The chapters
------------

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: kivi-steps

   .. grid-item-card:: Data and samples
      :link: data
      :link-type: doc

      Load the applications, split them into development and validation samples, and profile the candidate variables. ``profile_table``

   .. grid-item-card:: Transaction features
      :link: features
      :link-type: doc

      Derive features from the account transactions as of each application date, check that they are point in time, profile them and save their definitions. ``make_transactions`` · ``FeatureSpec`` · ``window_features`` · ``profile_table``

   .. grid-item-card:: Binning and WOE
      :link: binning
      :link-type: doc

      Compare the binning methods, group the categories, and fit every variable at once. ``bin_edges`` · ``chimerge_edges`` · ``tree_edges`` · ``monotonic_edges`` · ``chimerge_groups`` · ``group_categories`` · ``woe_table`` · ``fit_spec``

   .. grid-item-card:: Variable selection and the model
      :link: model
      :link-type: doc

      Encode the variables with WOE, check their correlations and fit the logistic regression on the approved applicants. ``woe_transform`` · ``correlation_filter`` · ``vif`` · ``stepwise`` · ``logistic_regression``

   .. grid-item-card:: Reject inference
      :link: reject_inference
      :link-type: doc

      Infer the outcomes of the declined applicants and refit the model on all applicants. ``fuzzy_augmentation`` · ``parceling`` · ``reweighting``

   .. grid-item-card:: The scorecard
      :link: scorecard
      :link-type: doc

      Scale the model to points, score the applicants and give the reasons for a low score. ``score_scaling`` · ``points_table`` · ``woe_to_points`` · ``score_to_pd`` · ``reason_codes``

   .. grid-item-card:: Model evaluation
      :link: evaluation
      :link-type: doc

      Measure the ranking on the validation sample and compare it with the vendor's PD. ``roc_auc`` · ``gini`` · ``ks`` · ``brier_score`` · ``roc_table`` · ``roc_auc_interval`` · ``delong_test`` · ``gains_table``

   .. grid-item-card:: Calibration and grades
      :link: calibration
      :link-type: doc

      Build a master scale, test the PD of each grade and diagnose the calibration. ``master_scale`` · ``assign_grades`` · ``calibration_table`` · ``jeffreys_test`` · ``binomial_test`` · ``hosmer_lemeshow`` · ``platt_fit`` · ``fit_grade_pd`` · ``herfindahl_index``

   .. grid-item-card:: Cut-off and strategy
      :link: strategy
      :link-type: doc

      Choose the cut-off, compare the new decisions with the old ones and add up the expected loss. ``cutoff_table`` · ``swap_set`` · ``expected_loss`` · ``expected_loss_table``

   .. grid-item-card:: Production and monitoring
      :link: monitoring
      :link-type: doc

      Save the binning for production, score a year of new applications and find the cause of a shift. ``BinningSpec`` · ``monitoring_table`` · ``stability_table`` · ``psi_table`` · ``csi_table``

   .. grid-item-card:: The loan book under IFRS 9
      :link: portfolio
      :link-type: doc

      Follow the booked loans, project their PDs and allocate them to stages with their loss allowance. ``vintage_table`` · ``roll_rates`` · ``transition_matrix`` · ``transition_pd`` · ``convert_pd`` · ``expected_credit_loss`` · ``significant_increase`` · ``assign_stages`` · ``stage_table`` · ``panel_features``

.. toctree::
   :hidden:
   :numbered: 1

   data
   features
   binning
   model
   reject_inference
   scorecard
   evaluation
   calibration
   strategy
   monitoring
   portfolio
