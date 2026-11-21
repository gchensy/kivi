:html_theme.sidebar_secondary.remove: true

.. rst-class:: kivi-home

Credit risk modeling with pandas
================================

.. container:: kivi-lead

   kivi is a pandas toolkit for credit risk and financial modeling. It profiles and bins the candidate variables, measures how well each bin separates a binary outcome such as default, selects the variables and fits a logistic scorecard, infers the outcomes of rejected applicants, scales the model to points, evaluates it and calibrates it on a master scale, prices the cut-offs and the expected loss, tracks how the population shifts once the model is live, and follows how the approved loans perform. Tables come back as pandas DataFrames with one row per bin, grade, cut-off or variable.

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
      from kivi.datasets import Dataset
      from kivi.stability import psi_table
      from kivi.woe import woe_table

      data = Dataset.bank_data()
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

   .. grid-item-card:: Univariate profile
      :link: user_guide/univariate
      :link-type: doc

      Missing values, distributions, event rates and a first IV ranking of every candidate in one table.

   .. grid-item-card:: Binning
      :link: user_guide/binning
      :link-type: doc

      Fine and coarse classing, special values and monotonic WOE for logistic regression scorecards.

   .. grid-item-card:: WOE and IV
      :link: user_guide/woe_iv
      :link-type: doc

      The WOE and IV of every bin, with smoothing that leaves the counts alone, and the WOE encoding of new data.

   .. grid-item-card:: Variable selection
      :link: user_guide/selection
      :link-type: doc

      Rank candidates by IV, drop correlated variables and select a logistic model whose coefficients all point the right way.

   .. grid-item-card:: Reject inference
      :link: user_guide/reject_inference
      :link-type: doc

      Infer the outcomes of declined applicants by fuzzy augmentation, parceling or reweighting, and refit the model on all applicants.

   .. grid-item-card:: Scorecard
      :link: user_guide/scorecard
      :link-type: doc

      Turn the model into points with PDO scaling, score new applications and explain each score with reason codes.

   .. grid-item-card:: Model performance
      :link: user_guide/metrics
      :link-type: doc

      AUC, Gini, KS, gains and lift on development and validation samples.

   .. grid-item-card:: Calibration
      :link: user_guide/calibration
      :link-type: doc

      Prior correction, master scales, binomial, Jeffreys and Hosmer-Lemeshow tests per grade, and Platt and isotonic recalibration.

   .. grid-item-card:: Credit strategy
      :link: user_guide/strategy
      :link-type: doc

      Approval and bad rates of every cut-off, swap sets between a current and a new score, and expected loss by grade.

   .. grid-item-card:: Binning specifications
      :link: user_guide/spec
      :link-type: doc

      Save the binning and WOE of every variable as versioned JSON, so production groups new data exactly as development did.

   .. grid-item-card:: Population stability
      :link: user_guide/psi
      :link-type: doc

      Compare live data with the development sample bin by bin, track PSI over time and measure each variable's effect on the score with CSI.

   .. grid-item-card:: Vintage and roll-rate analysis
      :link: user_guide/vintage
      :link-type: doc

      Vintage curves by origination cohort, roll rates and transition matrices between delinquency buckets.

How it works
------------

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: kivi-steps

   .. grid-item-card:: Profile and bin

      ``profile_table`` gives a first look at every candidate. ``bin_edges``, or a supervised method such as ``monotonic_edges``, fits edges on the development sample once, and every later sample is cut with them. The outer edges are infinite, so later values always fall into a bin.

   .. grid-item-card:: Encode and fit

      ``woe_table`` measures each bin against the target and ``woe_transform`` encodes it. ``stepwise`` and ``logistic_regression`` select the variables and fit the model, with ``weights`` for inferred rejects.

   .. grid-item-card:: Scale and evaluate

      ``points_table`` turns the model into points, and ``gains_table``, ``roc_auc`` and ``ks`` measure how well the score ranks applicants on a holdout sample.

   .. grid-item-card:: Calibrate

      ``master_scale`` and ``assign_grades`` group the PDs into grades, and ``calibration_table`` with the binomial and Jeffreys tests checks the PD of each grade.

   .. grid-item-card:: Decide

      ``cutoff_table`` shows what each cut-off approves and lets through, ``swap_set`` compares two scores, and ``expected_loss_table`` adds up the expected loss of each grade.

   .. grid-item-card:: Deploy and monitor

      ``BinningSpec`` saves the binning as JSON for production. ``psi_table`` and ``csi_table`` compare live data with development, and ``vintage_table`` follows the approved loans.

Documentation
-------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: getting_started/index
      :link-type: doc

      Install kivi and walk through a complete analysis of the bundled bank marketing data.

   .. grid-item-card:: User guide
      :link: user_guide/index
      :link-type: doc

      One page per step of a risk model, from the profile of the candidates to vintage analysis, with definitions, worked examples and how to read each result.

   .. grid-item-card:: API reference
      :link: api
      :link-type: doc

      Parameters, return values and examples for every public function.

   .. grid-item-card:: Release notes
      :link: release/index
      :link-type: doc

      What changed in each release and how to migrate code written for 0.2.x.

.. toctree::
   :hidden:

   getting_started/index
   user_guide/index
   api
   release/index
   development/index
