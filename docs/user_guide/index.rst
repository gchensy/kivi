User guide
==========

The user guide follows the steps of a credit risk model in the order of a project: profile and bin the candidate variables, encode them with WOE, select the variables and fit the model, infer the outcomes of rejected applicants, scale the model to a scorecard, evaluate and calibrate it, set the cut-offs, save the binning for production, and monitor the population and the loans that the model approves. :doc:`risk_workflow` runs these steps on the bundled data from start to finish.

kivi keeps fitting and applying apart. Bin edges, WOE tables, master scales and cut-offs are fitted on the development sample and then applied unchanged to every later sample, so that counts and rates stay comparable. The tables take the groups as they are, so category labels can be passed directly.

Every function follows the rules of :doc:`/api/conventions`, such as the order of its arguments and the meaning of ``weights``, and the :doc:`/api/glossary` defines the terms that these guides use.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Risk modeling workflow
      :link: risk_workflow
      :link-type: doc

      An end-to-end example: profile and screen the variables, bin and encode the shortlist, fit and scale a scorecard, evaluate and calibrate it, set a cut-off and report stability.

   .. grid-item-card:: Univariate profile
      :link: univariate
      :link-type: doc

      One row per candidate: missing values, distribution, event rates and the IV of a quick binning.

   .. grid-item-card:: Binning
      :link: binning
      :link-type: doc

      Quantile, uniform and custom edges, special values, merging fine bins, supervised binning and missing values.

   .. grid-item-card:: Weight of evidence and information value
      :link: woe_iv
      :link-type: doc

      Definitions, the smoothing rule for empty cells, how to read the result and how to encode new data.

   .. grid-item-card:: Variable selection
      :link: selection
      :link-type: doc

      Correlation filtering, variance inflation factors, logistic regression and stepwise selection with positive coefficients.

   .. grid-item-card:: Reject inference
      :link: reject_inference
      :link-type: doc

      Fuzzy augmentation, parceling and reweighting, to refit a model developed on approved applicants on all applicants.

   .. grid-item-card:: Scorecard
      :link: scorecard
      :link-type: doc

      PDO scaling, points per group, scores for new data and reason codes.

   .. grid-item-card:: Model performance
      :link: metrics
      :link-type: doc

      AUC, Gini, KS and gains tables, with the score oriented so that higher means riskier.

   .. grid-item-card:: Calibration
      :link: calibration
      :link-type: doc

      Prior correction, master scales built from PDs or fitted to default rates, tests of each grade's PD, and Platt and isotonic recalibration.

   .. grid-item-card:: Credit strategy
      :link: strategy
      :link-type: doc

      Approval and bad rates of each cut-off, swap sets between two scores, and expected loss by grade.

   .. grid-item-card:: Binning specifications
      :link: spec
      :link-type: doc

      Bin a whole data frame at once and save the edges and WOE tables as versioned JSON for production scoring.

   .. grid-item-card:: Population stability index
      :link: psi
      :link-type: doc

      Compare the distribution of two samples, including categories seen in only one of them, and measure each variable's effect on the score with CSI.

   .. grid-item-card:: Vintage and roll-rate analysis
      :link: vintage
      :link-type: doc

      Cumulative bad rates by cohort and month on book, roll rates and transition matrices between delinquency buckets.

   .. grid-item-card:: Example data
      :link: datasets
      :link-type: doc

      The bundled bank marketing data, a synthetic loan panel for portfolio analysis and synthetic applications for reject inference.

.. toctree::
   :hidden:
   :glob:

   risk_workflow
   univariate
   binning
   woe_iv
   selection
   reject_inference
   scorecard
   metrics
   calibration
   strategy
   spec
   psi
   vintage
   *
