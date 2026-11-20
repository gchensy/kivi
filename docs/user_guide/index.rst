User guide
==========

kivi covers the steps of credit risk modeling: binning, variable selection, scorecard scaling, model evaluation, calibration and monitoring. Use the same bin edges in development and monitoring so that counts remain comparable.

kivi keeps fitting and measuring apart. ``bin_edges`` fits edges on a reference sample, ``pandas.cut`` assigns every sample to those bins, and ``woe_table`` and ``psi_table`` summarize the resulting groups. The metric functions accept any grouping, so category labels can be passed directly.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Risk modeling workflow
      :link: risk_workflow
      :link-type: doc

      An end-to-end example: screen variables by IV, bin the shortlist, encode it with WOE, fit and scale a scorecard, evaluate and calibrate it, and report stability.

   .. grid-item-card:: Binning
      :link: binning
      :link-type: doc

      Quantile, uniform and custom edges, special values, merging fine bins, supervised binning and missing values.

   .. grid-item-card:: Weight of evidence and information value
      :link: woe_iv
      :link-type: doc

      Definitions, the smoothing rule for empty cells, how to read the result and how to encode new data.

   .. grid-item-card:: Population stability index
      :link: psi
      :link-type: doc

      Compare the distribution of two samples, including categories seen in only one of them.

   .. grid-item-card:: Variable selection
      :link: selection
      :link-type: doc

      Correlation filtering, variance inflation factors, logistic regression and stepwise selection with positive coefficients.

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

      Prior correction, master scales and statistical tests of each grade's PD.

   .. grid-item-card:: Example data
      :link: datasets
      :link-type: doc

      The bundled bank marketing data, its target and its license.

.. toctree::
   :hidden:
   :glob:

   risk_workflow
   binning
   woe_iv
   psi
   *
