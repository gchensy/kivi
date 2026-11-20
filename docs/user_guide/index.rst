User guide
==========

kivi covers the binning and stability steps of credit risk modeling: scorecard development, variable screening and model monitoring. Use the same bin edges in development and monitoring so that counts remain comparable.

kivi keeps fitting and measuring apart. ``bin_edges`` fits edges on a reference sample, ``pandas.cut`` assigns every sample to those bins, and ``woe_table`` and ``psi_table`` summarize the resulting groups. The metric functions accept any grouping, so category labels can be passed directly.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Risk modeling workflow
      :link: risk_workflow
      :link-type: doc

      An end-to-end example: screen variables by IV, bin the shortlist, encode it with WOE and report stability.

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
