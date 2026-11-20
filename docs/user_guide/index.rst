User guide
==========

Analyze a binary response and track changes in a population with pandas. Use the same bin edges in development and monitoring so that counts remain comparable.

kivi keeps fitting and measuring apart. ``bin_edges`` fits edges on a reference sample, ``pandas.cut`` assigns every sample to those bins, and ``woe_table`` and ``psi_table`` summarize the resulting groups. The metric functions accept any grouping, so category labels can be passed directly.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Binning
      :link: binning
      :link-type: doc

      Quantile and uniform edges, tied cut points, special values, values outside the reference range and missing values.

   .. grid-item-card:: Weight of evidence and information value
      :link: woe_iv
      :link-type: doc

      Definitions, the smoothing rule for empty cells and how to read the result.

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

   binning
   woe_iv
   psi
   datasets
