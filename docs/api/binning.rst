Binning
=======

.. module:: kivi.binning

Fit bin edges on a reference sample and group every sample with the same edges. :func:`~kivi.binning.bin_edges` fits quantile or uniform edges from the values alone; :func:`~kivi.binning.chimerge_edges`, :func:`~kivi.binning.tree_edges` and :func:`~kivi.binning.monotonic_edges` also use a binary target. :func:`~kivi.binning.cut` groups a sample and gives each special code a group of its own.

For categorical variables, :func:`~kivi.binning.category_groups` pools rare categories and :func:`~kivi.binning.chimerge_groups` merges categories with similar event rates; :func:`~kivi.binning.group_categories` applies the resulting dict to a sample.

See :doc:`/user_guide/binning` for examples and how to choose the edges.

.. autosummary::

   bin_edges
   cut
   chimerge_edges
   tree_edges
   monotonic_edges
   category_groups
   chimerge_groups
   group_categories

.. autofunction:: bin_edges

.. autofunction:: cut

.. autofunction:: chimerge_edges

.. autofunction:: tree_edges

.. autofunction:: monotonic_edges

.. autofunction:: category_groups

.. autofunction:: chimerge_groups

.. autofunction:: group_categories
