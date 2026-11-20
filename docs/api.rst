API reference
=============

The public functions take one-dimensional array-like inputs, such as lists, NumPy arrays or pandas Series, and return NumPy arrays or pandas DataFrames. Import them from their modules.

.. currentmodule:: kivi

.. autosummary::

   binning.bin_edges
   binning.cut
   binning.chimerge_edges
   binning.tree_edges
   binning.monotonic_edges
   woe.woe_table
   evaluate.psi_table
   datasets.Dataset.bank_data

Binning
-------

.. autofunction:: kivi.binning.bin_edges

.. autofunction:: kivi.binning.cut

.. autofunction:: kivi.binning.chimerge_edges

.. autofunction:: kivi.binning.tree_edges

.. autofunction:: kivi.binning.monotonic_edges

Weight of evidence
------------------

.. autofunction:: kivi.woe.woe_table

Population stability
--------------------

.. autofunction:: kivi.evaluate.psi_table

Example data
------------

.. automethod:: kivi.datasets.Dataset.bank_data
