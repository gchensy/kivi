Binning specifications
======================

.. module:: kivi.spec

Bin and encode every column of a data frame in one step. :func:`~kivi.spec.fit_spec` fits the binning and the WOE of each column, and the resulting :class:`~kivi.spec.BinningSpec` groups and encodes any other sample in the same way and saves itself as versioned JSON.

See :doc:`/user_guide/spec` for examples and the file format.

.. autosummary::

   fit_spec
   BinningSpec

.. autofunction:: fit_spec

.. autoclass:: BinningSpec
   :members: columns, cut, woe_transform, select, to_dict, from_dict, to_json, from_json
   :member-order: bysource
