Time-window features
====================

.. module:: kivi.features

Aggregate event data, such as transactions, and monthly repayment statuses over windows that end at an observation time, using only what is known then. :func:`~kivi.features.window_features` computes the number of events, sums, means, extremes and other aggregations of each window, :func:`~kivi.features.panel_features` the bad months, the longest runs of bad and good months and the worst status of a monthly panel, and :class:`~kivi.features.FeatureSpec` keeps the definitions of both, computes them again on any sample and saves them as versioned JSON.

See :doc:`/user_guide/features` for examples and the rules of the windows.

.. autosummary::

   window_features
   panel_features
   FeatureSpec

.. autofunction:: window_features

.. autofunction:: panel_features

.. autoclass:: FeatureSpec
   :members: columns, compute, to_dict, from_dict, to_json, from_json
   :member-order: bysource
