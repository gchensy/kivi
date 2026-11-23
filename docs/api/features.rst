Time-window features
====================

.. module:: kivi.features

Aggregate event data, such as transactions, over windows that end at an observation time, using only the events before it. :func:`~kivi.features.window_features` computes the number of events, sums, means, extremes and other aggregations of each window, and :class:`~kivi.features.FeatureSpec` keeps the definitions of the features, computes them again on any sample and saves them as versioned JSON.

See :doc:`/user_guide/features` for examples and the rules of the windows.

.. autosummary::

   window_features
   FeatureSpec

.. autofunction:: window_features

.. autoclass:: FeatureSpec
   :members: columns, compute, to_dict, from_dict, to_json, from_json
   :member-order: bysource
