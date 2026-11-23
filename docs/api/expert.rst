Expert scorecards
=================

.. module:: kivi.expert

Weigh the factors of an expert scorecard from pairwise comparisons, as the analytic hierarchy process does. :func:`~kivi.expert.ahp_weights` turns a matrix of comparisons into weights, :func:`~kivi.expert.ahp_consistency` measures how consistent the comparisons are, and :func:`~kivi.expert.combine_weights` joins the weights of factors and sub-factors into global weights.

See :doc:`/user_guide/expert` for a worked example and how to read the consistency ratio.

.. autosummary::

   ahp_weights
   ahp_consistency
   combine_weights

.. autofunction:: ahp_weights

.. autofunction:: ahp_consistency

.. autofunction:: combine_weights
