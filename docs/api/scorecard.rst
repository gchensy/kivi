Scorecard
=========

.. module:: kivi.scorecard

Turn a logistic regression on WOE values into points. :func:`~kivi.scorecard.score_scaling`, :func:`~kivi.scorecard.pd_to_score` and :func:`~kivi.scorecard.score_to_pd` set and apply the scale, :func:`~kivi.scorecard.points_table` and :func:`~kivi.scorecard.woe_to_points` give the points, and :func:`~kivi.scorecard.reason_codes` lists the variables that lowered a score most.

See :doc:`/user_guide/scorecard` for a scorecard built on the bundled data.

.. autosummary::

   score_scaling
   pd_to_score
   score_to_pd
   points_table
   woe_to_points
   reason_codes

.. autofunction:: score_scaling

.. autofunction:: pd_to_score

.. autofunction:: score_to_pd

.. autofunction:: points_table

.. autofunction:: woe_to_points

.. autofunction:: reason_codes
