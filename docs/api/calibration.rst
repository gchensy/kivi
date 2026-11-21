Calibration
===========

.. module:: kivi.calibration

Check and correct predicted probabilities. The functions correct probabilities for the sampling of the development data, build a master scale and assign its grades, count and test the grades, and recalibrate a score with Platt scaling or isotonic regression.

See :doc:`/user_guide/calibration` for examples on the bundled data.

.. autosummary::

   prior_correction
   corrected_intercept
   calibration_table
   binomial_test
   jeffreys_test
   hosmer_lemeshow
   herfindahl_index
   master_scale
   fit_grade_pd
   assign_grades
   platt_fit
   platt_predict
   isotonic_fit
   isotonic_predict

.. autofunction:: prior_correction

.. autofunction:: corrected_intercept

.. autofunction:: calibration_table

.. autofunction:: binomial_test

.. autofunction:: jeffreys_test

.. autofunction:: hosmer_lemeshow

.. autofunction:: herfindahl_index

.. autofunction:: master_scale

.. autofunction:: fit_grade_pd

.. autofunction:: assign_grades

.. autofunction:: platt_fit

.. autofunction:: platt_predict

.. autofunction:: isotonic_fit

.. autofunction:: isotonic_predict
