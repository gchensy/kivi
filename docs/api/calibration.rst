Calibration
===========

.. module:: kivi.calibration

Check and correct predicted probabilities. :func:`~kivi.calibration.prior_correction` and :func:`~kivi.calibration.corrected_intercept` correct them for the sampling of the development data, :func:`~kivi.calibration.master_scale`, :func:`~kivi.calibration.fit_grade_pd` and :func:`~kivi.calibration.assign_grades` build a master scale and assign its grades, :func:`~kivi.calibration.calibration_table` and the tests count and test the grades, and :func:`~kivi.calibration.platt_fit` and :func:`~kivi.calibration.isotonic_fit` recalibrate a score.

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
