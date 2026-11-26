Calibration
===========

.. module:: kivi.calibration

Check and correct predicted probabilities. :func:`~kivi.calibration.prior_correction` and :func:`~kivi.calibration.corrected_intercept` correct them for the sampling of the development data, :func:`~kivi.calibration.central_tendency_shift` and :func:`~kivi.calibration.quasi_moment_fit` bring them to a long-run average default rate, :func:`~kivi.calibration.master_scale`, :func:`~kivi.calibration.fit_grade_pd` and :func:`~kivi.calibration.assign_grades` build a master scale and assign its grades, :func:`~kivi.calibration.calibration_table` and the tests count and test the grades, and :func:`~kivi.calibration.platt_fit` and :func:`~kivi.calibration.isotonic_fit` recalibrate a score. :func:`~kivi.calibration.spiegelhalter_test` and :func:`~kivi.calibration.brier_decomposition` check the probabilities of single observations, :func:`~kivi.calibration.concentration_test` tests the concentration of the grades, and :func:`~kivi.calibration.backtest_table` back-tests continuous estimates such as LGDs.

See :doc:`/user_guide/calibration` for examples on the bundled data, and :doc:`/user_guide/validation` for the tests of a validation report.

.. autosummary::

   prior_correction
   corrected_intercept
   central_tendency_shift
   quasi_moment_fit
   calibration_table
   binomial_test
   jeffreys_test
   hosmer_lemeshow
   spiegelhalter_test
   brier_decomposition
   herfindahl_index
   concentration_test
   backtest_table
   master_scale
   fit_grade_pd
   assign_grades
   platt_fit
   platt_predict
   isotonic_fit
   isotonic_predict

.. autofunction:: prior_correction

.. autofunction:: corrected_intercept

.. autofunction:: central_tendency_shift

.. autofunction:: quasi_moment_fit

.. autofunction:: calibration_table

.. autofunction:: binomial_test

.. autofunction:: jeffreys_test

.. autofunction:: hosmer_lemeshow

.. autofunction:: spiegelhalter_test

.. autofunction:: brier_decomposition

.. autofunction:: herfindahl_index

.. autofunction:: concentration_test

.. autofunction:: backtest_table

.. autofunction:: master_scale

.. autofunction:: fit_grade_pd

.. autofunction:: assign_grades

.. autofunction:: platt_fit

.. autofunction:: platt_predict

.. autofunction:: isotonic_fit

.. autofunction:: isotonic_predict
