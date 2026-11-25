LGD and EAD
===========

.. module:: kivi.lgd

Measure and model the loss given default and the credit conversion factor of defaulted loans. :func:`~kivi.lgd.workout_lgd` discounts the recoveries and costs of each workout to the date of default, :func:`~kivi.lgd.conversion_factor` gives the realized CCF of a credit line, :func:`~kivi.lgd.lgd_table` summarizes realized LGDs by group and :func:`~kivi.lgd.fractional_regression` fits the fractional logit of an LGD on its drivers.

See :doc:`/user_guide/lgd` for examples and the rules of the Basel framework, the EBA and China's capital rules.

.. autosummary::

   workout_lgd
   conversion_factor
   lgd_table
   fractional_regression

.. autofunction:: workout_lgd

.. autofunction:: conversion_factor

.. autofunction:: lgd_table

.. autofunction:: fractional_regression
