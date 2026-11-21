Vintage and roll-rate analysis
==============================

.. module:: kivi.vintage

Analyse a loan panel as the loans age. :func:`~kivi.vintage.vintage_table` gives the cumulative bad rate of each cohort by month on book, and :func:`~kivi.vintage.transition_matrix` and :func:`~kivi.vintage.roll_rates` show how loans move between delinquency buckets or grades.

See :doc:`/user_guide/vintage` for examples on a synthetic loan panel.

.. autosummary::

   vintage_table
   transition_matrix
   roll_rates

.. autofunction:: vintage_table

.. autofunction:: transition_matrix

.. autofunction:: roll_rates
