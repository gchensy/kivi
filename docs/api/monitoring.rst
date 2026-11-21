Model monitoring
================

.. module:: kivi.monitoring

Track a score and its variables period by period. :func:`~kivi.monitoring.monitoring_table` gives the population, the events, the discrimination and the PSI of a score in every period, and :func:`~kivi.monitoring.stability_table` the PSI of every grouped variable.

See :doc:`/user_guide/monitoring` for examples on synthetic applications.

.. autosummary::

   monitoring_table
   stability_table

.. autofunction:: monitoring_table

.. autofunction:: stability_table
