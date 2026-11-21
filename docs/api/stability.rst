Population stability index
==========================

.. module:: kivi.stability

Measure how far a distribution has shifted between two samples. :func:`~kivi.stability.psi_table` and :func:`~kivi.stability.psi` give the population stability index, and :func:`~kivi.stability.csi_table` gives the change of a scorecard's average points that the shift causes.

See :doc:`/user_guide/psi` for examples and the usual thresholds.

.. autosummary::

   psi_table
   psi
   csi_table

.. autofunction:: psi_table

.. autofunction:: psi

.. autofunction:: csi_table
