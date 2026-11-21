Model performance
=================

.. module:: kivi.metrics

Measure how well a score that rises with the event probability separates events from non-events. :func:`~kivi.metrics.roc_auc`, :func:`~kivi.metrics.gini` and :func:`~kivi.metrics.ks` summarize the ranking, :func:`~kivi.metrics.roc_table` lists the ROC curve, :func:`~kivi.metrics.roc_auc_interval` and :func:`~kivi.metrics.delong_test` add DeLong's standard errors, :func:`~kivi.metrics.gains_table` breaks the ranking down by score band, and :func:`~kivi.metrics.brier_score` compares probabilities with outcomes.

See :doc:`/user_guide/metrics` for the definitions, examples and how to read a gains table.

.. autosummary::

   roc_auc
   gini
   ks
   roc_table
   roc_auc_interval
   delong_test
   gains_table
   brier_score

.. autofunction:: roc_auc

.. autofunction:: gini

.. autofunction:: ks

.. autofunction:: roc_table

.. autofunction:: roc_auc_interval

.. autofunction:: delong_test

.. autofunction:: gains_table

.. autofunction:: brier_score
