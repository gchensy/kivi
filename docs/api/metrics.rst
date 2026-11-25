Model performance
=================

.. module:: kivi.metrics

Measure how well a score that rises with the event probability separates events from non-events. :func:`~kivi.metrics.roc_auc`, :func:`~kivi.metrics.gini` and :func:`~kivi.metrics.ks` summarize the ranking, :func:`~kivi.metrics.roc_table` lists the ROC curve, :func:`~kivi.metrics.roc_auc_interval` and :func:`~kivi.metrics.delong_test` add DeLong's standard errors, :func:`~kivi.metrics.generalized_auc` extends the AUC with its standard error to ordered outcomes such as realized LGD segments, :func:`~kivi.metrics.gains_table` breaks the ranking down by score band, :func:`~kivi.metrics.brier_score` compares probabilities with outcomes, and :func:`~kivi.metrics.expected_maximum_profit` gives the expected profit of the best cut-off of a credit score.

See :doc:`/user_guide/metrics` for the definitions, examples and how to read a gains table.

.. autosummary::

   roc_auc
   gini
   ks
   roc_table
   roc_auc_interval
   delong_test
   generalized_auc
   gains_table
   brier_score
   expected_maximum_profit

.. autofunction:: roc_auc

.. autofunction:: gini

.. autofunction:: ks

.. autofunction:: roc_table

.. autofunction:: roc_auc_interval

.. autofunction:: delong_test

.. autofunction:: generalized_auc

.. autofunction:: gains_table

.. autofunction:: brier_score

.. autofunction:: expected_maximum_profit
