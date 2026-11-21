Credit strategy
===============

.. module:: kivi.strategy

Turn scores and PDs into credit decisions. :func:`~kivi.strategy.cutoff_table` shows what each cut-off approves, :func:`~kivi.strategy.swap_set` compares the decisions of two scores, and :func:`~kivi.strategy.expected_loss` and :func:`~kivi.strategy.expected_loss_table` give the expected loss of each exposure and of each grade.

See :doc:`/user_guide/strategy` for examples on the bundled data.

.. autosummary::

   cutoff_table
   swap_set
   expected_loss
   expected_loss_table

.. autofunction:: cutoff_table

.. autofunction:: swap_set

.. autofunction:: expected_loss

.. autofunction:: expected_loss_table
