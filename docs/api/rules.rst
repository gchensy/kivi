Policy rules
============

.. module:: kivi.rules

Measure the policy rules of a credit strategy. :func:`~kivi.rules.rule_table` counts the hits and the events of each rule of a set, and :func:`~kivi.rules.rule_set_table` applies the rules in order, as a decision waterfall, with the new hits of each rule and the book that the rules leave approved.

See :doc:`/user_guide/rules` for examples on the simulated applications, the overlap of two rules and the outcomes of declined applicants.

.. autosummary::

   rule_table
   rule_set_table

.. autofunction:: rule_table

.. autofunction:: rule_set_table
