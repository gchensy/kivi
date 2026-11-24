Weight of evidence and information value
========================================

.. module:: kivi.woe

Measure how each group of a variable separates a binary target, and encode groups with the result. :func:`~kivi.woe.woe_table` gives the WOE and IV of each group and :func:`~kivi.woe.woe_transform` encodes new data with them; :func:`~kivi.woe.woe` and :func:`~kivi.woe.iv` compute the same measures from aggregated counts, and :func:`~kivi.woe.woe_trend` names the shape of the WOE over the bins of a numeric variable.

See :doc:`/user_guide/woe_iv` for the definitions, examples and how to read the result.

.. autosummary::

   woe_table
   woe_transform
   woe
   iv
   woe_trend

.. autofunction:: woe_table

.. autofunction:: woe_transform

.. autofunction:: woe

.. autofunction:: iv

.. autofunction:: woe_trend
