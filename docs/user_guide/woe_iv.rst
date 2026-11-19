Weight of evidence and information value
========================================

``woe_table`` measures how each group separates a binary target. An event is target 1 and a non-event is target 0. Both target classes must be present, and the group and target series must share the same index.

Definitions
-----------

For group :math:`i`, let :math:`E_i` and :math:`N_i` be its numbers of events and non-events. Its shares of all events and of all non-events are

.. math::

   p_i = \frac{E_i}{\sum_j E_j}, \qquad q_i = \frac{N_i}{\sum_j N_j}.

The weight of evidence and the information value contribution of the group are

.. math::

   \mathrm{WOE}_i = \ln \frac{p_i}{q_i}, \qquad
   \mathrm{IV}_i = (p_i - q_i)\,\mathrm{WOE}_i, \qquad
   \mathrm{IV} = \sum_i \mathrm{IV}_i.

A positive WOE means that the group is relatively more common among events. Each IV contribution is zero or positive, because :math:`p_i - q_i` and :math:`\ln(p_i / q_i)` always share a sign.

A worked example
----------------

Two thirds of group ``a`` are events and two thirds of group ``b`` are non-events, so the two groups have opposite WOE values of :math:`\pm\ln 2`.

>>> import pandas as pd
>>> from kivi.woe import woe_table
>>> groups = pd.Series(["a", "a", "a", "b", "b", "b"])
>>> target = pd.Series([1, 1, 0, 1, 0, 0])
>>> result = woe_table(groups, target)
>>> result.round(4)
     non_event  event  non_event_share  event_share     woe     iv
bin
a            1      2           0.3333       0.6667  0.6931  0.231
b            2      1           0.6667       0.3333 -0.6931  0.231
>>> print(f"{result['iv'].sum():.4f}")
0.4621

Smoothing empty cells
---------------------

A group without events or without non-events would have an infinite WOE. Before the shares are computed, each zero count is replaced by the ``smoothing`` value :math:`s`:

.. math::

   \tilde E_i = \begin{cases} E_i & E_i > 0 \\ s & E_i = 0 \end{cases}, \qquad
   p_i = \frac{\tilde E_i}{\sum_j \tilde E_j},

and likewise for non-events. Smoothing replaces zero cells with 0.5 by default. Each column is then normalized separately. Counts in the result always report observations.

>>> groups = pd.Series(["a", "a", "b", "b", "c"])
>>> target = pd.Series([1, 0, 0, 0, 1])
>>> woe_table(groups, target).round(3)
     non_event  event  non_event_share  event_share    woe     iv
bin
a            1      1            0.286          0.4  0.336  0.038
b            2      0            0.571          0.2 -1.050  0.390
c            0      1            0.143          0.4  1.030  0.265

.. warning::

   Changing smoothing can materially change the result for sparse bins. Merge small bins, or compare several smoothing values, before relying on their WOE.

Reading the result
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Meaning
   * - ``non_event``, ``event``
     - Observed counts in each group.
   * - ``non_event_share``, ``event_share``
     - :math:`q_i` and :math:`p_i` after smoothing; each column sums to 1.
   * - ``woe``
     - :math:`\ln(p_i / q_i)`.
   * - ``iv``
     - The group's IV contribution. Sum the column for the variable's IV.

Read the per-bin contributions before interpreting a total. A group with few events can have a large WOE despite representing few customers.

.. note::

   A widely used rule of thumb reads an IV below 0.02 as not predictive, 0.02 to 0.1 as weak, 0.1 to 0.3 as medium and above 0.3 as strong. Treat these bands as a starting point, not as a test; a very high IV often points to a leak of the target into the variable.
