Population stability index
==========================

``psi_table`` compares the distribution of a variable in two samples, typically the development sample and a later monitoring sample. Group both samples with the same edges or labels before calling it.

Definition
----------

For group :math:`i`, let :math:`q_i` be its share of the reference sample and :math:`p_i` its share of the current sample. Then

.. math::

   \mathrm{PSI}_i = (p_i - q_i) \ln \frac{p_i}{q_i}, \qquad
   \mathrm{PSI} = \sum_i \mathrm{PSI}_i.

Each contribution is zero or positive. PSI describes distribution change; it does not use the outcome label.

Missing values and new categories
---------------------------------

Missing values form their own group, and categories appearing in only one sample are retained. Their empty cells are smoothed exactly as in :doc:`woe_iv`, so every contribution stays finite.

>>> import pandas as pd
>>> from kivi.evaluate import psi_table
>>> reference = pd.Series(["A", "A", None])
>>> current = pd.Series(["B", None, None])
>>> psi_table(reference, current).round(4)
     reference  current  reference_share  current_share     psi
bin
A            2        0           0.5714         0.1429  0.5941
NaN          1        2           0.2857         0.5714  0.1980
B            0        1           0.1429         0.2857  0.0990

Reading the result
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Meaning
   * - ``reference``, ``current``
     - Observed counts in each group.
   * - ``reference_share``, ``current_share``
     - :math:`q_i` and :math:`p_i` after smoothing; each column sums to 1.
   * - ``psi``
     - The group's contribution. Sum the column to obtain total drift.

.. note::

   A common convention reads a PSI below 0.1 as a small shift, 0.1 to 0.25 as a moderate shift and above 0.25 as a large shift. The thresholds depend on the number of bins and on the smoothing rule, so recheck them when moving analyses from 0.2.x.
