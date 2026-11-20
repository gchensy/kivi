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
>>> from kivi.stability import psi_table
>>> reference = pd.Series(["A", "A", None])
>>> current = pd.Series(["B", None, None])
>>> psi_table(reference, current).round(4)
     reference  current  reference_share  current_share     psi
bin
A            2        0           0.5714         0.1429  0.5941
NaN          1        2           0.2857         0.5714  0.1980
B            0        1           0.1429         0.2857  0.0990

When only the counts per group are available, for example from a monitoring report, ``psi`` gives the total from the counts of each sample in the same group order:

>>> from kivi.stability import psi
>>> round(psi([2, 1, 0], [0, 2, 1]), 4)
0.8912

Stable and shifted samples
--------------------------

Two samples from the same population give a PSI close to zero. When the population changes, the bins that moved stand out. Below, the current sample is replaced by single customers only, who are much younger than the reference.

>>> from kivi.binning import bin_edges
>>> from kivi.datasets import Dataset
>>> data = Dataset.bank_data()
>>> sample = data.iloc[:3000]
>>> edges = bin_edges(sample["age"], bins=5)
>>> before = pd.cut(sample["age"], edges)
>>> single = data[data["marital"] == "single"]
>>> shifted = psi_table(before, pd.cut(single["age"], edges)).sort_index()
>>> shifted[["reference_share", "current_share", "psi"]].round(3)
              reference_share  current_share    psi
bin
(-inf, 32.0]            0.234          0.521  0.230
(32.0, 36.0]            0.177          0.194  0.002
(36.0, 42.0]            0.196          0.154  0.010
(42.0, 51.0]            0.213          0.099  0.087
(51.0, inf]             0.180          0.032  0.258
>>> print(f"PSI = {shifted['psi'].sum():.3f}")
PSI = 0.586

.. plot::
   :alt: Reference and current age distributions for a random split and for single customers
   :caption: Left: the remaining customers look like the reference sample. Right: single customers crowd into the youngest bin, and the youngest and oldest bins contribute most of the PSI.

   kp.figure_psi_shift()

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

Monitoring over time
--------------------

In production, PSI is computed for every period against the development sample and compared with the conventions above. The bank data record the month of the last contact but not the year, so the months below mix several years; read them as an example of a monitoring report.

>>> psi_by_month = pd.Series({
...     month: psi_table(before, pd.cut(group["age"], edges))["psi"].sum()
...     for month, group in data.groupby("month")
... })
>>> psi_by_month.sort_values(ascending=False).round(3).head(4)
dec    0.753
mar    0.226
oct    0.217
sep    0.179
dtype: float64
>>> data["month"].value_counts()[["dec", "mar", "oct", "sep"]].tolist()
[20, 49, 80, 52]

.. plot::
   :alt: PSI of age for each contact month, with the 0.1 and 0.25 thresholds and the sample size of each month
   :caption: PSI of age for each contact month against the reference sample, with the number of customers under each month.

   kp.figure_psi_monitor()

The months that cross the thresholds are also the smallest. With 20 customers, the December value mostly reflects sampling noise. Set a minimum sample size before a high PSI raises an alert.
