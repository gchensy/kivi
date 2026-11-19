Binning
=======

``bin_edges`` fits numeric bin edges on a reference sample and returns them as an array for ``pandas.cut``. Keep the array: the same edges must be applied to every sample that is compared with the reference.

In credit scoring, binning is the classing step of scorecard development: a continuous variable becomes a handful of groups whose WOE can be read, explained to the business and monitored over time. This page compares the ways to choose the edges.

Quantile and uniform edges
--------------------------

Use quantile bins to divide the reference observations by rank, or uniform bins to divide their numeric range into equal widths.

>>> import pandas as pd
>>> from kivi.binning import bin_edges
>>> values = pd.Series([1, 2, 2, 3, 4, 5, 8, 13, 21, 34])
>>> bin_edges(values, bins=3)
array([-inf,   3.,   8.,  inf])
>>> bin_edges(values, bins=3, method="uniform")
array([-inf,  12.,  23.,  inf])

Quantile bins hold roughly equal numbers of observations, which suits skewed variables. Uniform bins keep equal widths, so a long tail can leave some bins nearly empty.

Account balances are typical of financial data: most customers hold small amounts and a few hold very large ones. On the bank data, quantile bins split the reference sample evenly, while uniform bins put almost everyone into the first bin.

>>> from kivi.datasets import Dataset
>>> data = Dataset.bank_data()
>>> reference = data.iloc[:3000]
>>> balance = reference["balance"]
>>> quantile = pd.cut(balance, bin_edges(balance, bins=5))
>>> uniform = pd.cut(balance, bin_edges(balance, bins=5, method="uniform"))
>>> quantile.value_counts(sort=False).tolist()
[600, 600, 602, 598, 600]
>>> uniform.value_counts(sort=False).tolist()
[2871, 111, 12, 5, 1]

.. plot::
   :alt: Histogram of account balance with quantile and uniform edges, and the number of customers in each bin for both methods
   :caption: Left: the balance distribution, cut at its 99th percentile, with the quantile edges (solid) and the first uniform edge (dashed). Right: customers per bin. Uniform bins leave the upper bins almost empty, so their WOE would rest on a handful of customers.

   kp.figure_binning_methods()

.. _custom-edges:

Custom edges and special values
-------------------------------

The edges do not have to come from ``bin_edges``. Any increasing array works with ``pandas.cut`` and the metric functions, so policy bands and expert edges fit the same workflow. Keep infinite outer edges so that every future value is covered.

>>> import numpy as np
>>> from kivi.woe import woe_table
>>> age_bands = np.array([-np.inf, 25, 35, 45, 55, 65, np.inf])
>>> by_band = woe_table(pd.cut(reference["age"], age_bands), reference["target"]).sort_index()
>>> by_band["woe"].round(2).tolist()
[0.79, -0.03, -0.17, -0.0, 0.04, 1.11]

Risk data often encode a state as a number, for example -1 for "never contacted" or 999 for "no record". In the bank data, 82% of the reference customers have ``pdays == -1``, so quantile binning collapses the variable into a single bin and its information value is lost. An edge at -1 gives the code a bin of its own.

>>> bin_edges(reference["pdays"], bins=5)
array([-inf,  inf])
>>> pdays_edges = np.array([-np.inf, -1, 100, 200, 300, np.inf])
>>> pdays = woe_table(pd.cut(reference["pdays"], pdays_edges), reference["target"]).sort_index()
>>> pdays[["event", "non_event", "woe"]].round(3)
                event  non_event    woe
bin
(-inf, -1.0]      228       2247 -0.257
(-1.0, 100.0]      47         48  2.010
(100.0, 200.0]     46        138  0.932
(200.0, 300.0]     10         77 -0.010
(300.0, inf]       17        142 -0.092
>>> print(f"IV = {pdays['iv'].sum():.3f}")
IV = 0.360

.. plot::
   :alt: WOE by age band and by days since the previous contact
   :caption: Left: age bands show a U shape, with the youngest and oldest customers more likely to subscribe. Right: customers contacted in a previous campaign within 200 days stand out once the -1 code has its own bin.

   kp.figure_custom_edges()

Merging fine bins
-----------------

Scorecard development usually starts from many fine bins and merges neighbours, a step known as coarse classing, until each bin is large enough to be stable and the WOE follows a trend that can be explained. Select a subset of the fitted edges to merge bins; keeping the first and last edge keeps the infinite ends.

>>> fine = bin_edges(balance, bins=10)
>>> fine_woe = woe_table(pd.cut(balance, fine), reference["target"]).sort_index()
>>> fine_woe["woe"].round(2).tolist()
[-0.52, -0.71, -0.46, 0.1, -0.24, 0.06, 0.05, 0.35, 0.37, 0.4]
>>> coarse = fine[[0, 3, 7, 10]]
>>> coarse
array([  -inf,  125.7, 1136.6,    inf])
>>> coarse_woe = woe_table(pd.cut(balance, coarse), reference["target"]).sort_index()
>>> coarse_woe["woe"].round(3).tolist()
[-0.522, -0.002, 0.373]
>>> bool(coarse_woe["woe"].is_monotonic_increasing)
True
>>> print(f"IV {fine_woe['iv'].sum():.3f} -> {coarse_woe['iv'].sum():.3f}")
IV 0.123 -> 0.115

.. plot::
   :alt: WOE of ten fine balance bins next to WOE of three merged bins
   :caption: Ten fine bins (left) zigzag; three merged bins (right) rise steadily with balance and give up little information value.

   kp.figure_coarse_classing()

.. note::

   Version 0.3 does not search for merges automatically. Supervised methods such as ChiMerge, decision-tree or monotonic binning are planned. Edges produced by any tool can already be passed to ``pandas.cut`` and the metric functions.

Ties and constant values
------------------------

Tied cut points are collapsed. A constant reference produces one bin.

>>> bin_edges(pd.Series([0, 0, 0, 0, 1, 2]), bins=4)
array([-inf, 0.75,  inf])
>>> bin_edges(pd.Series([7, 7, 7]))
array([-inf,  inf])

The number of bins can therefore be smaller than requested. Check ``len(edges) - 1`` when the count matters.

Applying the edges
------------------

Infinite endpoints cover finite future observations outside the training range. Use the saved edges with pandas.cut on every sample.

>>> edges = bin_edges(pd.Series([10, 20, 30, 40, 50, 60]), bins=2)
>>> edges
array([-inf,  35.,  inf])
>>> pd.cut(pd.Series([5, 35, 99, None]), edges)
0    (-inf, 35.0]
1    (-inf, 35.0]
2     (35.0, inf]
3             NaN
dtype: category
Categories (2, interval[float64, right]): [(-inf, 35.0] < (35.0, inf]]

Intervals are closed on the right, so a value equal to an edge belongs to the lower bin. Pass ``labels=False`` to ``pandas.cut`` for integer bin numbers instead of intervals.

Missing values and categories
-----------------------------

Missing reference values are ignored when the edges are fitted, and ``pandas.cut`` leaves them missing. The metric functions then count them as a separate group.

Pass categorical labels directly to the metric functions. A missing label is a separate group. Only observed groups appear, in first-observed order. For WOE the group and target indices must match exactly.

Invalid input
-------------

``bin_edges`` raises ``ValueError`` when:

* ``bins`` is not a positive integer;
* ``method`` is neither ``"quantile"`` nor ``"uniform"``;
* the reference values are not numeric;
* the reference has no observations, or contains an infinite value.
