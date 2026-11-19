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

Choosing the merges by hand works for a few variables. The supervised methods in the next section search for them automatically, and two of them start from fine quantile bins in the same way.

.. _supervised-binning:

Supervised binning
------------------

Quantile and uniform edges ignore the target. The supervised methods use it to place edges between customers with different event rates, which is what the WOE of each bin measures. They take the values and a 0/1 target on the same index, omit missing values while fitting, and return edges in the same form as ``bin_edges``: increasing, with infinite outer edges, ready for ``pandas.cut``, ``woe_table`` and ``psi_table``. Each returns at most ``max_bins`` bins, 5 by default.

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Function
     - How it fits the edges
     - Main options
   * - ``chimerge_edges``
     - Merges adjacent fine bins whose event rates do not differ significantly.
     - ``max_bins``, ``alpha``
   * - ``tree_edges``
     - Splits the values where a decision tree gains the most purity.
     - ``max_bins``, ``min_bin_share``, ``criterion``
   * - ``monotonic_edges``
     - Merges adjacent fine bins until the WOE rises or falls steadily.
     - ``max_bins``, ``direction``

As with ``bin_edges``, fit the edges on the reference sample only and apply the saved array to every other sample.

ChiMerge
^^^^^^^^

``chimerge_edges`` starts from fine quantile bins, 20 by default, and repeatedly merges the adjacent pair whose event rates are most alike, the pair with the smallest chi-square statistic. Merging continues while more than ``max_bins`` bins remain or while some adjacent pair does not differ significantly at level ``alpha``.

>>> from kivi.binning import chimerge_edges, monotonic_edges, tree_edges
>>> target = reference["target"]
>>> chi_edges = chimerge_edges(balance, target)
>>> chi_edges
array([  -inf,  68.75, 895.7 ,    inf])
>>> by_chi = woe_table(pd.cut(balance, chi_edges), target).sort_index()
>>> by_chi[["event", "non_event", "woe"]].round(3)
                event  non_event    woe
bin
(-inf, 68.75]      51        699 -0.587
(68.75, 895.7]    131       1069 -0.068
(895.7, inf]      166        884  0.358
>>> print(f"IV = {by_chi['iv'].sum():.3f}")
IV = 0.122

Five bins were allowed, but with five and with four bins some adjacent pair still had a chi-square below 3.84, the critical value at ``alpha=0.05``, so merging went on to three bins. Pass ``alpha=None`` to merge down to ``max_bins`` only, or a larger ``alpha`` to keep more bins.

.. plot::
   :alt: Smallest chi-square between adjacent balance bins as ChiMerge merges them, and the WOE of the three final bins
   :caption: Left: the smallest chi-square between adjacent bins as the fine bins are merged, read from left to right. Merging stops at the first count no larger than max_bins at which every pair lies above the alpha line. Right: the WOE of the final bins rises with balance.

   kp.figure_chimerge()

Decision tree
^^^^^^^^^^^^^

``tree_edges`` grows a decision tree on the one variable. Each split cuts a bin at the value that most reduces the impurity of the target, measured by entropy (the default) or by the Gini index with ``criterion="gini"``. The bin whose best split gains the most is split next, until there are ``max_bins`` bins or no allowed split reduces the impurity. ``min_bin_share`` keeps every bin above a share of the observations, 5% by default, so that its WOE rests on enough customers. The interior edges are observed values.

>>> tree = tree_edges(balance, target)
>>> tree
array([ -inf,   99.,  888., 3276., 6046.,   inf])
>>> by_tree = woe_table(pd.cut(balance, tree), target).sort_index()
>>> by_tree[["event", "non_event", "woe"]].round(3)
                  event  non_event    woe
bin
(-inf, 99.0]         56        767 -0.586
(99.0, 888.0]       123        996 -0.061
(888.0, 3276.0]     106        587  0.319
(3276.0, 6046.0]     47        168  0.757
(6046.0, inf]        16        134 -0.094
>>> print(f"IV = {by_tree['iv'].sum():.3f}")
IV = 0.158

The tree separates the classes best, but it does not look for a trend: customers above 6,046 subscribe less often than those just below them. A scorecard needs a WOE pattern that can be explained, so check the trend of tree bins before using them, and merge bins where it breaks.

.. plot::
   :alt: Entropy reduction of every possible first cut of balance, and the WOE of the five tree bins
   :caption: Left: the entropy reduction of every possible first cut. Cuts in the shaded areas would leave less than min_bin_share of the customers in a bin. The dashed lines are the tree's cuts, numbered in the order they were made. Right: the WOE of the five bins rises, then falls in the top bin.

   kp.figure_tree()

Monotonic binning
^^^^^^^^^^^^^^^^^

``monotonic_edges`` also starts from fine quantile bins. Whenever two adjacent bins break the trend, it merges them and recomputes the WOE (pool adjacent violators), so the WOE of the final bins is strictly increasing or strictly decreasing. If more than ``max_bins`` bins remain, it then merges the most similar adjacent pairs by chi-square. ``direction="auto"`` tries both trends and keeps the one with the larger information value; pass ``"increasing"`` or ``"decreasing"`` to impose the trend that the business expects.

>>> mono = monotonic_edges(balance, target)
>>> mono
array([   -inf,   68.75,  125.7 ,  895.7 , 1136.6 ,     inf])
>>> by_trend = woe_table(pd.cut(balance, mono), target).sort_index()
>>> by_trend["woe"].round(3).tolist()
[-0.587, -0.243, -0.045, 0.27, 0.373]
>>> bool(by_trend["woe"].is_monotonic_increasing)
True
>>> print(f"IV = {by_trend['iv'].sum():.3f}")
IV = 0.124

The trend is checked with the same smoothing as ``woe_table``, so the WOE that ``woe_table`` reports for these bins is monotonic as well, even when a bin has no events.

.. plot::
   :alt: WOE of the fine balance bins with the pooled monotonic WOE, and the WOE of the five final bins
   :caption: Left: the WOE of the fine quantile bins zigzags; pooling adjacent violators turns it into a rising step (max_bins=None). Right: merged further to max_bins=5, the WOE still rises with balance.

   kp.figure_monotonic()

Comparing the methods
^^^^^^^^^^^^^^^^^^^^^

Supervised edges are chosen to separate the sample they are fitted on, so the information value on that sample is optimistic. Compare the methods on a later sample as well, keeping the edges fixed.

>>> current = data.iloc[3000:]
>>> fitted = {
...     "bin_edges": bin_edges(balance, bins=5),
...     "chimerge_edges": chi_edges,
...     "tree_edges": tree,
...     "monotonic_edges": mono,
... }
>>> def information_value(sample, edges):
...     groups = pd.cut(sample["balance"], edges)
...     return woe_table(groups, sample["target"])["iv"].sum()
>>> rows = {}
>>> for name, edges in fitted.items():
...     rows[name] = {
...         "bins": len(edges) - 1,
...         "iv_reference": information_value(reference, edges),
...         "iv_current": information_value(current, edges),
...     }
>>> pd.DataFrame.from_dict(rows, orient="index").round(3)
                 bins  iv_reference  iv_current
bin_edges           5         0.099       0.084
chimerge_edges      3         0.122       0.016
tree_edges          5         0.158       0.065
monotonic_edges     5         0.124       0.020

On the reference sample all three supervised methods beat five quantile bins. On the later sample they fall behind: much of the extra separation they found in balance was noise, which is common for a weak variable and a few hundred events. Validate supervised edges on a later or held-out sample before adopting them, and keep the variable under PSI monitoring once it is in use.

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

``chimerge_edges``, ``tree_edges`` and ``monotonic_edges`` raise ``ValueError`` for the same problems with the values, for an option outside its range, and when:

* ``values`` and ``target`` do not have identical indices;
* ``target`` contains a label other than 0 and 1, or a missing label;
* the observed values do not cover both classes.
