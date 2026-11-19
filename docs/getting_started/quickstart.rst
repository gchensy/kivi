Quickstart
==========

This walkthrough bins a numeric and a categorical variable from the bundled data, measures how well each bin separates the target and checks whether the age distribution shifts between two samples. Every output below is produced by the code above it.

Load the data
-------------

The bundled bank marketing data records whether a customer subscribed to a term deposit. The split below demonstrates two samples; the rows do not represent a chronological validation split.

>>> import pandas as pd
>>> from kivi.datasets import Dataset
>>> from kivi.binning import bin_edges
>>> from kivi.woe import woe_table
>>> from kivi.evaluate import psi_table
>>> data = Dataset.bank_data()
>>> data.shape
(4521, 17)
>>> print(f"{data['target'].mean():.3f}")
0.115
>>> reference, current = data.iloc[:3000], data.iloc[3000:]

Fit bins on the reference sample
--------------------------------

Fit the edges on the reference sample only, then reuse them for every other sample so that the counts remain comparable.

>>> edges = bin_edges(reference["age"], bins=5)
>>> edges
array([-inf,  32.,  36.,  42.,  51.,  inf])
>>> before = pd.cut(reference["age"], edges)
>>> after = pd.cut(current["age"], edges)

The outer edges are infinite, so an age outside the reference range still falls into the first or last bin.

Weight of evidence
------------------

``woe_table`` returns one row per bin. Groups appear in the order they are first observed; ``sort_index`` puts interval bins in numeric order.

>>> evidence = woe_table(before, reference["target"]).sort_index()
>>> evidence[["event", "non_event", "woe", "iv"]].round(3)
              event  non_event    woe     iv
bin
(-inf, 32.0]     90        612  0.114  0.003
(32.0, 36.0]     55        475 -0.125  0.003
(36.0, 42.0]     58        529 -0.180  0.006
(42.0, 51.0]     71        569 -0.050  0.001
(51.0, inf]      74        467  0.189  0.007
>>> print(f"IV = {evidence['iv'].sum():.3f}")
IV = 0.019

A positive WOE means that the bin holds a larger share of the subscribers than of the non-subscribers. Age on its own separates the two groups only weakly.

Read the per-bin contributions before interpreting a total. A group with few events can have a large WOE despite representing few customers.

Categorical variables
---------------------

Pass category labels directly; they need no edges.

>>> jobs = woe_table(reference["job"], reference["target"])
>>> jobs.sort_values("woe", ascending=False)[["event", "non_event", "woe"]].round(3).head(3)
         event  non_event    woe
bin
student     15         44  0.955
retired     29        121  0.602
unknown      4         21  0.373
>>> print(f"IV = {jobs['iv'].sum():.3f}")
IV = 0.137

Population stability
--------------------

``psi_table`` compares the two samples bin by bin. It does not use the target.

>>> stability = psi_table(before, after).sort_index()
>>> stability[["reference", "current", "psi"]].round(4)
              reference  current     psi
bin
(-inf, 32.0]        702      353  0.0000
(32.0, 36.0]        530      255  0.0005
(36.0, 42.0]        587      281  0.0006
(42.0, 51.0]        640      337  0.0003
(51.0, inf]         541      295  0.0010
>>> print(f"PSI = {stability['psi'].sum():.4f}")
PSI = 0.0024

Next steps
----------

* :doc:`/user_guide/binning` explains quantile and uniform edges, ties and missing values.
* :doc:`/user_guide/woe_iv` and :doc:`/user_guide/psi` give the formulas and the smoothing rule.
* :doc:`/api` lists every parameter.
