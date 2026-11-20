Binning
=======

``bin_edges`` fits numeric bin edges on a reference sample and returns them as an array for ``pandas.cut``. The reference can be a list, a NumPy array or a pandas Series. Keep the array: the same edges must be applied to every sample that is compared with the reference.

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

Ties and constant values
------------------------

Tied cut points are collapsed. A constant reference produces one bin.

>>> bin_edges(pd.Series([0, 0, 0, 0, 1, 2]), bins=4)
array([-inf, 0.75,  inf])
>>> bin_edges(pd.Series([7, 7, 7]))
array([-inf,  inf])

The number of bins can therefore be smaller than requested, and ``bin_edges`` then issues a ``UserWarning``. A single value that covers much of the reference, such as a code for "not applicable", often causes this. Group that value separately before fitting edges on the remaining values.

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

Pass categorical labels directly to the metric functions. A missing label is a separate group. Only observed groups appear, in first-observed order. For WOE, group and target series must have identical indices; arrays and lists are matched by position.

Invalid input
-------------

``bin_edges`` raises ``ValueError`` when:

* ``bins`` is not a positive integer;
* ``method`` is neither ``"quantile"`` nor ``"uniform"``;
* the reference values are not one-dimensional or not numeric;
* the reference has no observations, or contains an infinite value.
