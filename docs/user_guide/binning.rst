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

The number of bins can therefore be smaller than requested, and ``bin_edges`` then issues a ``UserWarning``. A single value that covers much of the reference, such as a code for "not applicable", often causes this. Pass that value in ``special``, as described in the next section.

Special values
--------------

Many credit variables use codes such as ``-1`` or ``-999`` for "not applicable" or "no record". Such a code is not a measurement, so it should neither shape the edges nor share an interval with real values. Pass the codes in ``special`` to ``bin_edges`` to leave them out of the fit, and pass the same codes to ``kivi.binning.cut`` to give each one its own group.

In the bundled data, ``pdays`` is ``-1`` for the 82% of customers who were not contacted in an earlier campaign. Without ``special``, every quantile is tied at ``-1``: ``bin_edges`` returns a single bin and warns that ``-1`` makes up 82% of the fitted values. With ``special=[-1]`` the remaining values are divided into five bins:

>>> from kivi.binning import cut
>>> from kivi.datasets import Dataset
>>> from kivi.woe import woe_table
>>> data = Dataset.bank_data()
>>> edges = bin_edges(data["pdays"], bins=5, special=[-1])
>>> edges
array([-inf, 112., 179., 254., 343.,  inf])
>>> groups = cut(data["pdays"], edges, special=[-1])
>>> result = woe_table(groups, data["target"])
>>> result[["event", "non_event", "woe", "iv"]].round(3)
                event  non_event    woe     iv
bin
-1                337       3368 -0.264  0.051
(-inf, 112.0]      66        101  1.613  0.164
(112.0, 179.0]     22        138  0.202  0.002
(179.0, 254.0]     48        115  1.165  0.074
(254.0, 343.0]     21        144  0.113  0.000
(343.0, inf]       27        134  0.436  0.008
>>> print(f"IV = {result['iv'].sum():.3f}")
IV = 0.299

The single bin had an IV of zero; separating the code shows that ``pdays`` is informative. The categories of the result are the special values in the order given, followed by the intervals, and ``woe_table`` and ``psi_table`` list the rows in that order. Special values are matched by equality, so ``-1`` and ``-1.0`` are the same code. Without ``special``, ``cut`` returns the same result as ``pandas.cut``.

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

Pass categorical labels directly to the metric functions. A missing label is a separate group. Only observed groups appear. Categorical groups, including the result of ``pandas.cut``, follow the order of their categories with missing values last; other labels follow the order in which they first appear. For WOE, group and target series must have identical indices; arrays and lists are matched by position.

Invalid input
-------------

``bin_edges`` raises ``ValueError`` when:

* ``bins`` is not a positive integer;
* ``method`` is neither ``"quantile"`` nor ``"uniform"``;
* the reference values are not one-dimensional or not numeric;
* ``special`` is not one-dimensional or contains a missing value;
* the reference has no observations apart from missing and special values, or contains an infinite value that is not special.
