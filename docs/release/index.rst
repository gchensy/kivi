Release notes
=============

Unreleased
----------

New
^^^

* ``bin_edges`` takes a ``special`` argument. Codes such as ``-1`` or ``-999`` are left out when the edges are fitted.
* ``kivi.binning.cut`` assigns values to the fitted bins and gives each special value its own group. Without special values it returns the same result as ``pandas.cut``.
* ``chimerge_edges``, ``tree_edges`` and ``monotonic_edges`` fit edges from a binary target: ChiMerge merging of fine quantile bins, decision-tree splits by entropy or Gini with a minimum bin share, and merging until the WOE is monotonic. They accept the same array-like inputs as ``woe_table``, return infinite-ended edges like ``bin_edges`` and need only NumPy and pandas.
* The binning guide compares the three methods on the bank data, with a chart for each.
* ``chimerge_edges``, ``tree_edges`` and ``monotonic_edges`` take ``special`` like ``bin_edges``. The codes are left out of the fit together with their labels, and ``kivi.binning.cut`` with the same codes gives each one its own group.

Changed
^^^^^^^

* ``bin_edges``, ``woe_table`` and ``psi_table`` accept any one-dimensional array-like input, such as a list or a NumPy array. Group and target series passed to ``woe_table`` must still share an index; arrays and lists are matched by position.
* ``bin_edges`` issues a ``UserWarning`` when tied edges leave fewer bins than requested. The message names the most frequent value so that it can be passed in ``special``.
* ``woe_table`` and ``psi_table`` list categorical groups, including the result of ``pandas.cut``, in category order with missing values last. Other labels keep the order in which they first appear. Previously all groups followed first appearance, and ``sort_index`` was needed to put intervals in order.
* Error messages name the argument and report the value received.
* ``kivi.__version__`` is ``"0+unknown"`` when kivi is imported from a source tree that is not installed, instead of raising an error.
* Group counts use ``numpy.bincount``, which is faster on large samples.
* The API reference documents errors, warnings, formulas and related functions.

0.3.0 (2026-11-19)
------------------

A rewrite around a small pandas core. Read :doc:`migration` before upgrading code written for 0.2.x.

New
^^^

* ``bin_edges`` fits quantile or uniform edges on a reference sample. The outer edges are infinite, so the result can be passed to ``pandas.cut`` for any later sample.
* ``woe_table`` and ``psi_table`` return raw counts, smoothed shares and a contribution for each group, including missing values.
* Documentation in English and Simplified Chinese.

Changed
^^^^^^^

* NumPy and pandas are required dependencies. The 0.2.8 wheel placed them behind the ``all`` extra.
* The ``psi`` column holds the contribution of each bin. It previously repeated the overall total.
* Zero counts are replaced before each distribution is normalized, and reported counts are no longer modified.

Removed
^^^^^^^

* Spark, SQL, AHP, feature selection, model wrappers, weighted and supervised binning and scorecards. They remain in the repository history.

.. toctree::
   :hidden:

   migration
