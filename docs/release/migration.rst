Migrating from 0.2.x
====================

The functions below replace the 0.2.x classes. Dataset.bank_data() retains its import path and table schema. The target describes deposit subscription.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 0.2.x
     - 0.4
   * - ``DistanceBins``
     - ``bin_edges(values, method="uniform")``, then ``pandas.cut`` and ``woe_table``
   * - ``FrequencyBins``
     - ``bin_edges(values, method="quantile")``, then ``pandas.cut`` and ``woe_table``
   * - ``abnormal_vals=[...]``
     - ``special=[...]`` in ``bin_edges`` and ``kivi.binning.cut``
   * - ``CategoryBins``
     - ``woe_table(labels, target)``
   * - ``WOEScore``
     - ``woe_transform(groups, table)`` on groups formed with the development edges, from the release after 0.4.0
   * - ``PSI.fit()``
     - ``psi_table(before, after)`` on groups formed with the same edges
   * - ``Dataset.bank_data()``
     - Unchanged

Binning and WOE
---------------

DistanceBins and FrequencyBins previously calculated group assignments and statistics together. Use bin_edges with uniform or quantile, call pandas.cut, then pass the groups to woe_table. Codes listed in abnormal_vals become the special argument: pass them to bin_edges, and use kivi.binning.cut with the same codes in place of pandas.cut. CategoryBins users can pass the label series directly.

The release after 0.4.0 adds woe_transform in place of WOEScore. WOEScore matched raw values against the min_bin and max_bin columns of a WOE table. Form the groups with the development edges instead, then pass them to woe_transform with the woe_table result. A value that matches no group raises an error, as WOEScore did with error="error"; pass unseen to choose its WOE instead.

Version 0.4.0 adds ``chimerge_edges`` and ``tree_edges`` in place of ``Chi2Bins`` and ``TreeBins``. They are new implementations rather than ports, so the edges can differ. ``Chi2Bins`` started from 10 quantile bins, tested at a 0.9 confidence level, added 0.5 to every expected count and kept at least three bins; ``chimerge_edges(values, target, alpha=0.1, initial_bins=10)`` comes closest. ``TreeBins`` reported the midpoints that scikit-learn chooses as thresholds, while ``tree_edges`` reports the largest observed value below each cut. Compare the edges on a fixed reference sample before switching.

Codes listed in ``abnormal_vals`` of ``Chi2Bins`` and ``TreeBins`` become the ``special`` argument of ``chimerge_edges`` and ``tree_edges``. As before, the codes and their labels are left out of the fit; pass the same codes to ``kivi.binning.cut`` to give each one its own group.

Population stability
--------------------

For PSI.fit(), form both sets of groups using the same reference edges, then call psi_table. It returns raw counts and a contribution per bin. Previously the psi column repeated the overall total. The new smoothing rule preserves counts and normalizes each distribution after replacing zeros. Recheck thresholds when migrating saved analyses.

Not yet ported
--------------

Releases 0.3 and 0.4 cover unweighted counts only. Keep existing weighted binning and scorecard workflows on their original version until their outputs have been validated on fixed reference samples. Spark, SQL, AHP, feature selection, model wrappers and application code remain in the historical revisions.
