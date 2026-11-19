Migrating from 0.2.x
====================

The development release exposes four public operations. Dataset.bank_data() retains its import path and table schema. The target describes deposit subscription.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 0.2.x
     - 0.3
   * - ``DistanceBins``
     - ``bin_edges(values, method="uniform")``, then ``pandas.cut`` and ``woe_table``
   * - ``FrequencyBins``
     - ``bin_edges(values, method="quantile")``, then ``pandas.cut`` and ``woe_table``
   * - ``CategoryBins``
     - ``woe_table(labels, target)``
   * - ``PSI.fit()``
     - ``psi_table(before, after)`` on groups formed with the same edges
   * - ``Dataset.bank_data()``
     - Unchanged

Binning and WOE
---------------

DistanceBins and FrequencyBins previously calculated group assignments and statistics together. Use bin_edges with uniform or quantile, call pandas.cut, then pass the groups to woe_table. CategoryBins users can pass the label series directly.

Population stability
--------------------

For PSI.fit(), form both sets of groups using the same reference edges, then call psi_table. It returns raw counts and a contribution per bin. Previously the psi column repeated the overall total. The new smoothing rule preserves counts and normalizes each distribution after replacing zeros. Recheck thresholds when migrating saved analyses.

Not yet ported
--------------

The development release covers unweighted counts. Keep existing weighted, supervised binning and scorecard workflows on their original version until their outputs have been validated on fixed reference samples. Spark, SQL, AHP, feature selection, model wrappers and application code remain in the historical revisions.
