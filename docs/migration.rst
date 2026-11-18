Migrating from 0.2.x
====================

The development release exposes four public operations. Dataset.bank_data() retains its import path and table schema. The target describes deposit subscription.

DistanceBins and FrequencyBins previously calculated group assignments and statistics together. Use bin_edges with uniform or quantile, call pandas.cut, then pass the groups to woe_table. CategoryBins users can pass the label series directly.

For PSI.fit(), form both sets of groups using the same reference edges, then call psi_table. It returns raw counts and a contribution per bin. Previously the psi column repeated the overall total. The new smoothing rule preserves counts and normalizes each distribution after replacing zeros. Recheck thresholds when migrating saved analyses.

The development release covers unweighted counts. Keep existing weighted, supervised binning and scorecard workflows on their original version until their outputs have been validated on fixed reference samples. Spark, SQL, AHP, feature selection, model wrappers and application code remain in the historical revisions.
