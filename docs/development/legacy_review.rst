Legacy review
=============

This review of the 0.2.x code guided the rewrite in 0.3.0. Its findings describe the legacy implementation; :doc:`/release/migration` describes how each part was replaced.

Findings from the legacy implementation
---------------------------------------

* Packaging. The 0.2.8 wheel placed NumPy and pandas behind the all extra. A default installation could import kivi but failed on its dataset and WOE entry points.
* Dependencies. evaluate/BucketsUtils.py imported PySpark at module load even for pandas PSI. Its standalone CSI function also referenced an undefined self. The new metric functions depend on pandas and NumPy.
* Numerical results. The old WOE zero-cell logic adjusted one class with an if/elif and used the original totals. The old PSI helper modified the displayed counts. Tests now check raw counts, normalized shares and hand-calculated metrics.
* State. Model/Tree.py removed the target column from the caller's frame. Dataset generation also reset NumPy's global random state. Future estimators should preserve inputs and take a local random seed.
* Coverage. Legacy scorecard and WOE tests printed tables without asserting the expected numbers. Use frozen datasets to check fitted boundaries, score direction, special values and row counts before porting these workflows.
* Project scope. The former source tree mixed modeling, databases, distributed execution, notebooks and deployment configuration. Keep infrastructure-specific imports inside optional integrations and keep business SQL and local configuration outside the distribution.

Outcome
-------

The rewrite began with binning, WOE and PSI and then rebuilt the rest of the scorecard workflow, checking each part against the legacy outputs or reference implementations on fixed samples. :doc:`/release/index` lists the parts release by release. Spark, SQL and the model wrappers were not rebuilt. They are out of scope and remain in the repository history before the rewrite. The analytic hierarchy process of the expert scorecards was rebuilt later, in 1.2.0, as :mod:`kivi.expert`.
