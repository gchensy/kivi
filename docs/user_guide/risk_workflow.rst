Risk modeling workflow
======================

This example follows the steps of a scorecard project with kivi: profile and screen the candidate variables, bin the shortlist, encode it with WOE and save the binning for production, select the variables and fit the model, scale it to a scorecard, evaluate it, calibrate it on a master scale, set a cut-off with its expected loss, and prepare the stability report used in monitoring. The bundled data record a marketing response rather than a default, but the steps are the same for a credit default flag: use the bad flag as ``target``.

The bank data have no history of transactions or repayments, so the example starts from the profile of the columns as they are. Where such events exist, :doc:`features` first turns them into candidates, using only what was known at each observation date, and the profile then screens them with the other columns.

>>> import numpy as np
>>> import pandas as pd
>>> from kivi.binning import bin_edges
>>> from kivi.datasets import load_bank
>>> from kivi.stability import psi_table
>>> from kivi.woe import woe_table, woe_transform
>>> data = load_bank()
>>> development, monitoring = data.iloc[:3000], data.iloc[3000:]

Screen variables by IV
----------------------

Start with a profile of every candidate. :func:`~kivi.univariate.profile_table` gives each one row with its missing values, its most frequent value and, with the target, the IV of a quick, uniform binning: five quantile bins for numeric variables and the raw labels for categorical ones; see :doc:`univariate`. The IV of each variable gives a first ranking, and ``tied_bins`` marks the numeric variables whose tied values leave fewer bins than requested:

>>> from kivi.univariate import profile_table
>>> profile = profile_table(development.drop(columns="target"), development["target"])
>>> profile.index[profile["tied_bins"]].tolist()
['campaign', 'pdays', 'previous']
>>> iv = profile["iv"].sort_values(ascending=False)
>>> iv.round(3).head(8)
variable
duration    1.594
poutcome    0.423
month       0.334
contact     0.215
job         0.137
housing     0.108
balance     0.099
loan        0.059
Name: iv, dtype: float64

.. plot::
   :alt: Information value of every variable in the bank data, sorted, with the 0.02, 0.1 and 0.3 reading bands
   :caption: IV of every candidate variable with quick binning. The dashed lines mark the 0.02, 0.1 and 0.3 bands; ``duration`` is cut off at the right edge of the chart.

   kp.figure_iv_ranking()

Read the ranking critically before building the shortlist:

* ``duration`` is the length of the last call. It is only known after the call, when the outcome is largely decided, so its IV of 1.59 is a leak rather than a signal. The dataset's documentation recommends leaving it out of a realistic model. In credit risk, look for the same pattern in fields filled in after the decision, such as collection status.
* ``month`` describes when the campaign ran rather than who the customer is, so it is left out as well.
* ``pdays`` and ``previous`` show an IV of zero only because their tied values collapse into one quantile bin. Custom edges or the ``special`` argument recover ``pdays``; see :ref:`custom-edges` and :ref:`special-values`.

Bin the shortlist
-----------------

Each variable on the shortlist gets groups that can be explained: merged balance bins with a monotonic WOE, a bin of its own for the ``pdays == -1`` code, policy age bands, the labels of ``poutcome``, ``contact`` and ``housing``, and a few groups of jobs. Small categories give unstable WOE: a job held by a few dozen customers has only a handful of events, and on a later sample the WOE of four of the twelve jobs changes sign, as :ref:`category-groups` shows. :func:`~kivi.binning.chimerge_groups` merges the jobs with similar event rates into larger groups and returns a dict from job to group:

>>> from kivi.binning import chimerge_groups, group_categories
>>> job_groups = chimerge_groups(development["job"], development["target"])
>>> job_groups["unknown"]
'technician, admin., housemaid, management, unknown'

Each group is labelled with its jobs in increasing order of their WOE. A job that the development sample did not have is not in the dict, so decide before scoring new data which group it joins. Here a new job joins the group of ``unknown``, the customers whose job is not recorded, through the ``unseen`` argument of :func:`~kivi.binning.group_categories`. Keep the binning in one place so that development and monitoring data are grouped identically:

>>> fine = bin_edges(development["balance"], bins=10)
>>> numeric_edges = {
...     "balance": fine[[0, 3, 7, 10]],
...     "pdays": np.array([-np.inf, -1, 100, 200, 300, np.inf]),
...     "age": np.array([-np.inf, 25, 35, 45, 55, 65, np.inf]),
... }
>>> categorical = ["poutcome", "contact", "housing"]
>>> new_job = job_groups["unknown"]
>>> def to_groups(frame):
...     groups = {column: pd.cut(frame[column], edges) for column, edges in numeric_edges.items()}
...     groups.update({column: frame[column] for column in categorical})
...     groups["job"] = group_categories(frame["job"], job_groups, unseen=new_job)
...     return pd.DataFrame(groups)
>>> dev_groups = to_groups(development)
>>> tables = {column: woe_table(dev_groups[column], development["target"]) for column in dev_groups}
>>> print(tables["job"][["total", "event_rate", "woe"]].round(3).to_string())
                                                                total  event_rate    woe
bin
blue-collar, services, self-employed, entrepreneur, unemployed   1210       0.079 -0.420
technician, admin., housemaid, management, unknown               1581       0.132  0.144
retired, student                                                  209       0.211  0.709

The three groups of ``job`` hold between 209 and 1,581 development customers each, and their WOE rises from -0.420 to 0.709. ``tables`` holds one :func:`~kivi.woe.woe_table` result per variable. Review each one as a WOE chart (see :doc:`woe_iv`) before moving on. ``pdays`` and ``poutcome`` both describe the previous campaign, so check how strongly they are correlated before keeping both.

Encode with WOE
---------------

A logistic regression on WOE values is the classic scorecard model. Replace each group with the WOE from the development tables using :func:`~kivi.woe.woe_transform`:

>>> def to_woe(groups):
...     return pd.DataFrame({
...         column: woe_transform(groups[column], tables[column]) for column in groups
...     })
>>> X_dev = to_woe(dev_groups)
>>> X_dev.head(3).round(3)
   balance  pdays    age  poutcome  contact  housing    job
0    0.373 -0.257 -0.032    -0.257    0.241    0.333 -0.420
1    0.373 -0.092 -0.032     0.214    0.241   -0.327 -0.420
2    0.373 -0.092 -0.032     0.214    0.241   -0.327  0.144

``woe_transform`` raises an error for a group that never appeared in development, such as a new category of ``poutcome``. The grouping of ``job`` already gives new jobs a group; for the other variables, agree on a policy before scoring new data, for example ``unseen=0.0`` for a neutral WOE.

Save the binning
----------------

Production systems must group and encode new applications exactly as the development sample was. :func:`~kivi.spec.fit_spec` fits the edges and the WOE table of every column at once, and the resulting :class:`~kivi.spec.BinningSpec` saves them as versioned JSON; see :doc:`spec`. Passing the edges of the shortlist as ``bins``, the method ``"chimerge"`` for ``job`` and its group for new jobs as ``unseen`` reproduces the grouping and the encoding above:

>>> from kivi.spec import BinningSpec, fit_spec
>>> spec = fit_spec(
...     development,
...     development["target"],
...     columns=list(dev_groups),
...     method={"job": "chimerge"},
...     bins=numeric_edges,
...     unseen={"job": {"group": new_job}},
... )
>>> spec.groups["job"] == job_groups
True
>>> spec.woe_transform(development).equals(X_dev)
True
>>> restored = BinningSpec.from_json(spec.to_json())
>>> restored == spec
True

Store the JSON with the model. A scoring service that loads it with ``BinningSpec.from_json`` encodes new data with ``restored.woe_transform(frame)``, without the code of the development project, and gives a new job the WOE of the group of ``unknown``:

>>> restored.woe_transform(monitoring.head(1).assign(job="astronaut"))["job"].round(3).tolist()
[0.144]

Select variables and fit the model
----------------------------------

Strongly correlated variables carry the same information and make the coefficients unstable. :func:`~kivi.selection.correlation_filter` keeps the variable with the higher IV from each pair correlated above 0.7, and :func:`~kivi.selection.stepwise` then selects variables by the p-values of a logistic regression. With WOE inputs every coefficient should be positive, and ``stepwise`` rejects a variable that would make one negative; see :doc:`selection`.

>>> from kivi.selection import correlation_filter, logistic_regression, stepwise
>>> target = development["target"]
>>> iv_dev = pd.Series({column: table["iv"].sum() for column, table in tables.items()})
>>> kept = correlation_filter(X_dev, priority=iv_dev)
>>> X_dev.columns.difference(kept).tolist()
[]
>>> selected = stepwise(X_dev[kept], target)
>>> selected.tolist()
['balance', 'pdays', 'poutcome', 'contact', 'housing', 'job']
>>> model = logistic_regression(X_dev[selected], target)
>>> model.round(3)
            coef     se       z  p_value
variable
intercept -2.028  0.062 -32.730    0.000
balance    0.796  0.183   4.360    0.000
pdays      0.356  0.122   2.920    0.004
poutcome   0.657  0.112   5.876    0.000
contact    0.588  0.153   3.850    0.000
housing    0.635  0.189   3.358    0.001
job        0.592  0.180   3.285    0.001

No pair is correlated strongly enough to drop a variable, and ``stepwise`` leaves out ``age``. The model is fitted on the development sample only. :func:`~kivi.selection.vif_filter` checks the kept variables for collinearity among several of them, and :func:`~kivi.selection.selection_path` shows on a validation sample where adding variables in IV order stops paying off; :doc:`selection` shows both.

The bank data record the outcome of every customer. A credit development sample usually holds only the applications that the previous policy approved, and a model fitted on them alone underestimates the risk of the applicants it used to decline. :doc:`reject_inference` infers their outcomes and refits the model with ``weights``.

Scale to a scorecard
--------------------

:func:`~kivi.scorecard.points_table` turns the WOE tables and the coefficients into points for every group. Here 20 points double the odds of a good outcome, and a score of 600 stands for good:bad odds of 50; see :doc:`scorecard`. A higher WOE means more risk and therefore fewer points:

>>> from kivi.scorecard import points_table, woe_to_points
>>> scale = dict(pdo=20, base_score=600, base_odds=50)
>>> coefficients = model["coef"].drop("intercept")
>>> intercept = model.loc["intercept", "coef"]
>>> card = points_table({column: tables[column] for column in selected}, coefficients, intercept, **scale)
>>> card.loc["poutcome"].round({"woe": 3, "points": 1})
              bin    woe  points
variable
poutcome  unknown -0.257    95.8
poutcome  failure  0.214    86.9
poutcome    other  0.592    79.7
poutcome  success  2.548    42.6

Score the monitoring sample with the development tables: each customer's score is the sum of their points.

>>> mon_groups = to_groups(monitoring)
>>> mon_points = woe_to_points(to_woe(mon_groups)[selected], coefficients, intercept, **scale)
>>> mon_score = mon_points.sum(axis=1)
>>> mon_points.assign(score=mon_score).head(3).round(1)
variable  balance  pdays  poutcome  contact  housing   job  score
3000         91.0   93.6      95.8     87.4     84.8  88.5  541.1
3001        102.9   93.6      95.8    106.5     96.9  88.5  584.2
3002         82.4   93.6      95.8     86.9     96.9  98.1  553.7

Evaluate the model
------------------

A higher score means lower risk, while the measures in :doc:`metrics` expect a score that rises with the event probability. Pass the probability that each score stands for. The gains table uses bands fitted on the development scores, so that both samples are cut alike:

>>> from kivi.metrics import gains_table, gini, ks
>>> from kivi.scorecard import score_to_pd
>>> dev_score = woe_to_points(X_dev[selected], coefficients, intercept, **scale).sum(axis=1)
>>> dev_pd, mon_pd = score_to_pd(dev_score, **scale), score_to_pd(mon_score, **scale)
>>> y_mon = monitoring["target"]
>>> print(f"Gini = {gini(mon_pd, y_mon):.3f}, KS = {ks(mon_pd, y_mon):.3f}")
Gini = 0.375, KS = 0.281
>>> pd_edges = bin_edges(dev_pd, bins=5).round(3)
>>> gains = gains_table(mon_pd, y_mon, bins=pd_edges)
>>> gains[["total", "event_rate", "lift", "cum_event_share"]].round(3)
               total  event_rate   lift  cum_event_share
bin
(0.136, inf]     313       0.249  2.191            0.451
(0.1, 0.136]     294       0.102  0.897            0.624
(0.071, 0.1]     300       0.093  0.821            0.786
(0.05, 0.071]    235       0.077  0.673            0.890
(-inf, 0.05]     379       0.050  0.441            1.000

The riskiest band, about a fifth of the monitoring sample, holds 45% of its events at more than twice the average event rate.

A monitoring sample of this size measures the Gini with some uncertainty. :func:`~kivi.metrics.roc_auc_interval` gives the DeLong standard error and the 95% interval of the AUC and of the Gini:

>>> from kivi.metrics import roc_auc_interval
>>> roc_auc_interval(mon_pd, y_mon).round(3)
           estimate     se  lower  upper
statistic
auc           0.687  0.023  0.643  0.732
gini          0.375  0.046  0.285  0.464
>>> print(f"development Gini = {gini(dev_pd, target):.3f}")
development Gini = 0.429

With 173 events among 1,521 customers, the interval of the Gini runs from 0.285 to 0.464. It contains the development Gini of 0.429, which is itself optimistic because the binning, the variables and the coefficients were all fitted on the development sample, so the lower Gini of the monitoring sample is within the sampling noise. :doc:`metrics` also compares two scores on the same sample with the DeLong test.

Check calibration
-----------------

Rating grades give PDs from different models one meaning. :func:`~kivi.calibration.master_scale` builds a scale from the PD of each grade, here five grades in which each PD doubles the one before, and :func:`~kivi.calibration.assign_grades` gives every customer of the monitoring sample the grade of their PD; see :doc:`calibration`. Each grade is then compared with its PD. The Jeffreys test asks whether a grade has more events than its PD predicts, so a small p-value flags a grade that underestimates risk:

>>> from kivi.calibration import assign_grades, calibration_table, jeffreys_test, master_scale
>>> scale = master_scale(np.geomspace(0.025, 0.4, 5), labels=list("ABCDE"))
>>> scale.round(4)
        lower  probability   upper
grade
A      0.0000        0.025  0.0354
B      0.0354        0.050  0.0707
C      0.0707        0.100  0.1414
D      0.1414        0.200  0.2828
E      0.2828        0.400  1.0000
>>> grades = assign_grades(mon_pd, scale)
>>> grade_table = calibration_table(grades, y_mon, mon_pd).join(scale["probability"])
>>> grade_table["jeffreys"] = jeffreys_test(
...     grade_table["event"], grade_table["total"], grade_table["probability"]
... )
>>> grade_table.round(3)
     total  event  event_rate  mean_probability  probability  jeffreys
bin
A       80      2       0.025             0.028        0.025     0.452
B      534     35       0.066             0.053        0.050     0.054
C      595     58       0.097             0.108        0.100     0.574
D      222     38       0.171             0.187        0.200     0.859
E       90     40       0.444             0.460        0.400     0.194

No grade is rejected at the 5% level, but grade B comes close: its event rate of 6.6% exceeds its PD of 5% with a p-value of 0.054, and the monitoring of the next periods should watch it. Grade D has fewer events than its PD, which this one-sided test does not flag.

Set a cut-off
-------------

The scorecard ranks the applicants, and a cut-off decides whom to approve. :func:`~kivi.strategy.cutoff_table` shows, for each candidate PD cut-off, the share of applicants it approves, the bad rate of the approved book and the share of the events it turns away; see :doc:`strategy`. The upper bounds of the grades as cut-offs approve the grades up to each one:

>>> from kivi.strategy import cutoff_table, expected_loss_table
>>> policies = cutoff_table(mon_pd, y_mon, cutoffs=scale["upper"])
>>> columns = ["approval_rate", "approved_event_rate", "rejected_event_share"]
>>> policies.set_index(scale.index)[columns].round(3)
       approval_rate  approved_event_rate  rejected_event_share
grade
A              0.053                0.025                 0.988
B              0.404                0.060                 0.786
C              0.795                0.079                 0.451
D              0.941                0.093                 0.231
E              1.000                0.114                 0.000

Approving the grades up to C, a PD of at most 14.1%, approves 79.5% of the applicants at a bad rate of 7.9% and turns away 45.1% of the events. Approving grade D as well approves 94.1% at a bad rate of 9.3%. Which row to choose is a business decision; the table shows its price.

The expected loss of an exposure is its PD times its loss given default times its exposure at default. :func:`~kivi.strategy.expected_loss_table` adds it up for each grade. The bank data have no loan amounts, so the example takes the balance, floored at zero, as the exposure and one LGD of 45%:

>>> exposure = monitoring["balance"].clip(lower=0)
>>> by_grade = expected_loss_table(grades, mon_pd, 0.45, exposure)
>>> by_grade[["exposure", "expected_loss", "expected_loss_rate"]].round(
...     {"exposure": 0, "expected_loss": 0, "expected_loss_rate": 4}
... )
     exposure  expected_loss  expected_loss_rate
bin
A      2390.0           30.0              0.0125
B    440743.0        11065.0              0.0251
C    957512.0        46901.0              0.0490
D    769549.0        66399.0              0.0863
E    151415.0        31344.0              0.2070

Grade E holds 6.5% of the exposure but 20.1% of the expected loss of 155,740, and grades D and E together hold 40% of the exposure and 63% of the expected loss.

Report stability
----------------

Compare each variable of the monitoring sample with its development distribution. The status column applies the PSI conventions from :doc:`psi`.

>>> report = pd.DataFrame({
...     "iv": {column: tables[column]["iv"].sum() for column in dev_groups},
...     "psi": {column: psi_table(dev_groups[column], mon_groups[column])["psi"].sum()
...             for column in dev_groups},
... })
>>> report["status"] = pd.cut(report["psi"], [-np.inf, 0.1, 0.25, np.inf],
...                           labels=["stable", "watch", "shift"])
>>> from kivi.stability import csi_table
>>> report["csi"] = pd.Series({
...     column: csi_table(dev_groups[column], mon_groups[column], card.loc[[column]])["csi"].sum()
...     for column in selected
... })
>>> report.sort_values("iv", ascending=False).round(3)
             iv    psi  status    csi
poutcome  0.423  0.003  stable -0.336
pdays     0.360  0.005  stable -0.074
contact   0.215  0.008  stable  0.315
job       0.118  0.001  stable  0.123
balance   0.115  0.000  stable -0.145
housing   0.108  0.000  stable -0.035
age       0.061  0.005  stable    NaN

Every variable is stable here, as expected when the monitoring rows come from the same population. The IV of ``job`` is that of its three groups, 0.118, a little below the 0.137 of its twelve labels in the screening. The ``csi`` column weights each group's change of share with its points, so it is the change of the variable's average points; ``age`` is not in the scorecard and has none. Over the variables of the scorecard, the CSI adds up to the change of the average score:

>>> shift = mon_score.mean() - dev_score.mean()
>>> print(f"sum of csi = {report['csi'].sum():.3f}, score shift = {shift:.3f}")
sum of csi = -0.151, score shift = -0.151

The score itself is monitored the same way, with bands fitted on the development scores:

>>> score_edges = bin_edges(dev_score, bins=5).round(1)
>>> score_psi = psi_table(pd.cut(dev_score, score_edges), pd.cut(mon_score, score_edges))
>>> print(f"score PSI = {score_psi['psi'].sum():.4f}")
score PSI = 0.0020

:doc:`psi` shows what a shift looks like. In production, :func:`~kivi.monitoring.monitoring_table` repeats this PSI for every period together with the known outcomes and the Gini, and :func:`~kivi.monitoring.stability_table` the PSI of every variable; see :doc:`monitoring`. The PDs of the master scale also set the regulatory capital that :doc:`capital` computes. Once the approved loans are booked, :doc:`vintage` follows their delinquency, and :doc:`lifetime` turns it into PD term structures and the IFRS 9 loss allowance.
