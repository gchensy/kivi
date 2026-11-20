Risk modeling workflow
======================

This example follows the steps of a scorecard project with kivi: screen the candidate variables, bin the shortlist, encode it with WOE, select the variables and fit the model, scale it to a scorecard, evaluate and calibrate it, and prepare the stability report used in monitoring. The bundled data record a marketing response rather than a default, but the steps are the same for a credit default flag: use the bad flag as ``target``.

>>> import warnings
>>> import numpy as np
>>> import pandas as pd
>>> from kivi.binning import bin_edges
>>> from kivi.datasets import Dataset
>>> from kivi.stability import psi_table
>>> from kivi.woe import woe_table, woe_transform
>>> data = Dataset.bank_data()
>>> development, monitoring = data.iloc[:3000], data.iloc[3000:]

Screen variables by IV
----------------------

Start with a quick, uniform binning of every candidate: five quantile bins for numeric variables and the raw labels for categorical ones. The IV of each variable gives a first ranking. ``bin_edges`` warns when tied values leave fewer bins than requested; the loop records those variables instead of printing the warnings.

>>> iv, tied = {}, []
>>> for column in development.columns.drop("target"):
...     values = development[column]
...     if pd.api.types.is_numeric_dtype(values):
...         with warnings.catch_warnings(record=True) as caught:
...             warnings.simplefilter("always")
...             values = pd.cut(values, bin_edges(values, bins=5))
...         if caught:
...             tied.append(column)
...     iv[column] = woe_table(values, development["target"])["iv"].sum()
>>> tied
['campaign', 'pdays', 'previous']
>>> iv = pd.Series(iv).sort_values(ascending=False)
>>> iv.round(3).head(8)
duration    1.594
poutcome    0.423
month       0.334
contact     0.215
job         0.137
housing     0.108
balance     0.099
loan        0.059
dtype: float64

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

Each variable on the shortlist gets edges that can be explained: merged balance bins with a monotonic WOE, a bin of its own for the ``pdays == -1`` code, policy age bands, and the raw labels of the categorical variables. Keep the binning in one place so that development and monitoring data are grouped identically.

>>> fine = bin_edges(development["balance"], bins=10)
>>> numeric_edges = {
...     "balance": fine[[0, 3, 7, 10]],
...     "pdays": np.array([-np.inf, -1, 100, 200, 300, np.inf]),
...     "age": np.array([-np.inf, 25, 35, 45, 55, 65, np.inf]),
... }
>>> categorical = ["poutcome", "contact", "housing", "job"]
>>> def to_groups(frame):
...     groups = {column: pd.cut(frame[column], edges) for column, edges in numeric_edges.items()}
...     groups.update({column: frame[column] for column in categorical})
...     return pd.DataFrame(groups)
>>> dev_groups = to_groups(development)
>>> tables = {column: woe_table(dev_groups[column], development["target"]) for column in dev_groups}

``tables`` holds one ``woe_table`` result per variable. Review each one as a WOE chart (see :doc:`woe_iv`) before moving on. ``pdays`` and ``poutcome`` both describe the previous campaign, so check how strongly they are correlated before keeping both.

Encode with WOE
---------------

A logistic regression on WOE values is the classic scorecard model. Replace each group with the WOE from the development tables using ``woe_transform``:

>>> def to_woe(groups):
...     return pd.DataFrame({
...         column: woe_transform(groups[column], tables[column]) for column in groups
...     })
>>> X_dev = to_woe(dev_groups)
>>> X_dev.head(3).round(3)
   balance  pdays    age  poutcome  contact  housing    job
0    0.373 -0.257 -0.032    -0.257    0.241    0.333 -0.233
1    0.373 -0.092 -0.032     0.214    0.241   -0.327 -0.339
2    0.373 -0.092 -0.032     0.214    0.241   -0.327  0.271

``woe_transform`` raises an error for a group that never appeared in development, such as a new category. Agree on a policy before scoring new data, for example ``unseen=0.0`` for a neutral WOE.

Select variables and fit the model
----------------------------------

Strongly correlated variables carry the same information and make the coefficients unstable. ``correlation_filter`` keeps the variable with the higher IV from each pair correlated above 0.7, and ``stepwise`` then selects variables by the p-values of a logistic regression. With WOE inputs every coefficient should be positive, and ``stepwise`` rejects a variable that would make one negative; see :doc:`selection`.

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
intercept -2.028  0.062 -32.694    0.000
balance    0.776  0.183   4.245    0.000
pdays      0.346  0.122   2.834    0.005
poutcome   0.671  0.112   5.985    0.000
contact    0.580  0.153   3.793    0.000
housing    0.597  0.190   3.138    0.002
job        0.632  0.170   3.721    0.000

No pair is correlated strongly enough to drop a variable, and ``stepwise`` leaves out ``age``. The model is fitted on the development sample only.

Scale to a scorecard
--------------------

``points_table`` turns the WOE tables and the coefficients into points for every group. Here 20 points double the odds of a good outcome, and a score of 600 stands for good:bad odds of 50; see :doc:`scorecard`. A higher WOE means more risk and therefore fewer points:

>>> from kivi.scorecard import points_table, woe_to_points
>>> scale = dict(pdo=20, base_score=600, base_odds=50)
>>> coefficients = model["coef"].drop("intercept")
>>> intercept = model.loc["intercept", "coef"]
>>> card = points_table({column: tables[column] for column in selected}, coefficients, intercept, **scale)
>>> card.loc["poutcome"].round({"woe": 3, "points": 1})
              bin    woe  points
variable
poutcome  unknown -0.257    95.9
poutcome  failure  0.214    86.8
poutcome    other  0.592    79.5
poutcome  success  2.548    41.6

Score the monitoring sample with the development tables: each customer's score is the sum of their points.

>>> mon_groups = to_groups(monitoring)
>>> mon_points = woe_to_points(to_woe(mon_groups)[selected], coefficients, intercept, **scale)
>>> mon_score = mon_points.sum(axis=1)
>>> mon_points.assign(score=mon_score).head(3).round(1)
      balance  pdays  poutcome  contact  housing   job  score
3000     91.0   93.5      95.9     87.5     85.2  90.0  543.1
3001    102.6   93.5      95.9    106.3     96.6  86.0  580.9
3002     82.6   93.5      95.9     86.9     96.6  97.1  552.6

Evaluate the model
------------------

A higher score means lower risk, while the measures in :doc:`metrics` expect a score that rises with the event probability. Pass the probability that each score stands for. The gains table uses bands fitted on the development scores, so that both samples are cut alike:

>>> from kivi.metrics import gains_table, gini, ks
>>> from kivi.scorecard import score_to_pd
>>> dev_score = woe_to_points(X_dev[selected], coefficients, intercept, **scale).sum(axis=1)
>>> dev_pd, mon_pd = score_to_pd(dev_score, **scale), score_to_pd(mon_score, **scale)
>>> y_mon = monitoring["target"]
>>> print(f"Gini = {gini(mon_pd, y_mon):.3f}, KS = {ks(mon_pd, y_mon):.3f}")
Gini = 0.375, KS = 0.285
>>> pd_edges = bin_edges(dev_pd, bins=5).round(3)
>>> gains = gains_table(mon_pd, y_mon, bins=pd_edges)
>>> gains[["total", "event_rate", "lift", "cum_event_share"]].round(3)
                total  event_rate   lift  cum_event_share
bin
(0.144, inf]      356       0.233  2.050            0.480
(0.102, 0.144]    250       0.104  0.914            0.630
(0.076, 0.102]    296       0.095  0.832            0.792
(0.05, 0.076]     291       0.069  0.604            0.908
(-inf, 0.05]      328       0.049  0.429            1.000

The riskiest band, about a fifth of the monitoring sample, holds 48% of its events at about twice the average event rate.

Check calibration
-----------------

Group the monitoring sample into rating grades by PD and compare each grade's event rate with the mean PD of its customers; see :doc:`calibration`. The Jeffreys test asks whether a grade has more events than its PD predicts, so a small p-value flags a grade that underestimates risk:

>>> from kivi.calibration import calibration_table, jeffreys_test
>>> grades = pd.cut(mon_pd, [0, 0.05, 0.1, 0.2, 0.4, 1], labels=list("ABCDE"), include_lowest=True)
>>> grade_table = calibration_table(grades, y_mon, mon_pd)
>>> grade_table["jeffreys"] = jeffreys_test(
...     grade_table["event"], grade_table["total"], grade_table["mean_probability"]
... )
>>> grade_table.round(3)
     total  event  event_rate  mean_probability  jeffreys
bin
A      328     16       0.049             0.038     0.155
B      520     43       0.083             0.074     0.219
C      529     68       0.129             0.138     0.725
D       98     18       0.184             0.275     0.981
E       46     28       0.609             0.589     0.395

No grade is rejected at the 5% level. Grade D has fewer events than predicted, which this one-sided test does not flag.

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
>>> report.sort_values("iv", ascending=False).round(3)
             iv    psi  status
poutcome  0.423  0.003  stable
pdays     0.360  0.005  stable
contact   0.215  0.008  stable
job       0.137  0.005  stable
balance   0.115  0.000  stable
housing   0.108  0.000  stable
age       0.061  0.005  stable

Every variable is stable here, as expected when the monitoring rows come from the same population. The score itself is monitored the same way, with bands fitted on the development scores:

>>> score_edges = bin_edges(dev_score, bins=5).round(1)
>>> score_psi = psi_table(pd.cut(dev_score, score_edges), pd.cut(mon_score, score_edges))
>>> print(f"score PSI = {score_psi['psi'].sum():.4f}")
score PSI = 0.0029

:doc:`psi` shows what a shift looks like and how to track PSI period by period.
