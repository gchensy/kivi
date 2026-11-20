Risk modeling workflow
======================

This example follows the steps of a scorecard project with kivi: screen the candidate variables, bin the shortlist, encode it with WOE for a model and prepare the stability report used in monitoring. The bundled data record a marketing response rather than a default, but the steps are the same for a credit default flag: use the bad flag as ``target``.

>>> import warnings
>>> import numpy as np
>>> import pandas as pd
>>> from kivi.binning import bin_edges
>>> from kivi.datasets import Dataset
>>> from kivi.evaluate import psi_table
>>> from kivi.woe import woe_table
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

A logistic regression on WOE values is the classic scorecard model. Replace each group with the WOE from the development tables:

>>> def to_woe(groups):
...     return pd.DataFrame({
...         column: groups[column].map(tables[column]["woe"]).astype(float)
...         for column in groups
...     })
>>> X_dev = to_woe(dev_groups)
>>> X_dev.head(3).round(3)
   balance  pdays    age  poutcome  contact  housing    job
0    0.373 -0.257 -0.032    -0.257    0.241    0.333 -0.233
1    0.373 -0.092 -0.032     0.214    0.241   -0.327 -0.339
2    0.373 -0.092 -0.032     0.214    0.241   -0.327  0.271

Fit the model with the library of your choice, for example scikit-learn or statsmodels; kivi depends on neither. A category that never appeared in development maps to a missing value, so decide on a default, such as a WOE of 0, before scoring new data.

Report stability
----------------

Group the monitoring sample with the same function and compare each variable with its development distribution. The status column applies the PSI conventions from :doc:`psi`.

>>> mon_groups = to_groups(monitoring)
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

Every variable is stable here, as expected when the monitoring rows come from the same population. :doc:`psi` shows what a shift looks like and how to track PSI period by period.
