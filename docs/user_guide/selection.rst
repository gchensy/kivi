Variable selection
==================

A scorecard is a logistic regression on a short list of WOE-encoded variables. :mod:`kivi.selection` covers the usual steps from the candidates to that list: drop one variable of each strongly correlated pair, check the rest for collinearity and drop the most collinear ones, select variables by their p-values or compare the models of a fixed order, and read the fitted model. Every function takes a DataFrame with one WOE-encoded column per variable, such as the output of :func:`~kivi.woe.woe_transform`, and the functions that fit a model also take the 0/1 target.

Selection follows :doc:`woe_iv`, which encodes the candidates, and comes before :doc:`reject_inference` and the :doc:`scorecard`, which refit and scale the chosen model. :func:`~kivi.selection.correlation_filter` drops correlated variables, :func:`~kivi.selection.vif` measures collinearity, :func:`~kivi.selection.vif_filter` drops the most collinear variables, :func:`~kivi.selection.logistic_regression` fits the model, :func:`~kivi.selection.stepwise` selects its variables and :func:`~kivi.selection.selection_path` reports the models of the first 1, 2, 3 and more variables of an order.

Preparing the candidates
------------------------

Group every candidate as in :doc:`risk_workflow`: quantile bins for most numeric variables, explicit edges where tied values would collapse the quantiles, and the raw labels of categorical variables. Then fit one WOE table per variable on the development sample, rank the variables by IV and encode those with an IV of at least 0.02.

>>> import numpy as np
>>> import pandas as pd
>>> from kivi.binning import bin_edges
>>> from kivi.datasets import load_bank
>>> from kivi.selection import (
...     correlation_filter, logistic_regression, selection_path, stepwise, vif, vif_filter,
... )
>>> from kivi.woe import woe_table, woe_transform
>>> data = load_bank()
>>> development = data.iloc[:3000]
>>> target = development["target"]
>>> edges = {
...     "age": bin_edges(development["age"], bins=5),
...     "balance": bin_edges(development["balance"], bins=5),
...     "day": bin_edges(development["day"], bins=5),
...     "campaign": [-np.inf, 1, 2, 3, np.inf],
...     "pdays": [-np.inf, -1, 100, 200, 300, np.inf],
...     "previous": [-np.inf, 0, 1, 3, np.inf],
... }
>>> groups = pd.DataFrame({
...     column: pd.cut(development[column], column_edges) for column, column_edges in edges.items()
... })
>>> for column in ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]:
...     groups[column] = development[column]
>>> tables = {column: woe_table(groups[column], target) for column in groups}
>>> iv = pd.Series({column: table["iv"].sum() for column, table in tables.items()})
>>> iv = iv.sort_values(ascending=False)
>>> iv.round(3)
poutcome     0.423
pdays        0.360
previous     0.234
contact      0.215
job          0.137
housing      0.108
balance      0.099
loan         0.059
education    0.050
marital      0.049
campaign     0.025
age          0.019
day          0.011
default      0.000
dtype: float64
>>> candidates = iv.index[iv >= 0.02]
>>> X = pd.DataFrame({column: woe_transform(groups[column], tables[column]) for column in candidates})
>>> X.shape
(3000, 11)

Correlated variables
--------------------

Variables that describe the same behavior carry the same information twice. ``poutcome``, ``pdays`` and ``previous`` all describe the previous campaign:

>>> X[["poutcome", "pdays", "previous"]].corr().round(2)
          poutcome  pdays  previous
poutcome      1.00   0.68      0.67
pdays         0.68   1.00      0.71
previous      0.67   0.71      1.00

:func:`~kivi.selection.correlation_filter` visits the variables from the highest ``priority`` to the lowest and keeps a variable unless its absolute correlation with a variable already kept exceeds ``threshold`` (0.7 by default). Pass the IV as the priority, so that of two correlated variables the more predictive one stays:

>>> kept = correlation_filter(X, priority=iv)
>>> X.columns.difference(kept).tolist()
['previous']

.. plot::
   :alt: The eleven candidates in IV order with their largest absolute correlation with a variable kept before them, and previous dropped for pdays
   :caption: ``correlation_filter`` visits the variables from the highest IV to the lowest and keeps each one unless its absolute correlation with a variable already kept exceeds 0.7. ``previous`` is correlated at 0.71 with ``pdays``, which comes first, and is dropped.

   kd.figure_correlation_filter()

``previous`` is dropped for ``pdays``, which has a higher IV. ``poutcome`` and ``pdays`` stay together because their correlation of 0.68 is below the threshold. Thresholds between 0.6 and 0.8 are common; a lower one gives a shorter list.

``method`` chooses the correlation. The default, ``"pearson"``, measures how close two variables are to a straight line. ``"spearman"`` is Spearman's rank correlation, the Pearson correlation of the ranks of the values, as ``pandas.DataFrame.corr`` computes it: it counts any strictly monotone relation as a perfect one and is not moved by a few extreme values. That suits raw variables screened before binning, whose relations are often monotone but not linear and whose tails are long:

>>> raw = development[["age", "balance", "day", "campaign", "pdays", "previous"]]
>>> correlations = {
...     method: raw.corr(method=method).loc["pdays", ["previous", "campaign"]]
...     for method in ("pearson", "spearman")
... }
>>> pd.DataFrame(correlations).round(2)
          pearson  spearman
previous      0.6      0.99
campaign     -0.1     -0.15
>>> correlation_filter(raw, priority=iv).tolist()
['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
>>> correlation_filter(raw, method="spearman", priority=iv).tolist()
['age', 'balance', 'day', 'campaign', 'pdays']

Most customers have ``pdays`` -1 and ``previous`` 0, the codes of a customer who was never contacted before, so the ranks of the two variables agree almost everywhere, while the long right tails of the other customers hold their Pearson correlation down to 0.60. By rank, ``previous`` repeats ``pdays`` and is dropped.

Keep the default for WOE values. The logistic regression combines them linearly, and their linear correlation is what inflates the standard errors of its coefficients, as the VIF below measures. Ranks see only the order of the groups: ``poutcome``, ``pdays`` and ``previous`` all put the customers who were never contacted before in one group, so their Spearman correlations are 0.99, and the filter would keep ``poutcome`` alone.

Variance inflation factor
-------------------------

A correlation compares two variables, but a variable can also be explained by several others together. The variance inflation factor (VIF) regresses each variable on all the others: it is 1 for a variable that the others do not explain and grows without bound as they explain it completely. The square root of the VIF is the factor by which collinearity inflates the standard error of the variable's coefficient.

>>> pd.concat([vif(X).rename("candidates"), vif(X[kept]).rename("kept")], axis=1).round(2)
           candidates  kept
variable
poutcome         2.18  1.90
pdays            2.46  1.93
previous         2.47   NaN
contact          1.16  1.12
job              1.21  1.21
housing          1.12  1.09
balance          1.03  1.03
loan             1.02  1.02
education        1.15  1.15
marital          1.03  1.03
campaign         1.02  1.02

A common rule of thumb flags a VIF above 5, or above 10, as harmful collinearity. Every value here is well below that, and dropping ``previous`` lowers the VIF of the other two variables of the previous campaign.

Dropping collinear variables
----------------------------

:func:`~kivi.selection.vif_filter` drops variables until no VIF exceeds ``threshold``, 5 by default. Each round computes the VIFs of the variables that are still kept and drops the one with the largest VIF. Dropping a variable lowers the VIFs of the variables that it helped to explain, so the VIFs are computed again after every drop, and a variable above the threshold in the first round may stay in the end.

>>> vif_filter(X, priority=iv).equals(X.columns)
True
>>> low = vif_filter(X, threshold=2, priority=iv)
>>> X.columns.difference(low).tolist()
['previous']
>>> print(f"{vif(X[low]).max():.2f}")
1.93
>>> X.columns.difference(vif_filter(X, threshold=1.5, priority=iv)).tolist()
['pdays', 'previous']

Every candidate has a VIF below 5, so the default keeps them all. With a threshold of 2, ``previous`` goes first, with the largest VIF of 2.47, and the VIFs of ``poutcome`` and ``pdays`` then fall to 1.90 and 1.93: the result is the list that ``correlation_filter`` kept. A threshold of 1.5 drops ``pdays`` as well and leaves ``poutcome`` as the only variable of the previous campaign.

Unlike ``correlation_filter``, ``vif_filter`` drops the most collinear variable whatever its IV. ``priority`` decides only between variables whose VIFs tie for the largest. Every variable of an exact linear combination, such as a total next to its parts, has an infinite VIF; such variables go first, and of them the one with the lowest priority, here the lowest IV, is dropped. Without ``priority``, and between equal priorities, the last tied variable in the column order of ``X`` is dropped, so that earlier columns are kept, as in ``correlation_filter``: with the columns of ``X`` sorted by IV, the more predictive variable stays even without ``priority``. VIFs within a relative :math:`10^{-10}` of each other count as tied, so that rounding does not decide between equal VIFs, such as those of the two variables of a pair.

Logistic regression on WOE
--------------------------

:func:`~kivi.selection.logistic_regression` fits the model with an intercept and reports each coefficient with its standard error, its Wald z statistic and the two-sided p-value of that statistic. A model with a single WOE-encoded variable shows why WOE suits a logistic regression:

>>> logistic_regression(X[["poutcome"]], target).round(4)
             coef      se        z  p_value
variable
intercept -2.0309  0.0595 -34.1097      0.0
poutcome   1.0000  0.0824  12.1332      0.0
>>> print(f"{np.log(target.sum() / (len(target) - target.sum())):.4f}")
-2.0309

kivi defines the WOE of a group as :math:`\ln(p_i / q_i)`, the log of its share of events over its share of non-events. That equals the log odds of the group minus the log odds of the whole sample, so an intercept equal to the sample log odds and a coefficient of 1 reproduce the log odds of every group exactly. The coefficient is exactly 1 whenever no cell needed smoothing.

With several variables, each coefficient weighs a variable against the others. A higher WOE always means a higher event rate, so every coefficient should be positive. A negative one means that the other variables reverse the variable's effect, usually because of collinearity. In the model with every candidate, ``previous`` gets a negative coefficient:

>>> logistic_regression(X, target).loc[["poutcome", "pdays", "previous"]].round(3)
           coef     se      z  p_value
variable
poutcome  0.709  0.121  5.880    0.000
pdays     0.368  0.146  2.527    0.012
previous -0.116  0.189 -0.612    0.541

A scorecard built on that model would give fewer points to customers with more previous contacts, although they subscribe more often.

Stepwise selection
------------------

:func:`~kivi.selection.stepwise` selects the variables of the model by their p-values. With the default ``direction="both"``, it starts from the intercept alone and repeatedly adds the variable with the smallest p-value below ``p_enter`` (0.05), then removes any selected variable whose p-value has risen above ``p_remove`` (0.10). ``direction="forward"`` only adds variables and ``direction="backward"`` starts from every variable and only removes them.

>>> selected = stepwise(X[kept], target)
>>> selected.tolist()
['poutcome', 'pdays', 'contact', 'job', 'housing', 'balance', 'loan', 'marital']
>>> logistic_regression(X[selected], target).round(3)
            coef     se       z  p_value
variable
intercept -2.029  0.062 -32.484    0.000
poutcome   0.675  0.113   5.975    0.000
pdays      0.353  0.123   2.863    0.004
contact    0.576  0.153   3.754    0.000
job        0.546  0.172   3.175    0.001
housing    0.572  0.191   2.992    0.003
balance    0.692  0.199   3.479    0.001
loan       0.679  0.275   2.469    0.014
marital    1.005  0.275   3.657    0.000

.. plot::
   :alt: The z statistic of every variable outside the model in each forward step of stepwise, with the variable that enters in each step highlighted
   :caption: The forward steps of ``stepwise`` on the ten kept variables. Each step tries every variable outside the model, and the one with the largest absolute z statistic enters, provided its p-value is below ``p_enter`` and every coefficient stays positive. No variable is removed on the way, and in step 9 neither ``education`` nor ``campaign`` passes, so the selection stops.

   kd.figure_stepwise_path()

``education`` and ``campaign`` were left out because their p-values stay above 0.05.

By default, ``positive=True`` keeps every coefficient positive: a variable enters only if all coefficients of the extended model are positive, and a variable whose coefficient is not positive is removed first. Here ``previous`` fails on its p-value anyway. With lenient levels it would pass the p-value test, and only the sign rule keeps it out:

>>> lenient = {"p_enter": 0.7, "p_remove": 0.7}
>>> "previous" in stepwise(X, target, **lenient, positive=False)
True
>>> "previous" in stepwise(X, target, **lenient)
False

.. note::

   Stepwise selection tests many models on the same data, so the p-values of the selected model are optimistic. Treat the result as a proposal: review each variable with the business, and check the model on a validation sample before using it.

Selection path
--------------

:func:`~kivi.selection.selection_path` fits the models of a fixed order instead of choosing each variable: step :math:`k` is the logistic regression on the first :math:`k` variables of ``order``, by default the columns of ``X`` in their order. Each row reports the variable that the step adds, the ``auc``, ``gini`` and ``ks`` of the model's fitted probability on the development sample, the largest p-value of its variables and their smallest coefficient. With ``validation``, each model also scores a validation sample, which adds ``validation_auc``, ``validation_gini`` and ``validation_ks``.

Encode the other applicants of the bank data with the edges and WOE tables of the development sample, and follow the variables that ``correlation_filter`` kept in the order of ``X``, from the highest IV to the lowest:

>>> holdout = data.iloc[3000:]
>>> holdout_groups = holdout[groups.columns].copy()
>>> for column, column_edges in edges.items():
...     holdout_groups[column] = pd.cut(holdout[column], column_edges)
>>> X_valid = pd.DataFrame({
...     column: woe_transform(holdout_groups[column], tables[column]) for column in candidates
... })
>>> path = selection_path(X[kept], target, validation=(X_valid, holdout["target"]))
>>> path[["variable", "auc", "validation_auc", "max_p_value", "min_coefficient"]].round(3)
       variable    auc  validation_auc  max_p_value  min_coefficient
step
1      poutcome  0.607           0.616        0.000            1.000
2         pdays  0.610           0.614        0.000            0.521
3       contact  0.660           0.674        0.000            0.443
4           job  0.688           0.697        0.002            0.379
5       housing  0.697           0.697        0.002            0.373
6       balance  0.709           0.691        0.004            0.352
7          loan  0.715           0.699        0.011            0.346
8     education  0.718           0.696        0.081            0.332
9       marital  0.729           0.697        0.187            0.341
10     campaign  0.730           0.701        0.227            0.322

The AUC on the development sample rises with every variable, because each model is fitted to that sample. On the validation sample it reaches 0.697 when ``job`` enters at step 4 and stays within 0.01 of that afterwards: the last six variables add 0.042 to the development AUC and 0.004 to the validation AUC. From ``education`` on, the largest p-value is also above 0.05. ``path[["auc", "validation_auc"]].diff()`` shows the gain of each step. ``min_coefficient`` stays positive, so no variable works against its own event rates.

With ``previous`` in the order, right after ``pdays`` as its IV puts it, the smallest coefficient turns negative at step 3 and stays negative:

>>> selection_path(X, target).loc[2:4, ["variable", "max_p_value", "min_coefficient"]].round(3)
      variable  max_p_value  min_coefficient
step
2        pdays        0.000            0.521
3     previous        0.603           -0.095
4      contact        0.196           -0.237

``selection_path`` neither stops early nor removes a variable: the order decides, and the table shows every step, so that the cut is the analyst's choice. ``stepwise`` instead lets the p-values choose each variable. A model of the path whose classes are separated raises :exc:`~kivi.selection.SeparationError`, which names the step; end ``order`` before that variable, or merge its bins.

Sample weights
--------------

:func:`~kivi.selection.logistic_regression`, :func:`~kivi.selection.stepwise` and :func:`~kivi.selection.selection_path` take ``weights``: frequency weights that count each row as that many identical observations. Integer weights therefore give the same coefficients, standard errors and selection as repeating the rows:

>>> weights = np.random.default_rng(0).integers(1, 4, len(X))
>>> repeated = np.repeat(np.arange(len(X)), weights)
>>> weighted = logistic_regression(X[selected], target, weights=weights)
>>> np.allclose(weighted, logistic_regression(X[selected].iloc[repeated], target.iloc[repeated]))
True
>>> stepwise(X[kept], target, weights=weights).equals(
...     stepwise(X[kept].iloc[repeated], target.iloc[repeated])
... )
True

``selection_path`` weights the metrics of the development sample as well, while each row of its validation sample counts once.

The standard errors treat the total weight as the number of independent observations, so they depend on the scale of the weights: doubling every weight divides them by :math:`\sqrt 2` and lowers every p-value, as doubling the sample would.

>>> doubled = logistic_regression(X[selected], target, weights=2 * weights)
>>> print(f"{(weighted['se'] / doubled['se']).mean():.4f}")
1.4142

Weights that do not count observations do not give the standard errors of the data. Those of reject inference in :doc:`reject_inference` make them too small. A rejected applicant becomes a bad and a good row whose outcomes are inferred, not observed, and reweighting lets each approved applicant count as several, but all of these rows count as observations. Read the p-values of such a fit as optimistic, and select the variables on the approved applicants.
