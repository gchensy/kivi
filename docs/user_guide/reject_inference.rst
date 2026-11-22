Reject inference
================

A scorecard is developed on past applicants with a known outcome, and the outcome is known only for the applicants that the previous policy approved. The model learns from them and is then applied to every applicant, including the kind that used to be declined. Reject inference supplies outcomes for the rejected applicants so that the model can be refitted on all of them. :mod:`kivi.reject_inference` returns those outcomes as rows with weights, or returns weights for the approved applicants. The ``weights`` arguments of :func:`~kivi.selection.logistic_regression`, :func:`~kivi.woe.woe_table` and :func:`~kivi.spec.fit_spec` take either.

Reject inference follows :doc:`variable selection <selection>`, whose model it refits on all applicants, and comes before the :doc:`scorecard`, which scales the refitted model. :func:`~kivi.reject_inference.fuzzy_augmentation` and :func:`~kivi.reject_inference.parceling` return inferred rows, and :func:`~kivi.reject_inference.reweighting` returns weights for the approved applicants.

Every method rests on an assumption about the rejected applicants that the approved ones cannot confirm. The examples therefore use synthetic applications whose outcomes are known for everyone, so that each method can be checked against the model the full outcomes would give. Real data never allow that check.

Applications and approvals
--------------------------

:func:`~kivi.datasets.make_applications` simulates loan applications decided by an incumbent scorecard and two policy rules; :doc:`datasets` describes them. Policy declines would be declined whatever their score, so they are usually left out of the development population, as here. The approved applicants are the *accepts* and the others the *rejects*:

>>> import numpy as np
>>> import pandas as pd
>>> from kivi.datasets import make_applications
>>> applications = make_applications(rng=0)
>>> population = applications[~applications["policy_decline"]]
>>> accepts = population[population["approved"]]
>>> rejects = population[~population["approved"]]
>>> len(accepts), len(rejects)
(6770, 2467)

In practice, ``target`` would be missing for the rejects. Here it shows what reject inference tries to recover: the rejects default three times as often as the accepts.

>>> population.groupby("approved")["target"].mean().round(3)
approved
False    0.301
True     0.097
Name: target, dtype: float64

The accepts-only model
----------------------

All the models below use the same groups and the same WOE encoding, fitted on the accepts, so that their coefficients can be compared. Quantile edges need no outcome, so they are fitted on every applicant of the population. The number of inquiries takes few values and gets explicit edges:

>>> from kivi.binning import bin_edges
>>> from kivi.selection import logistic_regression
>>> from kivi.spec import fit_spec
>>> features = ["age", "income", "debt_to_income", "inquiries", "housing", "employment"]
>>> edges = {
...     column: bin_edges(population[column], bins=5)
...     for column in ["age", "income", "debt_to_income"]
... }
>>> edges["inquiries"] = [-np.inf, 0, 1, 2, np.inf]
>>> spec = fit_spec(accepts[features], accepts["target"], bins=edges)
>>> model = logistic_regression(spec.woe_transform(accepts), accepts["target"])
>>> model.round(3)
                 coef     se       z  p_value
variable
intercept      -2.230  0.043 -52.398      0.0
age             1.223  0.303   4.030      0.0
income          1.239  0.231   5.374      0.0
debt_to_income  1.356  0.182   7.436      0.0
inquiries       1.427  0.195   7.323      0.0
housing         1.057  0.204   5.174      0.0
employment      1.388  0.221   6.268      0.0

The PD of the model is the logistic function of the intercept plus the WOE values times the coefficients:

>>> def predict(model, frame, spec=spec):
...     coef = model["coef"]
...     log_odds = coef["intercept"] + spec.woe_transform(frame) @ coef.drop("intercept")
...     return 1 / (1 + np.exp(-log_odds))
>>> accept_pd, reject_pd = predict(model, accepts), predict(model, rejects)
>>> print(f"{reject_pd.mean():.3f}")
0.191

The model predicts a bad rate of 19.1% for the rejects, who actually default at 30.1%. The incumbent scorecard used information that is not among the features, like an underwriter who reads a credit bureau report, so the rejects are riskier than accepts with the same features. A model fitted on the accepts alone cannot see that.

Fuzzy augmentation
------------------

:func:`~kivi.reject_inference.fuzzy_augmentation` splits each reject into a bad row, weighted with its PD, and a good row with the rest of its weight of 1. The rows repeat the index of the rejects, so ``rejects.loc[inferred.index]`` gives their features:

>>> from kivi.reject_inference import fuzzy_augmentation, parceling, reweighting
>>> inferred = fuzzy_augmentation(reject_pd)
>>> inferred.head(4).round(3)
    target  weight
14       1   0.423
14       0   0.577
15       1   0.250
15       0   0.750

To refit, concatenate the accepts at weight 1 with the inferred rows and pass the weights. Only the features of the rejects enter the fit, never their ``target`` column:

>>> def refit(inferred, accept_weight=1.0):
...     frame = pd.concat([accepts[features], rejects.loc[inferred.index, features]])
...     target = pd.concat([accepts["target"], inferred["target"]])
...     weights = pd.concat([pd.Series(accept_weight, index=accepts.index), inferred["weight"]])
...     return logistic_regression(spec.woe_transform(frame), target, weights=weights)
>>> fuzzy = refit(inferred)
>>> np.allclose(fuzzy["coef"], model["coef"])
True
>>> (fuzzy["se"] / model["se"]).round(2)
variable
intercept         0.91
age               0.75
income            0.72
debt_to_income    0.80
inquiries         0.68
housing           0.78
employment        0.72
Name: se, dtype: float64

The refitted coefficients are those of the accepts-only model. At those coefficients, the bad row of each reject weighs exactly its fitted PD, so the inferred rows leave the fit where it was. Only the standard errors shrink, although the inferred rows add no information: they count as observations, so the standard errors of any fit with inferred rows are too small. Choose the variables on the accepts, not by the p-values of an augmented fit. With a factor of 1, fuzzy augmentation changes the model only if the encoding is refitted on the augmented sample or the variables change.

Sensitivity to the factor
-------------------------

``factor`` multiplies the PDs of the rejects, so a factor above 1 assumes that they are riskier than accepts with the same PD. The accepts cannot tell how much riskier, so compare several factors. ``summary`` gathers three results: the inferred bad rate of the rejects, their mean PD under the refitted model, and the Gini on a second sample from the same generator, where every applicant's outcome is known. The *known* model, fitted on all applicants with their true outcomes, is the benchmark that real data never provide; its inferred rate is the rejects' true bad rate:

>>> from kivi.metrics import gini
>>> holdout = make_applications(rng=1)
>>> holdout = holdout[~holdout["policy_decline"]]
>>> def summary(model, inferred=None):
...     rate = np.nan
...     if inferred is not None:
...         rate = np.average(inferred["target"], weights=inferred["weight"])
...     return {
...         "inferred": rate,
...         "reject_pd": predict(model, rejects).mean(),
...         "gini": gini(predict(model, holdout), holdout["target"]),
...     }
>>> known = logistic_regression(spec.woe_transform(population), population["target"])
>>> models = {"accepts": model, "known": known}
>>> results = {"accepts": summary(model), "known": summary(known, rejects.assign(weight=1.0))}
>>> for factor in [1.25, 1.5, 2]:
...     inferred = fuzzy_augmentation(reject_pd, factor=factor)
...     models[f"fuzzy {factor}"] = refit(inferred)
...     results[f"fuzzy {factor}"] = summary(models[f"fuzzy {factor}"], inferred)
>>> pd.DataFrame(results).T.round(3)
            inferred  reject_pd   gini
accepts          NaN      0.191  0.422
known          0.301      0.252  0.425
fuzzy 1.25     0.238      0.221  0.421
fuzzy 1.5      0.286      0.251  0.420
fuzzy 2        0.381      0.312  0.419

A factor of 1.5 brings the inferred bad rate of the rejects to 28.6%, near their true 30.1%, and their mean PD under the refitted model to 25.1%, close to the known model's 25.2%. Even the known model predicts less than 30.1% for the rejects: it is fitted on all applicants and lacks the information that set the rejects apart. The Gini hardly moves. On these data, reject inference corrects how risky the model takes the rejects to be, not how it ranks applicants.

.. plot::
   :alt: Inferred bad rate and mean refitted PD of the rejects for factors from 1 to 2.5, with the true bad rate of the rejects and the known model's mean PD as dashed lines
   :caption: The inferred bad rate and the mean refitted PD of the rejects rise with ``factor``. The dashed lines are the true bad rate of the rejects and the mean PD that the known model gives them; a factor of 1.5 meets the second.

   kp.figure_reject_factor()

Here the known outcomes say which factor fits. With real data, the factor is an assumption, best supported from outside the development sample: the bureau outcomes of rejects who borrowed elsewhere, a small sample of rejects approved on purpose, or business judgement. Report the results for a range of factors, as above.

Parceling
---------

:func:`~kivi.reject_inference.parceling` divides the scores of accepts and rejects into bands, by default ten quantile bands of both samples together. In each band, the rejects are labelled bad at the accepts' bad rate times ``factor``. The labels are drawn with a local generator, so a seed in ``rng`` makes them reproducible:

>>> parcels = parceling(accept_pd, accepts["target"], reject_pd, factor=1.5, rng=0)
>>> parcels["target"].value_counts()
target
0    1818
1     649
Name: count, dtype: int64

The rejects are labelled bad at 26.3%, less than the 28.6% that fuzzy augmentation infers with the same factor. Parceling knows only the band of a reject, not its features or its own PD. The riskiest band is open above and holds almost a third of the rejects, and they all get the bad rate of the band's accepts, which is below the mean PD that the model gives them.

The draw adds noise to the refit. Over ten seeds, the Gini on the second sample varies by several thousandths:

>>> ginis = [
...     summary(refit(parceling(accept_pd, accepts["target"], reject_pd, factor=1.5, rng=seed)))
...     for seed in range(10)
... ]
>>> print(f"{min(g['gini'] for g in ginis):.3f} to {max(g['gini'] for g in ginis):.3f}")
0.418 to 0.425

``expected=True`` instead returns the band's rate as the weight of a bad row and its complement as the weight of a good row, like :func:`~kivi.reject_inference.fuzzy_augmentation`. That removes the noise of the draw, so the comparison below uses it:

>>> parcels = parceling(accept_pd, accepts["target"], reject_pd, factor=1.5, expected=True)
>>> models["parceling"] = refit(parcels)
>>> results["parceling"] = summary(models["parceling"], parcels)

Reweighting
-----------

:func:`~kivi.reject_inference.reweighting` adds no rows. In each score band, it gives every accept the weight (accepts + rejects) / accepts, so that the weighted accepts stand for all applicants of their band:

>>> weights = reweighting(accept_pd, reject_pd)
>>> print(f"{weights.min():.2f} to {weights.max():.2f}, total {weights.sum():.0f}")
1.01 to 6.33, total 9237
>>> models["reweighting"] = logistic_regression(
...     spec.woe_transform(accepts), accepts["target"], weights=weights
... )
>>> results["reweighting"] = summary(models["reweighting"])

Reweighting assumes that rejects default like accepts with the same score, which is exactly what fails here. The weights add up to all 9,237 applicants of the population, although only the 6,770 accepts have an observed outcome, so the standard errors of the fit are too small. Each band must also hold accepts. On the incumbent PD, which decided the approval with a hard cut-off, the bands above the cut-off hold only rejects:

>>> reweighting(accepts["incumbent_pd"], rejects["incumbent_pd"])
Traceback (most recent call last):
...
ValueError: the score band (0.17118, 0.22683999999999996] holds 924 rejected applicant(s) but no accepted one, so no accepted applicant can stand in for them; use fewer bins or wider edges

Hard cut-off
------------

The simplest method needs no function. Score the rejects with the accepts-only model, label those above a PD cut-off bad and the others good, and refit. Here the cut-off labels the riskiest 30% of the rejects bad, close to their true bad rate; like the factor, that share is chosen with the known outcomes:

>>> cut_off = reject_pd.quantile(0.7)
>>> hard = pd.DataFrame({"target": (reject_pd > cut_off).astype(int), "weight": 1.0})
>>> models["cut-off"] = refit(hard)
>>> results["cut-off"] = summary(models["cut-off"], hard)

Comparison
----------

>>> pd.DataFrame(results).T.round(3)
             inferred  reject_pd   gini
accepts           NaN      0.191  0.422
known           0.301      0.252  0.425
fuzzy 1.25      0.238      0.221  0.421
fuzzy 1.5       0.286      0.251  0.420
fuzzy 2         0.381      0.312  0.419
parceling       0.263      0.229  0.421
reweighting       NaN      0.186  0.422
cut-off         0.300      0.311  0.418
>>> compared = ["accepts", "known", "fuzzy 1.5", "parceling", "reweighting", "cut-off"]
>>> pd.DataFrame({name: models[name]["coef"] for name in compared}).round(3)
                accepts  known  fuzzy 1.5  parceling  reweighting  cut-off
variable
intercept        -2.230 -2.067     -2.130     -2.104       -2.228   -2.574
age               1.223  1.640      1.652      1.404        1.201    2.644
income            1.239  1.345      1.624      1.373        1.146    2.980
debt_to_income    1.356  1.824      1.798      1.627        1.391    2.574
inquiries         1.427  1.899      1.901      1.596        1.286    3.459
housing           1.057  1.136      1.114      0.938        1.006    2.545
employment        1.388  1.504      1.817      1.542        1.393    3.192

With the accepts' WOE, the known model weighs every variable more than the accepts-only model: across all applicants, the features separate good from bad more strongly than among the accepts. The methods compare as follows:

* **Ranking.** The Gini on the second sample stays between 0.418 and 0.422 for every method in the table, and the known model reaches only 0.425; the random labels of parceling reach that only by chance. The extra risk of the rejects comes from information that no feature holds, so no method can recover it.
* **Fuzzy augmentation and parceling.** With a factor above 1, both raise the predicted bad rate of the rejects towards their true rate and move most coefficients towards those of the known model. With a factor of 1.5, fuzzy augmentation brings the intercept and four of the six variables within 0.07 of the known model, but overshoots ``income`` and ``employment``. Parceling lowers the coefficient of ``housing``, which the known model raises. Both methods help only as far as the factor is right.
* **Reweighting.** It does not help here: weights cannot make the rejects riskier than the accepts of their band.
* **Hard cut-off.** It treats the rejects above the cut-off as certainly bad and the others as certainly good, so the inferred outcomes follow the model's own score without error. The refit multiplies the coefficient of every variable by 1.9 to 2.4 and ranks slightly worse.

Refitting the WOE
-----------------

The comparisons keep the accepts' WOE encoding. In practice, the groups are often refitted on the augmented sample with the inferred weights, so that the rejects also inform the WOE of the groups they fill. :func:`~kivi.spec.fit_spec` takes the weights:

>>> inferred = fuzzy_augmentation(reject_pd, factor=1.5)
>>> frame = pd.concat([accepts[features], rejects.loc[inferred.index, features]])
>>> target = pd.concat([accepts["target"], inferred["target"]])
>>> weights = pd.concat([pd.Series(1.0, index=accepts.index), inferred["weight"]])
>>> augmented_spec = fit_spec(frame, target, bins=edges, weights=weights)
>>> augmented = logistic_regression(augmented_spec.woe_transform(frame), target, weights=weights)
>>> known_spec = fit_spec(population[features], population["target"], bins=edges)
>>> known_refit = logistic_regression(known_spec.woe_transform(population), population["target"])
>>> pd.Series({
...     "accepts": gini(predict(model, holdout), holdout["target"]),
...     "fuzzy 1.5, WOE refitted": gini(
...         predict(augmented, holdout, augmented_spec), holdout["target"]
...     ),
...     "known, WOE refitted": gini(predict(known_refit, holdout, known_spec), holdout["target"]),
... }).round(3)
accepts                    0.422
fuzzy 1.5, WOE refitted    0.425
known, WOE refitted        0.430
dtype: float64

Refitting the WOE raises the Gini from 0.422 to 0.425, against 0.430 for the known outcomes with their own WOE. It recovers part of a gap that is small to begin with, and it does about as well with a factor of 1: the gain comes from refitting the WOE of the groups that the rejects fill, not from the factor.

Limits
------

* The methods infer outcomes from the accepts, so they cannot reveal what the accepts do not show. A factor above 1 encodes how much riskier the rejects are, and the data cannot estimate it.
* The weights of every method make the standard errors of :func:`~kivi.selection.logistic_regression` too small and the p-values of :func:`~kivi.selection.stepwise` too low, because inferred rows and extra weight count as observations. Select the variables on the accepts and read the p-values of a refit as optimistic.
* kivi does not implement the Heckman two-stage model or other bivariate probit models of sample selection, nor iterative reclassification.
