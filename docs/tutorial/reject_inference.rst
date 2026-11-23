Reject inference
================

The model of :doc:`model` learned from the approved applicants only, but it will decide on every applicant, including the kind that the vendor's scorecard used to decline. This chapter infers the outcomes of the rejects and refits the model on all eligible applicants; :doc:`/user_guide/reject_inference` compares the methods in depth.

.. dropdown:: Code from the previous chapters

   >>> import numpy as np
   >>> import pandas as pd
   >>> from kivi.datasets import make_applications
   >>> from kivi.selection import logistic_regression, stepwise
   >>> from kivi.spec import fit_spec
   >>> applications = make_applications(n_applicants=40000, rng=0)
   >>> months = pd.period_range("2024-01", periods=12, freq="M")
   >>> applications["month"] = months[np.arange(40000) * 12 // 40000]
   >>> eligible = applications[~applications["policy_decline"]]
   >>> accepts = eligible[eligible["approved"]]
   >>> development = accepts[accepts["month"] <= "2024-09"]
   >>> validation = accepts[accepts["month"] >= "2024-10"]
   >>> rejects = eligible[~eligible["approved"] & (eligible["month"] <= "2024-09")]
   >>> features = ["age", "income", "debt_to_income", "inquiries", "housing", "employment", "incumbent_pd"]
   >>> method = {column: "monotonic" for column in ["age", "income", "debt_to_income", "incumbent_pd"]}
   >>> method["employment"] = "chimerge"
   >>> bins = {"inquiries": [-np.inf, 0, 1, 2, np.inf], "incumbent_pd": 8}
   >>> spec = fit_spec(development[features], development["target"], method=method, bins=bins)
   >>> selected = stepwise(spec.woe_transform(development), development["target"])
   >>> accepts_model = logistic_regression(spec.woe_transform(development)[selected], development["target"])
   >>> def predict(model, frame):
   ...     slopes = model["coef"].drop("intercept")
   ...     log_odds = model.loc["intercept", "coef"] + spec.woe_transform(frame)[slopes.index] @ slopes
   ...     return 1 / (1 + np.exp(-log_odds))

Score the rejects
-----------------

The accepts-only model gives the rejects of the development months a mean PD of about 20%, twice the default rate of the accepts:

>>> reject_pd = predict(accepts_model, rejects)
>>> print(f"{reject_pd.mean():.3f}")
0.199

The accepts cannot show whether that is right. The vendor's scorecard declined these applicants partly on bureau data that the model does not see, so they are probably riskier than accepts with the same characteristics.

.. admonition:: Simulation check
   :class: kivi-check

   The rejects default at 29.5%, so the accepts-only model underestimates their risk by a third:

   >>> print(f"{rejects['target'].mean():.3f}")
   0.295

Infer their outcomes
--------------------

:func:`~kivi.reject_inference.fuzzy_augmentation` splits each reject into a bad row, weighted with its PD times ``factor``, and a good row with the rest of its weight of 1. A factor above 1 assumes that the rejects are riskier than their PD says. :func:`~kivi.reject_inference.parceling` labels the rejects of each score band at the band's bad rate times ``factor``, and :func:`~kivi.reject_inference.reweighting` adds no rows and instead gives the accepts of each band the weight (accepts + rejects) / accepts. The inferred rows join the development sample with weight 1 for the accepts:

>>> from kivi.reject_inference import fuzzy_augmentation, parceling, reweighting
>>> def refit(inferred):
...     frame = pd.concat([development[features], rejects.loc[inferred.index, features]])
...     target = pd.concat([development["target"], inferred["target"]])
...     weights = pd.concat([pd.Series(1.0, index=development.index), inferred["weight"]])
...     return logistic_regression(spec.woe_transform(frame)[selected], target, weights=weights)

Each method gives the rejects an inferred bad rate and, after the refit, a mean PD:

>>> accept_pd = predict(accepts_model, development)
>>> candidates = {
...     f"fuzzy {factor:g}": fuzzy_augmentation(reject_pd, factor=factor) for factor in [1, 1.5, 2]
... }
>>> candidates["parceling 1.5"] = parceling(
...     accept_pd, development["target"], reject_pd, factor=1.5, expected=True
... )
>>> comparison = {
...     name: {
...         "inferred": np.average(inferred["target"], weights=inferred["weight"]),
...         "reject_pd": predict(refit(inferred), rejects).mean(),
...     }
...     for name, inferred in candidates.items()
... }
>>> weighted = logistic_regression(
...     spec.woe_transform(development)[selected],
...     development["target"],
...     weights=reweighting(accept_pd, reject_pd),
... )
>>> comparison["reweighting"] = {"inferred": np.nan, "reject_pd": predict(weighted, rejects).mean()}
>>> pd.DataFrame(comparison).T.round(3)
               inferred  reject_pd
fuzzy 1           0.199      0.199
fuzzy 1.5         0.299      0.265
fuzzy 2           0.399      0.333
parceling 1.5     0.297      0.263
reweighting         NaN      0.206

With a factor of 1, fuzzy augmentation infers exactly what the accepts-only model predicts and leaves the model where it was. Reweighting cannot make the rejects riskier than the accepts of their band. A factor of 1.5, for fuzzy augmentation or parceling, raises the inferred bad rate to about 30%.

The factor is an assumption that the approved applicants cannot confirm. The project assumes that the rejects default half as often again as their PD says, a factor of 1.5, and keeps fuzzy augmentation, which uses each reject's own PD rather than the band it falls into.

.. admonition:: Simulation check
   :class: kivi-check

   A factor of 1.5 brings the inferred bad rate within half a percentage point of the rejects' true 29.5%. On real data the factor is a judgment, and a sensitivity analysis as in :doc:`/user_guide/reject_inference` shows how much the results depend on it.

Refit the model
---------------

The final model is fitted on the accepts and the inferred rejects together:

>>> model = refit(fuzzy_augmentation(reject_pd, factor=1.5))
>>> model.round(3)
                 coef     se       z  p_value
variable
intercept      -2.106  0.024 -89.555      0.0
age             0.745  0.104   7.176      0.0
income          0.733  0.111   6.602      0.0
debt_to_income  0.870  0.101   8.603      0.0
inquiries       0.940  0.085  11.022      0.0
housing         1.075  0.053  20.333      0.0
employment      0.998  0.138   7.212      0.0
incumbent_pd    1.003  0.051  19.669      0.0

Every coefficient grows: across all eligible applicants, the characteristics separate good from bad more strongly than among the accepts alone. The intercept rises, because the sample now includes the riskier rejects. The standard errors, however, are too small: the inferred rows count as observations although they add no information. That is why the variables were selected on the accepts in :doc:`model`, not on the augmented sample.
