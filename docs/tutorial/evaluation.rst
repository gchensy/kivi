Model evaluation
================

The model was fitted on the applications of January to September. This chapter measures how well it ranks the approved applicants of October to December, which it has not seen, and whether it ranks them better than the vendor's PD; :doc:`/user_guide/metrics` defines each measure.

.. dropdown:: Code from the previous chapters

   >>> import numpy as np
   >>> import pandas as pd
   >>> from kivi.datasets import make_applications
   >>> from kivi.reject_inference import fuzzy_augmentation
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
   >>> inferred = fuzzy_augmentation(predict(accepts_model, rejects), factor=1.5)
   >>> frame = pd.concat([development[features], rejects.loc[inferred.index, features]])
   >>> target = pd.concat([development["target"], inferred["target"]])
   >>> weights = pd.concat([pd.Series(1.0, index=development.index), inferred["weight"]])
   >>> model = logistic_regression(spec.woe_transform(frame)[selected], target, weights=weights)

Measure the ranking
-------------------

The measures expect a score that rises with the risk, so the PD is passed as it is. :func:`~kivi.metrics.roc_auc`, :func:`~kivi.metrics.gini` and :func:`~kivi.metrics.ks` measure how well it separates defaults from non-defaults, and :func:`~kivi.metrics.brier_score` how close the PDs come to the outcomes:

>>> from kivi.metrics import brier_score, delong_test, gains_table, gini, ks, roc_auc
>>> from kivi.metrics import roc_auc_interval, roc_table
>>> probability = predict(model, validation)
>>> y = validation["target"]
>>> pd.Series(
...     {
...         "roc_auc": roc_auc(probability, y),
...         "gini": gini(probability, y),
...         "ks": ks(probability, y),
...         "brier_score": brier_score(probability, y),
...     }
... ).round(3)
roc_auc        0.686
gini           0.373
ks             0.290
brier_score    0.091
dtype: float64

A sample of 6,785 applications with 722 defaults measures the AUC with some uncertainty. :func:`~kivi.metrics.roc_auc_interval` gives its DeLong standard error and its 95% confidence interval, and those of the Gini:

>>> roc_auc_interval(probability, y).round(3)
           estimate    se  lower  upper
statistic
auc           0.686  0.01  0.666  0.706
gini          0.373  0.02  0.333  0.412

Compare with the vendor's PD
----------------------------

The validation sample was approved by the vendor's scorecard, so it also has the vendor's PD. :func:`~kivi.metrics.delong_test` compares the two AUCs on the same applicants, with the correlation between the two scores taken into account:

>>> delong_test(validation["incumbent_pd"], probability, y).round(4)
current_auc    0.6632
new_auc        0.6863
difference     0.0231
se             0.0060
lower          0.0113
upper          0.0349
z              3.8435
p_value        0.0001
dtype: float64

The new scorecard raises the AUC by 0.023, and its 95% interval runs from 0.011 to 0.035, so the gain is not chance. :func:`~kivi.metrics.roc_table` gives the points of both ROC curves:

>>> curves = {
...     "incumbent_pd": roc_table(validation["incumbent_pd"], y),
...     "probability": roc_table(probability, y),
... }

.. plot::
   :alt: ROC curves of the new scorecard's PD and of the vendor's PD on the validation sample
   :caption: ROC curves on the validation sample. The new scorecard's curve lies above the vendor's, most clearly among the riskiest applicants on the left.

   kt.figure_roc()

.. admonition:: Simulation check
   :class: kivi-check

   The validation sample holds only approved applicants. Over all eligible applicants of October to December, approved or not, the new scorecard also ranks better than the vendor's PD:

   >>> quarter = eligible[eligible["month"] >= "2024-10"]
   >>> quarter_pd = predict(model, quarter)
   >>> print(f"{gini(quarter_pd, quarter['target']):.3f} against {gini(quarter['incumbent_pd'], quarter['target']):.3f}")
   0.444 against 0.424

Break the ranking down by band
------------------------------

:func:`~kivi.metrics.gains_table` shows how the defaults spread over score bands. The bands are the PD quintiles of the development sample, so that both samples are cut alike:

>>> from kivi.binning import bin_edges
>>> edges = bin_edges(predict(model, development), bins=5).round(3)
>>> gains = gains_table(probability, y, bins=edges)
>>> gains[["total", "event", "event_rate", "lift", "cum_event_share"]].round(3)
                total  event  event_rate   lift  cum_event_share
bin
(0.175, inf]     1329    289       0.217  2.044            0.400
(0.115, 0.175]   1331    171       0.128  1.207            0.637
(0.076, 0.115]   1375    127       0.092  0.868            0.813
(0.045, 0.076]   1399     93       0.066  0.625            0.942
(-inf, 0.045]    1351     42       0.031  0.292            1.000

.. plot::
   :alt: Default rate and lift of the five PD bands of the validation sample, and the cumulative share of defaults against the share of applicants
   :caption: The five PD bands of the validation sample. The riskiest fifth of the applicants holds 40% of the defaults.

   kt.figure_gains()

The riskiest band defaults at 21.7%, twice the average, and the safest at 3%. The bands of the validation sample hold about a fifth of the applicants each, as they did in development, so the model's PDs have not shifted between the two periods.
