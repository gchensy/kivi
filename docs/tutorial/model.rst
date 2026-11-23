Variable selection and the model
================================

The scorecard is a logistic regression of the default flag on the WOE of each characteristic. This chapter encodes the development sample, checks that the characteristics do not repeat each other, selects them and fits the model on the approved applicants.

.. dropdown:: Code from the previous chapters

   >>> import numpy as np
   >>> import pandas as pd
   >>> from kivi.datasets import make_applications
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

Encode with WOE
---------------

:meth:`~kivi.spec.BinningSpec.woe_transform` groups every column of a sample and replaces each group with its WOE:

>>> X = spec.woe_transform(development)
>>> y = development["target"]
>>> X.head(3).round(3)
     age  income  debt_to_income  inquiries  housing  employment  incumbent_pd
0 -0.033  -0.038           -0.03     -0.188    0.341       0.186         0.490
2 -0.189   0.067           -0.03      0.066   -0.472      -0.081         0.660
3 -0.033  -0.038           -0.03      0.222    0.341      -0.081        -0.243

A higher WOE means a higher risk, so every coefficient of the model should come out positive.

Check the correlations
----------------------

Two strongly correlated characteristics carry the same information and make the coefficients unstable. :func:`~kivi.selection.correlation_filter` drops the weaker variable of every pair correlated above 0.7, and :func:`~kivi.selection.vif` measures how well the other variables together explain each one; see :doc:`/user_guide/selection`:

>>> from kivi.selection import correlation_filter, logistic_regression, stepwise, vif
>>> iv = pd.Series({column: table["iv"].sum() for column, table in spec.tables.items()})
>>> kept = correlation_filter(X, priority=iv)
>>> X.columns.difference(kept).tolist()
[]
>>> vif(X[kept]).round(2)
variable
age               1.15
income            1.19
debt_to_income    1.33
inquiries         1.22
housing           1.00
employment        1.15
incumbent_pd      1.73
Name: vif, dtype: float64

No pair comes close to the threshold. The vendor's PD has the highest variance inflation factor, because it uses the application data too, but 1.7 is far below the usual limit of 5 to 10.

Select the variables and fit the model
--------------------------------------

:func:`~kivi.selection.stepwise` adds and removes variables by the p-values of a logistic regression and, with WOE inputs, keeps only variables whose coefficient stays positive. :func:`~kivi.selection.logistic_regression` then fits the model:

>>> selected = stepwise(X[kept], y)
>>> selected.tolist()
['age', 'income', 'debt_to_income', 'inquiries', 'housing', 'employment', 'incumbent_pd']
>>> accepts_model = logistic_regression(X[selected], y)
>>> accepts_model.round(3)
                 coef     se       z  p_value
variable
intercept      -2.191  0.025 -88.805    0.000
age             0.529  0.151   3.495    0.000
income          0.466  0.159   2.931    0.003
debt_to_income  0.507  0.143   3.546    0.000
inquiries       0.617  0.134   4.607    0.000
housing         1.016  0.073  13.961    0.000
employment      0.615  0.204   3.009    0.003
incumbent_pd    0.805  0.059  13.643    0.000

Every characteristic is kept, and every coefficient is positive and significant. The vendor's PD and ``housing``, which the vendor's scorecard ignores, have the largest z statistics.

The PD of an applicant is the logistic function of the intercept plus the WOE values times the coefficients. The later chapters compute it with this function:

>>> def predict(model, frame):
...     slopes = model["coef"].drop("intercept")
...     log_odds = model.loc["intercept", "coef"] + spec.woe_transform(frame)[slopes.index] @ slopes
...     return 1 / (1 + np.exp(-log_odds))
>>> print(f"{predict(accepts_model, development).mean():.4f}, {y.mean():.4f}")
0.1008, 0.1008

On the development sample, the mean PD equals the default rate, as it does for any logistic regression with an intercept. The model is called ``accepts_model`` because it has seen only the approved applicants. :doc:`reject_inference` asks what it predicts for the others.
