The scorecard
=============

A scorecard gives every group of every characteristic a number of points, so that an applicant's score is a sum that anyone can retrace. This chapter scales the model of :doc:`reject_inference` to points, scores the validation sample and gives the reasons for a low score.

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

Choose the scale
----------------

The score is linear in the log of the good:bad odds. The lender keeps the scale of its other scorecards: 600 points stand for odds of 50 to 1, and every 20 points double the odds. :func:`~kivi.scorecard.score_scaling` gives the factor and the offset of that scale; see :doc:`/user_guide/scorecard`:

>>> from kivi.scorecard import points_table, reason_codes, score_scaling, score_to_pd, woe_to_points
>>> scale = dict(pdo=20, base_score=600, base_odds=50)
>>> factor, offset = score_scaling(**scale)
>>> print(f"factor = {factor:.2f}, offset = {offset:.2f}")
factor = 28.85, offset = 487.12

Build the points table
----------------------

:func:`~kivi.scorecard.points_table` turns the WOE tables and the coefficients into points. A higher WOE means more risk and therefore fewer points:

>>> coefficients = model["coef"].drop("intercept")
>>> intercept = model.loc["intercept", "coef"]
>>> card = points_table({column: spec.tables[column] for column in selected}, coefficients, intercept, **scale)
>>> card.loc[["housing", "employment", "incumbent_pd"]].round({"woe": 3, "points": 1})
                                   bin    woe  points
variable
housing                            own -0.472    92.9
housing                       mortgage -0.174    83.7
housing                           rent  0.341    67.7
employment                   permanent -0.081    80.6
employment    self_employed, temporary  0.186    72.9
incumbent_pd            (-inf, 0.0304] -1.623   125.2
incumbent_pd          (0.0304, 0.0377] -1.136   111.1
incumbent_pd          (0.0377, 0.0541] -0.692    98.3
incumbent_pd          (0.0541, 0.0793] -0.243    85.3
incumbent_pd          (0.0793, 0.0961]  0.037    77.2
incumbent_pd           (0.0961, 0.109]  0.199    72.5
incumbent_pd            (0.109, 0.123]  0.490    64.1
incumbent_pd              (0.123, inf]  0.660    59.2

Within a characteristic, the points of two groups differ by the factor times the coefficient times the difference of their WOE. The vendor's PD spans 66 points from its best bin to its worst, ``housing`` 25 points and ``employment`` 8.

Score the applicants
--------------------

:func:`~kivi.scorecard.woe_to_points` gives every applicant the points of their groups, and the points add up to the score:

>>> points = woe_to_points(spec.woe_transform(validation)[selected], coefficients, intercept, **scale)
>>> score = points.sum(axis=1)
>>> points.assign(score=score).head(3).T.round(1)
                30000  30001  30003
variable
age              75.5   82.3   82.3
income           76.0   70.1   76.9
debt_to_income   82.6   79.0   74.8
inquiries        76.5   76.5   76.5
housing          83.7   67.7   92.9
employment       72.9   80.6   80.6
incumbent_pd     72.5   59.2   72.5
score           539.7  515.4  556.5

The score is the scaled log odds of the model, so :func:`~kivi.scorecard.score_to_pd` turns it back into the model's PD:

>>> bool(np.allclose(score_to_pd(score, **scale), predict(model, validation)))
True
>>> score.describe()[["min", "25%", "50%", "75%", "max"]].round(1)
min    491.4
25%    536.0
50%    552.6
75%    569.6
max    637.3
dtype: float64

Explain a low score
-------------------

A declined applicant is told which characteristics lowered the score most. :func:`~kivi.scorecard.reason_codes` compares the points of each characteristic with those of its best group and lists the largest shortfalls. For the three lowest scores of the validation sample:

>>> lowest = score.nsmallest(3).index
>>> points.loc[lowest].T.round(1)
                35013  34278  36020
variable
age              69.0   69.0   75.5
income           79.1   76.0   70.1
debt_to_income   79.0   79.0   79.0
inquiries        64.5   64.5   72.3
housing          67.7   67.7   67.7
employment       72.9   80.6   72.9
incumbent_pd     59.2   59.2   59.2
>>> reason_codes(points.loc[lowest], card)
           reason_1 reason_2   reason_3
35013  incumbent_pd  housing  inquiries
34278  incumbent_pd  housing  inquiries
36020  incumbent_pd  housing     income

All three have a vendor PD in its worst bin and rent their home, so ``incumbent_pd`` and ``housing`` come first; the third reason differs. The reasons follow from the points table alone, so they are the same in every system that applies the scorecard.
