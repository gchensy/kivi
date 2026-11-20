Scorecard
=========

A scorecard turns a logistic regression on WOE values into points. Every applicant receives the points of their group in each variable, and the points add up to a score that stands for the odds of a good outcome. kivi takes the coefficients and the intercept as plain numbers, so the model can come from any library.

Scaling
-------

A score is linear in the log of the good:bad odds :math:`o = (1 - \mathrm{PD}) / \mathrm{PD}`. Three numbers set the scale: the base score, the odds at the base score and the points to double the odds (PDO):

.. math::

   \mathrm{score} = \mathrm{offset} + \mathrm{factor} \cdot \ln o,

   \mathrm{factor} = \frac{\mathrm{pdo}}{\ln 2}, \qquad
   \mathrm{offset} = \mathrm{base\_score} - \mathrm{factor} \cdot \ln \mathrm{base\_odds}.

Every function of :mod:`kivi.scorecard` takes the three numbers as keyword arguments, so keep them together in one dictionary:

>>> from kivi.scorecard import pd_to_score, score_scaling, score_to_pd
>>> scale = dict(pdo=20, base_score=600, base_odds=50)
>>> factor, offset = score_scaling(**scale)
>>> print(f"factor = {factor:.2f}, offset = {offset:.2f}")
factor = 28.85, offset = 487.12

On this scale, odds of 50:1, a PD of 1/51, score 600, and odds of 100:1 score 20 points more. ``score_to_pd`` converts in the other direction:

>>> pd_to_score([1 / 51, 1 / 101, 0.1], **scale).round(1)
array([600. , 620. , 550.5])
>>> score_to_pd([560, 600, 640], **scale).round(4)
array([0.0741, 0.0196, 0.005 ])

.. note::

   kivi's WOE is :math:`\ln(p_i / q_i)`, the event share over the non-event share, with the bad outcome as the event (see :doc:`woe_iv`). A higher WOE therefore means more risk, a logistic regression of the bad flag on WOE values has positive coefficients, and a higher WOE gives fewer points. Some textbooks define WOE the other way round; their coefficients and WOE values both change sign, and the points stay the same.

Points for each group
---------------------

The example below builds a small scorecard on the bundled bank data. The data record a marketing response rather than a default, so read target 1 as the bad outcome, as in :doc:`risk_workflow`. Group the development sample and fit the WOE of each variable:

>>> import numpy as np
>>> import pandas as pd
>>> from kivi.datasets import Dataset
>>> from kivi.woe import woe_table, woe_transform
>>> data = Dataset.bank_data()
>>> development, monitoring = data.iloc[:3000], data.iloc[3000:]
>>> age_edges = np.array([-np.inf, 25, 35, 45, 55, 65, np.inf])
>>> def to_groups(frame):
...     groups = frame[["poutcome", "contact", "housing"]].copy()
...     groups["age"] = pd.cut(frame["age"], age_edges)
...     return groups
>>> dev_groups = to_groups(development)
>>> target = development["target"]
>>> tables = {column: woe_table(dev_groups[column], target) for column in dev_groups}

A logistic regression of ``target`` on the four WOE columns of the development sample gives the coefficients and the intercept below, rounded to three decimals. ``points_table`` turns them into points:

>>> from kivi.scorecard import points_table
>>> coefficients = {"poutcome": 0.899, "contact": 0.658, "housing": 0.691, "age": 0.714}
>>> intercept = -2.028
>>> table = points_table(tables, coefficients, intercept, **scale)
>>> table.round({"woe": 4, "points": 1})
                   bin     woe  points
variable
poutcome       unknown -0.2571   143.1
poutcome       failure  0.2135   130.9
poutcome         other  0.5917   121.1
poutcome       success  2.5481    70.3
contact       cellular  0.2409   131.8
contact        unknown -0.9172   153.8
contact      telephone  0.2063   132.5
housing             no  0.3329   129.8
housing            yes -0.3273   142.9
age       (-inf, 25.0]  0.7941   120.0
age       (25.0, 35.0] -0.0324   137.1
age       (35.0, 45.0] -0.1664   139.8
age       (45.0, 55.0] -0.0001   136.4
age       (55.0, 65.0]  0.0384   135.6
age        (65.0, inf]  1.1146   113.4

Group :math:`i` of variable :math:`j`, one of :math:`m` variables, receives

.. math::

   \mathrm{points}_{ij} = \frac{\mathrm{offset} - \mathrm{factor} \cdot \beta_0}{m}
   - \mathrm{factor} \cdot \beta_j \, \mathrm{WOE}_{ij}.

The intercept :math:`\beta_0` and the offset are spread evenly across the variables, so each of the four variables above carries a quarter of :math:`487.12 + 28.85 \times 2.028 = 545.64` points. Within a variable, the points of two groups differ only by their WOE term. A group with a WOE of zero, which has the event rate of the whole sample, receives exactly the variable's share.

.. tip::

   A negative coefficient reverses a variable's direction, so that riskier groups receive more points. It usually points to correlated variables; review the signs before scaling a model.

Scoring applicants
------------------

Encode new applicants with the development WOE tables, as described in :doc:`woe_iv`, and give them the points of each variable with ``woe_to_points``. The row sums are the scores:

>>> from kivi.scorecard import woe_to_points
>>> mon_groups = to_groups(monitoring)
>>> woe = pd.DataFrame(
...     {column: woe_transform(mon_groups[column], tables[column]) for column in mon_groups}
... )
>>> points = woe_to_points(woe, coefficients, intercept, **scale)
>>> points.assign(score=points.sum(axis=1)).head(3).round(1)
      poutcome  contact  housing    age  score
3000     143.1    132.5    129.8  139.8  545.2
3001     143.1    153.8    142.9  139.8  579.7
3002     143.1    131.8    142.9  137.1  554.9

The score is the scaled log odds of the model, so it equals ``pd_to_score`` of the model's probability:

>>> probability = 1 / (1 + np.exp(-(intercept + woe @ pd.Series(coefficients))))
>>> bool(np.allclose(points.sum(axis=1), pd_to_score(probability, **scale)))
True

Deployed scorecards usually round the points to whole numbers. Round the points table and the applicant points in the same way, with ``table.round()`` and ``points.round()``: ``woe_to_points`` computes each point exactly as ``points_table`` does, so the rounded values match. The rounded total can differ from the exact score by up to half a point per variable.

Reason codes
------------

A declined applicant is told which variables lowered the score most. ``reason_codes`` matches the points of each variable to a group of the points table, compares them with those of the variable's best group and lists the variables from the largest shortfall:

>>> from kivi.scorecard import reason_codes
>>> reason_codes(points, table).head(3)
     reason_1 reason_2 reason_3
3000  contact  housing     None
3001     None     None     None
3002  contact      age     None

Applicant 3000 was contacted by telephone, 21.3 points below the best contact group, and has no housing loan, 13.2 points below the best housing group. Applicant 3001 is in the best group of every variable, so no variable lowered the score.

The scaling does not change the order of the reasons, because every shortfall is :math:`\mathrm{factor} \cdot \beta_j` times a difference in WOE. Points that match no group of the table within ``atol``, for example points computed with another intercept, raise an error instead of giving reasons against the wrong groups.

Applicants in a group without development observations, such as a new category, have no group of their own in the table. Their WOE comes from ``unseen`` in ``woe_transform``, a value chosen for the group rather than evidence about the applicant, so by default their points raise an error too. Pass their points as ``unseen_points`` to count them as costing nothing: the variable is then not a reason for those applicants. ``unseen_points`` holds points, not a WOE. Run ``woe_to_points`` on a one-row frame that holds the WOE given to ``unseen``; a WOE of 0 gives the base points, the same for every variable. Below, the first applicant has a housing status that the development sample did not have:

>>> new = pd.DataFrame({"housing": ["rented", "no"], "contact": ["telephone", "unknown"]})
>>> new_woe = pd.DataFrame({
...     "housing": woe_transform(new["housing"], tables["housing"], unseen=0.0),
...     "contact": woe_transform(new["contact"], tables["contact"]),
... })
>>> two = {"housing": coefficients["housing"], "contact": coefficients["contact"]}
>>> new_table = points_table(tables, two, intercept, **scale)
>>> new_points = woe_to_points(new_woe, two, intercept, **scale)
>>> zero = woe_to_points(pd.DataFrame({"housing": [0.0], "contact": [0.0]}), two, intercept, **scale)
>>> zero.round(2)
   housing  contact
0   272.82   272.82
>>> reason_codes(new_points, new_table, unseen_points=zero.iloc[0])
  reason_1 reason_2 reason_3
0  contact     None     None
1  housing     None     None

The first applicant's housing points are those of a WOE of 0, so only the telephone contact is a reason. The second applicant has no housing loan, the less favourable housing group, and is in the best contact group, ``unknown``. Only points near ``unseen_points`` are exempt: points from another model, or WOE values rounded more coarsely than ``atol`` allows, still raise an error.

.. note::

   A model developed on a sample that over- or undersamples bad outcomes has an intercept for that sample. Correct it with :func:`kivi.calibration.corrected_intercept` before building the points table; see :doc:`calibration`.
