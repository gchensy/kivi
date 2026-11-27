Scorecard
=========

A scorecard turns a logistic regression on WOE values into points. Every applicant receives the points of their group in each variable, and the points add up to a score that stands for the odds of a good outcome. kivi takes the coefficients and the intercept as plain numbers, so the model can come from any library.

The scorecard follows :doc:`variable selection <selection>` and :doc:`reject_inference`, which give the coefficients, and comes before :doc:`metrics`, which evaluates the score. Where there are too few defaults to fit a model, :doc:`expert` weighs the factors by judgment instead. :func:`~kivi.scorecard.score_scaling`, :func:`~kivi.scorecard.pd_to_score` and :func:`~kivi.scorecard.score_to_pd` set the scale, :func:`~kivi.scorecard.points_table` and :func:`~kivi.scorecard.woe_to_points` give the points, :func:`~kivi.scorecard.reason_codes` explains a score, and :class:`~kivi.scorecard.ScorecardSpec` keeps the whole scorecard for production and saves it as versioned JSON.

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

On this scale, odds of 50:1, a PD of 1/51, score 600, and odds of 100:1 score 20 points more. :func:`~kivi.scorecard.score_to_pd` converts in the other direction:

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
>>> from kivi.datasets import load_bank
>>> from kivi.woe import woe_table, woe_transform
>>> data = load_bank()
>>> development, monitoring = data.iloc[:3000], data.iloc[3000:]
>>> age_edges = np.array([-np.inf, 25, 35, 45, 55, 65, np.inf])
>>> def to_groups(frame):
...     groups = frame[["poutcome", "contact", "housing"]].copy()
...     groups["age"] = pd.cut(frame["age"], age_edges)
...     return groups
>>> dev_groups = to_groups(development)
>>> target = development["target"]
>>> tables = {column: woe_table(dev_groups[column], target) for column in dev_groups}

A logistic regression of ``target`` on the four WOE columns of the development sample gives the coefficients and the intercept below, rounded to three decimals. :func:`~kivi.scorecard.points_table` turns them into points:

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

Encode new applicants with the development WOE tables, as described in :doc:`woe_iv`, and give them the points of each variable with :func:`~kivi.scorecard.woe_to_points`. The row sums are the scores:

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

.. plot::
   :alt: The applicant with index 3001: values, groups, WOE, coefficients and points of four variables, and the score
   :caption: Scoring the applicant with index 3001: each value is grouped, encoded with the WOE of its group and turned into points, and the points add up to the score.

   kd.figure_scorecard_points()

The score is the scaled log odds of the model, so it equals :func:`~kivi.scorecard.pd_to_score` of the model's probability:

>>> probability = 1 / (1 + np.exp(-(intercept + woe @ pd.Series(coefficients))))
>>> bool(np.allclose(points.sum(axis=1), pd_to_score(probability, **scale)))
True

Deployed scorecards usually round the points to whole numbers. Round the points table and the applicant points in the same way, with ``table.round()`` and ``points.round()``: ``woe_to_points`` computes each point exactly as :func:`~kivi.scorecard.points_table` does, so the rounded values match. The rounded total can differ from the exact score by up to half a point per variable.

:class:`~kivi.scorecard.ScorecardSpec` keeps the rounded points table and scores with it; see :ref:`scorecard-saving`.

Reason codes
------------

A declined applicant is told which variables lowered the score most. :func:`~kivi.scorecard.reason_codes` matches the points of each variable to a group of the points table, compares them with those of the variable's best group and lists the variables from the largest shortfall:

>>> from kivi.scorecard import reason_codes
>>> reason_codes(points, table).head(3)
     reason_1 reason_2 reason_3
3000  contact  housing     None
3001     None     None     None
3002  contact      age     None

Applicant 3000 was contacted by telephone, 21.3 points below the best contact group, and has no housing loan, 13.2 points below the best housing group. Applicant 3001 is in the best group of every variable, so no variable lowered the score.

The scaling does not change the order of the reasons, because every shortfall is :math:`\mathrm{factor} \cdot \beta_j` times a difference in WOE. Points that match no group of the table within ``atol``, for example points computed with another intercept, raise an error instead of giving reasons against the wrong groups.

Applicants in a group without development observations, such as a new category, have no group of their own in the table. Their WOE comes from ``unseen`` in :func:`~kivi.woe.woe_transform`, a value chosen for the group rather than evidence about the applicant, so by default their points raise an error too. Pass their points as ``unseen_points`` to count them as costing nothing: the variable is then not a reason for those applicants. ``unseen_points`` holds points, not a WOE. Run :func:`~kivi.scorecard.woe_to_points` on a one-row frame that holds the WOE given to ``unseen``; a WOE of 0 gives the base points, the same for every variable. Below, the first applicant has a housing status that the development sample did not have:

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

.. _scorecard-saving:

Saving the scorecard
--------------------

Production needs more than the points table: the binning that groups each characteristic, the scaling, the points of values that development did not have and, often, a calibration of the PD and a master scale. :class:`~kivi.scorecard.ScorecardSpec` keeps them in one object, which scores any sample and saves itself as versioned JSON. :meth:`~kivi.scorecard.ScorecardSpec.from_model` builds it from a :class:`~kivi.spec.BinningSpec` and the coefficients of the model. :func:`~kivi.spec.fit_spec` with the age edges of the example gives the groups and the WOE of the tables above, and here gives housing statuses that development did not have a WOE of 0:

>>> from kivi.scorecard import ScorecardSpec
>>> from kivi.spec import fit_spec
>>> binning = fit_spec(
...     development[list(coefficients)], target, bins={"age": age_edges}, unseen={"housing": 0.0}
... )
>>> card = ScorecardSpec.from_model(binning, coefficients, intercept, **scale, decimals=0)
>>> card.table
                   bin       woe  points
variable
poutcome       unknown -0.257139   143.0
poutcome       failure  0.213512   131.0
poutcome         other  0.591650   121.0
poutcome       success  2.548123    70.0
contact       cellular  0.240895   132.0
contact        unknown -0.917237   154.0
contact      telephone  0.206318   132.0
housing             no  0.332926   130.0
housing            yes -0.327288   143.0
age       (-inf, 25.0]  0.794104   120.0
age       (25.0, 35.0] -0.032419   137.0
age       (35.0, 45.0] -0.166358   140.0
age       (45.0, 55.0] -0.000122   136.0
age       (55.0, 65.0]  0.038437   136.0
age        (65.0, inf]  1.114576   113.0

``from_model`` keeps the columns of the binning that have a coefficient, in the order of the coefficients, gives their groups the points of :func:`~kivi.scorecard.points_table` and rounds them to ``decimals`` with pandas ``round``, which rounds halves to even. To deploy points adjusted by hand, change the ``points`` column of ``card.table`` and pass the table back with ``dataclasses.replace``, which checks that ``bin`` and ``woe`` are still those of the binning.

:meth:`~kivi.scorecard.ScorecardSpec.score` adds the rounded points column by column rather than with a matrix product, so applicants in the same groups get exactly the same score on every computer:

>>> card.score(monitoring).head(3)
3000    545.0
3001    580.0
3002    555.0
Name: score, dtype: float64

These scores differ from the exact ones above, 545.2, 579.7 and 554.9, by less than half a point per variable, and the PD follows from the rounded score, as in production.

Unseen groups
^^^^^^^^^^^^^

A value whose group is not in its column's table is treated as the ``unseen`` policy of the binning says. A column with the policy ``"raise"`` raises an error, as :meth:`~kivi.spec.BinningSpec.woe_transform` does, and a grouped column whose policy names a group gives the value the points of that group. A column with an unseen WOE needs points for it, and rounded points cannot be turned back into a coefficient, so the specification keeps them in ``unseen_points``. :meth:`~kivi.scorecard.ScorecardSpec.from_model` computes them with :func:`~kivi.scorecard.woe_to_points` and rounds them with the table:

>>> dict(card.unseen_points)
{'housing': 136.0}
>>> new = pd.DataFrame({
...     "poutcome": ["unknown", "success"],
...     "contact": ["telephone", "cellular"],
...     "housing": ["rented", "no"],
...     "age": [30, 70],
... })
>>> card.points(new)
   poutcome  contact  housing    age
0     143.0    132.0    136.0  137.0
1      70.0    132.0    130.0  113.0
>>> card.reason_codes(new)
   reason_1 reason_2 reason_3
0   contact      age     None
1  poutcome      age  contact

:meth:`~kivi.scorecard.ScorecardSpec.reason_codes` ranks the costs as :func:`~kivi.scorecard.reason_codes` does, but it knows which values were unseen: an unseen value costs nothing, even when its points equal those of a group. The rented home of the first applicant is therefore not one of their reasons.

Calibration and grades
^^^^^^^^^^^^^^^^^^^^^^

A score :math:`s` stands for the log odds of the event :math:`\ell = (\mathrm{offset} - s) / \mathrm{factor}`. :meth:`~kivi.scorecard.ScorecardSpec.probability` computes them from the score directly, so that a PD near 0 or 1 is not rounded first, and without calibration gives :math:`\sigma(\ell) = 1 / (1 + e^{-\ell})`, the PD of :func:`~kivi.scorecard.score_to_pd`. A calibration changes the PD and leaves the points as they are:

.. list-table::
   :header-rows: 1
   :widths: 34 26 40

   * - ``calibration``
     - PD
     - Fitted with
   * - ``{"kind": "shift", "shift": shift}``
     - :math:`\sigma(\ell + \mathrm{shift})`
     - :func:`~kivi.calibration.central_tendency_shift` of the uncalibrated PDs
   * - ``{"kind": "platt", "slope": slope, "intercept": intercept}``
     - :math:`\sigma(\mathrm{slope} \cdot \ell + \mathrm{intercept})`
     - :func:`~kivi.calibration.platt_fit` or :func:`~kivi.calibration.quasi_moment_fit` of the log odds :math:`\ell`
   * - ``{"kind": "isotonic", "edges": edges, "probability": probability}``
     - The ``probability`` of the band of :math:`s`, as :func:`~kivi.calibration.isotonic_predict` gives it
     - :func:`~kivi.calibration.isotonic_fit` of the scores with ``increasing=False``, because a higher score means a lower risk

A Platt calibration fitted to the scores rather than to :math:`\ell` would differ only by the linear conversion between the two, so the specification takes the one on the log odds alone. ``scale``, a :func:`~kivi.calibration.master_scale` result, gives :meth:`~kivi.scorecard.ScorecardSpec.grade` the grade of each calibrated PD. Below, the PDs are shifted to a long-run average default rate of 15%, above the 11.6% of the development sample:

>>> import dataclasses
>>> from kivi.calibration import central_tendency_shift, master_scale
>>> shift = central_tendency_shift(card.probability(development), central_tendency=0.15)
>>> card = dataclasses.replace(
...     card,
...     calibration={"kind": "shift", "shift": shift},
...     scale=master_scale(np.geomspace(0.05, 0.4, 4), labels=list("ABCD")),
... )
>>> print(f"{card.probability(development).mean():.4f}")
0.1500
>>> card.grade(monitoring).value_counts(sort=False)
grade
A    328
B    456
C    634
D    103
Name: count, dtype: int64

``dataclasses.replace`` checks the new specification again, as the constructor does.

Saving and loading
^^^^^^^^^^^^^^^^^^

:meth:`~kivi.scorecard.ScorecardSpec.to_json` writes the whole scorecard as JSON text, and :meth:`~kivi.scorecard.ScorecardSpec.from_json` reads it back into an equal specification, which scores, calibrates and grades every sample identically:

>>> text = card.to_json(indent=2)
>>> loaded = ScorecardSpec.from_json(text)
>>> loaded == card
True
>>> loaded.grade(monitoring).equals(card.grade(monitoring))
True

Two specifications are equal when their JSON text is the same, which includes the version of kivi. Save the text as UTF-8, as for a :class:`~kivi.spec.BinningSpec`; see :ref:`spec-saving`. The time-window features of a :class:`~kivi.features.FeatureSpec` are computed from other tables before the scorecard scores an application, so they are not part of the specification; save the two files side by side.

The text always has these keys, with ``null`` for a field that is not set:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Content
   * - ``format``
     - Always ``"kivi.ScorecardSpec"``.
   * - ``format_version``
     - Version of the format, 1. ``ScorecardSpec.FORMAT_VERSION`` is the newest version. A file with a newer version raises an error that asks for a newer kivi, instead of being read in part.
   * - ``kivi_version``
     - Version of kivi that built the specification.
   * - ``scaling``
     - ``pdo``, ``base_score`` and ``base_odds``.
   * - ``binning``
     - The binning as :meth:`~kivi.spec.BinningSpec.to_dict` writes it, with its own format version.
   * - ``points``
     - For each column, the list of the points of its groups, in the order of the rows of its table in ``binning``.
   * - ``unseen_points``
     - The points of unseen groups, by column.
   * - ``calibration``
     - The calibration, with ``{"float": "inf"}`` and ``{"float": "-inf"}`` for the infinite outer edges of an isotonic calibration.
   * - ``scale``
     - One entry per grade, with its label ``grade`` and its ``lower``, ``probability`` and ``upper``.

:meth:`~kivi.scorecard.ScorecardSpec.from_json` checks every value and rejects ``NaN`` and ``Infinity`` in the text and objects that repeat a key, so a damaged or hand-edited file fails with a message instead of scoring with wrong values.
