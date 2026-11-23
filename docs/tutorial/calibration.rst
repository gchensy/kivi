Calibration and grades
======================

The scorecard will price loans, set limits and feed the loss allowance, so its PDs must match the default rates that follow, not just rank applicants. This chapter groups the PDs into the grades of a master scale, tests each grade on the validation sample and diagnoses what the tests show; :doc:`/user_guide/calibration` describes each test.

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

Build the master scale
----------------------

A master scale gives PDs from every model of the lender one meaning. :func:`~kivi.calibration.master_scale` builds one from the PD of each grade, here five grades in which each PD doubles the one before, with bounds halfway between them on a log scale:

>>> from kivi.calibration import assign_grades, binomial_test, calibration_table, fit_grade_pd
>>> from kivi.calibration import herfindahl_index, hosmer_lemeshow, jeffreys_test, master_scale, platt_fit
>>> master = master_scale(np.geomspace(0.02, 0.32, 5), labels=list("ABCDE"))
>>> master.round(4)
        lower  probability   upper
grade
A      0.0000         0.02  0.0283
B      0.0283         0.04  0.0566
C      0.0566         0.08  0.1131
D      0.1131         0.16  0.2263
E      0.2263         0.32  1.0000

Test each grade
---------------

:func:`~kivi.calibration.assign_grades` gives every applicant of the validation sample the grade of their PD, and :func:`~kivi.calibration.calibration_table` compares each grade's default rate with its mean PD. :func:`~kivi.calibration.jeffreys_test` asks whether a grade has more defaults than the PD of the master scale predicts, so a small p-value flags a grade that underestimates the risk. :func:`~kivi.calibration.binomial_test` asks the same question of the grade's mean PD with the exact binomial distribution:

>>> probability = predict(model, validation)
>>> y = validation["target"]
>>> grades = assign_grades(probability, master)
>>> table = calibration_table(grades, y, probability).join(master["probability"])
>>> table["jeffreys"] = jeffreys_test(table["event"], table["total"], table["probability"])
>>> table["binomial"] = binomial_test(table["event"], table["total"], table["mean_probability"])
>>> table[["total", "event", "event_rate", "mean_probability", "jeffreys", "binomial"]].round(3)
     total  event  event_rate  mean_probability  jeffreys  binomial
bin
A      679     17       0.025             0.018     0.172     0.126
B     1135     50       0.044             0.043     0.239     0.443
C     2225    189       0.085             0.082     0.194     0.328
D     2047    302       0.148             0.158     0.939     0.916
E      699    164       0.235             0.283     1.000     0.998

No grade underestimates the risk: every p-value is large. Grades A to D match their PDs closely, but grade E has 164 defaults among 699 approved applicants, 23.5%, against a mean PD of 28.3%. Both tests are one-sided and do not flag a PD that is too high. :func:`~kivi.calibration.hosmer_lemeshow` tests deviations in both directions, here over the PD deciles of the development sample:

>>> from kivi.binning import bin_edges
>>> deciles = pd.cut(probability, bin_edges(predict(model, development), bins=10))
>>> by_decile = calibration_table(deciles, y, probability)
>>> statistic, p_value = hosmer_lemeshow(by_decile["event"], by_decile["total"], by_decile["mean_probability"])
>>> print(f"H = {statistic:.2f}, p = {p_value:.3f}")
H = 17.02, p = 0.030

Diagnose the deviation
----------------------

:func:`~kivi.calibration.platt_fit` regresses the outcomes on the log odds of the PDs. A slope of 1 and an intercept of 0 would mean that the PDs are calibrated on the sample:

>>> slope, shift = platt_fit(np.log(probability / (1 - probability)), y)
>>> print(f"slope = {slope:.3f}, intercept = {shift:.3f}")
slope = 0.831, intercept = -0.386

A slope below 1 means that the PDs spread further than the outcomes. :func:`~kivi.calibration.platt_predict` shows what the fit would make of three PDs:

>>> from kivi.calibration import platt_predict
>>> example = np.array([0.02, 0.1, 0.3])
>>> platt_predict(np.log(example / (1 - example)), slope, shift).round(3)
array([0.026, 0.099, 0.252])

The fit raises a PD of 2% a little, leaves 10% where it is and lowers 30% to 25%: the highest PDs overstate the risk of the approved applicants most. That is what reject inference does by design. The model now predicts for all eligible applicants, and the approved ones in grade E are better than the declined applicants with the same PD, whom the vendor turned away on bureau data. Recalibrating the PDs to the approved applicants, with Platt scaling or isotonic regression, would undo the reject inference, so the project keeps them.

:func:`~kivi.calibration.fit_grade_pd` shows what the approved applicants alone would make of the grades: a log-linear fit of their default rates against the grade number:

>>> fit_grade_pd(table["event"], table["total"]).round(4)
bin
A    0.0258
B    0.0459
C    0.0815
D    0.1447
E    0.2571
dtype: float64

.. admonition:: Simulation check
   :class: kivi-check

   Over all eligible applicants of October to December, approved or not, each grade's default rate is close to its mean PD:

   >>> quarter = eligible[eligible["month"] >= "2024-10"]
   >>> quarter_pd = predict(model, quarter)
   >>> everyone = calibration_table(assign_grades(quarter_pd, master), quarter["target"], quarter_pd)
   >>> everyone.round(3)
        total  event  event_rate  mean_probability
   bin
   A      679     17       0.025             0.018
   B     1135     50       0.044             0.043
   C     2234    190       0.085             0.082
   D     2989    476       0.159             0.166
   E     2204    681       0.309             0.307

.. plot::
   :alt: Default rate against mean PD of each grade, for the approved applicants of the validation sample and for all eligible applicants of the quarter
   :caption: Default rate against mean PD by grade. The approved applicants of grade E default less often than their PD says; over all eligible applicants, which only simulated data reveal, the grades lie on the diagonal.

   kt.figure_calibration()

Check the concentration
-----------------------

A master scale should spread the applicants over its grades rather than put them all in one. :func:`~kivi.calibration.herfindahl_index` measures the concentration of the counts, from 0 for an even spread to 1 for a single grade:

>>> print(f"{herfindahl_index(table['total'], normalize=True):.3f}")
0.059

The approved applicants spread over the five grades with a normalized index of 0.06, far from concentrated.
