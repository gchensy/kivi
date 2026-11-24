Calibration
===========

A PD model is calibrated when its probabilities match the default rates that are later observed. Calibration is a separate question from ranking: a model can order applicants well and still predict default rates that are too low for the whole portfolio. :mod:`kivi.calibration` corrects probabilities for the sampling of the development data, builds a master scale, summarizes its grades and tests them, and maps a score to probabilities with Platt scaling or isotonic regression.

Calibration follows :doc:`metrics`, which measures the ranking, and comes before the :doc:`credit strategy <strategy>`, which sets cut-offs on the calibrated PDs. :func:`~kivi.calibration.prior_correction` and :func:`~kivi.calibration.corrected_intercept` correct for the sampling; :func:`~kivi.calibration.master_scale`, :func:`~kivi.calibration.fit_grade_pd` and :func:`~kivi.calibration.assign_grades` build and apply a master scale; :func:`~kivi.calibration.calibration_table`, :func:`~kivi.calibration.binomial_test`, :func:`~kivi.calibration.jeffreys_test`, :func:`~kivi.calibration.hosmer_lemeshow`, :func:`~kivi.calibration.spiegelhalter_test`, :func:`~kivi.calibration.brier_decomposition`, :func:`~kivi.calibration.herfindahl_index` and :func:`~kivi.calibration.concentration_test` count and test the grades; :func:`~kivi.calibration.backtest_table` back-tests continuous estimates such as LGDs; and :func:`~kivi.calibration.platt_fit`, :func:`~kivi.calibration.platt_predict`, :func:`~kivi.calibration.isotonic_fit` and :func:`~kivi.calibration.isotonic_predict` recalibrate a score. :doc:`validation` collects the tests that the European Central Bank asks for in a validation report.

Correcting for oversampling
---------------------------

Bad outcomes are often oversampled for development, for example to a sample with as many bads as goods. The model then predicts probabilities for that sample, not for the portfolio. With the development sample's event rate :math:`\bar y` and the portfolio's event rate :math:`\tau`, the prior correction of King and Zeng multiplies the odds of every probability by

.. math::

   \omega = \frac{\tau / (1 - \tau)}{\bar y / (1 - \bar y)}, \qquad
   \tilde p = \frac{\omega p}{\omega p + 1 - p}.

A model developed on a sample with 50% bads, for a portfolio with a 2% default rate, turns a probability of 0.5 into 0.02:

>>> from kivi.calibration import corrected_intercept, prior_correction
>>> prior_correction([0.5, 0.2, 0.05], sample_rate=0.5, population_rate=0.02).round(4)
array([0.02  , 0.0051, 0.0011])

The correction keeps the order of the probabilities. For a logistic regression it only shifts the intercept by :math:`\ln \omega`, so the coefficients and the points per WOE in :doc:`scorecard` stay the same:

>>> print(f"{corrected_intercept(-0.3, sample_rate=0.5, population_rate=0.02):.4f}")
-4.1918

Grades of a master scale
------------------------

A master scale divides the PD range into a fixed set of grades, so that ratings from different models share one meaning. Each grade has a PD, and the bounds between the grades decide which grade a predicted PD falls into. The examples below use the PDs of the scorecard from :doc:`scorecard` on the development and monitoring samples:

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
>>> coefficients = pd.Series(
...     {"poutcome": 0.899, "contact": 0.658, "housing": 0.691, "age": 0.714}
... )
>>> def predict(frame):
...     groups = to_groups(frame)
...     woe = pd.DataFrame(
...         {column: woe_transform(groups[column], tables[column]) for column in groups}
...     )
...     return 1 / (1 + np.exp(2.028 - woe @ coefficients))
>>> dev_prob, probability = predict(development), predict(monitoring)

Building a scale
^^^^^^^^^^^^^^^^

The PDs of a master scale usually grow by a constant factor from grade to grade. :func:`~kivi.calibration.master_scale` takes the PD of each grade and adds the bounds of the grades. Here the PDs are a geometric grid of six grades from 2% to 64%, each with twice the PD of the grade before:

>>> from kivi.calibration import assign_grades, fit_grade_pd, master_scale
>>> scale = master_scale(np.geomspace(0.02, 0.64, 6), labels=list("ABCDEF"))
>>> scale.round(4)
        lower  probability   upper
grade
A      0.0000         0.02  0.0283
B      0.0283         0.04  0.0566
C      0.0566         0.08  0.1131
D      0.1131         0.16  0.2263
E      0.2263         0.32  0.4525
F      0.4525         0.64  1.0000

Each bound is the geometric mean of the PDs on either side of it, their midpoint on a logarithmic scale, so a PD goes to the grade whose PD is nearer by ratio. ``boundary="arithmetic"`` takes the arithmetic mean instead, as the 0.2 versions of kivi did. The arithmetic mean lies above the geometric one, so each grade reaches further up than down. Grade A then holds the PDs up to 3% instead of 2.83%. A Series of PDs gives its index as the grade labels:

>>> arithmetic = master_scale(scale["probability"], boundary="arithmetic")
>>> arithmetic["upper"].round(4)
grade
A    0.03
B    0.06
C    0.12
D    0.24
E    0.48
F    1.00
Name: upper, dtype: float64

With arithmetic bounds, most PDs of a grade lie above its PD, so the observed default rate of a calibrated model tends to exceed the PD of the grade.

Assigning the grades
^^^^^^^^^^^^^^^^^^^^

:func:`~kivi.calibration.assign_grades` gives each PD the grade whose bounds hold it. A PD on a bound goes to the lower grade, as the intervals of ``pandas.cut`` are closed on the right, and a PD of 0 goes to the first grade. The grades are an ordered categorical of the labels, so :func:`~kivi.calibration.calibration_table` lists them in the order of the scale when it compares the observed default rate of each grade with its mean predicted PD. Joining the scale adds the PD of each grade:

>>> from kivi.calibration import calibration_table
>>> grades = assign_grades(probability, scale)
>>> scale_table = calibration_table(grades, monitoring["target"], probability)
>>> scale_table.join(scale["probability"]).round(3)
     total  event  event_rate  mean_probability  probability
bin
B      328     13       0.040             0.042         0.04
C      459     36       0.078             0.080         0.08
D      632     72       0.114             0.136         0.16
E       56     21       0.375             0.267         0.32
F       46     31       0.674             0.636         0.64

The model gives no customer a PD below 2.83%, so grade A is empty and ``calibration_table`` leaves it out.

Fitting the PDs of the grades
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The PDs of the grades can also come from the default rates observed in them. :func:`~kivi.calibration.fit_grade_pd` fits the logarithm of the default rate as a straight line in the grade number, by least squares weighted with the number of observations of each grade, so the fitted PDs again grow by a constant factor. A grade without defaults enters the fit with half a default, and a grade without observations is left out of the fit but still gets a PD. Count the development sample by the grades of the scale:

>>> dev_counts = calibration_table(assign_grades(dev_prob, scale), target, dev_prob)
>>> dev_counts["event_rate"].round(3)
bin
B    0.043
C    0.084
D    0.133
E    0.272
F    0.642
Name: event_rate, dtype: float64

The grades are numbered by their position in the input, so reindex the counts to every grade of the scale before the fit. The empty grade A then keeps its place with a count of 0:

>>> dev_counts = dev_counts.reindex(scale.index, fill_value=0)
>>> fitted_pd = fit_grade_pd(dev_counts["event"], dev_counts["total"])
>>> fitted = master_scale(fitted_pd)
>>> fitted.round(4)
        lower  probability   upper
grade
A      0.0000       0.0238  0.0321
B      0.0321       0.0432  0.0582
C      0.0582       0.0784  0.1056
D      0.1056       0.1423  0.1917
E      0.1917       0.2582  0.3477
F      0.3477       0.4683  1.0000

The fitted PDs grow by a factor of 1.81 from grade to grade and follow the default rates of grades B to E closely. Grade F, with a default rate of 64%, lies far above the line: it holds only 81 customers and so weighs little in the fit. The tests below show whether that matters.

.. plot::
   :alt: Observed default rate of each grade of the development sample, the PDs fitted by fit_grade_pd and the geometric grid of the scale, on a logarithmic axis
   :caption: Observed default rates of the development sample by grade, the PDs fitted by ``fit_grade_pd`` and the geometric grid of the scale, on a logarithmic axis. Larger points hold more customers; grade F lies above the fitted line.

   kp.figure_grade_fit()

Grades given by their bounds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A scale can also be given by its bounds alone, for example by an existing rating system. Cut the PDs at the bounds with ``pandas.cut``, whose intervals are closed on the right like those of :func:`~kivi.calibration.assign_grades`; ``include_lowest=True`` keeps a PD of exactly 0 in the first grade. The rest of this page uses these grades:

>>> pd_edges = [0, 0.05, 0.08, 0.12, 0.2, 0.4, 1]
>>> grades = pd.cut(probability, pd_edges, labels=list("ABCDEF"), include_lowest=True)
>>> table = calibration_table(grades, monitoring["target"], probability)
>>> table.round(3)
     total  event  event_rate  mean_probability
bin
A      328     13       0.040      0.042
B      234     13       0.056      0.071
C      276     28       0.101      0.093
D      572     66       0.115      0.136
E       65     22       0.338      0.260
F       46     31       0.674      0.636

:func:`~kivi.calibration.calibration_table` accepts any grouping, such as score bands or deciles of the PD. It counts the groups like :func:`~kivi.woe.woe_table`: categorical groups keep the order of their categories, and grades without observations are left out.

Testing each grade
------------------

Each grade is tested on its own against the hypothesis that its PD is not too low. Both tests are one-sided: a small p-value means that the grade has more defaults than its PD predicts.

* The **binomial test** treats the :math:`N` obligors of a grade as independent, each defaulting with probability :math:`p`. Its p-value is the probability of at least the observed :math:`d` defaults, :math:`P(D \ge d)`.
* The **Jeffreys test** starts from the Jeffreys prior :math:`\mathrm{Beta}(1/2, 1/2)` for the default rate. Its p-value is the posterior probability that the default rate is at most :math:`p`, the distribution function of :math:`\mathrm{Beta}(d + 1/2, N - d + 1/2)` at :math:`p`. The European Central Bank asks for this test in the validation of PD models.

>>> from kivi.calibration import binomial_test, jeffreys_test
>>> def grade_tests(table):
...     return table.assign(
...         binomial=binomial_test(table["event"], table["total"], table["mean_probability"]),
...         jeffreys=jeffreys_test(table["event"], table["total"], table["mean_probability"]),
...     )
>>> grade_tests(table)[["event_rate", "mean_probability", "binomial", "jeffreys"]].round(3)
     event_rate  mean_probability  binomial  jeffreys
bin
A         0.040      0.042     0.609     0.554
B         0.056      0.071     0.858     0.825
C         0.101      0.093     0.352     0.315
D         0.115      0.136     0.937     0.928
E         0.338      0.260     0.098     0.077
F         0.674      0.636     0.356     0.300

No grade is rejected at the usual 5% level. To see the tests at work, scale the PDs down as if the portfolio defaulted at 6% rather than at the 11.6% of the development sample. The monitoring sample still defaults at the development rate, so the PDs are now too low:

>>> low = prior_correction(probability, sample_rate=0.116, population_rate=0.06)
>>> low_grades = pd.cut(low, pd_edges, labels=list("ABCDEF"), include_lowest=True)
>>> low_table = calibration_table(low_grades, monitoring["target"], low)
>>> grade_tests(low_table)[["event_rate", "mean_probability", "binomial", "jeffreys"]].round(3)
     event_rate  mean_probability  binomial  jeffreys
bin
A         0.062      0.032     0.000     0.000
B         0.112      0.067     0.000     0.000
C         0.124      0.095     0.224     0.177
D         0.385      0.145     0.000     0.000
E         0.412      0.315     0.269     0.193
F         0.735      0.501     0.005     0.003

.. plot::
   :alt: Observed default rate against mean predicted PD for each grade, for the original and the scaled-down PDs
   :caption: The grades of the original PDs lie close to the diagonal, while those of the scaled-down PDs lie above it: more defaults than predicted. Larger points hold more customers.

   kp.figure_calibration()

For the same counts, the Jeffreys p-value is always a little lower than the binomial one, so the Jeffreys test rejects slightly more readily. Both tests assume independent defaults; defaults that move together in a downturn make them reject more often than their nominal level.

On a master scale, test each grade against its own PD as well, the PD that every customer of the grade is assigned. On the monitoring sample, the fitted scale from above passes for grades B to E, but grade F has more defaults than its PD of 47% predicts:

>>> fitted_table = calibration_table(
...     assign_grades(probability, fitted), monitoring["target"], probability
... ).join(fitted["probability"])
>>> fitted_table.assign(
...     binomial=binomial_test(
...         fitted_table["event"], fitted_table["total"], fitted_table["probability"]
...     ),
...     jeffreys=jeffreys_test(
...         fitted_table["event"], fitted_table["total"], fitted_table["probability"]
...     ),
... )[["total", "event", "event_rate", "probability", "binomial", "jeffreys"]].round(3)
     total  event  event_rate  probability  binomial  jeffreys
bin
B      328     13       0.040        0.043     0.664     0.611
C      456     36       0.079        0.078     0.509     0.474
D      620     70       0.113        0.142     0.986     0.984
E       70     23       0.329        0.258     0.115     0.092
F       47     31       0.660        0.468     0.006     0.004

The straight line understates the riskiest grade. Give grade F the PD of its own default rate instead, or merge it into grade E, and test the scale again.

Testing all grades at once
--------------------------

The Hosmer-Lemeshow test adds up the squared differences between observed and predicted defaults over all :math:`g` groups:

.. math::

   H = \sum_{k=1}^{g} \frac{(d_k - n_k \bar p_k)^2}{n_k \bar p_k (1 - \bar p_k)}.

On the sample the model was fitted on, form about ten groups of similar size from the sorted PDs and compare :math:`H` with a chi-square distribution with :math:`g - 2` degrees of freedom, the default:

>>> from kivi.binning import bin_edges
>>> from kivi.calibration import hosmer_lemeshow
>>> deciles = pd.cut(dev_prob, bin_edges(dev_prob, bins=10))
>>> dev_table = calibration_table(deciles, development["target"], dev_prob)
>>> statistic, pvalue = hosmer_lemeshow(
...     dev_table["event"], dev_table["total"], dev_table["mean_probability"]
... )
>>> print(f"H = {statistic:.2f}, p = {pvalue:.3f}")
H = 3.42, p = 0.905

PDs that were not fitted on the tested sample, such as the grades of the monitoring sample, lose no degrees of freedom, so pass ``ddof=0``:

>>> statistic, pvalue = hosmer_lemeshow(
...     table["event"], table["total"], table["mean_probability"], ddof=0
... )
>>> print(f"H = {statistic:.2f}, p = {pvalue:.3f}")
H = 5.60, p = 0.470
>>> statistic, pvalue = hosmer_lemeshow(
...     low_table["event"], low_table["total"], low_table["mean_probability"], ddof=0
... )
>>> print(f"H = {statistic:.1f}, p = {pvalue:.1e}")
H = 74.1, p = 6.0e-14

The Hosmer-Lemeshow statistic depends on how the groups are formed. Spiegelhalter's test needs no groups: it compares the Brier score of the probabilities with the score that they would have if they were the true probabilities, and standardizes the difference:

.. math::

   z = \frac{\sum_i (y_i - p_i)(1 - 2 p_i)}{\sqrt{\sum_i (1 - 2 p_i)^2 \, p_i (1 - p_i)}}.

:func:`~kivi.calibration.spiegelhalter_test` returns :math:`z` and its two-sided p-value. The PDs of the monitoring sample pass, and the scaled-down PDs fail clearly:

>>> from kivi.calibration import spiegelhalter_test
>>> for pds in (probability, low):
...     statistic, pvalue = spiegelhalter_test(pds, monitoring["target"])
...     print(f"z = {statistic:.2f}, p = {pvalue:.3f}")
z = -0.95, p = 0.343
z = 7.38, p = 0.000

A positive :math:`z` means that the outcomes lie further from the probabilities than chance allows, and a negative one that they lie closer. The test weighs the observations with probabilities close to 0 or 1 most, so a few confident mistakes can decide it.

Decomposing the Brier score
---------------------------

The Brier score of :func:`~kivi.metrics.brier_score` mixes calibration with discrimination. :func:`~kivi.calibration.brier_decomposition` splits it by groups, such as the grades of a scale. With :math:`n_k` of the :math:`N` observations in group :math:`k`, their mean probability :math:`\bar f_k`, their event rate :math:`\bar o_k` and the overall event rate :math:`\bar o`,

.. math::

   \mathrm{BS} = \mathrm{REL} - \mathrm{RES} + \mathrm{UNC} + \mathrm{WBV} - \mathrm{WBC}, \qquad
   \mathrm{REL} = \frac{1}{N} \sum_k n_k (\bar f_k - \bar o_k)^2, \qquad
   \mathrm{RES} = \frac{1}{N} \sum_k n_k (\bar o_k - \bar o)^2.

The reliability ``REL`` is lower for better calibrated groups, and the resolution ``RES`` is higher for groups whose event rates differ more. The uncertainty :math:`\mathrm{UNC} = \bar o (1 - \bar o)` depends on the outcomes alone. The within-group variance ``WBV`` and covariance ``WBC`` account for probabilities that vary within a group, and are 0 when every observation of a group has the same probability. The scaled-down PDs score worse than the PDs of the model because their reliability is nine times as high, while their resolution and uncertainty are almost the same:

>>> from kivi.calibration import brier_decomposition
>>> pd.DataFrame({
...     "model": brier_decomposition(grades, monitoring["target"], probability),
...     "scaled_down": brier_decomposition(low_grades, monitoring["target"], low),
... }).round(4)
                    model  scaled_down
statistic
brier              0.0877       0.0915
reliability        0.0005       0.0045
resolution         0.0134       0.0135
uncertainty        0.1008       0.1008
within_variance    0.0004       0.0003
within_covariance  0.0006       0.0006

On a master scale, every customer of a grade gets the PD of the grade, so the within-group terms vanish and the three terms of Murphy remain. Compare the PDs of the model with the PDs of the fitted scale, on the grades of that scale:

>>> fitted_grades = assign_grades(probability, fitted)
>>> grade_pd = fitted_grades.map(fitted["probability"]).astype(float)
>>> pd.DataFrame({
...     "model": brier_decomposition(fitted_grades, monitoring["target"], probability),
...     "grade_pd": brier_decomposition(fitted_grades, monitoring["target"], grade_pd),
... }).round(4)
                    model  grade_pd
statistic
brier              0.0877    0.0896
reliability        0.0005    0.0017
resolution         0.0129    0.0129
uncertainty        0.1008    0.1008
within_variance    0.0004    0.0000
within_covariance  0.0011    0.0000

The resolution is the same, because it depends only on the grades and the outcomes. The grade PDs lose the within-group covariance, which rewards the PDs of the model for rising with the default rate within a grade, and their reliability is three times as high, mostly from grade F, whose PD of 47% falls short of its default rate of 66%.

Concentration of the grades
---------------------------

A master scale differentiates risk poorly when most obligors fall into a few grades. The Herfindahl-Hirschman index sums the squared shares of the grades: it is :math:`1/k` when all :math:`k` grades are equally full and 1 when one grade holds everybody. With ``normalize=True`` it runs from 0 to 1 instead:

>>> from kivi.calibration import herfindahl_index
>>> print(f"{herfindahl_index(table['total']):.3f}")
0.247
>>> print(f"{herfindahl_index(table['total'], normalize=True):.3f}")
0.097

:func:`~kivi.calibration.calibration_table` leaves out empty grades. To count them, pass the counts of every grade of the scale, for example ``grades.value_counts(sort=False)``.

The European Central Bank measures the concentration with the HI instead, a logarithmic rescaling of the index for :math:`K` grades, :math:`\mathrm{HI} = 1 + \ln \mathrm{HHI} / \ln K`, which is also 0 for equally full grades and 1 for a single full grade. :func:`~kivi.calibration.concentration_test` computes it together with the coefficient of variation of the shares :math:`R_i` of the grades, and tests whether the coefficient has risen since the initial validation of the model:

.. math::

   \mathrm{CV} = \sqrt{K \sum_{i=1}^{K} \left(R_i - \frac{1}{K}\right)^2}, \qquad
   S = \frac{\sqrt{K - 1}\,(\mathrm{CV} - \mathrm{CV}_\mathrm{init})}{\sqrt{\mathrm{CV}^2 (0.5 + \mathrm{CV}^2)}}, \qquad
   p = 1 - \Phi(S).

Take the development sample as the initial validation, and count every grade of both samples, empty or not. The grades of the monitoring sample are as concentrated as at development, while the scaled-down PDs crowd into the low grades and more than double the HI:

>>> from kivi.calibration import concentration_test
>>> dev_grades = pd.cut(dev_prob, pd_edges, labels=list("ABCDEF"), include_lowest=True)
>>> initial = dev_grades.value_counts(sort=False)
>>> pd.DataFrame({
...     "model": concentration_test(grades.value_counts(sort=False), initial_counts=initial),
...     "scaled_down": concentration_test(
...         low_grades.value_counts(sort=False), initial_counts=initial
...     ),
... }).round(4)
             model  scaled_down
statistic
hi          0.2202       0.4876
cv          0.6954       1.1814
initial_hi  0.2244       0.2244
initial_cv  0.7036       0.7036
statistic  -0.0264       0.6569
p_value     0.5105       0.2556

Yet the test does not reject the scaled-down grades: its variance depends on the number of grades, not on the number of customers, so a scale of six grades leaves it little power. Read the HI and its change as well as the p-value. When only the coefficient of variation of the initial validation is known, pass it as ``initial_cv``. The European Central Bank also asks for the HI of the exposures, without a test; pass the exposure of each grade alone, here with the account balance standing in for the exposure:

>>> exposure = monitoring["balance"].clip(lower=0).groupby(grades, observed=False).sum()
>>> concentration_test(exposure)[["hi", "cv", "p_value"]].round(4)
statistic
hi         0.2306
cv         0.7152
p_value       NaN
dtype: float64

Back-testing continuous estimates
---------------------------------

LGD and CCF models estimate amounts rather than probabilities, so the counts of a calibration table do not apply to them. :func:`~kivi.calibration.backtest_table` compares the realized value :math:`r_i` of each observation with its estimate :math:`e_i` by the one-sample t-test for paired observations that the European Central Bank asks for. For the :math:`N` observations of a group, with the differences :math:`d_i = r_i - e_i`,

.. math::

   T = \frac{\sqrt{N}\,\bar d}{s}, \qquad s^2 = \frac{1}{N - 1} \sum_{i=1}^{N} (d_i - \bar d)^2,

and the p-value is the upper tail of Student's t distribution with :math:`N - 1` degrees of freedom. The test is one-sided: a small p-value means that the estimates of the group are too low. Here two facility grades of an LGD model are tested on seven closed defaults:

>>> from kivi.calibration import backtest_table
>>> lgd_grades = ["F1", "F1", "F1", "F2", "F2", "F2", "F2"]
>>> realized = [0.05, 0.35, 0.20, 0.55, 0.20, 0.90, 0.75]
>>> estimated = [0.15, 0.15, 0.15, 0.45, 0.45, 0.45, 0.45]
>>> backtest_table(lgd_grades, realized, estimated).round(4)
     total  estimated  realized  difference  variance  statistic  p_value
bin
F1       3       0.15       0.2        0.05    0.0225     0.5774   0.3110
F2       4       0.45       0.6        0.15    0.0917     0.9909   0.1974

In grade F1, the differences -0.1, 0.2 and 0.05 have the mean 0.05 and the variance 0.0225, so :math:`T = \sqrt{3} \cdot 0.05 / 0.15 = 0.577`. Both grades realize more than they estimate, but seven defaults are too few to reject either. For the whole portfolio, pass one group for every observation, such as ``["all"] * 7``. The realized values may lie outside [0, 1], as LGDs do after recovery costs or when the recoveries exceed the exposure. :doc:`validation` back-tests a simulated LGD portfolio, also in the twelve LGD segments of the European Central Bank.

Calibrating a score
-------------------

A score that ranks applicants well need not be a probability, and the PDs of a model may need recalibrating on recent data. :mod:`kivi.calibration` fits two common mappings from a score to probabilities. Like :func:`~kivi.binning.bin_edges` and :func:`~kivi.binning.cut`, each comes as a pair: the fit returns plain numbers, fitted once on a calibration sample, and the predict function applies them to the scores of any later sample.

Platt scaling
^^^^^^^^^^^^^

Platt scaling fits a logistic regression of the target on the score :math:`s`:

.. math::

   p(s) = \frac{1}{1 + \exp(-(a s + b))}.

For a score that is already the log odds of a model, this is the logistic recalibration of Cox: a slope of 1 and an intercept of 0 mean that the model is calibrated on the sample. Take the PDs that were scaled down to a 6% default rate above. On the development sample, the fit keeps the slope at 1 and raises the intercept by about 0.72, the log of the odds ratio that the correction applied:

>>> from kivi.calibration import platt_fit, platt_predict
>>> def log_odds(p):
...     return np.log(p / (1 - p))
>>> dev_low = prior_correction(dev_prob, sample_rate=0.116, population_rate=0.06)
>>> slope, intercept = platt_fit(log_odds(dev_low), target)
>>> print(f"slope = {slope:.4f}, intercept = {intercept:.4f}")
slope = 0.9996, intercept = 0.7192

Apply the slope and intercept to the low PDs of the monitoring sample. The recalibrated grades have the counts and rates of the grades of the original PDs again:

>>> recalibrated = platt_predict(log_odds(low), slope, intercept)
>>> recal_grades = pd.cut(recalibrated, pd_edges, labels=list("ABCDEF"), include_lowest=True)
>>> calibration_table(recal_grades, monitoring["target"], recalibrated).round(3)
     total  event  event_rate  mean_probability
bin
A      328     13       0.040             0.042
B      234     13       0.056             0.071
C      276     28       0.101             0.093
D      572     66       0.115             0.136
E       65     22       0.338             0.260
F       46     31       0.674             0.636

The score need not be log odds. Platt scaling fits any score, and a score that falls as the risk rises, such as scorecard points, gets a negative slope. When the score separates the classes, the slope that maximizes the likelihood is infinite, and :func:`~kivi.calibration.platt_fit` raises :class:`kivi.selection.SeparationError`.

Isotonic regression
^^^^^^^^^^^^^^^^^^^

Isotonic regression assumes only that the event rate rises with the score. The pool-adjacent-violators algorithm merges adjacent score bands until their event rates increase, and each band gets its event rate. :func:`~kivi.calibration.isotonic_fit` returns the band edges, from ``-inf`` to ``inf`` like those of :func:`~kivi.binning.bin_edges`, and the probability of each band:

>>> from kivi.calibration import isotonic_fit, isotonic_predict
>>> scores = list(range(1, 11))
>>> outcomes = [0, 1, 0, 0, 1, 0, 1, 1, 0, 1]
>>> edges, probability = isotonic_fit(scores, outcomes)
>>> edges
array([-inf,   1.,   4.,   6.,   9.,  inf])
>>> probability.round(3)
array([0.   , 0.333, 0.5  , 0.667, 1.   ])

The algorithm goes through the scores from the lowest to the highest. Each score starts a band of its own, and while a band has a higher event rate than the band after it, the two are pooled. Here the outcome 0 at score 3 follows an event at score 2, so scores 2 and 3 are pooled at 0.5, and the 0 at score 4 pools them again at 1/3. Each edge is the highest score of its band:

.. plot::
   :alt: Ten scores and outcomes, the bands after each pooling, and the fitted probabilities with the edges 1, 4, 6 and 9
   :caption: Pool-adjacent-violators on the ten scores. Each row shows the bands once the scores up to the one on its left are processed; the outlined band has just been pooled, and the gray outcomes are not processed yet. The last row holds the fitted probabilities.

   kd.figure_isotonic_pav()

On the development PDs, the fit gives 16 bands:

>>> edges, probability = isotonic_fit(dev_low, target)
>>> len(probability)
16
>>> probability[:4].round(4)
array([0.0397, 0.0459, 0.069 , 0.0805])

The bands are closed on the right, as in ``pandas.cut``, so they can also serve as grades. :func:`~kivi.calibration.isotonic_predict` gives each monitoring PD the probability of its band, and scores outside the development range get the probability of the lowest or the highest band. A score that falls between the observed scores of two bands gets the upper band; with ``increasing=False``, as for scorecard points, that is the band with the lower risk:

>>> iso = isotonic_predict(low, edges, probability)
>>> iso_table = calibration_table(pd.cut(low, edges), monitoring["target"], iso)
>>> statistic, pvalue = hosmer_lemeshow(
...     iso_table["event"], iso_table["total"], iso_table["mean_probability"], ddof=0
... )
>>> print(f"H = {statistic:.2f}, p = {pvalue:.3f}")
H = 13.94, p = 0.530
>>> int(iso_table["total"].min())
4

The step function follows the development sample closely, so some bands are small: the smallest holds four customers of the monitoring sample. Fit isotonic regression on a large sample, and check its bands on another sample as above. A band without events has a probability of 0, so floor the probabilities where a PD of zero is not allowed.

Sample weights
--------------

A development sample that keeps every bad but only some of the goods can be weighted back to the portfolio, with each observation weighted by the inverse of its sampling rate. :func:`~kivi.calibration.calibration_table`, :func:`~kivi.calibration.platt_fit` and :func:`~kivi.calibration.isotonic_fit` take such weights in ``weights``: counts become sums of weights, and integer weights give the same result as repeating rows. The binomial, Jeffreys and Hosmer-Lemeshow tests need whole numbers of observations and events, so run them on an unweighted table.

:func:`~kivi.calibration.spiegelhalter_test`, :func:`~kivi.calibration.brier_decomposition` and :func:`~kivi.calibration.backtest_table` take ``weights`` too, as frequency weights: integer weights give the same result as repeating rows, and the tests count the total weight as the number of observations, so inverse sampling rates make their p-values too small. :func:`~kivi.calibration.concentration_test` takes the counts or the exposures of the grades instead of weights.
