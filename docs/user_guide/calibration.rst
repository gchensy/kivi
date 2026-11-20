Calibration
===========

A PD model is calibrated when its probabilities match the default rates that are later observed. Calibration is a separate question from ranking: a model can order applicants well and still predict default rates that are too low for the whole portfolio. :mod:`kivi.calibration` corrects probabilities for the sampling of the development data, summarizes the grades of a master scale and tests them.

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

A master scale divides the PD range into a fixed set of grades, so that ratings from different models share one meaning. The examples below use the PDs of the scorecard from :doc:`scorecard` on the development and monitoring samples:

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
>>> coefficients = pd.Series(
...     {"poutcome": 0.899, "contact": 0.658, "housing": 0.691, "age": 0.714}
... )
>>> def predict(frame):
...     groups = to_groups(frame)
...     woe = pd.DataFrame(
...         {column: woe_transform(groups[column], tables[column]) for column in groups}
...     )
...     return 1 / (1 + np.exp(2.028 - woe @ coefficients))
>>> dev_prob, prob = predict(development), predict(monitoring)

Assign the grades with ``pandas.cut`` and the PD edges of the scale, then compare the observed default rate of each grade with its mean predicted PD using ``calibration_table``:

>>> from kivi.calibration import calibration_table
>>> pd_edges = [0, 0.05, 0.08, 0.12, 0.2, 0.4, 1]
>>> grades = pd.cut(prob, pd_edges, labels=list("ABCDEF"))
>>> table = calibration_table(grades, monitoring["target"], prob)
>>> table.round(3)
     count  event  event_rate  mean_prob
bin
A      328     13       0.040      0.042
B      234     13       0.056      0.071
C      276     28       0.101      0.093
D      572     66       0.115      0.136
E       65     22       0.338      0.260
F       46     31       0.674      0.636

``calibration_table`` accepts any grouping, such as score bands or deciles of the PD. Like ``woe_table``, it follows the order of categorical groups and leaves out grades without observations.

Testing each grade
------------------

Each grade is tested on its own against the hypothesis that its PD is not too low. Both tests are one-sided: a small p-value means that the grade has more defaults than its PD predicts.

* The **binomial test** treats the :math:`N` obligors of a grade as independent, each defaulting with probability :math:`p`. Its p-value is the probability of at least the observed :math:`d` defaults, :math:`P(D \ge d)`.
* The **Jeffreys test** starts from the Jeffreys prior :math:`\mathrm{Beta}(1/2, 1/2)` for the default rate. Its p-value is the posterior probability that the default rate is at most :math:`p`, the distribution function of :math:`\mathrm{Beta}(d + 1/2, N - d + 1/2)` at :math:`p`. The European Central Bank asks for this test in the validation of PD models.

>>> from kivi.calibration import binomial_test, jeffreys_test
>>> def grade_tests(table):
...     return table.assign(
...         binomial=binomial_test(table["event"], table["count"], table["mean_prob"]),
...         jeffreys=jeffreys_test(table["event"], table["count"], table["mean_prob"]),
...     )
>>> grade_tests(table)[["event_rate", "mean_prob", "binomial", "jeffreys"]].round(3)
     event_rate  mean_prob  binomial  jeffreys
bin
A         0.040      0.042     0.609     0.554
B         0.056      0.071     0.858     0.825
C         0.101      0.093     0.352     0.315
D         0.115      0.136     0.937     0.928
E         0.338      0.260     0.098     0.077
F         0.674      0.636     0.356     0.300

No grade is rejected at the usual 5% level. To see the tests at work, scale the PDs down as if the portfolio defaulted at 6% rather than at the 11.6% of the development sample. The monitoring sample still defaults at the development rate, so the PDs are now too low:

>>> low = prior_correction(prob, sample_rate=0.116, population_rate=0.06)
>>> low_grades = pd.cut(low, pd_edges, labels=list("ABCDEF"))
>>> low_table = calibration_table(low_grades, monitoring["target"], low)
>>> grade_tests(low_table)[["event_rate", "mean_prob", "binomial", "jeffreys"]].round(3)
     event_rate  mean_prob  binomial  jeffreys
bin
A         0.062      0.032     0.000     0.000
B         0.112      0.067     0.000     0.000
C         0.124      0.095     0.224     0.177
D         0.385      0.145     0.000     0.000
E         0.412      0.315     0.269     0.193
F         0.735      0.501     0.005     0.003

For the same counts, the Jeffreys p-value is always a little lower than the binomial one, so the Jeffreys test rejects slightly more readily. Both tests assume independent defaults; defaults that move together in a downturn make them reject more often than their nominal level.

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
...     dev_table["event"], dev_table["count"], dev_table["mean_prob"]
... )
>>> print(f"H = {statistic:.2f}, p = {pvalue:.3f}")
H = 3.42, p = 0.905

PDs that were not fitted on the tested sample, such as the grades of the monitoring sample, lose no degrees of freedom, so pass ``ddof=0``:

>>> statistic, pvalue = hosmer_lemeshow(
...     table["event"], table["count"], table["mean_prob"], ddof=0
... )
>>> print(f"H = {statistic:.2f}, p = {pvalue:.3f}")
H = 5.60, p = 0.470
>>> statistic, pvalue = hosmer_lemeshow(
...     low_table["event"], low_table["count"], low_table["mean_prob"], ddof=0
... )
>>> print(f"H = {statistic:.1f}, p = {pvalue:.1e}")
H = 74.1, p = 6.0e-14

Concentration of the grades
---------------------------

A master scale differentiates risk poorly when most obligors fall into a few grades. The Herfindahl-Hirschman index sums the squared shares of the grades: it is :math:`1/k` when all :math:`k` grades are equally full and 1 when one grade holds everybody. With ``normalize=True`` it runs from 0 to 1 instead:

>>> from kivi.calibration import herfindahl_index
>>> print(f"{herfindahl_index(table['count']):.3f}")
0.247
>>> print(f"{herfindahl_index(table['count'], normalize=True):.3f}")
0.097

``calibration_table`` leaves out empty grades. To count them, pass the counts of every grade of the scale, for example ``grades.value_counts(sort=False)``.
