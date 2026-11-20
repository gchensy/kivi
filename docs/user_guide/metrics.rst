Model performance
=================

``kivi.metrics`` measures how well a score separates events from non-events. ``roc_auc``, ``gini`` and ``ks`` summarize the ranking in one number, ``gains_table`` breaks it down by score band, and ``brier_score`` checks predicted probabilities against the outcomes.

Score direction
---------------

Every function expects a score that is higher for a higher event probability, such as a predicted probability of default (PD). Scorecard points run the other way, because more points mean a lower risk, so negate them before passing them in: ``roc_auc(-points, target)``.

Every function takes the score first and the target second, as ``woe_table(groups, target)`` takes the groups first. scikit-learn's metrics take the target first, so check the order when porting code from them.

As in ``woe_table``, the target is 1 for an event and 0 for a non-event. Two Series must share the same index; arrays and lists are matched by position. Scores must be finite, so decide how missing values are scored before evaluating the model.

Definitions
-----------

For :math:`n_1` events and :math:`n_0` non-events, let :math:`R_1` be the sum of the ranks of the events among all scores, where tied scores share their average rank. The area under the ROC curve and the Gini coefficient are

.. math::

   \mathrm{AUC} = \frac{R_1 - n_1 (n_1 + 1) / 2}{n_1 n_0}, \qquad
   \mathrm{Gini} = 2\,\mathrm{AUC} - 1.

The AUC is the probability that a randomly chosen event scores higher than a randomly chosen non-event, with ties counted as one half. A score unrelated to the target gives an AUC of about 0.5 and a Gini of about 0.

Let :math:`F_1` and :math:`F_0` be the empirical distribution functions of the scores of events and of non-events. The Kolmogorov-Smirnov statistic is their largest gap:

.. math::

   \mathrm{KS} = \max_t \, \lvert F_1(t) - F_0(t) \rvert.

The gap is taken in absolute value, as in the two-sample test. KS therefore says nothing about the direction of the score, and its largest gap can come from a range of scores where the ranking is reversed. Read it together with the Gini.

A worked example
----------------

Three of these six customers are events. Of the nine pairs of an event and a non-event, six are ordered correctly, two are not, and one is tied at 0.4, so the AUC is :math:`(6 + 0.5) / 9`:

>>> from kivi.metrics import gini, ks, roc_auc
>>> score = [0.1, 0.4, 0.4, 0.8, 0.7, 0.9]
>>> target = [0, 0, 1, 0, 1, 1]
>>> round(roc_auc(score, target), 4)
0.7222
>>> round(gini(score, target), 4)
0.4444

At a cut-off of 0.4, two thirds of the non-events but only one third of the events score at or below it. No cut-off separates the two classes further, so the KS is one third:

>>> round(ks(score, target), 4)
0.3333

Evaluating a WOE score
----------------------

The bank data give a more realistic example. The score below adds up the WOE values of four variables, fitted on the development sample. A higher WOE means a higher event rate, so the sum already runs in the direction the metrics expect. A scorecard would weight the variables with a logistic regression; the metrics treat any score in the same way.

>>> import numpy as np
>>> import pandas as pd
>>> from kivi.datasets import Dataset
>>> from kivi.woe import woe_table, woe_transform
>>> data = Dataset.bank_data()
>>> development, validation = data.iloc[:3000], data.iloc[3000:]
>>> def to_groups(frame):
...     return pd.DataFrame({
...         "poutcome": frame["poutcome"],
...         "contact": frame["contact"],
...         "housing": frame["housing"],
...         "age": pd.cut(frame["age"], [-np.inf, 25, 35, 45, 55, 65, np.inf]),
...     })
>>> dev_groups = to_groups(development)
>>> tables = {column: woe_table(dev_groups[column], development["target"]) for column in dev_groups}
>>> def woe_score(frame):
...     groups = to_groups(frame)
...     return pd.DataFrame({
...         column: woe_transform(groups[column], tables[column]) for column in groups
...     }).sum(axis=1)
>>> dev_score, val_score = woe_score(development), woe_score(validation)

Measure the score on both samples. The validation rows played no part in fitting the WOE values, so they show how the score holds up on new customers:

>>> samples = {
...     "development": (dev_score, development["target"]),
...     "validation": (val_score, validation["target"]),
... }
>>> pd.DataFrame({
...     name: {"auc": roc_auc(s, y), "gini": gini(s, y), "ks": ks(s, y)}
...     for name, (s, y) in samples.items()
... }).round(3)
      development  validation
auc         0.686       0.716
gini        0.372       0.431
ks          0.265       0.321

A clear drop from development to validation would point to overfitting. Here the validation values are even slightly higher, which is within the sampling noise of 173 validation events.

Points that fall as the risk rises rank the customers in the opposite order. Passed as they are, they give an AUC below 0.5 and a negative Gini; negated, they give the values above. KS is the same in both directions, so it cannot reveal a reversed score:

>>> points = 600 - 50 * val_score
>>> y = validation["target"]
>>> [round(roc_auc(points, y), 3), round(roc_auc(-points, y), 3)]
[0.284, 0.716]
>>> [round(ks(points, y), 3), round(ks(-points, y), 3)]
[0.321, 0.321]

.. note::

   No threshold makes a Gini or a KS good in general: the values that can be reached depend on the portfolio, the target definition and the data available. Compare a new score with the one it replaces on the same validation sample.

Gains table
-----------

``gains_table`` sorts the sample into score bands, from the highest scores to the lowest. Each row shows the event rate of its band and the share of all events found by reviewing that band and every band above it. The default ``bins=10`` cuts the sample at the edges of ``bin_edges(score, bins=10)``, and tied scores always share a band.

In monitoring, fix the bands on the development sample so that every later sample is cut at the same scores. Fit the edges with ``bin_edges`` and round them to readable cut-offs:

>>> from kivi.binning import bin_edges
>>> from kivi.metrics import gains_table
>>> edges = bin_edges(dev_score, bins=5).round(2)
>>> edges
array([ -inf, -1.46, -0.38,  0.15,  0.32,   inf])
>>> gains = gains_table(val_score, y, bins=edges)
>>> gains[["total", "event", "event_rate", "lift", "cum_event_share", "ks"]].round(3)
                total  event  event_rate   lift  cum_event_share     ks
bin
(0.32, inf]       264     72       0.273  2.398            0.416  0.274
(0.15, 0.32]      344     39       0.113  0.997            0.642  0.273
(-0.38, 0.15]     344     34       0.099  0.869            0.838  0.239
(-1.46, -0.38]    241     15       0.062  0.547            0.925  0.159
(-inf, -1.46]     328     13       0.040  0.348            1.000  0.000

The top band holds 17% of the validation customers (264 of 1,521) but 42% of the events, so its event rate is 2.4 times that of the whole sample. The table checks the gap between the two classes only at the four interior cut-offs, so its largest ``ks`` stays below the KS over all scores:

>>> print(f"{gains['ks'].max():.3f} < {ks(val_score, y):.3f}")
0.274 < 0.321

Every band is reported, including a band that no customer falls into, so two samples cut on the same edges give tables with the same rows. Their ``total`` columns then give the PSI of the score bands, here far below 0.1, the usual upper bound of a small shift (see :doc:`psi`):

>>> from kivi.stability import psi
>>> dev_gains = gains_table(dev_score, development["target"], bins=edges)
>>> round(psi(dev_gains["total"], gains["total"]), 4)
0.0025

Reading the result
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Meaning
   * - ``non_event``, ``event``, ``total``
     - Non-events, events and all observations in the band, as in ``woe_table``.
   * - ``event_rate``
     - ``event`` divided by ``total``; NaN for an empty band.
   * - ``lift``
     - The event rate of the band divided by the event rate of the whole sample.
   * - ``cum_non_event_share``, ``cum_event_share``
     - Shares of all non-events and of all events in this band and the bands above it.
   * - ``cum_total_share``
     - Share of all observations in this band and the bands above it. Plotting ``cum_event_share`` against ``cum_total_share`` gives the gains chart.
   * - ``ks``
     - The absolute gap between ``cum_event_share`` and ``cum_non_event_share`` at the lower edge of the band. Like ``ks``, it can come from a band where the score runs the wrong way, with ``cum_non_event_share`` above ``cum_event_share``.
   * - ``cum_lift``
     - ``cum_event_share`` divided by ``cum_total_share``: the lift of reviewing this band and every band above it.

Brier score
-----------

The ranking measures ignore the probability scale: any increasing transformation of the score gives the same AUC, Gini and KS. When a model predicts probabilities, for example a PD used for pricing or provisions, also compare them with the observed outcomes. ``brier_score`` is the mean squared difference between the probability :math:`p_i` and the outcome :math:`y_i`, so a lower value is better:

.. math::

   \mathrm{BS} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2.

The example below predicts the probability of each validation customer by the development event rate of the customer's ``poutcome`` group, and compares it with predicting the development event rate for everyone:

>>> from kivi.metrics import brier_score
>>> rates = development.groupby("poutcome")["target"].mean()
>>> by_group = validation["poutcome"].map(rates)
>>> overall = np.full(len(validation), development["target"].mean())
>>> [round(brier_score(by_group, y), 4), round(brier_score(overall, y), 4)]
[0.0907, 0.1008]

The group rates lower the Brier score by about 10% compared with the constant forecast. Unlike the ranking measures, the Brier score needs no events in the sample, so it can also be computed for a period without any.

Weighted samples
----------------

Validation samples often keep every event but only some of the non-events. Every function of :mod:`kivi.metrics` takes ``weights``, one non-negative weight per observation matched like the target, and then counts sums of weights instead of observations, so integer weights give the same result as repeating rows. To weight a sample back to the population, give each observation the inverse of its sampling rate. Below, every event of the validation sample is kept but only every second non-event:

>>> non_events_so_far = (y == 0).cumsum()
>>> kept = (y == 1) | (non_events_so_far % 2 == 0)
>>> sample_score, sample_y = val_score[kept], y[kept]
>>> weights = np.where(sample_y == 1, 1.0, 2.0)
>>> pd.DataFrame({
...     "all": gains_table(val_score, y, bins=edges)["event_rate"],
...     "sample": gains_table(sample_score, sample_y, bins=edges)["event_rate"],
...     "weighted": gains_table(sample_score, sample_y, bins=edges, weights=weights)[
...         "event_rate"
...     ],
... }).round(3)
                  all  sample  weighted
bin
(0.32, inf]     0.273   0.409     0.257
(0.15, 0.32]    0.113   0.203     0.113
(-0.38, 0.15]   0.099   0.185     0.102
(-1.46, -0.38]  0.062   0.125     0.067
(-inf, -1.46]   0.040   0.074     0.039

Without weights the sample has twice as many events per non-event as the population, and every band's event rate is inflated. The weights restore rates close to those of all customers. The ranking measures do not change, because a weight that is the same for every observation of a class cancels out of the AUC, the Gini and the KS:

>>> [round(roc_auc(sample_score, sample_y), 4), round(roc_auc(sample_score, sample_y, weights=weights), 4)]
[0.7073, 0.7073]

Weights that differ within a class, such as the inverse sampling rates of a stratified sample, do change them.

