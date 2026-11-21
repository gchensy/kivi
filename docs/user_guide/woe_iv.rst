Weight of evidence and information value
========================================

:func:`~kivi.woe.woe_table` measures how each group separates a binary target. An event is target 1 and a non-event is target 0. Both target classes must be present. Group and target series must share the same index; arrays and lists are matched by position, as :doc:`/api/conventions` describes for every function.

WOE and IV follow :doc:`binning`, which forms the groups, and come before :doc:`variable selection <selection>`, which fits a logistic regression on the WOE-encoded variables. :func:`~kivi.woe.woe_transform` encodes new data with the fitted WOE, and :func:`~kivi.woe.woe` and :func:`~kivi.woe.iv` compute the measures from counts that are already aggregated.

Definitions
-----------

For group :math:`i`, let :math:`E_i` and :math:`N_i` be its numbers of events and non-events. Its shares of all events and of all non-events are

.. math::

   p_i = \frac{E_i}{\sum_j E_j}, \qquad q_i = \frac{N_i}{\sum_j N_j}.

The weight of evidence and the information value contribution of the group are

.. math::

   \mathrm{WOE}_i = \ln \frac{p_i}{q_i}, \qquad
   \mathrm{IV}_i = (p_i - q_i)\,\mathrm{WOE}_i, \qquad
   \mathrm{IV} = \sum_i \mathrm{IV}_i.

A positive WOE means that the group is relatively more common among events. Each IV contribution is zero or positive, because :math:`p_i - q_i` and :math:`\ln(p_i / q_i)` always share a sign.

A worked example
----------------

Two thirds of group ``a`` are events and two thirds of group ``b`` are non-events, so the two groups have opposite WOE values of :math:`\pm\ln 2`.

>>> import pandas as pd
>>> from kivi.woe import woe_table
>>> groups = pd.Series(["a", "a", "a", "b", "b", "b"])
>>> target = pd.Series([1, 1, 0, 1, 0, 0])
>>> result = woe_table(groups, target)

The first columns describe each group on its own: its counts, its share of all observations and its event rate.

>>> result[["non_event", "event", "total", "total_share", "event_rate"]].round(4)
     non_event  event  total  total_share  event_rate
bin
a            1      2      3          0.5      0.6667
b            2      1      3          0.5      0.3333

The remaining columns compare the two classes:

>>> result[["non_event_share", "event_share", "woe", "iv"]].round(4)
     non_event_share  event_share     woe     iv
bin
a             0.3333       0.6667  0.6931  0.231
b             0.6667       0.3333 -0.6931  0.231
>>> print(f"{result['iv'].sum():.4f}")
0.4621

When the counts are already aggregated, for example from a database query, :func:`~kivi.woe.woe` and :func:`~kivi.woe.iv` take the event and non-event counts of each group and give the same values:

>>> from kivi.woe import iv, woe
>>> woe([2, 1], [1, 2]).round(4)
array([ 0.6931, -0.6931])
>>> round(iv([2, 1], [1, 2]), 4)
0.4621

Smoothing empty cells
---------------------

A group without events or without non-events would have an infinite WOE. Before the shares are computed, each zero count is replaced by the ``smoothing`` value :math:`s`:

.. math::

   \tilde E_i = \begin{cases} E_i & E_i > 0 \\ s & E_i = 0 \end{cases}, \qquad
   p_i = \frac{\tilde E_i}{\sum_j \tilde E_j},

and likewise for non-events. Smoothing replaces zero cells with 0.5 by default. Each column is then normalized separately. Counts in the result always report observations.

>>> groups = pd.Series(["a", "a", "b", "b", "c"])
>>> target = pd.Series([1, 0, 0, 0, 1])
>>> columns = ["non_event", "event", "non_event_share", "event_share", "woe", "iv"]
>>> woe_table(groups, target)[columns].round(3)
     non_event  event  non_event_share  event_share    woe     iv
bin
a            1      1            0.286          0.4  0.336  0.038
b            2      0            0.571          0.2 -1.050  0.390
c            0      1            0.143          0.4  1.030  0.265

.. warning::

   Changing smoothing can materially change the result for sparse bins. Merge small bins, or compare several smoothing values, before relying on their WOE.

Reading the result
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Meaning
   * - ``non_event``, ``event``, ``total``
     - Observed counts in each group.
   * - ``non_event_share``, ``event_share``
     - :math:`q_i` and :math:`p_i` after smoothing; each column sums to 1.
   * - ``total_share``
     - The group's share of all observations, without smoothing.
   * - ``event_rate``
     - Events divided by the total of the group, without smoothing. For a default target this is the bad rate.
   * - ``woe``
     - :math:`\ln(p_i / q_i)`.
   * - ``iv``
     - The group's IV contribution. Sum the column for the variable's IV.

Read the per-bin contributions before interpreting a total. A group with few events can have a large WOE despite representing few customers.

.. note::

   A widely used rule of thumb reads an IV below 0.02 as not predictive, 0.02 to 0.1 as weak, 0.1 to 0.3 as medium and above 0.3 as strong. Treat these bands as a starting point, not as a test; a very high IV often points to a leak of the target into the variable.

Charting WOE
------------

A WOE chart is the usual way to review a variable with the business: the share of customers and the event rate of each group on top, the WOE below. Categorical variables read best when sorted by WOE.

>>> from kivi.datasets import load_bank
>>> data = load_bank()
>>> reference = data.iloc[:3000]
>>> jobs = woe_table(reference["job"], reference["target"]).sort_values("woe")
>>> jobs[["event", "non_event", "event_rate", "woe"]].round(3).iloc[[0, 1, -2, -1]]
             event  non_event  event_rate    woe
bin
blue-collar     43        576       0.069 -0.564
services        23        246       0.086 -0.339
retired         29        121       0.193  0.602
student         15         44       0.254  0.955

.. plot::
   :alt: Share of customers, subscription rate and WOE for each job
   :caption: WOE by job. Students and retired customers subscribe far more often than blue-collar workers, but students are only 2% of the sample, so their WOE rests on few customers.

   kp.figure_woe_job()

.. dropdown:: Plotting code

   The documentation draws its WOE charts with this matplotlib function. It takes a :func:`~kivi.woe.woe_table` result.

   .. literalinclude:: ../_plotting.py
      :pyobject: woe_chart

To compare many variables at once, rank them by IV; see :doc:`risk_workflow`.

Encoding new data
-----------------

A scorecard model takes WOE values as inputs. :func:`~kivi.woe.woe_transform` replaces each group with its WOE from a development table, so the validation sample and later production data are encoded with exactly the same values. Group every sample with the edges fitted on the development sample:

>>> from kivi.binning import bin_edges
>>> from kivi.woe import woe_transform
>>> edges = bin_edges(reference["age"], bins=5)
>>> age_table = woe_table(pd.cut(reference["age"], edges), reference["target"])
>>> current = data.iloc[3000:]
>>> encoded = woe_transform(pd.cut(current["age"], edges), age_table)
>>> pd.DataFrame({"age": current["age"], "woe": encoded}).head(4).round(3)
      age    woe
3000   45 -0.050
3001   43 -0.050
3002   27  0.114
3003   33 -0.125

The result keeps the index of the groups. A missing value takes the WOE of the missing group when the development sample had missing values.

A group that is not in the table, such as a category that appears only in later data, raises an error by default, so a mapping problem cannot go unnoticed:

>>> new_jobs = pd.Series(["student", "gig worker", None])
>>> woe_transform(new_jobs, jobs)
Traceback (most recent call last):
    ...
ValueError: 2 group(s) are not in table, such as 'gig worker', nan; pass unseen to give them a WOE

Pass ``unseen`` to give such groups a WOE instead. A WOE of 0 is the neutral choice: it stands for the event rate of the whole development sample.

>>> woe_transform(new_jobs, jobs, unseen=0.0).round(3).tolist()
[0.955, 0.0, 0.0]

Sample weights
--------------

Development samples are often stratified: every bad account is kept, but only a fraction of the good ones. Pass the inverse of each observation's sampling rate as ``weights`` so that the counts describe the population. Below, the sample keeps every subscriber and every fourth other customer, so each non-subscriber stands for four customers.

>>> import numpy as np
>>> sample = data[(data["target"] == 1) | (np.arange(len(data)) % 4 == 0)]
>>> weights = np.where(sample["target"] == 1, 1.0, 4.0)
>>> columns = ["event_rate", "woe"]
>>> tables = {
...     "full": woe_table(data["housing"], data["target"])[columns],
...     "sample": woe_table(sample["housing"], sample["target"])[columns],
...     "weighted": woe_table(sample["housing"], sample["target"], weights=weights)[columns],
... }
>>> pd.concat(tables, axis=1).round(3)
          full            sample          weighted
    event_rate    woe event_rate    woe event_rate    woe
bin
no       0.153  0.330      0.431  0.365      0.159  0.365
yes      0.086 -0.326      0.270 -0.349      0.085 -0.349

The unweighted sample overstates the event rate, and the weights bring it back to the population level. The WOE does not change, because it compares shares within each class and the weights of a class scale all of its counts alike. That holds while no group is empty in either class: ``smoothing`` replaces an empty cell in units of weight, so with weights that do not sum to the number of observations, scale ``smoothing`` along with them. In a weighted result the counts are sums of weights, and groups whose weights sum to zero are left out; :func:`~kivi.woe.woe_transform` treats them as unseen.
