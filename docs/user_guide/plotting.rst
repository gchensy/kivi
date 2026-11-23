Charts
======

A risk report shows its tables as charts: the event rate of each bin, the ROC curve of a score, the shares that moved since development. :mod:`kivi.plotting` draws the result tables of the other modules with matplotlib, one function per table: :func:`~kivi.plotting.woe_chart` for a WOE table, :func:`~kivi.plotting.correlation_chart` for a correlation matrix, :func:`~kivi.plotting.roc_chart` and :func:`~kivi.plotting.gains_chart` for the ROC and gains tables, :func:`~kivi.plotting.calibration_chart` for a calibration table, :func:`~kivi.plotting.cutoff_chart` for a cut-off table, :func:`~kivi.plotting.psi_chart` for a PSI table, :func:`~kivi.plotting.monitoring_chart` for a monitoring report and :func:`~kivi.plotting.vintage_chart` for a vintage table.

The charts draw the tables of the earlier guides, from :doc:`woe_iv` to :doc:`vintage`, so this page comes last in the user guide, after :doc:`lifetime`.

matplotlib is an optional dependency of kivi. Install it with the ``plot`` extra:

.. code-block:: bash

   python -m pip install "kivi[plot]"

The other modules do not need it, and :mod:`kivi.plotting` imports it only when a chart is drawn.

.. testcleanup::

   import matplotlib.pyplot as plt

   plt.close("all")

A worked example
----------------

A chart draws the values of its table and nothing else. Ten observations, nine in three groups and one missing, give this WOE table:

>>> import pandas as pd
>>> from kivi.plotting import woe_chart
>>> from kivi.woe import woe_table
>>> groups = pd.Series(["a", "a", "b", "b", "b", "b", "c", "c", "c", None])
>>> target = pd.Series([1, 0, 1, 0, 0, 0, 1, 1, 0, 0])
>>> table = woe_table(groups, target)
>>> table[["total_share", "event_rate"]]
     total_share  event_rate
bin
a            0.2    0.500000
b            0.4    0.250000
c            0.3    0.666667
NaN          0.1    0.000000

:func:`~kivi.plotting.woe_chart` is the one chart with two y-axes, and so the one that returns two Axes: one for the bars, with the left y-axis, and one for the line, which shares the x-axis of the bars and has a y-axis of its own on the right. The bars are the ``total_share`` column, the line is the ``event_rate`` column, the tick labels are the groups of the index, and the legend gives the sum of the ``iv`` column:

>>> bars, line = woe_chart(table)
>>> [bar.get_height() for bar in bars.patches] == table["total_share"].tolist()
True
>>> line.get_lines()[0].get_ydata()
array([0.5       , 0.25      , 0.66666667, 0.        ])
>>> [label.get_text() for label in bars.get_xticklabels()]
['a', 'b', 'c', 'NaN']
>>> line.get_legend().get_title().get_text()
'iv = 0.536'

.. plot::
   :context: reset
   :alt: Bars of the share of groups a, b, c and NaN with a line of their event rates
   :caption: The bars read 20%, 40%, 30% and 10% on the left axis, and the line 50%, 25%, 67% and 0% on the right axis.

   import pandas as pd
   from kivi.plotting import woe_chart
   from kivi.woe import woe_table
   groups = pd.Series(["a", "a", "b", "b", "b", "b", "c", "c", "c", None])
   target = pd.Series([1, 0, 1, 0, 0, 0, 1, 1, 0, 0])
   woe_chart(woe_table(groups, target))

Figures and axes
----------------

Every function takes an optional ``ax``. Without it, the function creates a figure of its own, sized for the chart and with matplotlib's constrained layout. With it, the function draws on that Axes, so that several charts can share one figure:

>>> import matplotlib.pyplot as plt
>>> figure, (left, right) = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
>>> bars, line = woe_chart(table, ax=left)
>>> bars is left
True
>>> bars, line = woe_chart(table, line="woe", ax=right)
>>> len(figure.axes)
4

Each function returns the Axes it drew on, ``ax`` when it is given, so that titles, limits and labels can be changed afterwards with matplotlib, as in ``right.set_title("woe")``. :func:`~kivi.plotting.woe_chart` returns the Axes of the bars and that of the line, which is why the figure above has four Axes.

The functions never call ``matplotlib.pyplot.show``, never change ``matplotlib.rcParams`` and never modify the table. Save a figure with ``figure.savefig("woe.png")``, or show it with ``plt.show()`` outside a notebook. The curves that can be compared on one ``ax``, those of :func:`~kivi.plotting.roc_chart`, :func:`~kivi.plotting.gains_chart`, :func:`~kivi.plotting.calibration_chart`, :func:`~kivi.plotting.cutoff_chart` and :func:`~kivi.plotting.monitoring_chart`, take the next color of the Axes' color cycle. The other charts use fixed colors. Fonts, grids and spines follow the active matplotlib style, so a style set with ``plt.style.context`` applies to the charts drawn inside it.

Text inside a chart is limited to column names, such as ``total_share`` and ``event_rate``, and data values, such as the group labels and the IV. Each axis is labeled with the columns it shows, or with the name of the index for the groups and periods. The same chart therefore suits a report in any language; add titles in the language of the report.

Legends are placed so that they cover no data: :func:`~kivi.plotting.woe_chart` and :func:`~kivi.plotting.psi_chart` leave room for theirs at the top of the Axes, and :func:`~kivi.plotting.vintage_chart` and :func:`~kivi.plotting.monitoring_chart` put theirs to the right of the Axes.

WOE and event rate
------------------

The examples from here on use the bundled bank data, split as in :doc:`risk_workflow` into a development sample of 3,000 rows and a validation sample of 1,521 rows. A :class:`~kivi.spec.BinningSpec` bins five columns on the development sample:

>>> from kivi.datasets import load_bank
>>> from kivi.spec import fit_spec
>>> data = load_bank()
>>> development, validation = data.iloc[:3000], data.iloc[3000:]
>>> columns = ["age", "balance", "housing", "contact", "poutcome"]
>>> spec = fit_spec(development[columns], development["target"])

:func:`~kivi.plotting.woe_chart` draws one of its tables in the order of the table. The bins of a numeric column are already in order; sort the groups of a categorical column, for example by WOE, and draw the WOE as the line with ``line="woe"``:

>>> bars, line = woe_chart(spec.tables["age"])
>>> [label.get_text() for label in bars.get_xticklabels()]
['(-inf, 32.0]', '(32.0, 36.0]', '(36.0, 42.0]', '(42.0, 51.0]', '(51.0, inf]']
>>> bars, line = woe_chart(spec.tables["poutcome"].sort_values("woe"), line="woe")
>>> line.get_legend().get_title().get_text()
'iv = 0.423'

.. plot::
   :context: close-figs
   :alt: Share and event rate of the age bins, and share and WOE of the poutcome groups
   :caption: Left: the event rate of age is lowest in the middle bins. Right: four fifths of the customers have an unknown previous outcome, and the few with a previous success have by far the highest WOE.

   import matplotlib.pyplot as plt
   from kivi.datasets import load_bank
   from kivi.spec import fit_spec
   data = load_bank()
   development, validation = data.iloc[:3000], data.iloc[3000:]
   columns = ["age", "balance", "housing", "contact", "poutcome"]
   spec = fit_spec(development[columns], development["target"])
   figure, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
   woe_chart(spec.tables["age"], ax=left)
   woe_chart(spec.tables["poutcome"].sort_values("woe"), line="woe", ax=right)

Read the bars and the line together: a bin with a high event rate matters more when it holds many customers. The bars use the left axis and the line the right axis, which starts at 0 for ``event_rate`` and includes 0 for ``woe``.

Correlation
-----------

Before :doc:`selection`, :func:`~kivi.plotting.correlation_chart` shows which WOE-encoded candidates carry the same information. It takes a correlation matrix, such as the result of ``DataFrame.corr``, rather than the data, so that any method of correlation can be drawn:

>>> from kivi.plotting import correlation_chart
>>> woe = spec.woe_transform(development[columns])
>>> woe.corr().round(2)
           age  balance  housing  contact  poutcome
age       1.00     0.02     0.15     0.05      0.01
balance   0.02     1.00     0.04     0.03      0.09
housing   0.15     0.04     1.00     0.21      0.03
contact   0.05     0.03     0.21     1.00      0.19
poutcome  0.01     0.09     0.03     0.19      1.00
>>> ax = correlation_chart(woe.corr())
>>> len(ax.texts)
25

.. plot::
   :context: close-figs
   :alt: Heatmap of the correlations between the WOE values of five columns
   :caption: The largest correlation between two different columns is 0.21, between housing and contact.

   from kivi.plotting import correlation_chart
   woe = spec.woe_transform(development[columns])
   correlation_chart(woe.corr())

Each cell shows its value, blue for a positive correlation and amber for a negative one, darker towards 1 and -1. The colors always span -1 to 1, so charts of different samples compare directly, and a value outside that range, as in a covariance matrix, raises an error. For Spearman's rank correlation, pass ``woe.corr(method="spearman")``.

ROC curve
---------

:func:`~kivi.plotting.roc_chart` draws a :func:`~kivi.metrics.roc_table`. The score below adds up the WOE values of the five columns, so that it is higher for a higher risk, as :doc:`metrics` requires. A second call with the same ``ax`` adds the curve of ``poutcome`` alone, and ``label`` names both curves in the legend:

>>> from kivi.metrics import roc_table
>>> from kivi.plotting import roc_chart
>>> validation_woe = spec.woe_transform(validation[columns])
>>> score = validation_woe.sum(axis=1)
>>> y = validation["target"]
>>> ax = roc_chart(roc_table(score, y), label="score")
>>> ax = roc_chart(roc_table(validation_woe["poutcome"], y), ks=False, label="poutcome", ax=ax)
>>> [text.get_text() for text in ax.texts]
['ks = 0.299']

.. plot::
   :context: close-figs
   :alt: ROC curves of the WOE score and of poutcome alone, with the KS of the score marked
   :caption: The score lies above the curve of poutcome alone at most false positive rates. The dotted segment marks its KS of 0.299.

   from kivi.metrics import roc_table
   from kivi.plotting import roc_chart
   validation_woe = spec.woe_transform(validation[columns])
   score = validation_woe.sum(axis=1)
   y = validation["target"]
   ax = roc_chart(roc_table(score, y), label="score")
   roc_chart(roc_table(validation_woe["poutcome"], y), ks=False, label="poutcome", ax=ax)

The dotted segment marks the KS statistic, the largest gap between ``true_positive_rate`` and ``false_positive_rate``, which equals :func:`~kivi.metrics.ks` of the score. Each curve takes the next color of the Axes, and its KS segment shares its color; pass ``ks=False`` for the curves whose KS is not needed.

Gains and KS
------------

:func:`~kivi.plotting.gains_chart` draws a :func:`~kivi.metrics.gains_table`: the shares of all events and of all non-events in the bands reviewed so far, from the highest band down, against the share of the sample reviewed:

>>> from kivi.metrics import gains_table
>>> from kivi.plotting import gains_chart
>>> ax = gains_chart(gains_table(score, y))
>>> ax.texts[0].get_text()
'ks = 0.271'

.. plot::
   :context: close-figs
   :alt: Cumulative event and non-event shares of ten score bands, with the largest gap marked
   :caption: The riskiest 30% of the validation sample holds 54% of its events and 27% of its non-events, the largest gap of the ten bands.

   from kivi.metrics import gains_table
   from kivi.plotting import gains_chart
   gains_chart(gains_table(score, y))

The solid ``cum_event_share`` curve lies above the diagonal and the dashed ``cum_non_event_share`` curve below it when the score ranks well. The dotted segment marks the band with the largest ``ks``. The KS of the bands is at most that of :func:`~kivi.plotting.roc_chart`, which compares every score rather than the band edges. As in :func:`~kivi.plotting.roc_chart`, ``ks=False`` leaves the segment out, and ``label`` names the score in the legend, so that the gains of several scores can be drawn on one ``ax``.

Calibration
-----------

:func:`~kivi.plotting.calibration_chart` draws a :func:`~kivi.calibration.calibration_table`. Below, :func:`~kivi.calibration.platt_fit` turns the score into PDs on the development sample, and the validation PDs are grouped into the grades of a :func:`~kivi.calibration.master_scale`, as in :doc:`calibration`:

>>> import numpy as np
>>> from kivi.calibration import (
...     assign_grades,
...     calibration_table,
...     master_scale,
...     platt_fit,
...     platt_predict,
... )
>>> from kivi.plotting import calibration_chart
>>> slope, intercept = platt_fit(woe.sum(axis=1), development["target"])
>>> probability = platt_predict(score, slope, intercept)
>>> scale = master_scale(np.geomspace(0.02, 0.32, 5), labels=list("ABCDE"))
>>> table = calibration_table(assign_grades(probability, scale), y, probability)
>>> table
       total  event  event_rate  mean_probability
bin
A         43      1    0.023256          0.023619
B        368     17    0.046196          0.041415
C        466     46    0.098712          0.085882
D        558     69    0.123656          0.153627
E         86     40    0.465116          0.435768
>>> ax = calibration_chart(table, label="validation")

.. plot::
   :context: close-figs
   :alt: Event rate against mean predicted probability of grades A to E, next to the diagonal
   :caption: The five grades lie close to the diagonal; grade D has fewer events than predicted.

   import numpy as np
   from kivi.calibration import (
       assign_grades,
       calibration_table,
       master_scale,
       platt_fit,
       platt_predict,
   )
   from kivi.plotting import calibration_chart
   slope, intercept = platt_fit(woe.sum(axis=1), development["target"])
   probability = platt_predict(score, slope, intercept)
   scale = master_scale(np.geomspace(0.02, 0.32, 5), labels=list("ABCDE"))
   table = calibration_table(assign_grades(probability, scale), y, probability)
   calibration_chart(table, label="validation")

Points above the diagonal have more events than predicted, and points below it fewer. Both axes run from 0 to just above the largest rate, so that the diagonal stays at 45 degrees. The chart shows how far each grade lies from its prediction, not whether the difference is significant; :func:`~kivi.calibration.binomial_test` answers that.

Cut-offs
--------

:func:`~kivi.plotting.cutoff_chart` draws a :func:`~kivi.strategy.cutoff_table`: the bad rate of the approved book against the share of applicants approved, one point per cut-off of the PD:

>>> from kivi.plotting import cutoff_chart
>>> from kivi.strategy import cutoff_table
>>> ax = cutoff_chart(cutoff_table(probability, y, cutoffs=20))
>>> ax.get_xlabel(), ax.get_ylabel()
('approval_rate', 'approved_event_rate')

.. plot::
   :context: close-figs
   :alt: Event rate of the approved book against the approval rate for twenty PD cut-offs
   :caption: Approving more applicants raises the event rate of the approved book, from about 1% for the safest 5% of applicants to 11% when all of them are approved.

   from kivi.plotting import cutoff_chart
   from kivi.strategy import cutoff_table
   cutoff_chart(cutoff_table(probability, y, cutoffs=20))

The slope of the curve is the price in risk of a higher approval rate. Draw the tables of a current and a new score on the same ``ax``, each with a ``label``, to compare them at the same approval rate, as :doc:`strategy` does.

Population stability
--------------------

:func:`~kivi.plotting.psi_chart` draws a :func:`~kivi.stability.psi_table` as pairs of bars, the reference share and the current share of each group. Single customers, as in :doc:`psi`, are much younger than the development sample:

>>> from kivi.plotting import psi_chart
>>> from kivi.stability import psi_table
>>> reference = spec.cut(development[columns])
>>> single = spec.cut(data.loc[data["marital"] == "single", columns])
>>> ax = psi_chart(psi_table(reference["age"], single["age"]))
>>> ax.get_legend().get_title().get_text()
'psi = 0.586'

.. plot::
   :context: close-figs
   :alt: Reference and current shares of five age bins for single customers
   :caption: Half of the single customers fall into the youngest bin, against less than a quarter of the development sample.

   from kivi.plotting import psi_chart
   from kivi.stability import psi_table
   reference = spec.cut(development[columns])
   single = spec.cut(data.loc[data["marital"] == "single", columns])
   psi_chart(psi_table(reference["age"], single["age"]))

The groups whose bars differ most contribute most of the PSI, which the legend gives for the whole table.

Monitoring
----------

:func:`~kivi.plotting.monitoring_chart` draws one column of a :func:`~kivi.monitoring.monitoring_table` period by period. The applications below are simulated as in :doc:`monitoring`; the outcomes of the last three months are not known yet:

>>> from kivi.datasets import make_applications
>>> from kivi.monitoring import monitoring_table
>>> from kivi.plotting import monitoring_chart
>>> reference_applications = make_applications(n_applicants=10000, rng=0)
>>> applications = make_applications(n_applicants=24000, rng=1)
>>> months = pd.period_range("2025-01", periods=12, freq="M")
>>> applications["month"] = months[np.arange(24000) * 12 // 24000]
>>> known = applications["approved"] & (applications["month"] <= months[8])
>>> report = monitoring_table(
...     applications["incumbent_pd"],
...     applications["month"],
...     applications["target"].where(known),
...     reference_score=reference_applications["incumbent_pd"],
... )

Call the function once for each column, on the panels of one figure. ``kind`` draws a column as a line, the default, or as bars, and ``levels`` marks the values that a column is compared with, such as the PSI thresholds 0.1 and 0.25. ``label_outer`` then hides the x-axis labels of the upper panel, which shares the x-axis of the lower one:

>>> figure, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, layout="constrained")
>>> top = monitoring_chart(report, column="gini", ax=top)
>>> bottom = monitoring_chart(report, column="psi", kind="bar", levels=[0.1, 0.25], ax=bottom)
>>> top.label_outer()
>>> [text.get_text() for text in bottom.texts]
['0.1', '0.25']

.. plot::
   :context: close-figs
   :alt: Gini and PSI of the incumbent PD for each month of 2025, with the PSI thresholds 0.1 and 0.25
   :caption: The Gini stops in September, the last month with known outcomes. The PSI stays far below 0.1 in every month.

   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   from kivi.datasets import make_applications
   from kivi.monitoring import monitoring_table
   from kivi.plotting import monitoring_chart
   reference_applications = make_applications(n_applicants=10000, rng=0)
   applications = make_applications(n_applicants=24000, rng=1)
   months = pd.period_range("2025-01", periods=12, freq="M")
   applications["month"] = months[np.arange(24000) * 12 // 24000]
   known = applications["approved"] & (applications["month"] <= months[8])
   report = monitoring_table(
       applications["incumbent_pd"],
       applications["month"],
       applications["target"].where(known),
       reference_score=reference_applications["incumbent_pd"],
   )
   figure, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, layout="constrained")
   monitoring_chart(report, column="gini", ax=top)
   monitoring_chart(report, column="psi", kind="bar", levels=[0.1, 0.25], ax=bottom)
   top.label_outer()

A line suits rates and measures of discrimination, and several columns on one ``ax``; bars suit counts such as ``total`` and a single ``psi``. ``kind`` decides, not the name of the column, so any table with one row per period can be drawn either way.

A :func:`~kivi.monitoring.stability_table` holds the PSI of each variable by period. Draw its columns on one ``ax``, each with a ``label``. The y-axis is labeled ``psi``, because the columns of a stability table are named ``variable``:

>>> from kivi.monitoring import stability_table
>>> features = ["age", "income", "debt_to_income"]
>>> application_spec = fit_spec(
...     reference_applications[features], reference_applications["target"]
... )
>>> stability = stability_table(
...     application_spec.cut(applications[features]),
...     applications["month"],
...     reference_groups=application_spec.cut(reference_applications[features]),
... )
>>> ax = None
>>> for column in stability:
...     ax = monitoring_chart(stability, column=column, label=column, ax=ax)
>>> ax.get_ylabel()
'psi'

.. plot::
   :context: close-figs
   :alt: PSI of age, income and debt_to_income for each month of 2025
   :caption: The PSI of the three variables stays below 0.01 in every month.

   from kivi.monitoring import stability_table
   features = ["age", "income", "debt_to_income"]
   application_spec = fit_spec(
       reference_applications[features], reference_applications["target"]
   )
   stability = stability_table(
       application_spec.cut(applications[features]),
       applications["month"],
       reference_groups=application_spec.cut(reference_applications[features]),
   )
   ax = None
   for column in stability:
       ax = monitoring_chart(stability, column=column, label=column, ax=ax)

Vintage curves
--------------

:func:`~kivi.plotting.vintage_chart` draws a :func:`~kivi.vintage.vintage_table` with one curve per cohort, later cohorts in darker blue. The legend, to the right of the Axes, names eight of the twenty-four cohorts, evenly spread from the first to the last, since the shades of the others lie between theirs:

>>> from kivi.datasets import make_loan_panel
>>> from kivi.plotting import vintage_chart
>>> from kivi.vintage import vintage_table
>>> panel = make_loan_panel(rng=0)
>>> ever_90 = vintage_table(
...     panel["loan_id"], panel["cohort"], panel["month_on_book"], panel["status"] == "90+"
... )
>>> ax = vintage_chart(ever_90)
>>> [text.get_text() for text in ax.get_legend().get_texts()]
['2024-01', '2024-04', '2024-08', '2024-11', '2025-02', '2025-05', '2025-09', '2025-12']

.. plot::
   :context: close-figs
   :alt: Cumulative 90+ rate of twenty-four cohorts by month on book
   :caption: No cohort has a loan 90 days past due before its fourth month on book, and the curves of the later cohorts end where their observation ends.

   from kivi.datasets import make_loan_panel
   from kivi.plotting import vintage_chart
   from kivi.vintage import vintage_table
   panel = make_loan_panel(rng=0)
   ever_90 = vintage_table(
       panel["loan_id"], panel["cohort"], panel["month_on_book"], panel["status"] == "90+"
   )
   vintage_chart(ever_90)

A cohort whose curve rises above those of the cohorts before it is worth a closer look. The function takes any table of the same shape, such as the cumulative PDs of :func:`~kivi.lifetime.convert_pd`.

Special values
--------------

A special value keeps the group that :func:`~kivi.binning.cut` gave it, and the chart labels it with its code. For ``pdays``, where -1 means that the customer was not contacted before:

>>> pdays = fit_spec(development[["pdays"]], development["target"], special={"pdays": [-1]})
>>> bars, line = woe_chart(pdays.tables["pdays"])
>>> [label.get_text() for label in bars.get_xticklabels()]
['-1', '(-inf, 105.0]', '(105.0, 175.6]', '(175.6, 255.4]', '(255.4, 342.2]', '(342.2, inf]']

The bar of ``-1`` holds 82.5% of the sample. Charts do not reorder groups, so the special group stays where the table has it, ahead of the intervals.

Missing values
--------------

The group of missing values is labeled ``NaN``, as in the worked example. A missing value in a column, such as the ``gini`` of a month without known outcomes or a month on book that a cohort has not reached, leaves a gap in a line and no bar. A cut-off that approves nothing has a missing ``approved_event_rate`` and no point on the curve of :func:`~kivi.plotting.cutoff_chart`.

Limits
------

* The charts draw a table as it is. They have no ``weights`` argument: draw a table computed with weights to show weighted shares and rates.
* The functions choose the size of a new figure, the colors and the labels, and pass no other matplotlib properties through; change them on the returned Axes.
* A chart has no title, so that it fits a report in any language. Add one with ``ax.set_title``.
* :func:`~kivi.plotting.monitoring_chart` draws one column per call, and :func:`~kivi.plotting.correlation_chart` one matrix without a color bar; add one with ``figure.colorbar(ax.images[0], ax=ax)``.
* The charts are static matplotlib figures, not interactive ones.

Errors and warnings
-------------------

Without matplotlib, every chart raises an ``ImportError`` that asks for ``pip install "kivi[plot]"``.

A table that lacks a column a chart needs raises a ``ValueError`` that names the columns of the table it expects and those that are missing, as when a PSI table is passed to :func:`~kivi.plotting.woe_chart`:

>>> woe_chart(psi_table(reference["age"], single["age"]))
Traceback (most recent call last):
...
ValueError: table must have the columns ['total_share', 'event_rate', 'iv'] of kivi.woe.woe_table, missing ['total_share', 'event_rate', 'iv']

An ``ax`` that is not a matplotlib Axes, an empty table, a column that is not numeric and a correlation matrix with a value outside [-1, 1] raise a ``ValueError`` as well. A chart that raises does not leave an empty figure behind.

matplotlib warns when more than 20 figures are open. When a report draws a chart for each of many variables, save each figure and close it with ``plt.close(ax.figure)``.
