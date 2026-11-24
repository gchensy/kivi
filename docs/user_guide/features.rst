Time-window features
====================

Most behavioural variables of a scorecard summarize a history of events: the number of transactions in the last three months, the largest withdrawal in a year, the days since the last payment. Each is computed for an entity, such as a customer, at an observation time, such as the date of an application, and may only use what was known then. Code that filters and groups the events by hand easily lets a later event slip in, and code written again for production rarely gives the same numbers. :func:`~kivi.features.window_features` computes such variables for many observations at once and never uses an event at or after the observation time, :func:`~kivi.features.panel_features` does the same for monthly repayment statuses, such as the months in arrears in the last year, and a :class:`~kivi.features.FeatureSpec` keeps their definitions so that production computes them in the same way.

Window features come first in a project. :doc:`univariate` profiles the features that they give against the target, and :doc:`binning` bins the ones that are kept.

Definitions
-----------

An *observation* is an entity :math:`i` and a time :math:`t`, and a *window* of :math:`k` months is the interval :math:`[t - k, t)` before it. A feature aggregates the values :math:`v_e` of one column over the events :math:`e` of the entity whose times :math:`\tau_e` lie in the window:

.. math::

   x_{i, t} = \operatorname{agg} \{\, v_e : \mathrm{id}_e = i,\ t - k \le \tau_e < t \,\}.

The aggregation is a count, a sum, a mean, an extreme value or another of the aggregations below. The window ends just before :math:`t`, so events at the observation time and after it never count: the features are *point in time*. A pair of windows, such as from six to three months before :math:`t`, is the interval :math:`[t - 6, t - 3)`, the period before the most recent one.

Months are calendar months, counted as ``pandas.DateOffset(months=k)`` counts them: three months before 31 May is 28 February, or 29 February in a leap year, and the time of day is kept. A window of :math:`k` days starts :math:`k` days before :math:`t`.

A worked example
----------------

Two customers have made a few card payments and cash withdrawals, and three customers apply for a loan on 31 March 2024. Customer ``c`` has no history:

>>> import numpy as np
>>> import pandas as pd
>>> from kivi.features import FeatureSpec, window_features
>>> events = pd.DataFrame({
...     "customer": ["a", "a", "a", "a", "a", "b", "b", "b"],
...     "time": pd.to_datetime([
...         "2023-11-20", "2024-01-10", "2024-02-29", "2024-03-15", "2024-03-31",
...         "2024-01-05", "2024-03-20", "2024-03-20",
...     ]),
...     "amount": [300.0, 120.0, 80.0, 50.0, 200.0, 50.0, 40.0, 25.0],
...     "kind": ["card", "card", "cash", "card", "card", "cash", "cash", "card"],
... })
>>> applications = pd.DataFrame({
...     "customer": ["a", "b", "c"],
...     "application_date": pd.to_datetime(["2024-03-31", "2024-03-31", "2024-03-31"]),
... })
>>> keys = {"id": "customer", "time": "time", "observation_time": "application_date"}
>>> window_features(
...     events, applications, **keys,
...     windows=["1M", "3M"], aggregations={"amount": ["sum"]},
... )
   events_1m  amount_sum_1m  events_3m  amount_sum_3m
0          2          130.0          3          250.0
1          2           65.0          3          115.0
2          0            0.0          0            0.0

The month before 31 March starts on 29 February, so for customer ``a`` the window ``1m`` holds the withdrawal of 80 on that day and the payment of 50 on 15 March: two events with a sum of 130. The payment of 200 on 31 March is on the application date and counts in no window. The three months start on 31 December 2023 and add the payment of 120 on 10 January, while the payment of 300 in November is older than every window. Customer ``c`` has no events, so its counts and sums are 0; its mean or largest amount would be NaN.

The result has one row per application, on the index of ``applications``, so it can be put next to the applications with ``pd.concat([applications, features], axis=1)``.

.. plot::
   :alt: The events of customers a and b on a time line from November 2023 to April 2024, the windows of one and three months that end at the application date of 31 March 2024, and the number and the sum of the amounts of the events in each window
   :caption: The worked example. Each window starts one or three calendar months before the application date and ends just before it. The payment of 200 on the application date and the payment of 300 in November, drawn hollow, count in no window.

   kd.figure_window_features()

Windows
-------

``windows`` lists the windows as strings: ``"3M"`` for three calendar months and ``"90D"`` for 90 days. The default is ``("3M", "6M", "12M")``. A pair such as ``("3M", "1M")`` is the period from three months to one month before the observation; its bounds must have the same unit, and the first must reach further back. In the column names the windows are written in lower case, such as ``3m``, ``90d`` and ``3m_1m``:

>>> window_features(
...     events, applications, **keys,
...     windows=["30D", ("3M", "1M")], aggregations={"amount": ["sum"]},
... )
   events_30d  amount_sum_30d  events_3m_1m  amount_sum_3m_1m
0           1            50.0             1             120.0
1           2            65.0             1              50.0
2           0             0.0             0               0.0

Thirty days before 31 March is 1 March, so the withdrawal of 29 February is outside the window ``30d`` but inside ``1m``. The window ``3m_1m`` ends where ``1m`` begins, so the two together are the window ``3m``.

.. note::

   A pair is one window only as an item of the list of windows. ``windows=[("3M", "1M")]`` is the window ``3m_1m``, while ``windows=("3M", "1M")`` is the two windows ``3m`` and ``1m``:

   >>> list(window_features(events, applications, **keys, windows=("3M", "1M")).columns)
   ['events_3m', 'events_1m']

By default every window is closed on the left, :math:`[t - k, t)`. With ``closed="right"`` the windows are :math:`(t - k, t]` instead, and events at the observation time count, for events that are known at their own time, such as balances at the end of a day. Whether such an event is known at the observation time is the caller's judgment, because kivi cannot tell. Here the payment of 31 March counts, and the withdrawal on the lower bound of the window, 29 February, does not:

>>> window_features(events, applications, **keys, windows=["1M"], closed="right")
   events_1m
0          2
1          2
2          0

Time columns must hold datetimes. Both may be tz-naive, or both in one time zone. With a time zone, the bounds are computed on the local calendar and the events compared as instants. A bound in the hour that clocks skip in spring moves to the end of that hour, and one in the hour that they repeat in autumn takes its first occurrence.

Aggregations
------------

``aggregations`` maps each column of the events to its aggregations. A missing value is left out of the aggregations of its column, while the ``events`` columns count every event of the window:

.. list-table::
   :header-rows: 1
   :widths: 20 55 25

   * - Aggregation
     - Value
     - Empty window
   * - ``count``
     - Number of values
     - 0
   * - ``sum``
     - Sum of the values
     - 0
   * - ``mean``
     - Mean of the values
     - NaN
   * - ``min``, ``max``
     - Smallest and largest value
     - NaN
   * - ``std``
     - Sample standard deviation, as ``pandas.Series.std`` gives it; NaN for a single value
     - NaN
   * - ``nunique``
     - Number of distinct values
     - 0
   * - ``first``, ``last``
     - Value of the earliest and of the latest event; events of one time are taken in the order of the rows
     - NaN
   * - ``days_since_last``
     - Days from the latest event to the observation time, with fractions of a day
     - NaN

``count``, ``nunique`` and ``days_since_last`` take columns of any dtype, such as the kind of each event; the other aggregations need numbers, and booleans count as 0 and 1, so the ``sum`` of a flag counts the events that have it. Counts are integers and all other columns floats. The table is transposed to fit the page:

>>> features = window_features(
...     events, applications, **keys,
...     windows=["3M"],
...     aggregations={
...         "amount": ["mean", "min", "std", "first", "last", "days_since_last"],
...         "kind": ["nunique"],
...     },
... )
>>> features.T.round(2)
                               0      1    2
events_3m                    3.00   3.00  0.0
amount_mean_3m              83.33  38.33  NaN
amount_min_3m               50.00  25.00  NaN
amount_std_3m               35.12  12.58  NaN
amount_first_3m            120.00  50.00  NaN
amount_last_3m              50.00  25.00  NaN
amount_days_since_last_3m   16.00  11.00  NaN
kind_nunique_3m              2.00   2.00  0.0

Customer ``b`` made two transactions on 20 March, a withdrawal of 40 and a payment of 25; ``last`` takes the payment, which comes later in the rows. Both are eleven days before the application. For a pair of windows ``days_since_last`` is still measured from the observation time. It is the time that has passed, so with a time zone it can differ by the hour that clocks change from the local calendar days that windows such as ``"90D"`` count.

Ratios and changes
------------------

``ratios`` divides two features, named by their columns. A ratio is NaN where its denominator is 0. It is named ``{numerator}_over_{window}`` when the two features differ only in their windows, and ``{numerator}_over_{denominator}`` otherwise. The share of the last month in the last three months shows whether activity is recent, and the ratio of the last month to the two months before it measures the change from one period to the next:

>>> window_features(
...     events, applications, **keys,
...     windows=["1M", "3M", ("3M", "1M")],
...     aggregations={"amount": ["sum"]},
...     ratios=[("amount_sum_1m", "amount_sum_3m"), ("amount_sum_1m", "amount_sum_3m_1m")],
... ).iloc[:, -2:].round(4)
   amount_sum_1m_over_3m  amount_sum_1m_over_3m_1m
0                 0.5200                    1.0833
1                 0.5652                    1.3000
2                    NaN                       NaN

The change from the same quarter a year earlier is the ratio of the window ``3M`` to the pair ``("15M", "12M")``, such as ``("amount_sum_3m", "amount_sum_15m_12m")``. A difference needs no option of its own: subtract the two columns with pandas.

Filtered events
---------------

kivi has no expressions to filter events. To compute features of some of the events only, such as the cash withdrawals, select them with pandas and give their features a ``prefix``:

>>> cash = events[events["kind"] == "cash"]
>>> window_features(
...     cash, applications, **keys,
...     windows=["3M"], aggregations={"amount": ["sum"]}, prefix="cash_",
... )
   cash_events_3m  cash_amount_sum_3m
0               1                80.0
1               2                90.0
2               0                 0.0

The prefix is put in front of every column, and ratios name the columns with it.

Observations over time
----------------------

An entity may appear in several rows of the observations, such as one row per month end to follow its features over time, or the same row twice. Each row gets the features of its own time:

>>> month_ends = pd.DataFrame({
...     "customer": ["a", "a", "a"],
...     "application_date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]),
... })
>>> window_features(
...     events, month_ends, **keys, windows=["3M"], aggregations={"amount": ["sum"]}
... )
   events_3m  amount_sum_3m
0          2          420.0
1          1          120.0
2          3          250.0

Three months before 29 February is 29 November, after the payment of 20 November, which therefore counts only at the end of January. One call computes the features of every date, which is how past samples are rebuilt. The events are sorted once and the bounds of every window are found by binary search, so counts, sums, means, first and last values and ``days_since_last`` take a time that grows with the number of rows of the two tables. ``min``, ``max``, ``std`` and ``nunique`` look at every event of every window instead, so their time grows with the number of observation dates of an entity times its events per window.

Reading the result
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Column
     - Meaning
     - Type and empty window
   * - ``{prefix}events_{window}``
     - Number of events in the window, with or without values
     - Integer, 0
   * - ``{prefix}{column}_count_{window}``, ``…_nunique_{window}``
     - Number of values, and of distinct values, of the column
     - Integer, 0
   * - ``{prefix}{column}_sum_{window}``
     - Sum of the values
     - Float, 0
   * - ``{prefix}{column}_{aggregation}_{window}``
     - The other aggregations
     - Float, NaN
   * - ``{numerator}_over_{window}``, ``{numerator}_over_{denominator}``
     - Ratios, last
     - Float, NaN for a denominator of 0

The columns come window by window, then column by column in the order of ``aggregations``, then aggregation by aggregation, and the ratios last. Column names and prefixes keep their case: only kivi's own parts of the names, the aggregations and the windows, are in lower case, so the features of ``Amount`` and ``amount`` stay apart. A NaN means that the window had no value to aggregate, which is information in its own right: profile the features with :func:`~kivi.univariate.profile_table` to see how often they are missing and how well they separate the classes, and bin them with :doc:`binning`, where missing values form their own group.

.. _repayment-behaviour:

Repayment behaviour
-------------------

A monthly panel of repayment statuses, such as the result of :func:`~kivi.vintage.delinquency_panel` or :func:`~kivi.datasets.make_loan_panel`, gives the behaviour of each borrower on its earlier loans. :func:`~kivi.features.panel_features` counts, for each window of months before the month of the observation time, the months with a status, the bad months, the longest runs of consecutive bad and good months and the worst status, and adds the months since the latest bad month. A status is bad from ``bad`` on, ``"30-59"`` by default, in the order of the categories of the status column, which must therefore be an ordered categorical.

Customer ``a`` fell 30 days behind in December and January, and customer ``b`` has two loans in January, one current and one late:

>>> from kivi.features import panel_features
>>> panel = pd.DataFrame({
...     "customer": ["a"] * 7 + ["b"] * 7,
...     "month": pd.PeriodIndex(
...         ["2023-09", "2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03"]
...         + ["2023-10", "2023-11", "2023-12", "2024-01", "2024-01", "2024-02", "2024-03"],
...         freq="M",
...     ),
...     "status": pd.Categorical(
...         ["current", "current", "1-29", "30-59", "30-59", "current", "current"]
...         + ["current", "current", "current", "current", "1-29", "60-89", "90+"],
...         categories=["current", "1-29", "30-59", "60-89", "90+"],
...         ordered=True,
...     ),
... })
>>> behaviour = panel_features(
...     panel, applications, id="customer", month="month", status="status",
...     observation_time="application_date", windows=["3M", "6M"],
... )
>>> behaviour.T
                         0      1    2
months_3m                3      3    0
bad_months_3m            2      1    0
longest_bad_run_3m       2      1    0
longest_good_run_3m      1      2    0
worst_status_3m      30-59  60-89  NaN
months_6m                6      5    0
bad_months_6m            2      1    0
longest_bad_run_6m       2      1    0
longest_good_run_6m      3      4    0
worst_status_6m      30-59  60-89  NaN
months_since_bad       2.0    1.0  NaN

The applications are dated 31 March, but the status at the end of March is not known then, so the windows end with February: the three months are December to February and the six months September to February. The 90 days behind of customer ``b`` in March therefore count nowhere. In January customer ``b`` has two loans, and the worse status, ``1-29``, counts for the month. Its six months hold only five months with a status, because its first loan began in October: only months with a status count, so the columns ``months_*`` show how much of a window the panel covers. A month without a status also ends a run, so a bad month, a month without a loan and a bad month are two runs of one month. A run that began before a window counts only with its months inside it, and ``months_since_bad`` looks at all the months before the observation, whatever the windows.

.. plot::
   :alt: The monthly statuses of customers a, b and c from September 2023 to March 2024 as tinted cells, with the bad months outlined, March hatched, the windows of three and six months that end at the start of March, and the result of panel_features for each customer
   :caption: The repayment example. The windows end where the month of the application date begins, so the statuses of March, hatched, count nowhere. In January customer ``b`` has two loans, and the worse status counts. The bad months, from ``30-59`` on, are outlined.

   kd.figure_panel_features()

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Column
     - Meaning
     - Type and empty window
   * - ``{prefix}months_{window}``
     - Number of months of the window with a status
     - Integer, 0
   * - ``{prefix}bad_months_{window}``
     - Number of bad months
     - Integer, 0
   * - ``{prefix}longest_bad_run_{window}``, ``…_longest_good_run_{window}``
     - Longest runs of consecutive bad and good months
     - Integer, 0
   * - ``{prefix}worst_status_{window}``
     - Worst status of the window
     - The dtype of the status, missing
   * - ``{prefix}months_since_bad``
     - Months from the latest bad month to the month of the observation time
     - Float, NaN if never bad

The windows count months only, as ``"6M"`` or ``("12M", "6M")``; windows of days raise an error, because the statuses are monthly. The month column may hold monthly periods or datetimes, which stand for their month. Months and observation times are each taken on their own calendar, so tz-aware months and tz-aware observation times must share one time zone: 1 April in Tokyo is still 31 March in New York, before the status at the end of March there exists.

On the simulated book of :func:`~kivi.datasets.make_loan_panel`, the features of every loan in January 2025 summarize its last year:

>>> from kivi.datasets import make_loan_panel
>>> loans = make_loan_panel(rng=0)
>>> book = pd.DataFrame({"loan_id": loans["loan_id"].unique(), "date": pd.Timestamp("2025-01-15")})
>>> year = panel_features(
...     loans, book, id="loan_id", month="month", status="status", observation_time="date",
...     windows=["12M"],
... )
>>> year["worst_status_12m"].value_counts(sort=False, dropna=False)
worst_status_12m
current    1976
1-29        205
30-59        48
60-89        20
90+          40
NaN        2711
Name: count, dtype: int64

The 2,711 loans whose first month on book is in 2025 have no month before January 2025, so their windows are empty and their worst status is missing. Of the others, 108 were 30 days or more behind at some month end of 2024.

Feature specifications
----------------------

A :class:`~kivi.features.FeatureSpec` holds the definitions of the features: the column of the entities, the column of the observation times, and one or more groups of window features, each with the keyword arguments of one call of :func:`~kivi.features.window_features`. A group that reads other events, such as the cash withdrawals, names its event table with ``table``. Ratios may divide features of different groups:

>>> spec = FeatureSpec(
...     id="customer",
...     observation_time="application_date",
...     events=[
...         {"time": "time", "windows": ["1M", "3M"], "aggregations": {"amount": ["sum"]}},
...         {
...             "table": "cash",
...             "time": "time",
...             "windows": ["3M"],
...             "aggregations": {"amount": ["sum"]},
...             "prefix": "cash_",
...         },
...     ],
...     ratios=[("cash_amount_sum_3m", "amount_sum_3m")],
... )
>>> list(spec.columns)
['events_1m', 'amount_sum_1m', 'events_3m', 'amount_sum_3m', 'cash_events_3m',
 'cash_amount_sum_3m', 'cash_amount_sum_3m_over_amount_sum_3m']

:meth:`~kivi.features.FeatureSpec.compute` takes the observations and the event tables by name. A single DataFrame serves when every group reads the same table:

>>> features = spec.compute(applications, events={"events": events, "cash": cash})
>>> features[["amount_sum_3m", "cash_amount_sum_3m", "cash_amount_sum_3m_over_amount_sum_3m"]]
   amount_sum_3m  cash_amount_sum_3m  cash_amount_sum_3m_over_amount_sum_3m
0          250.0                80.0                               0.320000
1          115.0                90.0                               0.782609
2            0.0                 0.0                                    NaN

Each group gives the columns that :func:`~kivi.features.window_features` gives for its definitions, so the specification reproduces the calls of development exactly. A specification holds definitions only: nothing in it is fitted to data, so it can be written by hand or read from a configuration file.

Groups of panel features go in ``panel``, each with the keyword arguments of one call of :func:`~kivi.features.panel_features` and the name of its panel in ``table``, ``"panel"`` by default, and :meth:`~kivi.features.FeatureSpec.compute` takes the panels as ``panel``. A specification may hold only panel features, and its ratios may divide window features by panel features. Two groups with different ``bad`` statuses and prefixes give the features of two definitions of default:

>>> both = FeatureSpec(
...     id="customer",
...     observation_time="application_date",
...     events=[{"time": "time", "windows": ["3M"]}],
...     panel=[
...         {"month": "month", "status": "status", "windows": ["6M"]},
...         {"month": "month", "status": "status", "windows": ["6M"], "bad": "90+",
...          "prefix": "severe_"},
...     ],
... )
>>> list(both.columns)
['events_3m', 'months_6m', 'bad_months_6m', 'longest_bad_run_6m', 'longest_good_run_6m',
 'worst_status_6m', 'months_since_bad', 'severe_months_6m', 'severe_bad_months_6m',
 'severe_longest_bad_run_6m', 'severe_longest_good_run_6m', 'severe_worst_status_6m',
 'severe_months_since_bad']
>>> both.compute(applications, events=events, panel=panel)[["bad_months_6m", "severe_bad_months_6m"]]
   bad_months_6m  severe_bad_months_6m
0              2                     0
1              1                     0
2              0                     0

The window features come first, then the panel features and the ratios last.

:meth:`~kivi.features.FeatureSpec.to_json` writes the specification as JSON text, and :meth:`~kivi.features.FeatureSpec.from_json` reads it back into an equal specification. Two specifications are equal when their JSON text is the same, which includes the version of kivi that wrote them:

>>> loaded = FeatureSpec.from_json(spec.to_json(indent=2))
>>> loaded == spec
True
>>> loaded.compute(applications, events={"events": events, "cash": cash}).equals(features)
True

The file has the keys below. After ``kivi_version`` they are the arguments of :class:`~kivi.features.FeatureSpec` as it takes them, with lists for tuples and every key of each group, so :meth:`~kivi.features.FeatureSpec.to_dict` gives what the constructor takes:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Content
   * - ``format``
     - Always ``"kivi.FeatureSpec"``.
   * - ``format_version``
     - Version of the format, 1. ``FeatureSpec.FORMAT_VERSION`` is the newest version that kivi reads. A file with a newer version raises an error that asks for a newer kivi, instead of being read in part.
   * - ``kivi_version``
     - Version of kivi that wrote the specification.
   * - ``id`` and ``observation_time``
     - The columns of the entities and of the observation times.
   * - ``events``
     - One dict per group, in order, with ``table``, ``time``, ``windows`` (strings and pairs of strings), ``aggregations`` (a dict from each column to the list of its aggregations, in the order of the columns of the result), ``closed`` and ``prefix``. A JSON object that repeats a key raises an error, because it would otherwise lose a value.
   * - ``ratios``
     - The pairs of column names of the ratios.
   * - ``panel``
     - One dict per group of panel features, in order, with ``table``, ``month``, ``status``, ``windows``, ``bad`` and ``prefix``. ``bad`` is written as the string or integer of its category.

Features of synthetic transactions
----------------------------------

:func:`~kivi.datasets.make_transactions` simulates the card purchases, cash withdrawals and instalment repayments of the twelve months before each application of :func:`~kivi.datasets.make_applications`, as :doc:`datasets` describes. The transactions depend on a risk factor of the applicants that no lender sees, which ``include_risk=True`` adds to the applications as the column ``risk``. The applications have no dates, so they are spread over 2024 in the order of the rows, and the observations are the applicants and their application dates, on the index of the applications:

>>> from kivi.datasets import make_applications, make_transactions
>>> sample = make_applications(n_applicants=10000, include_risk=True, rng=0)
>>> days = pd.to_timedelta(np.arange(10000) * 366 // 10000, unit="D")
>>> sample["application_date"] = pd.Timestamp("2024-01-01") + days
>>> transactions = make_transactions(sample, application_date="application_date", rng=0)
>>> observations = pd.DataFrame(
...     {"applicant_id": sample.index, "application_date": sample["application_date"]},
...     index=sample.index,
... )

A :class:`~kivi.features.FeatureSpec` with one group for each kind of transaction gives the cash withdrawals and the repayments of the last three months and of the three months before them, and the purchases of the last three months with the number of their merchant categories. Two ratios compare the last three months with the three months before them:

>>> two_periods = {
...     "time": "timestamp", "windows": ["3M", ("6M", "3M")], "aggregations": {"amount": ["sum"]},
... }
>>> transaction_spec = FeatureSpec(
...     id="applicant_id",
...     observation_time="application_date",
...     events=[
...         {"table": "cash", "prefix": "cash_", **two_periods},
...         {"table": "repayments", "prefix": "repayment_", **two_periods},
...         {
...             "table": "purchases",
...             "prefix": "purchase_",
...             "time": "timestamp",
...             "windows": ["3M"],
...             "aggregations": {"amount": ["sum"], "merchant_category": ["nunique"]},
...         },
...     ],
...     ratios=[
...         ("cash_events_3m", "cash_events_6m_3m"),
...         ("repayment_amount_sum_3m", "repayment_amount_sum_6m_3m"),
...     ],
... )
>>> kind = transactions["kind"]
>>> transaction_features = transaction_spec.compute(
...     observations,
...     events={
...         "cash": transactions[kind == "cash_withdrawal"],
...         "repayments": transactions[kind == "repayment"],
...         "purchases": transactions[kind == "purchase"],
...     },
... )
>>> transaction_features.shape
(10000, 13)

The features are candidates like the columns of the applications, so :func:`~kivi.univariate.profile_table` screens them against the outcome, as :doc:`univariate` describes:

>>> from kivi.univariate import profile_table
>>> screening = profile_table(transaction_features, sample["target"])
>>> screening[["missing_share", "iv", "gini"]].sort_values("iv", ascending=False).round(3)
                                       missing_share     iv   gini
variable
cash_events_3m                                 0.000  0.310  0.316
repayment_events_3m                            0.000  0.199 -0.257
cash_amount_sum_3m                             0.000  0.192  0.249
cash_events_3m_over_6m_3m                      0.241  0.185  0.185
repayment_amount_sum_3m_over_6m_3m             0.004  0.185 -0.218
cash_events_6m_3m                              0.000  0.180  0.247
purchase_amount_sum_3m                         0.000  0.104 -0.181
cash_amount_sum_6m_3m                          0.000  0.085  0.177
repayment_events_6m_3m                         0.000  0.082 -0.123
purchase_events_3m                             0.000  0.048 -0.121
repayment_amount_sum_3m                        0.000  0.031 -0.089
purchase_merchant_category_nunique_3m          0.000  0.030 -0.094
repayment_amount_sum_6m_3m                     0.000  0.005  0.037

The number of cash withdrawals in the last three months separates the classes best, and more withdrawals mean more defaults, as the positive ``gini`` shows. The repayments of the last three months come next with the opposite sign: applicants at risk miss more of them, so fewer repayments, and a smaller sum than in the three months before, mean more defaults. Both follow from how the transactions are simulated, with a stress that builds up before the application, and both are stronger in the last three months than in the three months before them. The purchases say little. The ratio of the cash withdrawals is missing for the 24% of applicants who withdrew no cash in the three months before the last three, because its denominator is 0. The profile counts them as a group of their own, which defaults less often than the others: a missing value is information too.

The signal is clear but modest, and only simulated data can tie it to a factor that no lender sees: the transactions show how window features work, not how well behaviour predicts default in a real portfolio. Bin the features that are kept as :doc:`binning` describes and select them with the rest of the candidates, as :doc:`selection` describes.

Missing values
--------------

Missing values of an aggregated column are left out of its aggregations, so ``count`` counts the values and ``events`` the rows:

>>> partial = events.assign(amount=events["amount"].where(events["kind"] == "card"))
>>> window_features(
...     partial, applications, **keys, windows=["3M"], aggregations={"amount": ["count", "mean"]}
... )
   events_3m  amount_count_3m  amount_mean_3m
0          3                2            85.0
1          3                1            25.0
2          0                0             NaN

A missing entity or time raises an error, because the event could not be placed in a window.

Limits
------

* The windows compare times only. Events that are recorded late, such as transactions booked days after they were made, are known later than their time says; shift their times, or the observation times, before computing the features.
* kivi has no expressions to filter events and no ranks of an entity within a population; filter with pandas and rank the features afterwards.
* The features are not weighted: each event counts once.
* An entity has one id column, shared by the events and the panels. To combine several keys, such as a customer and a product, join them into one column first.
* Panel features need monthly statuses. A panel with another frequency, such as weekly snapshots, has to be reduced to month ends first.

Errors and warnings
-------------------

Time columns must hold datetimes; convert strings first:

>>> text = applications.assign(application_date=["2024-03-31"] * 3).astype({"application_date": object})
>>> try:
...     window_features(events, text, **keys)
... except ValueError as error:
...     print(error)
observation_time column 'application_date' of observations must hold datetimes, got dtype object; convert it with pandas.to_datetime

A single window is still a list:

>>> try:
...     window_features(events, applications, **keys, windows="3M")
... except ValueError as error:
...     print(error)
windows must be a list of windows such as '3M', '90D' or ('6M', '3M'), got '3M'; pass ['3M'] for one window

A time column with a time zone and one without, or two time zones, raise an error that names both; convert them to one zone with ``Series.dt.tz_convert`` or remove the zones with ``Series.dt.tz_localize(None)``. For a panel, tz-aware months and tz-aware observation times in two time zones raise the same way. The aggregations other than ``count``, ``nunique`` and ``days_since_last`` raise an error for a column of text, and every aggregation of numbers raises one for infinite values.

A status column of strings has no order, so :func:`~kivi.features.panel_features` asks for an ordered categorical, and ``bad`` must be one of its categories:

>>> try:
...     panel_features(
...         panel.astype({"status": object}), applications, id="customer", month="month",
...         status="status", observation_time="application_date",
...     )
... except ValueError as error:
...     print(error)
status column 'status' of panel must be an ordered categorical, from the best status to the worst, got dtype object; convert it with pandas.Categorical(values, categories=[...], ordered=True)
>>> try:
...     panel_features(
...         panel, applications, id="customer", month="month", status="status",
...         observation_time="application_date", bad="30+",
...     )
... except ValueError as error:
...     print(error)
bad must be one of the categories of status column 'status' of panel, got '30+'; the categories are ['current', '1-29', '30-59', '60-89', '90+']
