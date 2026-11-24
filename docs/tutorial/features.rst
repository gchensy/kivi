Transaction features
====================

The applicants hold current accounts with the lender, and the transactions on them show how each applicant handles money before applying. This chapter dates the applications, derives features from the transactions as of each application date with a feature specification, checks that no later transaction enters them, profiles them next to the candidates of :doc:`data`, and saves their definitions for the scoring system; :doc:`/user_guide/features` explains each step.

.. dropdown:: Code from the previous chapters

   >>> import numpy as np
   >>> import pandas as pd
   >>> from kivi.datasets import make_applications
   >>> applications = make_applications(n_applicants=40000, rng=0)
   >>> months = pd.period_range("2024-01", periods=12, freq="M")
   >>> applications["month"] = months[np.arange(40000) * 12 // 40000]
   >>> eligible = applications[~applications["policy_decline"]]
   >>> accepts = eligible[eligible["approved"]]
   >>> development = accepts[accepts["month"] <= "2024-09"]
   >>> features = ["age", "income", "debt_to_income", "inquiries", "housing", "employment", "incumbent_pd"]

Date the applications
---------------------

The windows of the features end at the application date, but :doc:`data` gave each application only a month. The applications of each month are spread evenly over its days, in the order of the rows: ``position`` is the place of a row within its month, from 0 up to 1:

>>> position = np.arange(40000) * 12 % 40000 / 40000
>>> days = np.floor(position * applications["month"].dt.days_in_month)
>>> start = applications["month"].dt.start_time
>>> applications["application_date"] = start + pd.to_timedelta(days, unit="D")
>>> applications.loc[[0, 3333, 3334, 39999], ["month", "application_date"]]
         month application_date
0      2024-01       2024-01-01
3333   2024-01       2024-01-31
3334   2024-02       2024-02-01
39999  2024-12       2024-12-31

The dates keep the months of :doc:`data`, so the samples stay the same.

Simulate the transactions
-------------------------

For the project, the lender extracted the transactions of its applicants' accounts: card purchases, cash withdrawals and the repayments of their instalments, from a year before each application to three months after it. :func:`~kivi.datasets.make_transactions` simulates them. It draws the transactions from a risk factor of each applicant that no lender sees, so the simulation reads the same applications with ``include_risk=True``, which adds that factor as the column ``risk``; the steps of the project never use it. The first call gives the year before each application, and the second the three months before a date three months after it:

>>> from kivi.datasets import make_transactions
>>> simulated = make_applications(n_applicants=40000, include_risk=True, rng=0)
>>> simulated["application_date"] = applications["application_date"]
>>> later = simulated["application_date"] + pd.DateOffset(months=3)
>>> extract = pd.concat([
...     make_transactions(simulated, application_date="application_date", rng=0),
...     make_transactions(simulated, application_date=later, n_months=3, rng=1),
... ]).sort_values(["applicant_id", "timestamp"], ignore_index=True)
>>> extract.head()
   applicant_id           timestamp  amount       kind merchant_category
0             0 2023-01-05 12:52:45   60.01   purchase            online
1             0 2023-01-05 22:29:20  113.02   purchase            online
2             0 2023-01-10 08:59:08  664.73  repayment               NaN
3             0 2023-01-13 13:54:59   87.74   purchase            health
4             0 2023-01-26 22:03:14  352.16   purchase           grocery

``applicant_id`` is the index label of the application. The extract holds 2.2 million transactions, more than half of them purchases.

Define the features
-------------------

A :class:`~kivi.features.FeatureSpec` holds the definitions of the features: the column of the applicants, the column of the application dates, and one group of window features for each kind of transaction, each with the arguments of one call of :func:`~kivi.features.window_features`. The groups count and add up the cash withdrawals of the last three and twelve months, the repayments of the last three months and of the three months before them, and the purchases of the last three months. Two ratios compare the cash with the purchases, and the repayments of the last three months with those of the three months before:

>>> from kivi.features import FeatureSpec
>>> feature_spec = FeatureSpec(
...     id="applicant_id",
...     observation_time="application_date",
...     events=[
...         {"table": "cash", "prefix": "cash_", "time": "timestamp", "windows": ["3M", "12M"],
...          "aggregations": {"amount": ["sum"]}},
...         {"table": "repayments", "prefix": "repayment_", "time": "timestamp",
...          "windows": ["3M", ("6M", "3M")], "aggregations": {"amount": ["sum"]}},
...         {"table": "purchases", "prefix": "purchase_", "time": "timestamp", "windows": ["3M"],
...          "aggregations": {"amount": ["sum"]}},
...     ],
...     ratios=[
...         ("cash_amount_sum_3m", "purchase_amount_sum_3m"),
...         ("repayment_amount_sum_3m", "repayment_amount_sum_6m_3m"),
...     ],
... )
>>> list(feature_spec.columns)
['cash_events_3m', 'cash_amount_sum_3m', 'cash_events_12m', 'cash_amount_sum_12m',
 'repayment_events_3m', 'repayment_amount_sum_3m', 'repayment_events_6m_3m',
 'repayment_amount_sum_6m_3m', 'purchase_events_3m', 'purchase_amount_sum_3m',
 'cash_amount_sum_3m_over_purchase_amount_sum_3m', 'repayment_amount_sum_3m_over_6m_3m']

The observations are the applicants and their application dates, on the index of the applications. :meth:`~kivi.features.FeatureSpec.compute` takes the transactions of each group by the name of its table, so a small function splits the extract by kind:

>>> observations = pd.DataFrame(
...     {"applicant_id": applications.index, "application_date": applications["application_date"]},
...     index=applications.index,
... )
>>> def by_kind(transactions):
...     kind = transactions["kind"]
...     return {
...         "cash": transactions[kind == "cash_withdrawal"],
...         "repayments": transactions[kind == "repayment"],
...         "purchases": transactions[kind == "purchase"],
...     }
>>> account_features = feature_spec.compute(observations, events=by_kind(extract))
>>> account_features.head(3).T.round(2)
                                                      0        1        2
cash_events_3m                                     4.00     1.00     5.00
cash_amount_sum_3m                               596.36    65.84   683.87
cash_events_12m                                   13.00     5.00    16.00
cash_amount_sum_12m                             1754.90   335.42  1891.98
repayment_events_3m                                2.00     3.00     0.00
repayment_amount_sum_3m                         1214.41  5097.31     0.00
repayment_events_6m_3m                             2.00     3.00     2.00
repayment_amount_sum_6m_3m                      1211.13  5194.59  1917.27
purchase_events_3m                                13.00    12.00     1.00
purchase_amount_sum_3m                          2781.96  2082.14   112.82
cash_amount_sum_3m_over_purchase_amount_sum_3m     0.21     0.03     6.06
repayment_amount_sum_3m_over_6m_3m                 1.00     0.98     0.00

The third applicant made two repayments in the three months before the last three and none since, while withdrawing cash five times and spending little by card. Its ratio of repayments is 0, and its cash is six times its purchases.

Check that the features are point in time
-----------------------------------------

A fifth of the extract lies on or after the application date of its applicant. The scoring system could not have seen those transactions when it decided the application:

>>> date = extract["applicant_id"].map(applications["application_date"])
>>> print(f"{(extract['timestamp'] >= date).mean():.3f}")
0.212

The windows end just before the application date, so those transactions count in no feature. To check it, remove them by hand and compute the features again. They are the same, the counts exactly and the sums up to the rounding of their last digits, because a table with fewer rows adds the same amounts in another order:

>>> known = extract[extract["timestamp"] < date]
>>> np.allclose(
...     feature_spec.compute(observations, events=by_kind(known)), account_features, equal_nan=True
... )
True

A count of each applicant's transactions grouped by hand, without the dates, would include them, and a model developed on it would rank applicants on information that it never has in production.

Profile the features
--------------------

The features are candidates like the fields of the application form. :func:`~kivi.univariate.profile_table` profiles them next to the seven candidates of :doc:`data`, on the development sample:

>>> from kivi.univariate import profile_table
>>> candidates = pd.concat([development[features], account_features.loc[development.index]], axis=1)
>>> profile = profile_table(candidates, development["target"])
>>> profile[["missing_share", "iv", "gini"]].sort_values("iv", ascending=False).round(3)
                                                missing_share     iv   gini
variable
incumbent_pd                                            0.000  0.272  0.295
housing                                                 0.000  0.110    NaN
cash_events_12m                                         0.000  0.099  0.183
cash_events_3m                                          0.000  0.089  0.176
cash_amount_sum_3m_over_purchase_amount_sum_3m          0.010  0.073  0.162
repayment_amount_sum_3m_over_6m_3m                      0.001  0.063 -0.138
repayment_events_3m                                     0.000  0.058 -0.123
cash_amount_sum_3m                                      0.000  0.050  0.139
cash_amount_sum_12m                                     0.000  0.047  0.124
debt_to_income                                          0.000  0.031  0.101
income                                                  0.000  0.023 -0.085
age                                                     0.000  0.023 -0.088
inquiries                                               0.000  0.021  0.101
employment                                              0.000  0.017    NaN
purchase_amount_sum_3m                                  0.000  0.015 -0.054
purchase_events_3m                                      0.000  0.005 -0.027
repayment_amount_sum_6m_3m                              0.000  0.003  0.017
repayment_amount_sum_3m                                 0.000  0.003 -0.034
repayment_events_6m_3m                                  0.000  0.000 -0.050
>>> list(profile.index[profile["tied_bins"]])
['inquiries', 'cash_events_3m', 'cash_amount_sum_3m', 'repayment_events_3m',
 'repayment_events_6m_3m', 'cash_amount_sum_3m_over_purchase_amount_sum_3m']

.. plot::
   :alt: Information value of the seven application candidates and the twelve transaction features on the development sample, sorted, with the 0.02, 0.1 and 0.3 reading bands
   :caption: IV of each candidate with a quick binning on the development sample. The cash withdrawals and the change of the repayments are stronger than most fields of the application form.

   kt.figure_features_iv()

The profile ranks the transaction features among the application data:

* The cash withdrawals carry the most information: their numbers in the last twelve and three months have an IV of 0.099 and 0.089, more than any field of the application form except ``housing``. More withdrawals mean more defaults, as the positive ``gini`` shows.
* The repayments matter by their change, not by their level. The sums of the last three months and of the three months before are weak on their own, because they follow the size of the instalments, but their ratio has an IV of 0.063 with a negative ``gini``: applicants who repaid less recently than before default more often. So do applicants with fewer repayments in the last three months.
* The purchases say little.
* The ratio of cash to purchases is missing for the 1% of applicants without a purchase in the last three months, because its denominator is 0. The profile counts them as a group of their own.
* Five transaction features are marked in ``tied_bins``, like ``inquiries`` in :doc:`data`: the numbers of withdrawals and repayments in three months take few values, and a fifth of the applicants withdrew no cash in the last three months, so their quantile bins collapse. A binning would give them explicit edges.

.. admonition:: Simulation check
   :class: kivi-check

   :func:`~kivi.datasets.make_transactions` draws the transactions from the hidden risk factor, of which the vendor's scorecard sees only a part. The withdrawals of the last year follow that factor more closely than the vendor's PD does, and share only part of their variation with the vendor's PD, which is why they carry information that it lacks:

   >>> simulated[["risk", "incumbent_pd"]].join(account_features["cash_events_12m"]).corr().round(2)
                    risk  incumbent_pd  cash_events_12m
   risk             1.00          0.44             0.67
   incumbent_pd     0.44          1.00             0.53
   cash_events_12m  0.67          0.53             1.00

Save the specification
----------------------

A feature specification holds definitions only: nothing in it is fitted to data. :meth:`~kivi.features.FeatureSpec.to_json` writes it as JSON for the scoring system, and :meth:`~kivi.features.FeatureSpec.from_json` reads it back into an equal specification that computes the same features:

>>> text = feature_spec.to_json(indent=2)
>>> restored = FeatureSpec.from_json(text)
>>> restored == feature_spec
True
>>> restored.compute(observations, events=by_kind(extract)).equals(account_features)
True

Each group is written with every argument of its call, the defaults included, so the file states the windows, the aggregations and the side on which each window is closed:

>>> import json
>>> print(json.dumps(restored.to_dict()["events"][0]))
{"table": "cash", "time": "timestamp", "windows": ["3M", "12M"],
 "aggregations": {"amount": ["sum"]}, "closed": "left", "prefix": "cash_"}

Keep them for the next scorecard
--------------------------------

Several transaction features are stronger than most fields of the application form, but the project builds its first scorecard from the application data and the vendor's PD, as it set out to do. A scorecard can rely on a characteristic only once the scoring system computes it as development did, so the lender first runs the specification in production, as :doc:`monitoring` describes for the binning, and tests the features in a challenger scorecard of its next development. Whether they add to the vendor's PD, which they partly repeat, is a question for that model, not for the profile. The chapters that follow use the seven candidates of :doc:`data`.
