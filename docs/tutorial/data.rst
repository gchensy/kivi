Data and samples
================

The project starts from the applications of 2024 and their decisions. This chapter loads them, sets aside the applications that a policy rule declined, splits the approved ones into a development sample and a later validation sample, and takes a first look at every candidate variable.

Load the applications
---------------------

:func:`~kivi.datasets.make_applications` returns one row per application. The simulated applications have no date, so the example spreads them evenly over the twelve months of 2024 in the order of the rows:

>>> import numpy as np
>>> import pandas as pd
>>> from kivi.datasets import make_applications
>>> applications = make_applications(n_applicants=40000, rng=0)
>>> months = pd.period_range("2024-01", periods=12, freq="M")
>>> applications["month"] = months[np.arange(40000) * 12 // 40000]
>>> applications.head(3).T
                        0          1          2
age                    41         39         47
income             3500.0     2630.0     3100.0
debt_to_income       0.21       0.61       0.28
inquiries               0          1          1
housing              rent       rent        own
employment      temporary  permanent  permanent
incumbent_pd       0.1095     0.2349      0.124
policy_decline      False       True      False
approved             True      False       True
target                  0          1          0
month             2024-01    2024-01    2024-01

``incumbent_pd`` is the PD of the vendor's scorecard, ``policy_decline`` flags the applications that a policy rule declined, and ``approved`` the applications that the lender approved. ``target`` is 1 if the borrower defaulted within the outcome window.

Split the samples
-----------------

A policy decline does not depend on the score, so a scorecard is not developed on those applications: they are declined whatever it says. The other applications are *eligible*. The vendor's scorecard approved the eligible applications with a PD below 15%, the *accepts*, and declined the others, the *rejects*:

>>> pd.crosstab(applications["policy_decline"], applications["approved"])
approved        False  True
policy_decline
False           10001  27012
True             2987      0
>>> eligible = applications[~applications["policy_decline"]]
>>> accepts = eligible[eligible["approved"]]

The model is developed on the accepts of January to September, and the accepts of October to December are kept apart to validate it out of time, as a model in production is applied to applications that come after its development sample. The rejects of the development months enter the project in :doc:`reject_inference`:

>>> development = accepts[accepts["month"] <= "2024-09"]
>>> validation = accepts[accepts["month"] >= "2024-10"]
>>> rejects = eligible[~eligible["approved"] & (eligible["month"] <= "2024-09")]
>>> pd.DataFrame(
...     {
...         "applications": [len(development), len(validation), len(rejects)],
...         "event_rate": [development["target"].mean(), validation["target"].mean(), np.nan],
...     },
...     index=["development", "validation", "rejects"],
... ).round(3)
             applications  event_rate
development         20227       0.101
validation           6785       0.106
rejects              7545         NaN

About one approved borrower in ten defaulted, in both samples. The outcome of the rejects is unknown: they never received a loan.

.. admonition:: Simulation check
   :class: kivi-check

   The simulated rejects do have an outcome. They default three times as often as the accepts, partly because the vendor's scorecard saw bureau data that the application form does not hold:

   >>> print(f"{rejects['target'].mean():.3f}")
   0.295

Profile the candidates
----------------------

The candidate characteristics are the six fields of the application form and the vendor's PD. :func:`~kivi.univariate.profile_table` gives each of them one row on the development sample, with its distribution and the IV of a quick binning; see :doc:`/user_guide/univariate`:

>>> from kivi.univariate import profile_table
>>> features = ["age", "income", "debt_to_income", "inquiries", "housing", "employment", "incumbent_pd"]
>>> profile = profile_table(development[features], development["target"])
>>> profile[["kind", "distinct", "missing", "top_share", "iv", "tied_bins"]].round(3)
                       kind  distinct  missing  top_share     iv  tied_bins
variable
age                 numeric        55        0      0.039  0.023      False
income              numeric      1049        0      0.004  0.023      False
debt_to_income      numeric        61        0      0.031  0.031      False
inquiries           numeric         7        0      0.471  0.021       True
housing         categorical         3        0      0.399  0.110      False
employment      categorical         3        0      0.719  0.017      False
incumbent_pd        numeric      1378        0      0.002  0.272      False

.. plot::
   :alt: Information value of the seven candidate characteristics on the development sample, sorted, with the 0.02, 0.1 and 0.3 reading bands
   :caption: IV of each candidate with a quick binning on the development sample. The vendor's PD stands out; the fields of the application form are weak on their own.

   kt.figure_profile_iv()

The profile shapes the next chapters:

* No variable has missing values or special codes, so neither needs a group of its own.
* ``incumbent_pd`` is by far the strongest candidate, followed by ``housing``. The other fields of the application form have an IV between 0.017 and 0.031: weak on their own, but they may add to the vendor's PD.
* The approved applicants are a selected sample: the vendor's scorecard turned away the riskiest applications, so every variable separates the outcomes less among the accepts than among all applicants.
* ``inquiries`` is marked in ``tied_bins``: almost half of the applicants had no inquiry, so its quantile bins collapse. :doc:`binning` gives it explicit edges.
