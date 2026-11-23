Example data
============

:mod:`kivi.datasets` provides four kinds of example data: real bank marketing data with one row per customer and a binary target, a synthetic panel of loans with one row per loan and month, synthetic loan applications with one row per applicant and an approval decision, and synthetic account transactions of the applicants with one row per transaction. :func:`~kivi.datasets.load_bank`, :func:`~kivi.datasets.make_loan_panel`, :func:`~kivi.datasets.make_applications` and :func:`~kivi.datasets.make_transactions` return them.

Bank marketing data
-------------------

:func:`~kivi.datasets.load_bank` loads 4,521 customers from the Bank Marketing data set. It has sixteen features and an integer ``target``: 1 means that the customer subscribed to a term deposit.

>>> from kivi.datasets import load_bank
>>> data = load_bank()
>>> list(data.columns)  # doctest: +NORMALIZE_WHITESPACE
['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous',
 'poutcome', 'target']
>>> data["target"].value_counts().to_dict()
{0: 4000, 1: 521}

The examples in this documentation split the rows by position only to demonstrate two samples; the split is not chronological.

Synthetic loan panel
--------------------

:func:`~kivi.datasets.make_loan_panel` simulates the monthly history of a portfolio of instalment loans, for the vintage and roll-rate analysis in :doc:`vintage`. The data is generated, not taken from a real portfolio, and the same ``rng`` seed always gives the same panel:

>>> from kivi.datasets import make_loan_panel
>>> panel = make_loan_panel(n_loans=5000, n_months=24, rng=0)
>>> panel.shape
(59295, 8)
>>> panel.groupby("cohort")["loan_id"].nunique().agg(["min", "max"]).to_dict()
{'min': 188, 'max': 234}

The loans are spread at random over ``n_months`` origination months from January 2024, and each loan has one row for every month end it is on book, up to the month after the last origination month. The columns are:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Meaning
   * - ``loan_id``
     - Identifier of the loan, increasing with the origination month.
   * - ``cohort``
     - Origination month, a monthly ``pandas.Period``.
   * - ``month``
     - Month of the observation, ``cohort + month_on_book``.
   * - ``month_on_book``
     - Months since origination, from 1.
   * - ``term``
     - Number of monthly instalments: 12, 24 or 36.
   * - ``amount``
     - Amount lent.
   * - ``balance``
     - Outstanding principal at the end of the month.
   * - ``status``
     - Days past due at the end of the month, an ordered categorical: ``current``, ``1-29``, ``30-59``, ``60-89`` and ``90+``.

How the panel is simulated
^^^^^^^^^^^^^^^^^^^^^^^^^^

Each loan is repaid in equal monthly instalments at 1% interest a month. Its status counts the months since its oldest unpaid instalment fell due: one is ``1-29``, two ``30-59``, three ``60-89`` and four or more ``90+``. Within the term this is the number of instalments past due. Every month:

* a current loan misses its instalment with a probability that peaks at its sixth month on book and then declines, scaled by a risk factor of the loan and one of its cohort;
* a delinquent loan pays all its arrears and becomes current, pays one instalment and keeps its status, or pays nothing and rolls to the next bucket, also after its term, and the later the bucket, the less likely it cures and the more likely it rolls;
* a ``90+`` loan pays nothing more, so its status and balance stay the same.

A loan leaves the panel in the month it pays its last instalment, which can come after its term if it was behind. Loans are not charged off or prepaid.

Synthetic loan applications
---------------------------

:func:`~kivi.datasets.make_applications` simulates loan applications that an incumbent scorecard and two policy rules decided, for the reject inference in :doc:`reject_inference`. Every applicant has a true default flag, also the rejected ones, so that the inferred outcomes can be compared with the real ones. The same ``rng`` seed always gives the same applications:

>>> from kivi.datasets import make_applications
>>> applications = make_applications(n_applicants=10000, rng=0)
>>> applications.shape
(10000, 10)
>>> print(applications["approved"].mean())
0.677

The columns are:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Meaning
   * - ``age``
     - Age in years, from 18 to 75.
   * - ``income``
     - Monthly income, rounded to tens.
   * - ``debt_to_income``
     - Monthly debt payments over income, rounded to hundredths.
   * - ``inquiries``
     - Number of credit inquiries in the last six months.
   * - ``housing``
     - A categorical: ``own``, ``mortgage`` or ``rent``.
   * - ``employment``
     - A categorical: ``permanent``, ``temporary`` or ``self_employed``.
   * - ``incumbent_pd``
     - PD of the scorecard that decided the application, rounded to four decimals.
   * - ``policy_decline``
     - True if a policy rule declined the application, whatever its PD: a debt-to-income ratio above 0.6 or an age below 21.
   * - ``approved``
     - True if ``incumbent_pd`` is below 0.15 and no policy rule declined the application.
   * - ``target``
     - 1 if the applicant defaults, else 0.

How the applications are simulated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default probability rises with the debt-to-income ratio and the number of inquiries, falls with age and income, and is higher for renters and for temporary and self-employed applicants. A hidden risk factor that no column holds also raises it, and the number of inquiries rises with that factor.

The incumbent scorecard weighs the features differently and ignores ``housing``. Like an underwriter with a credit bureau report, it also sees part of the hidden risk factor. Rejected applicants are therefore riskier than approved applicants with the same features, so a model fitted on the approved applicants alone underestimates how often the rejected ones default, as :doc:`reject_inference` shows. Overall, the rejected applicants default three times as often:

>>> eligible = applications[~applications["policy_decline"]]
>>> eligible.groupby("approved")["target"].mean().round(3)
approved
False    0.301
True     0.097
Name: target, dtype: float64

.. warning::

   In practice, the outcome is known only for approved applicants. ``target`` is given for the rejected applicants only to evaluate reject inference; fitting a model on it would make reject inference pointless.

Synthetic transactions
----------------------

:func:`~kivi.datasets.make_transactions` simulates the transactions on an account that came before each application of :func:`~kivi.datasets.make_applications`: card purchases, cash withdrawals and instalment repayments over the preceding twelve months. They are the events from which features of time windows, such as the number of cash withdrawals in the last three months, are computed. The data is generated, not taken from real accounts, and the same ``rng`` seed always gives the same transactions. The function returns two tables, the transactions and the observations that their windows end at:

>>> import pandas as pd
>>> from kivi.datasets import make_transactions
>>> transactions, observations = make_transactions(applications, rng=0)
>>> transactions.shape
(431711, 5)
>>> transactions.head(3)
   applicant_id           timestamp  amount             kind merchant_category
0             0 2023-01-03 03:22:53   82.58         purchase           grocery
1             0 2023-01-08 05:23:51  130.78         purchase            dining
2             0 2023-01-16 18:23:05 1047.86        repayment               NaN
>>> observations.head(3)
   applicant_id application_date
0             0       2024-01-01
1             1       2024-01-01
2             2       2024-01-01

The applications have neither an applicant id nor a date. The applicant id is therefore the index label of the application, and the observations have the same index, so that features computed for them join back to the applications. Without ``application_date``, the applications are spread evenly over the twelve months of 2024 in the order of their rows, the same months that the tutorial assigns to them: the first twelfth of the rows in January, the next in February and so on, and within a month evenly over its days. Pass ``application_date`` for other dates, as the name of a column of the applications, as one date, or as a Series with the index of the applications.

The columns of the transactions are:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Meaning
   * - ``applicant_id``
     - Index label of the applicant in the applications.
   * - ``timestamp``
     - Time of the transaction, to the second.
   * - ``amount``
     - Amount, positive and rounded to cents.
   * - ``kind``
     - A categorical: ``purchase``, ``cash_withdrawal`` or ``repayment``.
   * - ``merchant_category``
     - A categorical with eight categories, such as ``grocery`` and ``travel``, for a purchase, and missing for the other kinds.

The columns of the observations are ``applicant_id``, the same index label, and ``application_date``, the date of the application at midnight. A time of day in ``application_date`` is ignored, and a time zone is dropped, as in :func:`~kivi.vintage.delinquency_panel`.

The transactions of an applicant lie in the ``months`` calendar months before the application date, twelve by default: from the date ``months`` months earlier, counted as ``pandas.DateOffset`` counts them, up to but not including the application date. A window such as the last three months before the application date therefore cuts the transactions exactly, and no transaction on or after the date exists to leak into it.

How the transactions are simulated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The transactions of each month are drawn independently. In a year an applicant has on average about 26 purchases, 6 cash withdrawals and 12 repayments, 43 transactions in all. A purchase costs about 4.5% of the monthly income, a cash withdrawal about 3%, and a repayment is the instalment that ``income * debt_to_income`` implies, which varies by about 20% between applicants. Applicants with a higher ``income`` make a few more purchases and withdrawals, and purchases are more frequent in the months that start in November and December. An applicant whom the incumbent scorecard rates as riskier, by ``incumbent_pd``, withdraws more cash and misses more repayments.

The outcome drives a stress that builds up before the application. For a defaulter it starts between three and nine months before the application and is strongest in the last month, and 25% of the other applicants have an episode of the same shape that is a little milder. Under stress, the cash withdrawals are more frequent and larger, the purchases a little rarer, and a repayment is more likely to be missed, or paid up to a tenth short. Applicants differ in their baseline and in the size of their stress, which is capped at three times its typical size, so no feature predicts the outcome well.

The number of cash withdrawals in the last three months shows the signal. It is counted here with pandas, on the transactions and observations above; the default rate rises from 9% of the applicants without a withdrawal to 41% of those with five or more:

>>> merged = transactions.merge(observations, on="applicant_id")
>>> last_3_months = merged["timestamp"] >= merged["application_date"] - pd.DateOffset(months=3)
>>> cash = merged[last_3_months & (merged["kind"] == "cash_withdrawal")]
>>> cash_3m = cash.groupby("applicant_id").size().reindex(applications.index, fill_value=0)
>>> cash_3m.name = "cash_withdrawals_3m"
>>> rate = applications["target"].groupby(cash_3m.clip(upper=5)).agg(["size", "mean"])
>>> rate.round(3)
                     size   mean
cash_withdrawals_3m
0                    2750  0.090
1                    2899  0.140
2                    1981  0.148
3                    1114  0.212
4                     586  0.265
5                     670  0.407

The IV of this feature, which :func:`~kivi.univariate.profile_table` computes as in :doc:`univariate`, is a clear but modest 0.25:

>>> from kivi.univariate import profile_table
>>> profile_table(cash_3m.to_frame(), applications["target"])[["iv"]].round(3)
                        iv
variable
cash_withdrawals_3m  0.255

Other window features are weaker or stronger. With the default five bins, the IV is 0.04 for the number of purchases in the last three months, 0.25 for the repayments of the last three months relative to the three months before them, and 0.30 for the cash withdrawn in the last three months relative to income.

.. warning::

   The transactions are drawn from the outcome, which a lender learns only after the transactions. Use them to learn how window features work, not to estimate how well behaviour predicts default.

Dataset and attribution
-----------------------

Bank Marketing by S. Moro, P. Rita and P. Cortez (2014) is distributed under CC BY 4.0. The bundled CSV was converted from the legacy workbook. The loader encodes subscription as 1 and non-subscription as 0. See the `dataset record <https://doi.org/10.24432/C5K306>`_ and `license <https://creativecommons.org/licenses/by/4.0/>`_. The loan panel, the applications and the transactions are synthetic and have no source to attribute.
