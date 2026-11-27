Example data
============

:mod:`kivi.datasets` provides five kinds of example data: real bank marketing data with one row per customer and a binary target, a synthetic panel of loans with one row per loan and month, synthetic loan applications with one row per applicant and an approval decision, synthetic account transactions of the applicants with one row per transaction, and synthetic defaulted loans with one row per loan and the cash flows of their workouts with one row per cash flow. :func:`~kivi.datasets.load_bank`, :func:`~kivi.datasets.make_loan_panel`, :func:`~kivi.datasets.make_applications`, :func:`~kivi.datasets.make_transactions`, :func:`~kivi.datasets.make_defaults` and :func:`~kivi.datasets.make_recoveries` return them.

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

A loan leaves the panel in the month it pays its last instalment, which can come after its term if it was behind. Loans are not charged off, and they repay early only with ``prepayment``.

Early repayments
^^^^^^^^^^^^^^^^

``prepayment`` is the monthly probability that a current loan repays its whole balance before its last instalment is due, for the survival tables in :doc:`lifetime`. The default, 0, gives the panel above, the same as in earlier releases. Every month, a loan that was current at the end of the month before, as every loan is at origination, and that has more instalments left than the one now due repays early with that probability, whatever its risk. Its balance is 0 in that month, and it has no rows after it, so the columns stay the same: a balance of 0 before the term marks an early repayment, and a balance of 0 at or after the term a repayment on schedule.

>>> repaying = make_loan_panel(n_loans=5000, n_months=24, prepayment=0.02, rng=0)
>>> last = repaying.groupby("loan_id").tail(1)
>>> repaid = last["balance"] == 0
>>> early = repaid & (last["month_on_book"] < last["term"])
>>> int(early.sum()), int((repaid & ~early).sum()), int((last["status"] == "90+").sum())
(962, 454, 182)

A 2% chance a month repays 962 of the 5,000 loans early. The early repayments use random numbers of their own, drawn after all the others, so with the same ``rng`` every loan follows the same path as without ``prepayment`` until it repays early. Fewer loans therefore stay on book long enough to reach their term or 90+: 454 loans repay on schedule and 182 reach 90+, against 584 and 205 without early repayments.

An economic cycle
^^^^^^^^^^^^^^^^^

``correlation`` and ``factors`` add an economic cycle, for the transition matrices of :doc:`lifetime`. Every month a systematic factor moves the payments of all loans together, in the one-factor model of :func:`~kivi.lifetime.conditional_matrix`: a loan that is not yet 90 days past due misses its instalment, or fails to pay its arrears, more often when the factor of the month is negative and less often when it is positive, and averaged over the factor it keeps its probabilities. ``correlation`` is the asset correlation of the factor; the default, 0, gives the panel above, the same as in earlier releases. ``factors`` holds the factor of each month from February 2024 on, so that an estimate can be compared with it. Without ``factors``, they are drawn from ``rng`` after all the other random numbers. A cycle stuck in a bad economy or in a good one changes how many loans reach 90+:

>>> def reached_90(panel):
...     return int((panel.groupby("loan_id").tail(1)["status"] == "90+").sum())
>>> bad = make_loan_panel(n_loans=5000, n_months=24, correlation=0.05, factors=[-1.0] * 24, rng=0)
>>> good = make_loan_panel(n_loans=5000, n_months=24, correlation=0.05, factors=[1.0] * 24, rng=0)
>>> reached_90(bad), reached_90(good)
(450, 68)

With a factor of −1 in every month, 450 loans reach 90+, and with a factor of 1, 68, against 205 without the cycle. The factor moves the misses of current loans and the cures and rolls of delinquent ones, but not the early repayments of ``prepayment``.

Synthetic loan applications
---------------------------

:func:`~kivi.datasets.make_applications` simulates loan applications that an incumbent scorecard and two policy rules decided, for the reject inference in :doc:`reject_inference` and the policy rules in :doc:`rules`. Every applicant has a true default flag, also the rejected ones, so that the inferred outcomes can be compared with the real ones. The same ``rng`` seed always gives the same applications:

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
   * - ``risk``
     - Only with ``include_risk=True``: the hidden risk factor, a standard normal draw that is not rounded.

How the applications are simulated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default probability rises with the debt-to-income ratio and the number of inquiries, falls with age and income, and is higher for renters and for temporary and self-employed applicants. A hidden risk factor that no column holds, unless you ask for it with ``include_risk=True``, also raises it, and the number of inquiries rises with that factor.

The incumbent scorecard weighs the features differently and ignores ``housing``. Like an underwriter with a credit bureau report, it also sees part of the hidden risk factor. Rejected applicants are therefore riskier than approved applicants with the same features, so a model fitted on the approved applicants alone underestimates how often the rejected ones default, as :doc:`reject_inference` shows. Overall, the rejected applicants default three times as often:

>>> eligible = applications[~applications["policy_decline"]]
>>> eligible.groupby("approved")["target"].mean().round(3)
approved
False    0.301
True     0.097
Name: target, dtype: float64

.. warning::

   In practice, the outcome is known only for approved applicants. ``target`` is given for the rejected applicants only to evaluate reject inference; fitting a model on it would make reject inference pointless. The same holds for ``risk``, the information that no data set holds: it is there to simulate data that depend on it, such as the transactions below, and a model must not be fitted on it.

Synthetic transactions
----------------------

:func:`~kivi.datasets.make_transactions` simulates the transactions on an account that came before each application of :func:`~kivi.datasets.make_applications`: card purchases, cash withdrawals and instalment repayments over the preceding twelve months. They are the events from which features of time windows, such as the number of cash withdrawals in the last three months, are computed. The data is generated, not taken from real accounts, and the same ``rng`` seed always gives the same transactions.

The transactions depend on the hidden risk factor of the applicants and on their income, and not on the outcome. ``make_applications`` returns the factor as the column ``risk`` when it is called with ``include_risk=True``, and the other columns and the draws behind them stay the same. The applications have no date either, so the example gives each of them one, spread evenly over 2024 in the order of the rows, and passes the name of that column as ``application_date``:

>>> import numpy as np
>>> import pandas as pd
>>> from kivi.datasets import make_transactions
>>> applications = make_applications(n_applicants=10000, include_risk=True, rng=0)
>>> days = pd.to_timedelta(np.arange(10000) * 366 // 10000, unit="D")
>>> applications["application_date"] = pd.Timestamp("2024-01-01") + days
>>> transactions = make_transactions(applications, application_date="application_date", rng=0)
>>> transactions.shape
(443130, 5)
>>> transactions.head(3)
   applicant_id           timestamp   amount       kind merchant_category
0             0 2023-01-02 14:45:14   201.00   purchase            dining
1             0 2023-01-06 12:12:03  1032.63  repayment               NaN
2             0 2023-02-08 04:22:55   421.14   purchase            online

``application_date`` is required. It is the name of a column of the applications, as here, one date for every application, or a Series with the index of the applications, and the dates are read as in :func:`~kivi.vintage.delinquency_panel`: a time of day is ignored, a time zone is dropped, and numbers and periods raise an error, so a period column needs ``.dt.to_timestamp()`` first. The applicant id of a transaction is the index label of the application. The observations that the windows end at, one row per application with its id and date, follow from the applications and their dates. They keep the index of the applications, so that the features computed for them join back to the applications:

>>> observations = pd.DataFrame(
...     {"applicant_id": applications.index, "application_date": applications["application_date"]},
...     index=applications.index,
... )
>>> observations.head(2)
   applicant_id application_date
0             0       2024-01-01
1             1       2024-01-01

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

The transactions of an applicant lie in the ``n_months`` calendar months before the application date, twelve by default: from the date ``n_months`` months earlier, counted as ``pandas.DateOffset`` counts them, up to but not including the application date. A window such as the last three months before the application date therefore cuts the transactions exactly, and no transaction on or after the date exists to leak into it.

How the transactions are simulated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The transactions of each month are drawn independently. In a year an applicant has on average about 25 purchases, 8 cash withdrawals and 11 repayments, 44 transactions in all. A purchase costs about 4.5% of the monthly income, a cash withdrawal about 3%, and a repayment is the instalment that ``income * debt_to_income`` implies, which varies by about 20% between applicants. Applicants with a higher ``income`` make a few more purchases and withdrawals, and purchases are more frequent in the months that start in November and December. An applicant whom the incumbent scorecard rates as riskier, by ``incumbent_pd``, withdraws more cash and misses more repayments, and so does an applicant with a higher ``risk``.

The ``risk`` also drives a stress that builds up before the application. It starts between three and nine months before the application and is strongest in the last month. Its level rises with the risk plus noise, from almost none to a bounded maximum. Under stress, the cash withdrawals are more frequent and larger, the purchases a little rarer, and a repayment is more likely to be missed, or paid up to a tenth short. Applicants differ in their baseline and the stress reflects the risk with noise, so no feature predicts the outcome well. The transactions do not depend on ``target``: they are drawn from the risk before the outcome, as a lender would observe them.

The number of cash withdrawals in the last three months shows the signal. It is counted here with pandas, on the transactions and observations above; the default rate rises from 8% of the applicants without a withdrawal to 27% of those with five or more:

>>> merged = transactions.merge(observations, on="applicant_id")
>>> last_3_months = merged["timestamp"] >= merged["application_date"] - pd.DateOffset(months=3)
>>> cash = merged[last_3_months & (merged["kind"] == "cash_withdrawal")]
>>> cash_3m = cash.groupby("applicant_id").size().reindex(applications.index, fill_value=0)
>>> cash_3m.name = "cash_withdrawals_3m"
>>> rate = applications["target"].groupby(cash_3m.clip(upper=5)).agg(["size", "mean"])
>>> rate.round(3)
                     size   mean
cash_withdrawals_3m
0                    1704  0.081
1                    2010  0.105
2                    1539  0.129
3                    1154  0.137
4                     767  0.192
5                    2826  0.268

The IV of this feature, which :func:`~kivi.univariate.profile_table` computes as in :doc:`univariate`, is a clear but modest 0.31:

>>> from kivi.univariate import profile_table
>>> profile_table(cash_3m.to_frame(), applications["target"])[["iv"]].round(3)
                       iv
variable
cash_withdrawals_3m  0.31

The incumbent scorecard sees only part of the risk, so the transactions say something about the default of applicants that it rates alike. Within each half of the applicants by ``incumbent_pd``, the default rate still rises with the number of withdrawals:

>>> above = applications["incumbent_pd"] > applications["incumbent_pd"].median()
>>> half = pd.Series(np.where(above, "above", "below"), index=applications.index, name="incumbent_pd")
>>> applications["target"].groupby([half, cash_3m.clip(upper=4)]).mean().unstack(0).round(3)
incumbent_pd         above  below
cash_withdrawals_3m
0                    0.163  0.052
1                    0.167  0.070
2                    0.194  0.081
3                    0.201  0.073
4                    0.310  0.098

Other window features are weaker or stronger. With the default five bins, the IV is 0.05 for the number of purchases in the last three months, 0.18 for the repayments of the last three months relative to the three months before them, and 0.31 for the cash withdrawn in the last three months relative to income. :func:`~kivi.features.window_features` computes such features without the merge, for many windows and columns at once, as :doc:`features` shows.

.. warning::

   The transactions depend on ``risk``, which no lender has: it exists only in simulated applications. Use the transactions to learn how window features work, not to estimate how well behaviour predicts default.

Synthetic defaults and recoveries
---------------------------------

:func:`~kivi.datasets.make_defaults` simulates loans that defaulted from 2018 to 2024, and :func:`~kivi.datasets.make_recoveries` the cash flows of their workouts: the recoveries, the costs and the cures from which the realized LGD of each loan follows. The revolving lines also have their limit and drawn amount 12 months before default, from which their realized CCF follows. The data is generated, not taken from a real portfolio, and the same ``rng`` seed always gives the same loans and cash flows:

>>> from kivi.datasets import make_defaults, make_recoveries
>>> defaults = make_defaults(n_defaults=2000, rng=0)
>>> defaults.shape
(2000, 9)
>>> columns = ["loan_id", "segment", "default_date", "ead", "collateral_value", "closing_date"]
>>> defaults[columns].head(3)
   loan_id    segment default_date        ead  collateral_value closing_date
0        1    secured   2018-01-01  307147.39          312800.0   2019-08-17
1        2  revolving   2018-01-01    2471.68               0.0   2020-11-21
2        3  unsecured   2018-01-01   23456.26               0.0   2018-12-29
>>> revolving = defaults[defaults["segment"] == "revolving"]
>>> revolving[["loan_id", "ead", "limit", "drawn", "discount_rate"]].head(3)
    loan_id       ead    limit     drawn  discount_rate
1         2   2471.68   5100.0   2043.04           0.05
5         6   3545.43   4700.0   2119.49           0.05
15       16  21057.35  21500.0  20902.37           0.05

The columns of the defaults are:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Meaning
   * - ``loan_id``
     - Identifier of the loan, from 1 and increasing with the default date.
   * - ``segment``
     - A categorical: ``secured``, ``unsecured`` or ``revolving``, about 40%, 35% and 25% of the loans.
   * - ``default_date``
     - Date of default, from 2018 to 2024.
   * - ``ead``
     - Exposure at default, the amount outstanding on the default date, rounded to cents.
   * - ``limit``
     - Limit of a revolving line on the reference date, 12 months before default, rounded to hundreds; missing for the other segments.
   * - ``drawn``
     - Amount drawn on a revolving line on the reference date, rounded to cents; missing for the other segments.
   * - ``collateral_value``
     - Value of the collateral of a secured loan at default, rounded to hundreds; 0 for the other segments.
   * - ``discount_rate``
     - Annual rate at which to discount the cash flows of the workout to the default date: a simulated three-month money-market rate of the month of default plus five percentage points, as paragraph 143 of EBA/GL/2017/16 prescribes.
   * - ``closing_date``
     - Date the workout ended, with the cure, the sale of the collateral or the write-off of what was not recovered; missing for the workouts still open at the data cut-off, 30 June 2025.

:func:`~kivi.datasets.make_recoveries` takes the defaults and returns one row per cash flow, sorted by loan and date. The first loan is secured: after three legal costs and two payments by the borrower, the collateral was sold on the closing date, at a cost of 3% of the proceeds:

>>> flows = make_recoveries(defaults, rng=0)
>>> flows.shape
(11497, 4)
>>> flows.head(7)
   loan_id       date     amount      kind
0        1 2018-04-10    -808.71      cost
1        1 2018-11-01   -1543.90      cost
2        1 2019-05-14    -552.91      cost
3        1 2019-07-11    3595.22  recovery
4        1 2019-08-02    2012.10  recovery
5        1 2019-08-17  233970.35  recovery
6        1 2019-08-17   -7019.11      cost

The columns of the cash flows are:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Meaning
   * - ``loan_id``
     - The ``loan_id`` of the loan in the defaults.
   * - ``date``
     - Date of the cash flow.
   * - ``amount``
     - Amount, rounded to cents: positive for recoveries and cures, negative for costs.
   * - ``kind``
     - A categorical: ``recovery``, ``cost`` or ``cure``.

A ``cure`` is the amount outstanding on the closing date of a loan that returned to performing status. Paragraph 135 of EBA/GL/2017/16 treats it as if the borrower had repaid that amount on the closing date, so it is a cash flow like a recovery. :func:`~kivi.datasets.make_recoveries` reads only the columns ``loan_id``, ``segment``, ``default_date``, ``ead``, ``collateral_value`` and ``closing_date``, so it also simulates the workouts of a subset of the defaults or of loans with other identifiers. The dates must have a datetime64 dtype.

How the workouts are simulated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A workout lasts at least three months, the probation of a cure. About 30% of the secured loans, 10% of the unsecured loans and 15% of the revolving lines cure, after seven months on average, and the others are liquidated:

* a cured loan pays part of its arrears during the workout, sometimes with a small cost, and the cure follows on the closing date;
* a secured loan pays a little during the workout, and its collateral is sold on the closing date. The bank recovers the proceeds up to its claim, the exposure plus default interest of 4% to 10% a year, and pays 3% of the proceeds as the costs of the sale and a few legal costs. The workout lasts about two and a half years on average;
* an unsecured loan or a revolving line recovers about 17% or 22% of its exposure on average, in several payments, with a few collection costs. The workout lasts a year and a half or 15 months on average.

About 5% of the loans also have a legal cost up to 90 days before default, which paragraph 144 of EBA/GL/2017/16 counts in the economic loss, and 3% of the liquidations have litigation costs of 10% to 40% of the exposure. The secured loans have a median loan-to-value ratio of 75% at default, and some owe more than their collateral is worth. The cash flows stop at the data cut-off, 30 June 2025: a workout still open then has its cash flows up to the cut-off and none after it.

Realized LGDs
^^^^^^^^^^^^^

The realized LGD of a loan is its economic loss over its exposure at default, ``(ead + PV(costs) - PV(recoveries)) / ead``, where the cures count as recoveries and the present values discount each cash flow to the default date at the annual ``discount_rate`` over the days in between, divided by 365. Because the costs are negative amounts, the LGD is one minus the present value of all the cash flows over the exposure. It is computed here with pandas:

>>> dated = flows.merge(defaults[["loan_id", "default_date", "discount_rate"]], on="loan_id")
>>> years = (dated["date"] - dated["default_date"]).dt.days / 365
>>> present_value = dated["amount"] * (1 + dated["discount_rate"]) ** -years
>>> loans = defaults.set_index("loan_id")
>>> net = present_value.groupby(dated["loan_id"]).sum().reindex(loans.index, fill_value=0)
>>> lgd = (1 - net / loans["ead"]).rename("lgd")

Only the workouts that have closed have a realized LGD. Their LGDs are bimodal: most secured loans lose less than 10% of their exposure, and most unsecured loans and revolving lines more than 70%:

>>> import numpy as np
>>> import pandas as pd
>>> closed = loans["closing_date"].notna()
>>> labels = ["below 0", "0-0.1", "0.1-0.3", "0.3-0.7", "0.7-0.9", "0.9-1", "1 and above"]
>>> edges = [-np.inf, 0, 0.1, 0.3, 0.7, 0.9, 1, np.inf]
>>> band = pd.cut(lgd[closed], edges, right=False, labels=labels)
>>> pd.crosstab(band, loans.loc[closed, "segment"])
segment      secured  unsecured  revolving
lgd
below 0           38          0          0
0-0.1            478         68         82
0.1-0.3          121          1          3
0.3-0.7           67         49         62
0.7-0.9            0        210        155
0.9-1              0        183         86
1 and above        0         69         50

The cures make the mode near 0 in every segment. A cured loan loses only the discounting of the time until the cure and a few costs, while the loans that are liquidated keep the shape of their segment:

>>> cured = pd.Series(loans.index.isin(flows.loc[flows["kind"] == "cure", "loan_id"]), loans.index)
>>> by_outcome = lgd[closed].groupby([loans["segment"], cured.rename("cured")], observed=True)
>>> by_outcome.agg(["size", "mean", "median"]).round(3)
                 size   mean  median
segment   cured
secured   False   442  0.130   0.069
          True    262  0.034   0.029
unsecured False   511  0.877   0.899
          True     69  0.039   0.033
revolving False   356  0.835   0.869
          True     82  0.035   0.031

The 38 negative LGDs are secured loans whose collateral covered the claim with its default interest, so that they recovered more than their exposure is worth at the discount rate. The 119 LGDs above 1 are loans whose costs exceeded their recoveries. Paragraphs 160 and 162 of EBA/GL/2017/16 keep both in the data: a negative LGD counts as 0 in the averages, and an LGD far above 1 calls for an explanation. Neither is clipped here.

An open workout is not over, so the LGD of its cash flows so far is not a realized LGD. Above all, no collateral of an open workout has been sold yet:

>>> workout = closed.map({True: "closed", False: "open"}).rename("workout")
>>> lgd.groupby([loans["segment"], workout], observed=True).mean().unstack().round(3)
workout    closed   open
segment
secured     0.094  1.000
unsecured   0.777  0.924
revolving   0.685  0.903

The long workouts are the ones still open, so the closed workouts also hold more cures than all the workouts do. Leave out the open workouts before averaging or back-testing realized LGDs, as paragraph 154 of EBA/GL/2017/16 and the back-tests of the European Central Bank in :doc:`validation` ask.

Realized CCFs
^^^^^^^^^^^^^

The realized CCF of a revolving line is the share of its undrawn limit on the reference date that it drew by default, ``(ead - drawn) / (limit - drawn)``, with the limit and the drawn amount of the reference date. It is undefined for a line drawn to or beyond its limit:

>>> undrawn = revolving["limit"] - revolving["drawn"]
>>> ccf = ((revolving["ead"] - revolving["drawn"]) / undrawn).where(undrawn > 0)
>>> int(ccf.isna().sum()), int((ccf < 0).sum()), int(ccf.between(0, 1).sum()), int((ccf > 1).sum())
(25, 43, 376, 45)
>>> print(round(ccf.median(), 3))
0.554

Of the 464 lines that had an undrawn limit on the reference date, 376 have a CCF from 0 to 1, with a median of 0.55. A line with a negative CCF repaid part of its balance before default, and a line with a CCF above 1 ended above its limit. Article 182(1) of CRR3 counts negative CCFs as 0 when the CCF is estimated, while the CCFs above 1 stay, since the reference data must not be capped at 100%.

.. warning::

   The defaults and their cash flows are simulated to show the computations of realized LGDs and CCFs, with effects whose size is set by the simulation. Use them to learn how the computations work, not to estimate the LGDs or CCFs of a real portfolio.

Dataset and attribution
-----------------------

Bank Marketing by S. Moro, P. Rita and P. Cortez (2014) is distributed under CC BY 4.0. The bundled CSV was converted from the legacy workbook. The loader encodes subscription as 1 and non-subscription as 0. See the `dataset record <https://doi.org/10.24432/C5K306>`_ and `license <https://creativecommons.org/licenses/by/4.0/>`_. The loan panel, the applications, the transactions, the defaults and their cash flows are synthetic and have no source to attribute.
