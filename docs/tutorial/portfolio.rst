The loan book under IFRS 9
==========================

The approved applications become loans, and the lender follows them until they are repaid. This chapter tracks how the loans go bad by origination month and how they move between delinquency buckets, turns those movements into the PD term structures, stages and loss allowance that IFRS 9 requires, and reads the repayment history of each loan; :doc:`/user_guide/vintage` and :doc:`/user_guide/lifetime` explain each step.

The loan book is the synthetic panel of :func:`~kivi.datasets.make_loan_panel`: 5,000 instalment loans originated from January 2024 to December 2025, with their status at every month end up to January 2026. It stands in for the loans that the lender booked; it is simulated apart from the applications of the earlier chapters, so this chapter needs none of their code.

Follow the vintages
-------------------

:func:`~kivi.vintage.vintage_table` gives, for each origination month and month on book, the share of the loans that have been 90 days past due at least once:

>>> import numpy as np
>>> import pandas as pd
>>> from kivi.datasets import make_loan_panel
>>> from kivi.vintage import roll_rates, transition_matrix, vintage_table
>>> panel = make_loan_panel(rng=0)
>>> ever_90 = vintage_table(
...     panel["loan_id"], panel["cohort"], panel["month_on_book"], panel["status"] == "90+"
... )
>>> ever_90[[6, 12, 18, 24]].iloc[::6].round(3)
month_on_book     6      12     18     24
cohort
2024-01        0.000  0.068  0.094  0.094
2024-07        0.005  0.051  0.081    NaN
2025-01        0.005  0.034    NaN    NaN
2025-07        0.005    NaN    NaN    NaN

.. plot::
   :alt: Cumulative 90+ bad rate by month on book for one cohort in each half year
   :caption: Vintage curves of the cohorts in the table: each stops at the cohort's latest month on book.

   kt.figure_vintage()

Each row follows one cohort and never decreases, because a loan that went bad stays counted. Months that a cohort has not reached are missing, not zero.

Measure the roll rates
----------------------

A transition matrix counts how loans move between the delinquency buckets from one month end to the next. Pair each row of the panel with the row of the same loan one month later, then pass the two statuses to :func:`~kivi.vintage.transition_matrix` and :func:`~kivi.vintage.roll_rates`:

>>> later = panel[["loan_id", "month_on_book", "status"]].assign(
...     month_on_book=panel["month_on_book"] - 1
... )
>>> pairs = panel.merge(later, on=["loan_id", "month_on_book"], suffixes=("", "_next"))
>>> monthly = transition_matrix(pairs["status"], pairs["status_next"], normalize=True)
>>> roll_rates(pairs["status"], pairs["status_next"]).round(3)
before
current    0.023
1-29       0.295
30-59      0.553
60-89      0.712
Name: roll_rate, dtype: float64

Each month 2.3% of the current loans miss an instalment, and seven in ten loans at 60 to 89 days past due reach 90+ a month later.

Project the PDs
---------------

If the movements follow the same monthly matrix every month, its powers give the probability of reaching 90+ within any horizon from each bucket. :func:`~kivi.lifetime.transition_pd` makes 90+ absorbing and returns these cumulative PDs, and :func:`~kivi.lifetime.convert_pd` turns them into the marginal PD of each month:

>>> from kivi.lifetime import assign_stages, convert_pd, expected_credit_loss
>>> from kivi.lifetime import significant_increase, stage_table, transition_pd
>>> cumulative = transition_pd(monthly, horizons=36)
>>> cumulative[[3, 12, 36]].round(4)
horizon      3       12      36
state
current  0.0000  0.0393  0.1440
1-29     0.1159  0.2420  0.3246
30-59    0.5381  0.6041  0.6472
60-89    0.8635  0.8731  0.8869
90+      1.0000  1.0000  1.0000
>>> marginal = convert_pd(cumulative, kind="cumulative", to="marginal")

A current loan needs four months to reach 90+, so its PD is 0 for three months; a loan at 60 to 89 days past due reaches 90+ within the year with a probability of 87%.

Measure the expected credit loss
--------------------------------

The reporting date is the end of January 2026, the last month of the panel, and the book holds the loans that still have a balance. Each loan gets the marginal PDs of its bucket, one column per month after the reporting date:

>>> book = panel[(panel["month"] == "2026-01") & (panel["balance"] > 0)].set_index("loan_id")
>>> account_pd = marginal.loc[book["status"]].set_axis(book.index)

The exposure of each month is the balance that the loan is expected to owe then, repaid by its remaining instalments of an annuity at 1% a month; :doc:`/user_guide/lifetime` derives it. A loan at 90+ has defaulted at the reporting date, so its loss is not discounted:

>>> remaining = np.maximum(book["term"] - book["month_on_book"], 1).to_numpy()[:, None]
>>> paid = np.minimum(np.arange(36), remaining)
>>> ead = book[["balance"]].to_numpy() * (1.01**remaining - 1.01**paid) / (1.01**remaining - 1)
>>> defaulted = book["status"] == "90+"
>>> rate = np.where(defaulted, 0.0, 0.01)
>>> twelve_month = expected_credit_loss(account_pd, 0.45, ead, rate=rate, horizon=12)
>>> lifetime = expected_credit_loss(account_pd, 0.45, ead, rate=rate)

Allocate the stages
-------------------

A loan whose credit risk has increased significantly since it was granted is in stage 2 and carries its lifetime ECL; a loan in default is in stage 3. :func:`~kivi.lifetime.significant_increase` compares the 12-month PD of each loan's bucket with that of a current loan, as every loan was current when granted, and :func:`~kivi.lifetime.assign_stages` adds the backstops of 30 and 90 days past due. The panel records buckets, so the example takes the middle of each bucket as the days past due:

>>> increase = significant_increase(
...     cumulative.loc[book["status"], 12].to_numpy(),
...     np.full(len(book), cumulative.loc["current", 12]),
...     relative=2.0,
...     absolute=0.01,
... )
>>> days_past_due = np.array([0, 15, 45, 75, 105])[book["status"].cat.codes]
>>> stage = assign_stages(increase, days_past_due, defaulted)

:func:`~kivi.lifetime.stage_table` adds up the exposure and the loss allowance of each stage: the 12-month ECL in stage 1 and the lifetime ECL in stages 2 and 3:

>>> allowance = np.where(stage == 1, twelve_month, lifetime)
>>> table = stage_table(stage, book["balance"], allowance)
>>> table.round(4)
       total     exposure  exposure_share  expected_credit_loss  coverage
stage
1       4076  22660955.57          0.9126           219165.5680    0.0097
2        135    739820.77          0.0298           129643.3610    0.1752
3        205   1431123.43          0.0576           644005.5435    0.4500
>>> print(f"{table['expected_credit_loss'].sum() / table['exposure'].sum():.4f}")
0.0400

Stage 1 holds nine tenths of the exposure at a coverage below 1%, while stage 3 holds the defaulted loans at the LGD of 45%. The loss allowance of the whole book is 4% of its exposure.

Read the repayment history
--------------------------

The stages above look only at the bucket of each loan at the reporting date, so a loan returns to stage 1 as soon as it has paid its arrears. Many lenders keep a loan that has cured in stage 2 for a probation period instead. :func:`~kivi.features.panel_features` reads the history of each loan from the panel: for each window of months before the month of the reporting date, the months with a status, the months in arrears, the longest runs of bad and good months and the worst status, and the months since the latest month in arrears. With ``bad="1-29"`` every month end in arrears counts as bad, as every bucket in arrears puts a loan in stage 2 or 3 above; :doc:`/user_guide/features` describes the features:

>>> from kivi.features import panel_features
>>> observations = pd.DataFrame(
...     {"loan_id": book.index, "reporting_date": pd.Timestamp("2026-01-31")}, index=book.index
... )
>>> history = panel_features(
...     panel, observations, id="loan_id", month="month", status="status",
...     observation_time="reporting_date", windows=["6M", "12M"], bad="1-29",
... )
>>> pd.crosstab(stage, history["worst_status_6m"])
worst_status_6m  current  1-29  30-59  60-89  90+
stage
1                   3558   240     43      9    0
2                     54    42     28      7    0
3                      0     0      0     14  191

The windows end before January, the month of the reporting date, so the six months are July to December 2025, and the status of January is the bucket that the stages already use. Of the loans in stage 1, 292 were in arrears at one of those month ends and are current again. The 230 loans originated in December 2025 have no month end before January, so their worst status is missing and the table leaves them out.

A probation of six months keeps those 292 loans in stage 2: a loan in arrears at any of the six month ends before the reporting date counts as a significant increase too:

>>> probation = history["bad_months_6m"].to_numpy() > 0
>>> revised = assign_stages(increase | probation, days_past_due, defaulted)
>>> revised_table = stage_table(revised, book["balance"], np.where(revised == 1, twelve_month, lifetime))
>>> revised_table.round(4)
       total     exposure  exposure_share  expected_credit_loss  coverage
stage
1       3784  21218424.37          0.8545           205727.2235    0.0097
2        427   2182351.97          0.0879           148631.8793    0.0681
3        205   1431123.43          0.0576           644005.5435    0.4500
>>> print(f"{revised_table['expected_credit_loss'].sum() - table['expected_credit_loss'].sum():.0f}")
5550

The rule moves 1.4 million of exposure into stage 2 but adds only 5,550 to the allowance, and the coverage of stage 2 falls from 17.5% to 6.8%. The cured loans are current, so the transition matrix gives them the PDs of every other current loan, and only the horizon of their ECL changes, from 12 months to the lifetime. The matrix forgets the history of a loan, so it cannot tell whether a loan that was recently in arrears is riskier than one that never was. A model with the features of the history can: they are the characteristics of a behaviour scorecard, which the steps of this tutorial would build from the book as they built the application scorecard.

Close the loop
--------------

The project has gone from the applications to the loss allowance of the loans they became. In production the steps repeat: the monitoring of :doc:`monitoring` watches the applicants and the scorecard, the vintages, roll rates and repayment histories of this chapter watch the loans, and when either drifts, the project returns to the data of :doc:`data` with the new outcomes.
