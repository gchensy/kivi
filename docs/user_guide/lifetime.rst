Lifetime PD and expected credit loss
====================================

IFRS 9 requires a lender to hold a loss allowance for the credit losses that it expects on its loans. While the credit risk of a loan has not increased significantly since it was granted, the allowance covers the losses from defaults in the next 12 months; once it has, and once the loan is in default, it covers the losses over the remaining life of the loan. Both need the probability of a default in each period after the reporting date, a PD term structure, rather than the single 12-month PD of a scorecard.

The term structures come from the vintage curves and the transition matrices of :doc:`vintage`, and the loss allowance is the last step of this user guide. :func:`~kivi.lifetime.convert_pd` converts a term structure between cumulative, marginal and conditional PDs, :func:`~kivi.lifetime.transition_pd` projects one from a transition matrix, :func:`~kivi.lifetime.expected_credit_loss` gives the 12-month and the lifetime expected credit loss of each account, :func:`~kivi.lifetime.significant_increase` and :func:`~kivi.lifetime.assign_stages` allocate the accounts to the three stages, and :func:`~kivi.lifetime.stage_table` sums the loss allowance of each stage.

Definitions
-----------

A PD term structure describes the risk of each period after a starting point, such as the origination of a loan or the reporting date. For period :math:`k`, the *cumulative PD* :math:`C_k` is the probability of a default in periods 1 to :math:`k`, the *marginal PD* :math:`m_k` the probability of a default in period :math:`k`, and the *conditional PD* :math:`h_k`, also called the hazard rate, the probability of a default in period :math:`k` given that there was none before. With :math:`C_0 = 0`, each of them gives the others:

.. math::

   m_k = C_k - C_{k-1}, \qquad
   h_k = \frac{m_k}{1 - C_{k-1}}, \qquad
   C_k = \sum_{j \le k} m_j = 1 - \prod_{j \le k} (1 - h_j).

The expected credit loss (ECL) of an account adds up, over the periods :math:`t = 1, \ldots, H` after the reporting date, the marginal PD of each period times the loss given default (LGD) and the exposure at default (EAD) in that period, discounted to the reporting date at the effective interest rate :math:`r` per period:

.. math::

   \mathrm{ECL} = \sum_{t=1}^{H} m_t \, \mathrm{LGD}_t \, \mathrm{EAD}_t \, (1 + r)^{-t}.

The 12-month ECL adds up the periods of the next 12 months, and the lifetime ECL every period up to maturity. :func:`~kivi.lifetime.expected_credit_loss` uses these conventions:

- Period 1 starts at the reporting date and ends one period later, so with monthly periods the 12-month ECL has ``horizon=12``.
- The marginal PDs are seen from the reporting date. They are not conditional on survival, and they add up to the cumulative PD.
- A default in period :math:`t` occurs at its end, so its loss is discounted over :math:`t` periods. The rate is the effective interest rate per period, such as :math:`(1 + r_\text{annual})^{1/12} - 1` for monthly periods.

IFRS 9 allocates each exposure to one of three stages, which decide its horizon:

.. list-table::
   :header-rows: 1
   :widths: 15 55 30

   * - Stage
     - Exposures
     - Loss allowance
   * - 1
     - Credit risk not increased significantly since initial recognition
     - 12-month ECL
   * - 2
     - Credit risk increased significantly since initial recognition
     - Lifetime ECL
   * - 3
     - Credit-impaired
     - Lifetime ECL

IFRS 9 defines credit-impaired exposures by events that hurt their expected cash flows, such as a default or the significant financial difficulty of the borrower. In practice, lenders usually align them with their definition of default.

A worked example
----------------

A loan defaults in its first year with probability 0.02, by the end of its second year with probability 0.05 and by the end of its third year with probability 0.08:

>>> from kivi.lifetime import convert_pd, expected_credit_loss
>>> cumulative = [0.02, 0.05, 0.08]
>>> convert_pd(cumulative, kind="cumulative", to="marginal")
array([0.02, 0.03, 0.03])
>>> convert_pd(cumulative, kind="cumulative", to="conditional").round(4)
array([0.02  , 0.0306, 0.0316])

The marginal PD of the second year is 0.05 − 0.02 = 0.03. Of the 98% of loans without a default in the first year, 0.03 / 0.98 = 3.06% default in the second, the conditional PD.

With an LGD of 50%, an exposure that amortizes from 1,000 to 700 and 400, and an effective interest rate of 5% a year, the expected credit losses of the loan are:

>>> marginal = [[0.02, 0.03, 0.03]]
>>> ead = [[1000.0, 700.0, 400.0]]
>>> expected_credit_loss(marginal, 0.5, ead, rate=0.05).round(2)
array([24.23])
>>> expected_credit_loss(marginal, 0.5, ead, rate=0.05, horizon=1).round(2)
array([9.52])

The first year contributes 0.02 × 0.5 × 1,000 / 1.05 = 9.52, which is the 12-month ECL. The second contributes 0.03 × 0.5 × 700 / 1.05² = 9.52 and the third 0.03 × 0.5 × 400 / 1.05³ = 5.18, so the lifetime ECL is 24.23. Each row of the PDs is one account, so a portfolio is computed in one call.

Term structures from vintage curves
-----------------------------------

A vintage table holds the cumulative bad rate of each cohort by month on book: a cumulative PD term structure from origination. The examples use the synthetic loan panel of :func:`~kivi.datasets.make_loan_panel`, which :doc:`datasets` describes, and the 90+ bad definition of :doc:`vintage`. The conditional PDs of each cohort follow from :func:`~kivi.vintage.vintage_table` and :func:`~kivi.lifetime.convert_pd`:

>>> import numpy as np
>>> import pandas as pd
>>> from kivi.datasets import make_loan_panel
>>> from kivi.vintage import vintage_table
>>> panel = make_loan_panel(rng=0)
>>> ever_90 = vintage_table(
...     panel["loan_id"], panel["cohort"], panel["month_on_book"], panel["status"] == "90+"
... )
>>> hazard = convert_pd(ever_90, kind="cumulative", to="conditional")
>>> hazard[[6, 12, 18, 24]].iloc[::3].round(4)
month_on_book      6       12      18   24
cohort
2024-01        0.0000  0.0091  0.0093  0.0
2024-04        0.0050  0.0051  0.0106  NaN
2024-07        0.0051  0.0053  0.0000  NaN
2024-10        0.0000  0.0000     NaN  NaN
2025-01        0.0000  0.0050     NaN  NaN
2025-04        0.0049     NaN     NaN  NaN
2025-07        0.0000     NaN     NaN  NaN
2025-10           NaN     NaN     NaN  NaN

Each entry is the share of a cohort's loans that went bad at that month on book among the loans that had not gone bad before. The months on book that a cohort has not reached stay missing. A cohort has about 200 loans, so a single bad loan moves its rate by half a percentage point, and many entries are 0.

Conditional PDs at the same month on book are comparable across cohorts, which makes them the quantity to pool. The mean of each column averages the cohorts that reached that month on book and skips the others, because ``mean`` ignores missing values. The early months rest on many cohorts and the late ones on few:

>>> pooled = hazard.mean()
>>> hazard[[6, 12, 18, 24]].count()
month_on_book
6     19
12    13
18     7
24     1
dtype: int64
>>> curve = convert_pd(pooled, kind="conditional", to="cumulative")
>>> pd.DataFrame({"pooled": curve, "2024-01": ever_90.loc["2024-01"]}).loc[[6, 12, 18, 24]].round(4)
               pooled  2024-01
month_on_book
6              0.0101   0.0000
12             0.0417   0.0684
18             0.0717   0.0940
24             0.0734   0.0940

.. plot::
   :alt: Cumulative 90+ bad rate of every cohort by month on book as thin lines, and the cumulative PD of the pooled conditional PDs as a bold line
   :caption: Cumulative 90+ bad rate of each cohort by month on book (thin) and the curve of the pooled conditional PDs (bold). The young cohorts stop after a few months on book, while the pooled curve runs to the last month on book of the panel.

   kp.figure_lifetime_term_structure()

The pooled curve reaches the oldest month on book of the panel for every cohort. From month on book 12 on, it lies below the curve of the oldest cohort, 2024-01, which was riskier than the average one, and beyond month on book 18 it rests on seven cohorts or fewer.

From the reporting date
^^^^^^^^^^^^^^^^^^^^^^^

:func:`~kivi.lifetime.expected_credit_loss` takes PDs seen from the reporting date, for a loan that has not defaulted by then. For a loan of a given age, convert the conditional PDs of the months on book after that age to marginal PDs. A loan at month on book 6 has these marginal PDs for the next 12 months:

>>> ahead = convert_pd(pooled.loc[7:18], kind="conditional", to="marginal")
>>> ahead.round(4).head(3)
month_on_book
7    0.0063
8    0.0054
9    0.0048
dtype: float64
>>> print(f"{ahead.sum():.4f}")
0.0623

Their sum, 6.23%, is the probability that a loan without a default by month on book 6 defaults by month on book 18, :math:`(C_{18} - C_6) / (1 - C_6)` on the pooled curve. A vintage table counts every loan of a cohort, also the loans repaid early, which can no longer go bad. These PDs therefore already allow for early repayment, and they go with the balance of a loan that is still on book, such as its contractual schedule.

Term structures from transition matrices
----------------------------------------

A transition matrix gives the probabilities of moving from each delinquency bucket or rating grade to each other one in a period, as in :doc:`vintage`. If the movements follow a time-homogeneous Markov chain, the powers of the matrix give a term structure for each starting state. :func:`~kivi.lifetime.transition_pd` makes the default state absorbing, so that loans that cure from default still count as defaulted, and returns the cumulative PD of each state at each horizon:

>>> from kivi.lifetime import transition_pd
>>> from kivi.vintage import transition_matrix
>>> later = panel[["loan_id", "month_on_book", "status"]].assign(
...     month_on_book=panel["month_on_book"] - 1
... )
>>> pairs = panel.merge(later, on=["loan_id", "month_on_book"], suffixes=("", "_next"))
>>> monthly = transition_matrix(pairs["status"], pairs["status_next"], normalize=True)
>>> cumulative = transition_pd(monthly, horizons=36)
>>> cumulative[[1, 3, 6, 12, 24, 36]].round(4)
horizon      1       3       6       12      24      36
state
current  0.0000  0.0000  0.0113  0.0393  0.0932  0.1440
1-29     0.0000  0.1159  0.2177  0.2420  0.2845  0.3246
30-59    0.0000  0.5381  0.5922  0.6041  0.6263  0.6472
60-89    0.7118  0.8635  0.8694  0.8731  0.8802  0.8869
90+      1.0000  1.0000  1.0000  1.0000  1.0000  1.0000

The default state is the last one, here ``90+``, unless ``default_state`` names another. A loan falls at most one bucket further behind per month, so a current loan needs four months to reach 90+ and its PD is 0 for three months. The PD of every state rises with the horizon, and the worse the bucket, the higher it starts: six in ten loans at 30 to 59 days past due reach 90+ within a year.

Because :math:`P^{a+b} = P^a P^b`, the PD at :math:`a + b` months is the PD at :math:`b` months of the states reached after :math:`a` months. A monthly matrix therefore gives the PD over any number of months, but only as far as the movements are a Markov chain with the same probabilities in every month. The current state against the pooled vintage curve shows how far that holds:

>>> pd.DataFrame({"matrix": cumulative.loc["current"], "vintage": curve}).loc[[6, 12, 18, 24]].round(4)
    matrix  vintage
6   0.0113   0.0101
12  0.0393   0.0417
18  0.0666   0.0717
24  0.0932   0.0734

.. plot::
   :alt: Cumulative PD by horizon for the states current, 1-29, 30-59, 60-89 and 90+, with the pooled vintage curve as a dashed line next to the current state
   :caption: Cumulative PD of each state by horizon from the monthly transition matrix. The dashed line is the pooled vintage curve, which follows the curve of the current state in the first year and levels off in the second.

   kp.figure_lifetime_transition()

The two agree in the first year and part in the second. The matrix mixes loans of every age, while the chance of missing an instalment depends on the month on book, and the vintage curve beyond month on book 18 rests on a few cohorts.

:func:`~kivi.lifetime.convert_pd` gives the marginal PDs of each state, which add up to the PD at the last horizon:

>>> marginal = convert_pd(cumulative, kind="cumulative", to="marginal")
>>> marginal.sum(axis=1).round(4)
state
current    0.1440
1-29       0.3246
30-59      0.6472
60-89      0.8869
90+        1.0000
dtype: float64

Expected credit loss
--------------------

The reporting date here is the end of January 2026, the last month of the panel, and the book holds the loans that still have a balance:

>>> book = panel[(panel["month"] == "2026-01") & (panel["balance"] > 0)].set_index("loan_id")
>>> book["status"].value_counts(sort=False)
status
current    4076
1-29         82
30-59        21
60-89        32
90+         205
Name: count, dtype: int64

Each loan gets the marginal PDs of its delinquency bucket, with one row per loan and one column per month after the reporting date:

>>> account_pd = marginal.loc[book["status"]].set_axis(book.index)
>>> account_pd.iloc[:3, :6].round(4)
horizon    1    2    3       4       5       6
loan_id
2        0.0  0.0  0.0  0.0026  0.0041  0.0046
3        0.0  0.0  0.0  0.0026  0.0041  0.0046
4        1.0  0.0  0.0  0.0000  0.0000  0.0000

The exposure at default of each month is the balance that the loan is expected to have outstanding in that month: the balance at the reporting date, repaid by the remaining instalments of an annuity at the loans' interest rate of 1% a month. ``paid`` counts the instalments paid before each month. Loans past their term, which keep rolling in this data, owe their balance at once. After the last instalment the exposure is 0, so the months after a loan's term add nothing:

>>> remaining = np.maximum(book["term"] - book["month_on_book"], 1).to_numpy()[:, None]
>>> paid = np.minimum(np.arange(36), remaining)
>>> growth = 1.01
>>> ead = book[["balance"]].to_numpy() * (growth**remaining - growth**paid) / (growth**remaining - 1)

The losses are discounted at the loans' effective interest rate of 1% a month. A loan already at 90+ has defaulted at the reporting date, so its loss is measured at that date without discounting, and its rate is 0. With an LGD of 45%, :func:`~kivi.lifetime.expected_credit_loss` gives the 12-month and the lifetime ECL of each loan:

>>> defaulted = book["status"] == "90+"
>>> rate = np.where(defaulted, 0.0, 0.01)
>>> twelve_month = expected_credit_loss(account_pd, 0.45, ead, rate=rate, horizon=12)
>>> lifetime = expected_credit_loss(account_pd, 0.45, ead, rate=rate)
>>> losses = pd.DataFrame(
...     {
...         "status": book["status"],
...         "remaining": remaining[:, 0],
...         "balance": book["balance"],
...         "twelve_month": twelve_month,
...         "lifetime": lifetime,
...     }
... )
>>> losses[losses["remaining"] > 24].groupby("status", observed=True).head(1).round(2)
          status  remaining   balance  twelve_month  lifetime
loan_id
2697     current         25   2999.08         36.00     56.27
2723       30-59         25   4069.08       1017.32   1028.66
2790         90+         25  10371.94       4667.37   4667.37
2809        1-29         25   3331.36        307.70    325.47
3383       60-89         28   6964.51       2680.77   2688.99

These are the first loans of each bucket with more than two years to run. The lifetime ECL of the current loan is half as large again as its 12-month ECL: its PD is 0 for three months and its risk builds up over the following ones. The delinquent loans default mostly within the year, so their two ECLs are close. For a loan already at 90+, the PD is 1 in the first month and the rate is 0, so the ECL is the LGD times the balance: 0.45 × 10,371.94 = 4,667.37.

Stage allocation
----------------

A significant increase in credit risk is judged against the risk expected at initial recognition. :func:`~kivi.lifetime.significant_increase` compares the PD at the reporting date with the PD expected at origination, as a ratio, a difference or both. With ``relative=2.0`` and ``absolute=0.01``, a PD must have more than doubled and risen by more than one percentage point:

>>> from kivi.lifetime import assign_stages, significant_increase, stage_table
>>> significant_increase(
...     [0.004, 0.012, 0.03, 0.09], [0.001, 0.010, 0.01, 0.05], relative=2.0, absolute=0.01
... )
array([False, False,  True, False])

The first PD quadrupled but rose by only 0.3 percentage points, the second barely rose, and the last rose by 4 points without doubling. IFRS 9 does not set these thresholds. They are the lender's policy, like the definition of default, so both arguments are required, and either can be None to leave its test out. For a rule under which either test suffices, combine two calls, each with one threshold, with ``|``; qualitative criteria, such as a watch list, combine with the flags in the same way.

Compare PDs of the same horizon. In the panel, a loan's delinquency bucket is the only risk driver, and every loan was current at origination, so the 12-month PD of each bucket against that of a current loan gives the test:

>>> probability = cumulative.loc[book["status"], 12].to_numpy()
>>> origination_probability = np.full(len(book), cumulative.loc["current", 12])
>>> increase = significant_increase(
...     probability, origination_probability, relative=2.0, absolute=0.01
... )

:func:`~kivi.lifetime.assign_stages` adds the backstops: an exposure more than 30 days past due is at least in stage 2, and one 90 days past due or more, or flagged as defaulted, is in stage 3. The defaults follow the rebuttable presumptions of IFRS 9: credit risk has increased significantly when contractual payments are more than 30 days past due (paragraph 5.5.11), and default occurs no later than when a financial asset is 90 days past due (paragraph B5.5.37). Pass ``stage_2_days`` and ``stage_3_days`` for other thresholds. The panel records buckets rather than days, so the example takes the middle of each bucket, and it flags the 90+ loans as defaulted:

>>> days_past_due = np.array([0, 15, 45, 75, 105])[book["status"].cat.codes]
>>> stage = assign_stages(increase, days_past_due, defaulted)
>>> pd.crosstab(book["status"], stage)
stage       1   2    3
status
current  4076   0    0
1-29        0  82    0
30-59       0  21    0
60-89       0  32    0
90+         0   0  205

The PD test flags every delinquent bucket, so here the stages follow the buckets. A behavioural score or a rating would also separate the current loans from each other.

The wording of the two presumptions sets the boundaries. Like the PD thresholds, the 30 days must be exceeded, so an exposure exactly 30 days past due stays in stage 1, while an exposure exactly 90 days past due is already in stage 3:

>>> assign_stages([False] * 5, days_past_due=[0, 30, 31, 89, 90])
array([1, 1, 2, 2, 3])

Loss allowance by stage
-----------------------

The loss allowance of a loan in stage 1 is its 12-month ECL, and that of a loan in stage 2 or 3 its lifetime ECL. :func:`~kivi.lifetime.stage_table` sums the exposures and the allowances of each stage:

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

Reading the result
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Meaning
   * - ``total``
     - Number of exposures in the stage, or their total weight with ``weights``.
   * - ``exposure``
     - Sum of the exposures, such as the gross carrying amounts.
   * - ``exposure_share``
     - The stage's share of the exposure of all stages.
   * - ``expected_credit_loss``
     - Sum of the loss allowances.
   * - ``coverage``
     - ``expected_credit_loss`` divided by ``exposure``, the coverage ratio. NaN for a stage without exposure.

The coverage ratio rises from stage to stage, here from 1% in stage 1 to 18% in stage 2 and 45% in stage 3. In stage 3 the default has happened and the loss is not discounted, so the coverage is the LGD. In stage 1 it is below the LGD times the 12-month PD of a current loan, 0.45 × 0.0393 = 1.8%, because the balances amortize and many loans have less than a year to run. The coverage of the portfolio, 4.0% here, is the sum of the loss allowances divided by the sum of the exposures. The share of the exposure in stage 2 shows how much of the book carries lifetime losses; recompute the table with other thresholds to see how much the allowance depends on them.

Missing values
--------------

:func:`~kivi.lifetime.convert_pd` treats a missing value as a period that is not observed, and never fills it. The marginal and conditional PDs of a period need the cumulative PDs at its end and at the end of the period before, and a cumulative PD needs every period before it. A curve that starts with missing values, as in a panel that starts after origination, gives marginal and conditional PDs from its second observed period on, but no cumulative PDs. Fill the periods that are not observed first, for example by pooling the cohorts as above or by extending the curve beyond its last month, before passing the PDs on.

:func:`~kivi.lifetime.expected_credit_loss` converts each account on its own, so as :doc:`/api/conventions` describes, an account with a missing PD, LGD or EAD in a period that is added up, or with a missing rate, gets a missing ECL; the periods after ``horizon`` are not used. :func:`~kivi.lifetime.transition_pd` needs a complete row for every state other than the default state. The staging functions reject missing values, because a flag or a stage cannot be missing: decide how to stage an exposure without a PD or without its days past due before passing it in.

Sample weights
--------------

:func:`~kivi.lifetime.stage_table` takes ``weights``, such as the inverse of a sampling rate. The counts, exposures and losses become sums of weights, so integer weights give the same table as repeated rows. The other functions work on single accounts or states and take no weights; to weight the loans of a vintage table by amount, pass ``weights`` to :func:`~kivi.vintage.vintage_table`.

Limits
------

- A transition matrix gives a term structure only as far as the movements are a time-homogeneous Markov chain: the same probabilities in every period, whatever the age of a loan or the time it has spent in its state.
- IFRS 9 asks for an unbiased, probability-weighted amount that reflects forecasts of economic conditions. These functions compute the ECL of one set of PDs, LGDs and EADs: compute it for each economic scenario and weight the results with the probabilities of the scenarios.
- Early repayment enters only through the EAD schedule and through the vintage curves, which count repaid loans as not bad.
- For revolving facilities, such as credit cards, IFRS 9 measures the expected credit losses over the period in which the lender is exposed to credit risk, which can be longer than the contractual period (paragraphs B5.5.39 and B5.5.40). Choose the periods and the EAD schedule, including drawings of undrawn limits, for that period.
- :func:`~kivi.lifetime.significant_increase` does not apply the exemption for exposures with low credit risk at the reporting date (paragraph 5.5.10); combine its flags with a low-risk criterion of your own.
- :func:`~kivi.lifetime.assign_stages` looks at one reporting date. Probation periods, during which an exposure that no longer meets a criterion stays in its stage, need the history of each exposure and are not applied, and neither are the rules for assets that are credit-impaired when they are purchased or originated.

Errors and warnings
-------------------

Published transition matrices are rounded, so their rows rarely add up to 1. :func:`~kivi.lifetime.transition_pd` allows a rounding error of 1e-9 only, and asks to divide each row by its sum:

>>> states = ["A", "B", "D"]
>>> rounded = pd.DataFrame(
...     [[0.9, 0.08, 0.019], [0.1, 0.7, 0.2], [0.0, 0.0, 1.0]], index=states, columns=states
... )
>>> transition_pd(rounded, horizons=12)
Traceback (most recent call last):
...
ValueError: the rows of matrix must add up to 1, got 0.999 for state 'A'; divide each row by its sum
>>> transition_pd(rounded.div(rounded.sum(axis=1), axis=0), horizons=[1, 12])
horizon        1         12
state
A        0.019019  0.478693
B        0.200000  0.773781
D        1.000000  1.000000

``relative`` is a ratio, not an increase: for a PD that rose by more than 50%, pass 1.5 rather than 0.5.

>>> significant_increase([0.03], [0.01], relative=0.5, absolute=None)
Traceback (most recent call last):
...
ValueError: relative must be None or a ratio greater than 1, such as 2 for a PD that doubled, got 0.5

A horizon longer than the term structure raises an error, because leaving out the missing periods would understate the loss:

>>> expected_credit_loss(account_pd.iloc[:, :6], 0.45, ead[:, :6], rate=0.01, horizon=12)
Traceback (most recent call last):
...
ValueError: horizon must be at most the number of periods of probability, 6, got 12
