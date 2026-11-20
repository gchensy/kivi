Vintage and roll-rate analysis
==============================

Once a scorecard is in use, the loans it approves are monitored as they age. :mod:`kivi.vintage` covers the two standard views of a loan portfolio: vintage curves, which follow the bad rate of each origination cohort by month on book, and roll rates and transition matrices, which show how loans move between delinquency buckets from one month end to the next.

Both work on a loan panel in long format, with one row per loan and month on book. The examples use the synthetic panel of ``make_loan_panel``: 5,000 instalment loans originated from January 2024 to December 2025 and observed at every month end up to January 2026.

>>> import numpy as np
>>> import pandas as pd
>>> from kivi.datasets import make_loan_panel
>>> panel = make_loan_panel(rng=0)
>>> panel.head(3)
   loan_id   cohort    month  month_on_book  term   amount   balance   status
0        1  2024-01  2024-02              1    24  18300.0  17621.56  current
1        1  2024-01  2024-03              2    24  18300.0  16936.33  current
2        1  2024-01  2024-04              3    24  18300.0  16244.25  current
>>> panel["status"].cat.categories.tolist()
['current', '1-29', '30-59', '60-89', '90+']

The status is an ordered categorical of days past due, so a comparison such as ``panel["status"] >= "60-89"`` follows the order of the buckets. :doc:`datasets` describes how the panel is simulated.

Vintage curves
--------------

A loan is *ever bad* by month on book :math:`k` if it was bad at any month on book up to :math:`k`. A vintage table holds, for each cohort and month on book, the share of the cohort's loans that are ever bad by then. ``vintage_table`` takes the loan, the cohort, the month on book and whether the loan is bad on each row, so the bad definition is a condition on the status. Here a loan is bad once it is 90 days past due:

>>> from kivi.vintage import vintage_table
>>> ever_90 = vintage_table(
...     panel["loan_id"], panel["cohort"], panel["month_on_book"], panel["status"] == "90+"
... )
>>> ever_90[[6, 12, 18, 24]].iloc[::3].round(3)
month_on_book     6      12     18    24
cohort
2024-01        0.000  0.068  0.090  0.09
2024-04        0.020  0.034  0.079   NaN
2024-07        0.005  0.051  0.076   NaN
2024-10        0.010  0.025    NaN   NaN
2025-01        0.005  0.034    NaN   NaN
2025-04        0.010    NaN    NaN   NaN
2025-07        0.005    NaN    NaN   NaN
2025-10          NaN    NaN    NaN   NaN

Each row follows one cohort and never decreases from left to right, because a loan that was bad stays counted as bad even if it cures later. The entries a cohort has not reached are NaN, not 0: the 2025-04 cohort has been on book for ten months, so its bad rate at month 12 is not known yet. Reading down a column compares the cohorts at the same age, which separates the quality of new lending from the seasoning of the portfolio. The latest value of each cohort mixes the two, because older cohorts have had more time to go bad.

Every cohort here has about 200 loans, so one bad loan moves its rate by half a percentage point. Compare small differences only after averaging several cohorts.

Bad definitions and amounts
^^^^^^^^^^^^^^^^^^^^^^^^^^^

An earlier bad definition gives an earlier signal. On the ordered status, ``>= "60-89"`` counts the loans that reached 60 days past due or worse:

>>> ever_60 = vintage_table(
...     panel["loan_id"], panel["cohort"], panel["month_on_book"], panel["status"] >= "60-89"
... )
>>> pd.DataFrame({"ever_60": ever_60[6], "ever_90": ever_90[6]}).head(4).round(3)
         ever_60  ever_90
cohort
2024-01    0.021    0.000
2024-02    0.029    0.020
2024-03    0.025    0.015
2024-04    0.030    0.020

To weight each loan by the amount lent, pass ``weights``. The table then reports the share of the cohort's amount lent that went to loans that turned bad:

>>> by_amount = vintage_table(
...     panel["loan_id"],
...     panel["cohort"],
...     panel["month_on_book"],
...     panel["status"] == "90+",
...     weights=panel["amount"],
... )
>>> by_amount[[6, 12, 18, 24]].iloc[::3].round(3)
month_on_book     6      12     18     24
cohort
2024-01        0.000  0.068  0.091  0.091
2024-04        0.026  0.035  0.081    NaN
2024-07        0.009  0.058  0.078    NaN
2024-10        0.010  0.018    NaN    NaN
2025-01        0.003  0.030    NaN    NaN
2025-04        0.018    NaN    NaN    NaN
2025-07        0.007    NaN    NaN    NaN
2025-10          NaN    NaN    NaN    NaN

The weight must be the same on every row of a loan, so pass the amount lent rather than the outstanding balance.

Roll rates
----------

A roll rate is the share of the loans in a delinquency bucket at one month end that are in the next worse bucket at the next month end. To pair each row with the status a month later, shift the status within each loan. The last row of a loan has no next month, because the loan was repaid or the panel ends, so keep only the rows with a next status:

>>> from kivi.vintage import roll_rates, transition_matrix
>>> next_status = panel.groupby("loan_id")["status"].shift(-1)
>>> observed = next_status.notna()
>>> before, after = panel["status"][observed], next_status[observed]
>>> roll_rates(before, after).round(3)
before
current    0.023
1-29       0.287
30-59      0.536
60-89      0.674
Name: roll_rate, dtype: float64

The states are the categories of the status, from the best to the worst. Each month about 2.3% of the current loans miss an instalment, and the share that rolls on rises with the days past due: two thirds of the loans at 60 to 89 days are 90 days past due a month later.

A loan falls at most one bucket further behind per month, so the only way from current to 90+ in four months is to roll every month. If the movements follow a Markov chain, the product of the roll rates is the share of current loans that are 90+ four months later:

>>> print(f"{roll_rates(before, after).prod():.5f}")
0.00236

Balance roll rates weight each loan by its balance at the first month end, as loss forecasts do:

>>> roll_rates(before, after, weights=panel["balance"][observed]).round(3)
before
current    0.023
1-29       0.291
30-59      0.552
60-89      0.719
Name: roll_rate, dtype: float64

Transition matrices
-------------------

``transition_matrix`` shows all the movements, not only the roll to the next bucket. Row ``i``, column ``j`` counts the loans that moved from state ``i`` to state ``j``:

>>> transition_matrix(before, after)
after    current  1-29  30-59  60-89   90+
before
current    49436  1150      0      0     0
1-29         723   290    408      0     0
30-59        127     2     96    260     0
60-89         29     0      3     63   196
90+            0     0      0      0  1489

With ``normalize=True``, each row holds the shares of its loans and sums to 1:

>>> monthly = transition_matrix(before, after, normalize=True)
>>> monthly.round(3)
after    current   1-29  30-59  60-89    90+
before
current    0.977  0.023  0.000  0.000  0.000
1-29       0.509  0.204  0.287  0.000  0.000
30-59      0.262  0.004  0.198  0.536  0.000
60-89      0.100  0.000  0.010  0.216  0.674
90+        0.000  0.000  0.000  0.000  1.000

The diagonal holds the loans that stayed in their bucket, the ``current`` column the cures and the entries just right of the diagonal the roll rates. A loan that reaches 90+ stays there in this data, so the last row is all 90+.

Migration between two dates
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same function compares any two observations of the same entities, such as the rating grades of obligors at two annual reviews or the delinquency buckets of a portfolio at two month ends. Here are the loans on book at the end of both June and December 2025:

>>> june = panel[panel["month"] == "2025-06"].set_index("loan_id")["status"]
>>> december = panel[panel["month"] == "2025-12"].set_index("loan_id")["status"]
>>> both = june.index.intersection(december.index)
>>> six_months = transition_matrix(june[both], december[both], normalize=True)
>>> six_months.round(3)
after    current   1-29  30-59  60-89    90+
before
current    0.954  0.019  0.011  0.007  0.009
1-29       0.775  0.034  0.000  0.000  0.191
30-59      0.211  0.000  0.000  0.000  0.789
60-89      0.190  0.000  0.048  0.000  0.762
90+        0.000  0.000  0.000  0.000  1.000

Missing values raise an error instead of being counted, so first decide what to do with the entities observed only once: drop them, as here, or give them a state of their own, such as ``"closed"``.

If the monthly movements followed a time-homogeneous Markov chain, the sixth power of the monthly matrix would give the six-month matrix. Comparing the shares that end in 90+ shows how far that holds:

>>> projected = np.linalg.matrix_power(monthly.to_numpy(), 6)
>>> pd.DataFrame({"projected": projected[:, -1], "observed": six_months["90+"]}).round(3)
         projected  observed
before
current      0.010     0.009
1-29         0.208     0.191
30-59        0.581     0.789
60-89        0.868     0.762
90+          1.000     1.000

The projection is of the right size but not exact: loans differ in risk, and the chance of missing an instalment depends on the month on book, so the movements are not a time-homogeneous Markov chain. Few loans are 30 to 89 days past due in June, so those rows are also uncertain.
