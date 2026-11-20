Example data
============

:mod:`kivi.datasets` provides two kinds of example data: real bank marketing data with one row per customer and a binary target, and a synthetic panel of loans with one row per loan and month.

Bank marketing data
-------------------

``load_bank()`` loads 4,521 customers from the Bank Marketing data set. It has sixteen features and an integer ``target``: 1 means that the customer subscribed to a term deposit.

>>> from kivi.datasets import load_bank
>>> data = load_bank()
>>> list(data.columns)  # doctest: +NORMALIZE_WHITESPACE
['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous',
 'poutcome', 'target']
>>> data["target"].value_counts().to_dict()
{0: 4000, 1: 521}

The examples in this documentation split the rows by position only to demonstrate two samples; the split is not chronological. Earlier examples call ``Dataset.bank_data()``, which returns the same data and keeps working.

Synthetic loan panel
--------------------

``make_loan_panel`` simulates the monthly history of a portfolio of instalment loans, for the vintage and roll-rate analysis in :doc:`vintage`. The data is generated, not taken from a real portfolio, and the same ``rng`` seed always gives the same panel:

>>> from kivi.datasets import make_loan_panel
>>> panel = make_loan_panel(n_loans=5000, n_months=24, rng=0)
>>> panel.shape
(59272, 8)
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

Each loan is repaid in equal monthly instalments at 1% interest a month. Its status counts the instalments past due: one is ``1-29``, two ``30-59``, three ``60-89`` and four or more ``90+``. Every month:

* a current loan misses its instalment with a probability that peaks at its sixth month on book and then declines, scaled by a risk factor of the loan and one of its cohort;
* a delinquent loan pays all its arrears and becomes current, pays one instalment and keeps its status, or pays nothing and rolls to the next bucket, and the later the bucket, the less likely it cures and the more likely it rolls. After the term no instalment falls due, so paying one moves the loan up a bucket;
* a ``90+`` loan pays nothing more, so its status and balance stay the same.

A loan leaves the panel in the month it pays its last instalment, which can come after its term if it was behind. Loans are not charged off or prepaid.

Dataset and attribution
-----------------------

Bank Marketing by S. Moro, P. Rita and P. Cortez (2014) is distributed under CC BY 4.0. The bundled CSV was converted from the legacy workbook. The loader encodes subscription as 1 and non-subscription as 0. See the `dataset record <https://doi.org/10.24432/C5K306>`_ and `license <https://creativecommons.org/licenses/by/4.0/>`_. The loan panel is synthetic and has no source to attribute.
