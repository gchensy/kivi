Example data
============

:mod:`kivi.datasets` provides three kinds of example data: real bank marketing data with one row per customer and a binary target, a synthetic panel of loans with one row per loan and month, and synthetic loan applications with one row per applicant and an approval decision. :func:`~kivi.datasets.load_bank`, :func:`~kivi.datasets.make_loan_panel` and :func:`~kivi.datasets.make_applications` return them.

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

Dataset and attribution
-----------------------

Bank Marketing by S. Moro, P. Rita and P. Cortez (2014) is distributed under CC BY 4.0. The bundled CSV was converted from the legacy workbook. The loader encodes subscription as 1 and non-subscription as 0. See the `dataset record <https://doi.org/10.24432/C5K306>`_ and `license <https://creativecommons.org/licenses/by/4.0/>`_. The loan panel and the applications are synthetic and have no source to attribute.
