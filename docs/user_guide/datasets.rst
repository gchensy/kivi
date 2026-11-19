Example data
============

``Dataset.bank_data()`` loads 4,521 customers from the Bank Marketing data set. It has sixteen features and an integer ``target``: 1 means that the customer subscribed to a term deposit.

>>> from kivi.datasets import Dataset
>>> data = Dataset.bank_data()
>>> list(data.columns)  # doctest: +NORMALIZE_WHITESPACE
['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous',
 'poutcome', 'target']
>>> data["target"].value_counts().to_dict()
{0: 4000, 1: 521}

The examples in this documentation split the rows by position only to demonstrate two samples; the split is not chronological.

Dataset and attribution
-----------------------

Bank Marketing by S. Moro, P. Rita and P. Cortez (2014) is distributed under CC BY 4.0. The bundled CSV was converted from the legacy workbook. The loader encodes subscription as 1 and non-subscription as 0. See the `dataset record <https://doi.org/10.24432/C5K306>`_ and `license <https://creativecommons.org/licenses/by/4.0/>`_.
