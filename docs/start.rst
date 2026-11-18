Getting started
===============

Install the development version from a local checkout. Python 3.10 or newer is required.

.. code-block:: bash

   python -m pip install .

Fit bins on the reference sample
--------------------------------

The bundled bank marketing data records whether a customer subscribed to a term deposit. The split below demonstrates two samples; the rows do not represent a chronological validation split.

>>> import pandas as pd
>>> from kivi.datasets import Dataset
>>> from kivi.binning import bin_edges
>>> from kivi.woe import woe_table
>>> from kivi.evaluate import psi_table
>>> data = Dataset.bank_data()
>>> reference, current = data.iloc[:3000], data.iloc[3000:]
>>> edges = bin_edges(reference["age"], bins=5)
>>> before = pd.cut(reference["age"], edges)
>>> after = pd.cut(current["age"], edges)
>>> evidence = woe_table(before, reference["target"])
>>> drift = psi_table(before, after)
>>> int(evidence["event"].sum() + evidence["non_event"].sum())
3000
>>> bool(evidence["iv"].sum() >= 0 and drift["psi"].sum() >= 0)
True

Read the per-bin contributions before interpreting a total. A group with few events can have a large WOE despite representing few customers.
