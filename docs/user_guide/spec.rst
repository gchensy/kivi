Binning specifications
======================

A scorecard must group and encode every variable in the same way in development, validation and production. ``fit_spec`` bins all the columns of a development sample in one call and measures the WOE of every group. The resulting ``BinningSpec`` groups and encodes any other sample with the same edges and WOE values. It can be saved as JSON and loaded again in production without changing either.

Each column is fitted with the functions of :doc:`binning` and :doc:`woe_iv`, so a column of the specification has exactly the edges and table that those steps give it alone.

Screen the candidates
---------------------

With the defaults, every numeric column gets five quantile bins and every other column keeps its labels. The information value of each column gives a first ranking. A warning names each column whose tied quantiles leave fewer bins than requested:

>>> import dataclasses
>>> import warnings
>>> import numpy as np
>>> import pandas as pd
>>> from kivi.datasets import Dataset
>>> from kivi.spec import BinningSpec, fit_spec
>>> data = Dataset.bank_data()
>>> development, monitoring = data.iloc[:3000], data.iloc[3000:]
>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter("always")
...     quick = fit_spec(development.drop(columns="target"), development["target"])
>>> [str(warning.message).split(":")[0] for warning in caught]
["column 'campaign'", "column 'pdays'", "column 'previous'"]
>>> iv = pd.Series({column: table["iv"].sum() for column, table in quick.tables.items()})
>>> iv.sort_values(ascending=False).round(3).head(5)
duration    1.594
poutcome    0.423
month       0.334
contact     0.215
job         0.137
dtype: float64

``quick.tables`` maps each column to its ``woe_table`` result. The columns of ``frame`` are fitted unless ``columns`` names some of them. Leave the target out: ``fit_spec`` raises an error if a column has the name of the ``target`` series.

Which columns are binned depends on their dtype. Numeric columns, apart from booleans, are binned; columns of any other dtype, such as strings, categories or booleans, keep their labels as groups. To keep integer codes as labels, convert the column to the ``category`` dtype before fitting.

Choose the binning of each column
---------------------------------

``method``, ``bins``, ``special`` and ``options`` can each be a dict by column. Here ``balance`` is merged until its WOE is monotonic, ``pdays`` is split by a decision tree with the ``-1`` code ("not contacted before") in its own group, and ``age`` gets policy bands. A list of edges in ``bins`` is used as given; it must run from ``-inf`` to ``inf``, so that every future value falls into a bin:

>>> shortlist = ["balance", "pdays", "age", "poutcome", "contact", "housing", "job"]
>>> spec = fit_spec(
...     development,
...     development["target"],
...     columns=shortlist,
...     method={"balance": "monotonic", "pdays": "tree"},
...     bins={"balance": 4, "age": [-np.inf, 25, 35, 45, 55, 65, np.inf]},
...     special={"pdays": [-1]},
...     options={"pdays": {"min_bin_share": 0.1}},
... )
>>> spec
BinningSpec(columns=['balance', 'pdays', 'age', 'poutcome', 'contact', 'housing', 'job'])
>>> spec.edges["pdays"]
array([-inf, 100., 176., 192., 356.,  inf])
>>> spec.tables["pdays"][["total", "event_rate", "woe", "iv"]].round(3)
                total  event_rate    woe     iv
bin
-1               2475       0.092 -0.257  0.049
(-inf, 100.0]      95       0.495  2.010  0.235
(100.0, 176.0]    121       0.149  0.287  0.004
(176.0, 192.0]     53       0.509  2.069  0.140
(192.0, 356.0]    188       0.080 -0.414  0.009
(356.0, inf]       68       0.191  0.588  0.010

The methods are ``"quantile"`` and ``"uniform"`` for ``bin_edges`` and ``"chimerge"``, ``"tree"`` and ``"monotonic"`` for the supervised functions. ``bins`` is passed as ``bins`` to ``bin_edges`` and as ``max_bins`` to the others, and ``options`` holds their remaining arguments, such as ``alpha`` or ``min_bin_share``. ``"monotonic"`` uses the ``smoothing`` of the specification, so its trend holds in the tables. ``weights`` passes sample weights to every binning function and to ``woe_table``.

.. note::

   ``fit_spec`` never re-bins a column on its own. The legacy ``WOEBatch`` lowered the number of bins until the WOE was monotonic; here the method and its options decide, and a column that should be monotonic is fitted with ``"monotonic"``.

Group and encode a sample
-------------------------

``cut`` groups every column of a sample, and ``woe_transform`` replaces each group with its WOE. Both return the columns of the specification in its order, with the index of the sample:

>>> spec.cut(monitoring)[["balance", "pdays", "age", "job"]].head(3)
             balance pdays           age         job
3000  (125.7, 895.7]    -1  (35.0, 45.0]      admin.
3001   (-inf, 68.75]    -1  (35.0, 45.0]  management
3002    (895.7, inf]    -1  (25.0, 35.0]    services
>>> X_mon = spec.woe_transform(monitoring)
>>> X_mon.head(3).round(3)
      balance  pdays    age  poutcome  contact  housing    job
3000   -0.045 -0.257 -0.166    -0.257    0.206    0.333  0.052
3001   -0.587 -0.257 -0.166    -0.257   -0.917   -0.327  0.271
3002    0.358 -0.257 -0.032    -0.257    0.241   -0.327 -0.339

A value whose group is not in its column's table, such as a new category, raises an error by default, so it never turns into a silent NaN:

>>> new = monitoring.head(2).assign(job="astronaut")
>>> spec.woe_transform(new)
Traceback (most recent call last):
...
ValueError: column 'job': 1 group(s) are not in table, such as 'astronaut'; ...

The ``unseen`` argument of ``fit_spec`` sets a WOE for such values, for all columns or by column. The same holds for a special code or an interval that has no development observations, and for missing values when the development sample had none. A specification is immutable, so change the policy of a fitted one with ``dataclasses.replace``:

>>> lenient = dataclasses.replace(spec, unseen={**spec.unseen, "job": 0.0})
>>> lenient.woe_transform(new)["job"].tolist()
[0.0, 0.0]

Keep the model variables
------------------------

After variable selection, keep only the variables of the model before saving the specification for production:

>>> from kivi.selection import stepwise
>>> selected = stepwise(spec.woe_transform(development), development["target"])
>>> production = spec.select(selected)
>>> production
BinningSpec(columns=['balance', 'pdays', 'poutcome', 'contact', 'housing', 'job'])

``production.tables`` can be passed to ``points_table`` as it is; see :doc:`scorecard`.

Save and load a specification
-----------------------------

``to_json`` writes the specification as JSON text, and ``from_json`` reads it back into an equal specification that groups and encodes every sample identically:

>>> text = production.to_json(indent=2)
>>> loaded = BinningSpec.from_json(text)
>>> loaded == production
True
>>> loaded.woe_transform(monitoring).equals(production.woe_transform(monitoring))
True

The same specification always gives the same text, so it can be kept under version control and compared. Labels are written as they are, including non-ASCII ones, so save the text as UTF-8:

.. code-block:: python

   from pathlib import Path

   Path("scorecard_spec.json").write_text(production.to_json(indent=2), encoding="utf-8")
   production = BinningSpec.from_json(Path("scorecard_spec.json").read_text(encoding="utf-8"))

``to_dict`` gives the same content as a dict. Each column is one entry of ``variables``:

>>> entry = production.to_dict()["variables"][1]
>>> entry["name"], entry["kind"], entry["unseen"], entry["special"]
('pdays', 'numeric', 'raise', [-1])
>>> entry["edges"]
[{'float': '-inf'}, 100.0, 176.0, 192.0, 356.0, {'float': 'inf'}]
>>> entry["table"][1]["bin"]
{'interval': [{'float': '-inf'}, 100.0], 'closed': 'right'}
>>> entry["table"][1]["woe"]
2.0098134623334465

The format
----------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Content
   * - ``format``
     - Always ``"kivi.BinningSpec"``.
   * - ``format_version``
     - Version of the format, currently 1. A file with a newer version raises an error that asks for a newer kivi, instead of being read in part.
   * - ``kivi_version``
     - Version of kivi that fitted the specification.
   * - ``smoothing``
     - Smoothing of the WOE tables.
   * - ``variables``
     - One entry per column, in order, with ``name``, ``kind`` (``"numeric"`` or ``"categorical"``), ``unseen`` (``"raise"`` or a number) and ``table``, plus ``edges`` and ``special`` for numeric columns.
   * - ``table``
     - One entry per group with its label ``bin`` and the columns of ``woe_table``.

JSON has no infinite numbers and does not keep the types of labels apart in every language, so labels and edges are written as follows:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value
     - JSON
   * - Strings, integers and booleans
     - As they are: ``"yes"``, ``1``, ``true``.
   * - Floats
     - With a decimal point or an exponent, such as ``1.0``, so that they stay apart from integers.
   * - Infinite numbers
     - ``{"float": "inf"}`` and ``{"float": "-inf"}``.
   * - Intervals
     - ``{"interval": [left, right], "closed": "right"}``.
   * - The missing group
     - ``null``.

Labels of categorical columns must be strings, integers, floats, booleans or intervals; ``fit_spec`` rejects other labels, such as dates. ``from_json`` checks every value and rejects ``NaN`` or ``Infinity`` in the text, so a damaged file fails with a message instead of encoding wrong values.

.. note::

   The interval labels of a table are those of ``pandas.cut``, which rounds their bounds to three decimals, while ``edges`` keeps the exact values that assign values to bins. Software outside kivi that reads the file should bin with ``edges`` and use ``special`` codes before intervals, as :func:`kivi.binning.cut` does. Intervals are closed on the right, so ``-inf`` falls into no interval; ``woe_transform`` treats it as unseen.
