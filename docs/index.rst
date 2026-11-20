:html_theme.sidebar_secondary.remove: true

.. rst-class:: kivi-home

Binning, WOE, IV and PSI for risk modeling
==========================================

.. container:: kivi-lead

   kivi is a pandas toolkit for credit risk and financial modeling. It fits bin edges on a development sample, measures how well each bin separates a binary outcome such as default, and tracks how the population shifts once a model is live. Every result is a pandas DataFrame with one row per bin.

.. container:: kivi-actions

   .. button-ref:: getting_started/quickstart
      :ref-type: doc
      :color: primary

   .. button-ref:: api
      :ref-type: doc
      :color: primary
      :outline:

.. code-block:: bash

   python -m pip install kivi

At a glance
-----------

.. container:: kivi-showcase

   .. testcode::

      import pandas as pd
      from kivi.binning import bin_edges
      from kivi.datasets import Dataset
      from kivi.evaluate import psi_table
      from kivi.woe import woe_table

      data = Dataset.bank_data()
      reference, current = data.iloc[:3000], data.iloc[3000:]

      edges = bin_edges(reference["age"], bins=5)
      before = pd.cut(reference["age"], edges)
      after = pd.cut(current["age"], edges)

      evidence = woe_table(before, reference["target"])
      stability = psi_table(before, after)

.. plot::
   :alt: WOE chart for age with the share of customers and the event rate per bin
   :caption: The WOE chart for ``age``, drawn from ``evidence``.

   kp.figure_quickstart_woe()

Built for risk teams
--------------------

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Scorecard development
      :link: user_guide/binning
      :link-type: doc

      Fine and coarse classing, special values and monotonic WOE for logistic regression scorecards.

   .. grid-item-card:: Variable screening
      :link: user_guide/risk_workflow
      :link-type: doc

      Rank candidates by IV and catch leaks before they reach the model.

   .. grid-item-card:: Model monitoring
      :link: user_guide/psi
      :link-type: doc

      Compare live data with the development sample bin by bin and track PSI over time.

How it works
------------

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: kivi-steps

   .. grid-item-card:: Fit edges once

      Call ``bin_edges`` on the reference sample, or a supervised method such as ``monotonic_edges`` that also uses the target. The outer edges are infinite, so later values always fall into a bin.

   .. grid-item-card:: Cut every sample

      Apply the saved edges with ``pandas.cut``. Categorical columns skip this step and are passed as they are.

   .. grid-item-card:: Measure

      ``woe_table`` scores each bin against the target and ``psi_table`` compares two samples. Sum a column for the total.

Documentation
-------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: getting_started/index
      :link-type: doc

      Install kivi and walk through a complete analysis of the bundled bank marketing data.

   .. grid-item-card:: User guide
      :link: user_guide/index
      :link-type: doc

      How bins are fitted and how WOE, IV and PSI are defined, including missing values and smoothing.

   .. grid-item-card:: API reference
      :link: api
      :link-type: doc

      Parameters, return values and examples for every public function.

   .. grid-item-card:: Release notes
      :link: release/index
      :link-type: doc

      What changed in each release and how to migrate code written for 0.2.x.

.. toctree::
   :hidden:

   getting_started/index
   user_guide/index
   api
   release/index
   development/index
