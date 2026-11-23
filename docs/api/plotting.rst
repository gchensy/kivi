Charts
======

.. module:: kivi.plotting

Draw the result tables of the other modules with matplotlib, one function per table, such as :func:`~kivi.plotting.woe_chart` for a WOE table, :func:`~kivi.plotting.roc_chart` for a ROC table and :func:`~kivi.plotting.monitoring_chart` for a monitoring report. Each function draws on the Axes passed as ``ax``, or on a new figure, and returns the Axes it drew on. matplotlib is an optional dependency: install it with ``pip install "kivi[plot]"``.

See :doc:`/user_guide/plotting` for examples on the bundled data.

.. autosummary::

   woe_chart
   correlation_chart
   roc_chart
   gains_chart
   calibration_chart
   cutoff_chart
   psi_chart
   monitoring_chart
   vintage_chart

.. autofunction:: woe_chart

.. autofunction:: correlation_chart

.. autofunction:: roc_chart

.. autofunction:: gains_chart

.. autofunction:: calibration_chart

.. autofunction:: cutoff_chart

.. autofunction:: psi_chart

.. autofunction:: monitoring_chart

.. autofunction:: vintage_chart

.. testcleanup::

   import matplotlib.pyplot as plt

   plt.close("all")
