Variable selection
==================

.. module:: kivi.selection

Select the variables of a scorecard and fit its logistic regression on WOE-encoded columns. :func:`~kivi.selection.correlation_filter` and :func:`~kivi.selection.vif` check the candidates for collinearity, and :func:`~kivi.selection.logistic_regression` and :func:`~kivi.selection.stepwise` fit and select the model.

See :doc:`/user_guide/selection` for a selection on the bundled data.

.. autosummary::

   correlation_filter
   vif
   logistic_regression
   stepwise
   SeparationError

.. autofunction:: correlation_filter

.. autofunction:: vif

.. autofunction:: logistic_regression

.. autofunction:: stepwise

.. autoexception:: SeparationError
