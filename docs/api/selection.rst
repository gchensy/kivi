Variable selection
==================

.. module:: kivi.selection

Select the variables of a scorecard and fit its logistic regression on WOE-encoded columns. :func:`~kivi.selection.correlation_filter`, :func:`~kivi.selection.vif` and :func:`~kivi.selection.vif_filter` check the candidates for collinearity and drop the collinear ones, :func:`~kivi.selection.logistic_regression` and :func:`~kivi.selection.stepwise` fit and select the model, and :func:`~kivi.selection.selection_path` compares the models of a fixed order of variables.

See :doc:`/user_guide/selection` for a selection on the bundled data.

.. autosummary::

   correlation_filter
   vif
   vif_filter
   logistic_regression
   stepwise
   selection_path
   SeparationError

.. autofunction:: correlation_filter

.. autofunction:: vif

.. autofunction:: vif_filter

.. autofunction:: logistic_regression

.. autofunction:: stepwise

.. autofunction:: selection_path

.. autoexception:: SeparationError
