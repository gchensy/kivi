Lifetime PD and expected credit loss
====================================

.. module:: kivi.lifetime

Turn PD term structures into the expected credit loss and the stages of IFRS 9. :func:`~kivi.lifetime.convert_pd` converts a term structure between cumulative, marginal and conditional PDs, :func:`~kivi.lifetime.transition_pd` projects one from a transition matrix, :func:`~kivi.lifetime.conditional_matrix` shifts a transition matrix to the economy of a scenario, :func:`~kivi.lifetime.matrix_factor` and :func:`~kivi.lifetime.matrix_correlation` estimate its factor and correlation from observed movements, :func:`~kivi.lifetime.survival_table` estimates one from loans that are censored, repaid early or observed only from some age on, :func:`~kivi.lifetime.expected_credit_loss` gives the 12-month and the lifetime expected credit loss of each account, and :func:`~kivi.lifetime.significant_increase`, :func:`~kivi.lifetime.assign_stages` and :func:`~kivi.lifetime.stage_table` allocate the stages and sum the loss allowance of each stage.

See :doc:`/user_guide/lifetime` for examples on the bundled data.

.. autosummary::

   convert_pd
   transition_pd
   conditional_matrix
   matrix_factor
   matrix_correlation
   survival_table
   expected_credit_loss
   significant_increase
   assign_stages
   stage_table

.. autofunction:: convert_pd

.. autofunction:: transition_pd

.. autofunction:: conditional_matrix

.. autofunction:: matrix_factor

.. autofunction:: matrix_correlation

.. autofunction:: survival_table

.. autofunction:: expected_credit_loss

.. autofunction:: significant_increase

.. autofunction:: assign_stages

.. autofunction:: stage_table
