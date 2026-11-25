Regulatory capital
==================

.. module:: kivi.capital

Compute the regulatory capital of credit exposures under the internal ratings-based approach. :func:`~kivi.capital.asset_correlation` and :func:`~kivi.capital.capital_requirement` evaluate the risk-weight functions of the Basel framework and of China's 2023 capital rules, :func:`~kivi.capital.conditional_pd`, :func:`~kivi.capital.systematic_factor` and :func:`~kivi.capital.vasicek_fit` work with the one-factor model behind them, and :func:`~kivi.capital.most_prudent_pd` bounds the PDs of a portfolio with few defaults.

See :doc:`/user_guide/capital` for examples and the parameters of both jurisdictions.

.. autosummary::

   asset_correlation
   capital_requirement
   conditional_pd
   systematic_factor
   vasicek_fit
   most_prudent_pd

.. autofunction:: asset_correlation

.. autofunction:: capital_requirement

.. autofunction:: conditional_pd

.. autofunction:: systematic_factor

.. autofunction:: vasicek_fit

.. autofunction:: most_prudent_pd
