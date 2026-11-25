Regulatory capital
==================

A bank that uses the internal ratings-based (IRB) approach sets its capital requirement for credit risk from its own estimates of the probability of default (PD), the loss given default (LGD), the exposure at default (EAD) and the maturity of each exposure. Capital covers the unexpected loss: the loss of a year in which the economy is as bad as once in a thousand years, minus the expected loss that pricing and provisions already cover. The risk-weight functions that turn the estimates into that requirement are fixed by the Basel framework and by national rules such as China's capital rules of 2023, and both rest on the one-factor model of Vasicek.

The PDs come from the master scale of :doc:`calibration`, and the expected loss that capital leaves out is that of :doc:`strategy`; :doc:`lifetime` turns the same PDs into the loss allowance of IFRS 9. :func:`~kivi.capital.asset_correlation` gives the asset correlation of each asset class and :func:`~kivi.capital.capital_requirement` the capital requirement. :func:`~kivi.capital.conditional_pd` gives the PD of the one-factor model in a scenario of the economy, :func:`~kivi.capital.systematic_factor` the state of the economy that an observed default rate implies, and :func:`~kivi.capital.vasicek_fit` estimates the long-run PD and the correlation from a series of default rates. :func:`~kivi.capital.most_prudent_pd` bounds the PDs of a portfolio with too few defaults to calibrate them.

Definitions
-----------

In the one-factor model, the assets of a borrower are :math:`\sqrt{\rho}\,Z + \sqrt{1 - \rho}\,\varepsilon`, with a systematic factor :math:`Z` that all borrowers share, an idiosyncratic factor :math:`\varepsilon`, both standard normal, and the asset correlation :math:`\rho`. The borrower defaults when its assets fall below :math:`\Phi^{-1}(\mathrm{PD})`, where :math:`\Phi` is the standard normal distribution function. Given the factor :math:`z`, defaults are independent and occur with the conditional PD

.. math::

   \mathrm{PD}(z) = \Phi\!\left( \frac{\Phi^{-1}(\mathrm{PD}) - \sqrt{\rho}\,z}{\sqrt{1 - \rho}} \right).

A positive :math:`z` stands for an economy better than on average. In a large portfolio the default rate equals the conditional PD, so the loss at the :math:`q` quantile of the economy is that of :math:`z = -\Phi^{-1}(q)`. The IRB approach takes :math:`q = 0.999` and sets the capital requirement :math:`K`, as a share of the EAD, to that loss minus the expected loss:

.. math::

   K = \mathrm{LGD} \left[ \Phi\!\left(
   \frac{\Phi^{-1}(\mathrm{PD}) + \sqrt{R}\,\Phi^{-1}(0.999)}{\sqrt{1 - R}}
   \right) - \mathrm{PD} \right] \frac{1 + (M - 2.5)\,b}{1 - 1.5\,b},
   \qquad b = (0.11852 - 0.05478 \ln \mathrm{PD})^2.

The asset correlation :math:`R` depends on the asset class, and for corporate exposures falls from 0.24 to 0.12 as the PD rises:

.. math::

   w = \frac{1 - e^{-50\,\mathrm{PD}}}{1 - e^{-50}}, \qquad R = 0.12\,w + 0.24\,(1 - w).

The last factor is the maturity adjustment, with the effective maturity :math:`M` in years; retail exposures have none. The risk-weighted assets are :math:`\mathrm{RWA} = 12.5\,K\,\mathrm{EAD}`, and :math:`12.5\,K` is the risk weight: the capital ratio of 8% applied to the RWA gives back :math:`K\,\mathrm{EAD}`.

A worked example
----------------

A corporate loan with a PD of 1%, an LGD of 45% and an effective maturity of 2.5 years:

>>> from kivi.capital import asset_correlation, capital_requirement, conditional_pd
>>> r = asset_correlation(0.01, asset_class="corporate")
>>> stressed = conditional_pd(0.01, correlation=r, level=0.999)
>>> k = capital_requirement(0.01, 0.45, asset_class="corporate")
>>> print(f"R = {r:.6f}, stressed PD = {stressed:.4%}, K = {k:.6f}, risk weight = {12.5 * k:.2%}")
R = 0.192784, stressed PD = 14.0273%, K = 0.073853, risk weight = 92.32%

With :math:`w = (1 - e^{-0.5}) / (1 - e^{-50}) = 0.3935`, the correlation is 0.12 × 0.3935 + 0.24 × 0.6065 = 0.1928. In the worst year of a thousand, 14.03% of such loans default instead of 1%, so the loss beyond the expected one is 0.45 × (0.1403 − 0.01) = 0.0586 of the EAD. At 2.5 years the maturity adjustment is :math:`1 / (1 - 1.5\,b) = 1.2598` with :math:`b = 0.1375`, which gives K = 0.0739 and a risk weight of 92.32%, the value of Annex 5 of Basel II. An exposure of 1,000,000 has an RWA of 923,168 and needs 73,853 of capital.

Asset classes
-------------

The ``asset_class`` argument selects the correlation and whether the maturity adjustment applies:

- ``"corporate"``: corporate, sovereign and bank exposures, with the correlation above and the maturity adjustment. ``turnover`` lowers the correlation of small and medium-sized enterprises (SMEs) by up to 0.04, and ``financial=True`` multiplies it by 1.25 for large or unregulated financial institutions.
- ``"hvcre"``: high-volatility commercial real estate, with 0.30 instead of 0.24 and the maturity adjustment.
- ``"mortgage"``: residential mortgages, with :math:`R = 0.15`.
- ``"revolving"``: qualifying revolving retail exposures, such as credit cards, with :math:`R = 0.04`.
- ``"other_retail"``: other retail exposures, with weights :math:`e^{-35\,\mathrm{PD}}` and bounds 0.03 and 0.16 instead of 0.12 and 0.24.

At a PD of 1% and an LGD of 45%, the classes differ by a factor of seven:

>>> import pandas as pd
>>> classes = {
...     "corporate": {"asset_class": "corporate"},
...     "SME, S = 5": {"asset_class": "corporate", "turnover": 5.0},
...     "financial": {"asset_class": "corporate", "financial": True},
...     "hvcre": {"asset_class": "hvcre"},
...     "mortgage": {"asset_class": "mortgage"},
...     "revolving": {"asset_class": "revolving"},
...     "other_retail": {"asset_class": "other_retail"},
... }
>>> weights = pd.DataFrame(
...     {
...         name: {
...             "correlation": asset_correlation(0.01, **options),
...             "risk_weight": 12.5 * capital_requirement(0.01, 0.45, **options),
...         }
...         for name, options in classes.items()
...     }
... ).T
>>> weights.round(4)
              correlation  risk_weight
corporate          0.1928       0.9232
SME, S = 5         0.1528       0.7239
financial          0.2410       1.1795
hvcre              0.2292       1.1150
mortgage           0.1500       0.5640
revolving          0.0400       0.1722
other_retail       0.1216       0.4577

A turnover :math:`S` within the range :math:`[l, u]` of ``turnover_range`` lowers the corporate correlation by :math:`0.04\,(1 - (S - l) / (u - l))`: by the full 0.04 at or below the lower bound, and not at all from the upper bound on. Pass ``numpy.inf`` for the borrowers of a portfolio that are not SMEs. With ``financial=True``, a turnover below the upper bound raises an error, because the printed formula for financial institutions has no turnover term in either jurisdiction.

Parameters of Basel and China
-----------------------------

The risk-weight functions are the same in the current Basel framework (the CRE chapters in force since 1 January 2023) and in China's *Rules on capital management of commercial banks* (NFRA order 2023 No. 4). The parameters around them differ, and kivi builds in neither set: the keywords take the Basel values by default, and the other rules are applied to the inputs before the call, such as a PD floor with ``numpy.maximum``.

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * - Parameter
     - Basel framework
     - China, 2023 rules
   * - Scaling factor 1.06
     - Not applied; only Basel II of 2006 multiplied the RWA by 1.06, which the reforms of December 2017 removed (d424, footnote 3 of the table in the introduction). ``scaling`` defaults to 1.0.
     - Not applied: RWA = K × 12.5 × EAD (annex 6, part I (4)).
   * - SME adjustment
     - :math:`-0.04\,(1 - (S - 5) / 45)`, with :math:`S` the annual sales in millions of euros, clipped to [5, 50] (CRE31.8); supervisors may allow total assets instead (CRE31.9). ``turnover_range=(5.0, 50.0)``.
     - :math:`-0.04\,(1 - (S - 3) / 27)`, with :math:`S` the average operating revenue of the last three years in tens of millions of yuan, and 3 below 3 (annex 6); an SME has at most 300 million yuan (annex 4, part II (3)). ``turnover_range=(3.0, 30.0)``.
   * - Financial institutions, × 1.25
     - Regulated financial institutions with total assets of at least USD 100 billion, and unregulated financial institutions of any size (CRE31.7).
     - Every financial institution except non-systemically important banks, which use the corporate correlation, with no threshold on assets (annex 6, part I (1) 2).
   * - High-volatility commercial real estate
     - :math:`R = 0.12\,w + 0.30\,(1 - w)` (CRE31.11).
     - The same (annex 8, part I (2)).
   * - PD floors
     - 0.05% for all exposures except sovereigns (CRE32.4); for retail 0.05%, and 0.1% for revolving QRRE (CRE32.58). Basel II had 0.03%.
     - 0.05% for corporate exposures and financial institutions, none for sovereigns (article 91); for retail 0.05%, and 0.1% for general revolving retail. The rules of 2012 had 0.03%.
   * - LGD floors of the advanced approach
     - Unsecured corporate 25%; secured by financial collateral 0%, by receivables or real estate 10%, by other physical collateral 15% (CRE32.16–32.17). Residential mortgages 5%, QRRE 50%, other unsecured retail 30% (CRE32.58–32.59).
     - The same, except 10% for residential mortgages (article 92).
   * - LGDs of the foundation approach
     - Senior unsecured exposures to sovereigns, banks, securities firms and other financial institutions 45%, to other corporates 40%; subordinated exposures 75% (CRE32.6–32.7).
     - The same: 45%, 40% and 75%.
   * - Effective maturity
     - 2.5 years in the foundation approach, 0.5 years for repo-style transactions (CRE32.44); otherwise clipped to [1, 5] years (CRE32.46), with exceptions down to one day for short-term exposures (CRE32.51–32.54); 2.5 years when no other rule applies (CRE32.55).
     - 2.5 years in the foundation approach, 0.5 years for repo-style transactions; in the advanced approach the larger of the internal estimate and one year, at most five years, and 2.5 years allowed for SMEs (article 94).
   * - Scope of the advanced approach
     - Not for corporates with consolidated revenues above 500 million euros, nor for banks and other financial institutions (CRE30.34–30.35).
     - Not for financial institutions, nor for corporates with annual revenues above 3 billion yuan (article 89).
   * - Output floor
     - The RWA of the IRB approach may not fall below 72.5% of those of the standardized approach.
     - RWA = max(RWA of the IRB approach, 72.5% × RWA of the standardized approach) (annex 21, part IV).

``maturity`` is not clipped to [1, 5] years, because of the short-term exceptions, so clip it with ``numpy.clip`` where the rules ask for it. The output floor applies to a whole portfolio and needs the RWA of the standardized approach, so kivi leaves it out.

The SME adjustment of China's rules, with ``turnover`` in tens of millions of yuan:

>>> asset_correlation(
...     0.01, asset_class="corporate", turnover=[2.0, 10.0, 30.0], turnover_range=(3.0, 30.0)
... ).round(4)
array([0.1528, 0.1632, 0.1928])

Risk-weighted assets of a portfolio
-----------------------------------

The value-by-value functions take arrays and Series, so one call per asset class covers a portfolio. A small book of corporate loans, mortgages and credit cards, with the PD floor of 0.05% applied first:

>>> import numpy as np
>>> book = pd.DataFrame(
...     {
...         "asset_class": ["corporate", "corporate", "corporate", "mortgage", "mortgage", "revolving"],
...         "probability": [0.0003, 0.004, 0.02, 0.006, 0.03, 0.015],
...         "lgd": [0.45, 0.40, 0.45, 0.15, 0.20, 0.80],
...         "ead": [5_000_000.0, 2_000_000.0, 800_000.0, 300_000.0, 250_000.0, 20_000.0],
...         "maturity": [1.0, 2.5, 4.0, np.nan, np.nan, np.nan],
...     }
... )
>>> floored = np.maximum(book["probability"], 0.0005)
>>> requirement = pd.Series(np.nan, index=book.index)
>>> for asset_class, rows in book.groupby("asset_class"):
...     options = {"maturity": rows["maturity"]} if asset_class == "corporate" else {}
...     requirement[rows.index] = capital_requirement(
...         floored[rows.index], rows["lgd"], asset_class=asset_class, **options
...     )
>>> book["risk_weight"] = 12.5 * requirement
>>> book["rwa"] = book["risk_weight"] * book["ead"]
>>> book[["asset_class", "risk_weight", "rwa"]].round({"risk_weight": 4, "rwa": 0})
  asset_class  risk_weight        rwa
0   corporate       0.1122   560871.0
1   corporate       0.5575  1114981.0
2   corporate       1.3394  1071502.0
3    mortgage       0.1328    39848.0
4    mortgage       0.4977   124431.0
5   revolving       0.4159     8318.0
>>> book.groupby("asset_class")["rwa"].sum().round(0)
asset_class
corporate    2747354.0
mortgage      164279.0
revolving       8318.0
Name: rwa, dtype: float64

The first loan has a PD of 0.03%, below the floor, so it is weighted at 0.05%, and its maturity of one year gives it a risk weight of 11.22% instead of the 19.65% of 2.5 years. The capital requirement is 8% of the RWA, 233,596 in total. :func:`~kivi.strategy.expected_loss` gives the expected loss of the same exposures, which the IRB approach compares with the provisions.

Stressed PDs and the systematic factor
--------------------------------------

The conditional PD of the one-factor model turns a long-run PD into the PD of a scenario of the economy. A factor of 0 is the median economy, which is better than the average one, and the 0.1% worst economy has :math:`z = -\Phi^{-1}(0.999) = -3.09`:

>>> conditional_pd(0.02, correlation=0.12, factor=[1.0, 0.0, -1.0, -3.0902]).round(4)
array([0.0053, 0.0143, 0.0344, 0.1473])
>>> print(f"{conditional_pd(0.02, correlation=0.12, level=0.999):.4f}")
0.1473

``level=0.999`` gives the PD that the capital formula uses. The factor of each year follows from its default rate with :func:`~kivi.capital.systematic_factor`, the inverse of :func:`~kivi.capital.conditional_pd`, and :func:`~kivi.capital.vasicek_fit` estimates the long-run PD and the correlation from the mean and the variance of the default rates. Twenty years of a portfolio of 5,000 borrowers, simulated with a long-run PD of 2% and a correlation of 0.1:

>>> from kivi.capital import systematic_factor, vasicek_fit
>>> rng = np.random.default_rng(0)
>>> factor = rng.standard_normal(20)
>>> defaults = rng.binomial(5000, conditional_pd(0.02, correlation=0.1, factor=factor))
>>> default_rate = defaults / 5000
>>> probability, correlation = vasicek_fit(default_rate)
>>> print(f"PD = {probability:.4f}, correlation = {correlation:.4f}")
PD = 0.0217, correlation = 0.0927
>>> implied = systematic_factor(default_rate, probability, correlation=correlation)
>>> print(f"{np.corrcoef(implied, factor)[0, 1]:.3f}")
0.989

Twenty years estimate the correlation only roughly: repeated simulations of this portfolio give estimates that scatter by about a third of the true correlation. The implied factors follow the simulated ones closely, and a scenario for the factor, such as a recession of :math:`z = -2`, gives the stressed PD of each grade with :func:`~kivi.capital.conditional_pd`.

Most prudent PDs
----------------

A portfolio of banks, sovereigns or large corporates has so few defaults that its default rates cannot calibrate a PD. Pluto and Tasche bound the PD of each grade by the largest PD that the defaults observed in the grade and in all riskier grades do not reject at a confidence level, assuming only that the PD does not fall from one grade to the next riskier one. With :math:`n_k` obligors and :math:`d_k` defaults in grade :math:`k` and the riskier grades together, the bound :math:`p_k` solves

.. math::

   P(D \le d_k) = 1 - q, \qquad D \sim \mathrm{Bin}(n_k, p_k),

at the level :math:`q`, so that without defaults :math:`p_k = 1 - (1 - q)^{1/n_k}`. The example of their paper has three grades of 100, 400 and 300 obligors:

>>> from kivi.capital import most_prudent_pd
>>> grades = pd.Index(["A", "B", "C"], name="grade")
>>> total = pd.Series([100, 400, 300], index=grades)
>>> pd.DataFrame(
...     {
...         "no_defaults": most_prudent_pd([0, 0, 0], total, level=0.9),
...         "defaults_0_2_1": most_prudent_pd(pd.Series([0, 2, 1], index=grades), total, level=0.9),
...         "correlated": most_prudent_pd([0, 0, 0], total, level=0.9, correlation=0.12),
...     }
... ).mul(100).round(2)
       no_defaults  defaults_0_2_1  correlated
grade
A             0.29            0.83        0.86
B             0.33            0.95        0.96
C             0.76            1.29        1.89

Without defaults, grade A pools all 800 obligors: 1 − 0.1^(1/800) = 0.29%. Two defaults in B and one in C raise the bounds of every grade from C down, because they belong to the pooled grades. With the correlation of 0.12 that the paper takes from the Basel corporate formula, a year without defaults is less informative, since a good year makes defaults rare for all borrowers at once. The numbers are those of tables 1 to 9 of the paper at 90%.

Reading the result
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Result
     - Meaning
   * - ``correlation``
     - The asset correlation :math:`R` of each exposure. A higher correlation means that defaults cluster in bad years, which raises the capital requirement.
   * - ``capital_requirement``
     - :math:`K` as a share of the EAD. Multiply by 12.5 for the risk weight, by 12.5 × EAD for the RWA and by the EAD for the capital. It rises with the PD up to a PD of about 30% to 40%, beyond which the expected loss takes over the loss of the worst year, and in proportion to the LGD; for ``"corporate"`` and ``"hvcre"`` it also rises with the maturity.
   * - ``conditional_pd``
     - The PD in the scenario of the factor or the level. It lies below the long-run PD at a factor of 0, the median economy, and above it in bad enough years.
   * - ``factor``
     - The state of the economy that a default rate implies, in standard deviations; negative in years worse than on average.
   * - ``vasicek_fit``
     - The long-run PD and the asset correlation that the default rates imply, for comparison with the regulatory correlation or as inputs of a stress test.
   * - ``most_prudent_pd``
     - An upper bound of the PD of each grade at the level. Compare it with the PD of the grade: a PD above the bound is prudent enough for the data, and the bound shrinks as defaults stay absent and obligors accumulate.

Missing values
--------------

The value-by-value functions give a missing result for a missing PD, LGD, default rate, factor or level, so a portfolio with gaps keeps its shape. A missing ``turnover``, ``maturity`` or ``correlation`` raises an error instead, because it decides the formula: pass ``numpy.inf`` as the turnover of a borrower that is not an SME, and the maturity that the rules prescribe where it is unknown. Sums such as the RWA of a portfolio skip missing values in pandas, so check them before adding up. :func:`~kivi.capital.vasicek_fit` and :func:`~kivi.capital.most_prudent_pd` fit their inputs as a whole and raise on missing values.

Limits
------

- The functions compute the risk-weight functions of exposures that are not in default. A defaulted exposure needs :math:`K = \max(0, \mathrm{LGD} - \mathrm{ELBE})` with the best estimate of its expected loss (CRE31.3; China, annex 6, part II), which the bank computes directly.
- Credit risk mitigation, the substitution approach for guarantees, the EAD and its credit conversion factors, purchased receivables, specialized lending slotting and the output floor are outside kivi.
- :func:`~kivi.capital.vasicek_fit` assumes a portfolio large enough for its default rate to equal the conditional PD. The binomial noise of a small portfolio adds to the variance, and a short series estimates the correlation with a large error.
- :func:`~kivi.capital.most_prudent_pd` gives the bounds of one period. Over several years with correlated factors, the bounds of Pluto and Tasche need simulation, which kivi leaves out.

Errors and warnings
-------------------

- **A PD at or below about 2.927e-6 for the corporate classes.** The maturity adjustment divides by :math:`1 - 1.5\,b`, which reaches 0 there, and below a PD of about 1e-5 the risk weight rises again as the PD falls: 2.8% at 1e-5, 46% at 3e-6 and 145% at 2.95e-6. Earlier versions of the framework set a negative requirement of a sovereign to zero; the current text has no such rule, so kivi raises an error. Apply the PD floor first, such as ``numpy.maximum(probability, 0.0005)``; sovereigns, which have no floor, still need a PD above the bound.
- **A PD of 1.** The error names the formula of defaulted exposures above.
- **financial=True with a turnover below the upper bound.** The formula for financial institutions has no SME term in either jurisdiction, so the two adjustments do not combine.
- **turnover, financial or maturity for a class that does not use them.** ``turnover`` and ``financial`` belong to ``"corporate"`` only, and ``maturity`` to ``"corporate"`` and ``"hvcre"``.
