"""Seeded generator of synthetic loan applications with an approval decision."""

import numpy as np
import numpy.typing as npt
import pandas as pd

from .. import _validation

HOUSING = ("own", "mortgage", "rent")
EMPLOYMENT = ("permanent", "temporary", "self_employed")
CUT_OFF = 0.15
MAX_DEBT_TO_INCOME = 0.6
MIN_AGE = 21

_HOUSING_SHARES = [0.25, 0.35, 0.40]
_EMPLOYMENT_SHARES = [0.65, 0.20, 0.15]
_HOUSING_RISK = np.array([-0.35, -0.10, 0.35])
_EMPLOYMENT_RISK = np.array([-0.20, 0.45, 0.25])
_MEDIAN_INCOME = 3200.0
# Weight of the hidden risk factor in the true log odds, and the share of it that the
# incumbent score sees through information that is not in the data.
_HIDDEN_RISK = 0.6
_INCUMBENT_SHARE = 0.5
_INCUMBENT_NOISE = 0.3


def make_applications(
    *, n_applicants: int = 10000, rng: int | np.random.Generator | None = None
) -> pd.DataFrame:
    """Generate synthetic loan applications with an approval decision.

    Every applicant has a few numeric and categorical features, the PD of an
    incumbent scorecard and a true default flag. The incumbent PD and two
    policy rules decide who is approved, so the data show the selection that
    reject inference corrects for. The data is simulated, not drawn from a
    real portfolio.

    Parameters
    ----------
    n_applicants : int, default 10000
        Number of applications.
    rng : {None, int, numpy.random.Generator}, optional
        Seed or generator passed to `numpy.random.default_rng`. The global
        NumPy random state is not used, so the same seed always returns the
        same applications.

    Returns
    -------
    pandas.DataFrame
        One row per application with the columns described in the Notes.

    Raises
    ------
    ValueError
        If `n_applicants` is not a positive integer.

    See Also
    --------
    kivi.reject_inference.fuzzy_augmentation : Infer the outcomes of rejected applicants.
    make_loan_panel : Synthetic monthly history of a loan portfolio.
    load_bank : Real data with a binary target.

    Notes
    -----
    The columns are:

    - ``age``: age in years, from 18 to 75;
    - ``income``: monthly income, rounded to tens;
    - ``debt_to_income``: monthly debt payments over income, rounded to
      hundredths;
    - ``inquiries``: number of credit inquiries in the last six months;
    - ``housing``: a categorical with the categories ``own``, ``mortgage``
      and ``rent``;
    - ``employment``: a categorical with the categories ``permanent``,
      ``temporary`` and ``self_employed``;
    - ``incumbent_pd``: the PD of the scorecard that decided the
      application, rounded to four decimals;
    - ``policy_decline``: True if a policy rule declined the application,
      whatever its PD: a debt-to-income ratio above 0.6 or an age below 21;
    - ``approved``: True if ``incumbent_pd`` is below 0.15 and no policy
      rule declined the application;
    - ``target``: 1 if the applicant defaults, else 0.

    The default probability rises with the debt-to-income ratio and the
    number of inquiries, falls with age and income, and is higher for
    renters and for temporary and self-employed applicants. A hidden risk
    factor that no column holds also raises it, and the number of inquiries
    rises with that factor.

    The incumbent scorecard weighs the features differently, ignores
    ``housing`` and, like an underwriter with a credit bureau report, sees
    part of the hidden risk factor. Rejected applicants are therefore
    riskier than approved applicants with the same features, so a model
    fitted on the approved applicants alone underestimates their default
    rate.

    In practice, the outcome is known only for approved applicants. Here
    ``target`` is also given for rejected applicants, but only to evaluate
    reject inference: a model must not be fitted on it, or reject inference
    would be pointless.

    Examples
    --------
    >>> from kivi.datasets import make_applications
    >>> applications = make_applications(rng=0)
    >>> applications.shape
    (10000, 10)

    The rejected applicants default three times as often as the approved
    ones, an outcome that only synthetic data can show:

    >>> by_decision = applications.groupby(["policy_decline", "approved"])["target"]
    >>> by_decision.agg(["size", "mean"]).round(3)
                             size   mean
    policy_decline approved
    False          False     2467  0.301
                   True      6770  0.097
    True           False      763  0.274
    """
    n_applicants = _validation.check_count(n_applicants, "n_applicants")
    rng = np.random.default_rng(rng)

    age = np.clip(np.round(rng.normal(40, 11, n_applicants)), 18, 75).astype(np.int64)
    log_income = rng.normal(np.log(_MEDIAN_INCOME) + 0.01 * (age - 40), 0.45)
    income = np.round(np.exp(log_income), -1)
    debt_to_income = np.round(rng.beta(2.0, 5.0, n_applicants), 2)
    housing = rng.choice(len(HOUSING), size=n_applicants, p=_HOUSING_SHARES)
    employment = rng.choice(len(EMPLOYMENT), size=n_applicants, p=_EMPLOYMENT_SHARES)
    hidden = rng.standard_normal(n_applicants)
    inquiries = rng.poisson(0.9 * np.exp(0.35 * hidden))

    relative_income = np.log(income / _MEDIAN_INCOME)
    log_odds = (
        -2.2
        - 0.025 * (age - 40)
        - 0.6 * relative_income
        + 3.0 * (debt_to_income - 0.29)
        + 0.25 * inquiries
        + _HOUSING_RISK[housing]
        + _EMPLOYMENT_RISK[employment]
        + _HIDDEN_RISK * hidden
    )
    target = (rng.random(n_applicants) < _logistic(log_odds)).astype(np.int64)

    incumbent = (
        -2.3
        - 0.02 * (age - 40)
        - 0.5 * relative_income
        + 2.5 * (debt_to_income - 0.29)
        + 0.2 * inquiries
        + 0.8 * _EMPLOYMENT_RISK[employment]
        + _INCUMBENT_SHARE * _HIDDEN_RISK * hidden
        + rng.normal(0, _INCUMBENT_NOISE, n_applicants)
    )
    incumbent_pd = np.round(_logistic(incumbent), 4)
    policy_decline = (debt_to_income > MAX_DEBT_TO_INCOME) | (age < MIN_AGE)
    return pd.DataFrame(
        {
            "age": age,
            "income": income,
            "debt_to_income": debt_to_income,
            "inquiries": inquiries.astype(np.int64),
            "housing": pd.Categorical.from_codes(housing, categories=HOUSING),  # type: ignore[call-overload]  # categories may be any list-like
            "employment": pd.Categorical.from_codes(employment, categories=EMPLOYMENT),  # type: ignore[call-overload]  # categories may be any list-like
            "incumbent_pd": incumbent_pd,
            "policy_decline": policy_decline,
            "approved": (incumbent_pd < CUT_OFF) & ~policy_decline,
            "target": target,
        }
    )


def _logistic(log_odds: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 1 / (1 + np.exp(-log_odds))
