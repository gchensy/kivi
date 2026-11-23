"""Example data for credit risk modeling.

``load_bank`` loads a real data set with a binary target,
``make_loan_panel`` simulates the monthly history of a loan portfolio,
``make_applications`` simulates loan applications with an approval decision,
and ``make_transactions`` simulates the account transactions of the applicants
before their applications.
"""

from importlib import resources

import pandas as pd

from ._applications import make_applications
from ._loan_panel import make_loan_panel
from ._transactions import make_transactions

__all__ = ["load_bank", "make_applications", "make_loan_panel", "make_transactions"]


def load_bank() -> pd.DataFrame:
    """Load 4,521 observations from UCI Bank Marketing.

    The ``y`` response of the original data is returned as the binary
    ``target`` column.

    Returns
    -------
    pandas.DataFrame
        Sixteen features and the integer target. A value of 1 indicates a
        term deposit subscription.

    See Also
    --------
    kivi.woe.woe_table : Relate a feature to ``target``.
    make_loan_panel : Synthetic monthly history of a loan portfolio.

    Notes
    -----
    Bank Marketing by S. Moro, P. Rita and P. Cortez (2014) is distributed
    under CC BY 4.0. The bundled CSV was converted from the legacy workbook.

    References
    ----------
    .. [1] S. Moro, P. Rita and P. Cortez, "Bank Marketing", UCI Machine
       Learning Repository, 2014. https://doi.org/10.24432/C5K306

    Examples
    --------
    >>> from kivi.datasets import load_bank
    >>> data = load_bank()
    >>> data.shape
    (4521, 17)
    >>> data["target"].value_counts().to_dict()
    {0: 4000, 1: 521}
    """
    with resources.files(__package__).joinpath("bank.csv").open("rb") as source:
        data = pd.read_csv(source, sep=";")
    data["target"] = data.pop("y").map({"yes": 1, "no": 0})
    return data
