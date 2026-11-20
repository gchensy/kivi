"""Example data for credit risk modeling.

``load_bank`` loads a real data set with a binary target, and
``make_loan_panel`` simulates the monthly history of a loan portfolio.
"""

from importlib.resources import files

import pandas as pd

from ._loan_panel import make_loan_panel

__all__ = ["load_bank", "make_loan_panel", "Dataset"]


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
    with files(__package__).joinpath("bank.csv").open("rb") as source:
        data = pd.read_csv(source, sep=";")
    data["target"] = data.pop("y").map({"yes": 1, "no": 0})
    return data


class Dataset:
    """Example datasets bundled with kivi.

    New code should call the functions of `kivi.datasets` directly; this
    class remains so that existing code keeps working.
    """

    @staticmethod
    def bank_data() -> pd.DataFrame:
        """Load 4,521 observations from UCI Bank Marketing.

        This is the same as `load_bank`, which new code should use instead.
        It is kept for existing code and does not warn.

        Returns
        -------
        pandas.DataFrame
            Sixteen features and the integer target. A value of 1 indicates a
            term deposit subscription.

        See Also
        --------
        load_bank : The same data.

        Examples
        --------
        >>> from kivi.datasets import Dataset, load_bank
        >>> Dataset.bank_data().equals(load_bank())
        True
        """
        return load_bank()
