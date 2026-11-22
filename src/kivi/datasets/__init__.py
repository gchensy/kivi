"""Example data for credit risk modeling.

``load_bank`` loads a real data set with a binary target,
``make_loan_panel`` simulates the monthly history of a loan portfolio, and
``make_applications`` simulates loan applications with an approval decision.
"""

from importlib import resources

import pandas as pd

from .. import _deprecation
from ._applications import make_applications
from ._loan_panel import make_loan_panel

__all__ = ["load_bank", "make_applications", "make_loan_panel", "Dataset"]


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


class Dataset:
    """Example datasets bundled with kivi.

    Call the functions of `kivi.datasets` instead. This class is deprecated
    and is removed in kivi 1.0.
    """

    @staticmethod
    def bank_data() -> pd.DataFrame:
        """Load 4,521 observations from UCI Bank Marketing.

        .. deprecated:: 0.9.0
           Use `load_bank`, which returns the same frame. ``bank_data``
           warns with a ``FutureWarning`` and is removed in kivi 1.0.

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
        >>> import warnings
        >>> from kivi.datasets import Dataset, load_bank
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     data = Dataset.bank_data()
        >>> caught[0].category.__name__
        'FutureWarning'
        >>> data.equals(load_bank())
        True
        """
        _deprecation.warn(
            "Dataset.bank_data() is deprecated and is removed in kivi 1.0; "
            "use kivi.datasets.load_bank(), which returns the same frame"
        )
        return load_bank()
