"""Bundled examples for tabular modeling."""

from importlib.resources import files

import pandas as pd

__all__ = ["Dataset"]


class Dataset:
    """Example datasets bundled with kivi."""

    @staticmethod
    def bank_data() -> pd.DataFrame:
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
        >>> from kivi.datasets import Dataset
        >>> data = Dataset.bank_data()
        >>> data.shape
        (4521, 17)
        >>> data["target"].value_counts().to_dict()
        {0: 4000, 1: 521}
        """
        with files(__package__).joinpath("bank.csv").open("rb") as source:
            data = pd.read_csv(source, sep=";")
        data["target"] = data.pop("y").map({"yes": 1, "no": 0})
        return data
