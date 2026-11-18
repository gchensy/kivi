"""Bundled examples for tabular modeling."""

from importlib.resources import files

import pandas as pd


class Dataset:
    """读取随包分发的示例数据。"""

    @staticmethod
    def bank_data() -> pd.DataFrame:
        """Load 4,521 observations from UCI Bank Marketing.

        Returns
        -------
        pandas.DataFrame
            Sixteen features and the integer target. A value of 1 indicates a term deposit subscription.
        """
        with files(__package__).joinpath("bank.csv").open("rb") as source:
            data = pd.read_csv(source, sep=";")
        data["target"] = data.pop("y").map({"yes": 1, "no": 0})
        return data
