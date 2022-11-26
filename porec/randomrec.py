from copy import deepcopy

import numpy as np

from . import baserec
from .dataset import DataSet


class RandomRecommender(baserec.BaseRecommender):
    """Recommendation algorithm that recommend randomly"""

    def fit(self, data: DataSet, *args, **kwargs) -> None:
        """Get and set data

        Args:
            data (DataSet): content data(ex: movies list, item list)

        Returns:
            None
        """
        self.data: DataSet = data

    def recommend(
        self,
        K: int = 1,
        low: float = 0.0,
        high: float = 5.0,
        *args,
        **kwargs,
    ) -> DataSet:
        """Get top K evaluation values (low ~ high)

        Args:
            K (int): number of top items
            low (float): minimum evaluation values
            high (float): maximum evaluation values

        Returns:
            DataSet: the data which get in fit() with evaluation values

        Examples:
            >>> rand = RandomRecommender()
            >>> rand.fit(data)
            >>> rand.recommend(3)
            |id|title|values|
            |5|GHI|0.8|
            |1|ABC|0.6|
            |8|DEF|0.2|
        """

        rec = deepcopy(self.data)
        rec["values"] = np.random.uniform(
            low=low,
            high=high,
            size=len(self.data),
        )
        return rec.sort_values(
            by=["values"],
            ascending=False,
        ).head(K)
