from copy import deepcopy

import numpy as np

from . import baserec
from .dataset import DataSet


class RandomRecommender(baserec.BaseRecommender):
    """Recommendation algorithm that recommend randomly"""

    def fit(self, data: DataSet, *args, **kwargs) -> None:
        self.data: DataSet = data

    def recommend(
        self,
        K: int = 1,
        low: float = 0.0,
        high: float = 5.0,
        *args,
        **kwargs,
    ) -> DataSet:
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
