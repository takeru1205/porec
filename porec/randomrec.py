import pandas as pd

from . import base
from .dataset import DataSet


class RandomRecommender(base.BaseRecommender):
    """Recommendation algorithm that recommend randomly"""

    def fit(self, data: DataSet) -> None:
        self.data: DataSet = data

    def recommend(self, K: int = 1) -> DataSet:
        return self.data.sample(n=K)
