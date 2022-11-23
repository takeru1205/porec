from . import base
from .dataset import DataSet


class RandomRecommender(base.BaseRecommender):
    """Recommendation algorithm that recommend randomly"""

    def fit(self, data: DataSet, *args, **kwargs) -> None:
        self.data: DataSet = data

    def recommend(self, K: int = 1, *args, **kwargs) -> DataSet:
        return self.data.sample(n=K)
