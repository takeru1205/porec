from abc import ABC, abstractmethod

from .dataset import DataSet


class BaseRecommender(ABC):
    """Base class of recommendation algorithms."""

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        """Fit algorithm to dataset"""
        raise NotImplementedError

    @abstractmethod
    def recommend(self, *args, **kwargs) -> DataSet:
        """recommend items from args"""
        raise NotImplementedError
