from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    """Base class of recommendation algorithms."""

    @abstractmethod
    def fit(self):
        """Fit algorithm to dataset"""
        raise NotImplementedError

    @abstractmethod
    def recommend(self):
        """recommend items from args"""
        raise NotImplementedError
