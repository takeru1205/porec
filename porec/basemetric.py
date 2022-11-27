from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Metric:
    """Evaluation results"""

    rmse: float = 0.0
    precision_k: float = 0.0
    recall: float = 0.0

    def __str__(self):
        return f"""
Root Mean Squared Error: {self.rmse}
Precision@K: {self.precision_k}
Recall: {self.recall}"""


class BaseMetrics(ABC):
    """Base evaluate the recommendation results"""

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Metric:
        """Fit algorithm to dataset"""
        raise NotImplementedError
