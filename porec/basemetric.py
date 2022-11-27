from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Metric:
    """Evaluation results

    Args:
        rmse (float): Root Mean Squared Error
        precision_k (float): Precision@K
        recall (float): Recall
    """

    rmse: float = 0.0
    precision_k: float = 0.0
    recall: float = 0.0

    def __str__(self):
        """when calling method, to align the metrics and stdout"""
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
