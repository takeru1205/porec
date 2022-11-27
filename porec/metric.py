from typing import List

import numpy as np
from sklearn.metrics import mean_squared_error

from .basemetric import Metric


class Metrics:
    """Evaluate the recommendation results"""

    def __init__(
        self,
        true_values: List[float],
        estimate_values: List[float],
        *args,
        **kwargs,
    ):
        """Set true values and estimate values

        Args:
            true_values (List[float]): the ground truth values
            estimate_values (List[float]): the estimated values

        Returns:
            None
        """
        self.true_values = true_values
        self.estimate_values = estimate_values

    def evaluate(
        self,
        *args,
        **kwargs,
    ) -> Metric:
        """Calculate RMSE, Precision@K and Recall

        Returns:
            Metric: returns contain RMSE, Precision@K and Recall.
            But currently Precision@K and Recall is not available.

        Examples:
            >>> met = Metrics(true_values, estimate_values)
            >>> met.evaluate()
            Metric(rmse=XXX, precision_k=YYY, recall=ZZZ)
        """
        rmse = self._calc_rmse()
        evals = Metric(rmse=rmse)
        return evals

    def _calc_rmse(self):
        """Calculate root mean squared error true_values and estimate_values"""
        return np.sqrt(
            mean_squared_error(
                self.true_values,
                self.estimate_values,
            )
        )

    def _calc_precision_k(self):
        """Calculate precision@K true_values and estimate_values"""
        raise NotImplementedError

    def _calc_recall(self):
        """Calculate recall true_values and estimate_values"""
        raise NotImplementedError
