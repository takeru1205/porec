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
        self.true_values = true_values
        self.estimate_values = estimate_values

    def evaluate(
        self,
        *args,
        **kwargs,
    ) -> Metric:
        """Calculate RMSE, Precision@K and Recall"""
        rmse = self._calc_rmse()
        evals = Metric(rmse=rmse)
        return evals

    def _calc_rmse(self):
        return np.sqrt(
            mean_squared_error(
                self.true_values,
                self.estimate_values,
            )
        )

    def _calc_precision_k(self):
        raise NotImplementedError

    def _calc_recall(self):
        raise NotImplementedError
