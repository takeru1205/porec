import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

from porec import basemetric, metric


class TestMetric:
    @pytest.fixture
    def init_data(self):
        self.true_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.estimate_values = [0.5, 0.4, 0.3, 0.2, 0.1]
        self.mes = metric.Metrics(
            self.true_values,
            self.estimate_values,
        )

    def test_init(self, init_data):
        assert self.mes.true_values == self.true_values
        assert self.mes.estimate_values == self.estimate_values

    def test_evaluste(self, init_data):
        evals = self.mes.evaluate()
        assert isinstance(evals, basemetric.Metric)
        rmse = np.sqrt(
            mean_squared_error(
                self.true_values,
                self.estimate_values,
            )
        )
        assert evals.rmse == rmse
        assert evals.precision_k == 0.0
        assert evals.recall == 0.0
        print(evals)
