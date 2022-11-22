import pytest
from porec import dataset


class TestDataSet:
    @pytest.fixture
    def init_dataset(self):
        self.dataset = dataset.read_csv("tests/contents.csv")

    def test_read_csv(self, init_dataset):
        assert isinstance(self.dataset, dataset.DataSet)

    def test_dataset_length(self, init_dataset):
        assert len(self.dataset) == 8

    def test_dataset_shape(self, init_dataset):
        assert self.dataset.shape == (8, 3)
