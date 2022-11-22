import pytest
from porec import dataset, randomrec


class TestRandomRecommender:
    @pytest.fixture
    def init_recommender(self):
        self.data = dataset.read_csv("tests/contents.csv")
        self.recommender = randomrec.RandomRecommender()
        self.recommender.fit(self.data)

    def test_fit(self):
        data = dataset.read_csv("tests/contents.csv")
        recommender = randomrec.RandomRecommender()
        recommender.fit(data)

    def test_length(self, init_recommender):
        len(self.recommender.data) == 8

    def test_recommend(self, init_recommender):
        len(self.recommender.recommend(1)) == 1
        len(self.recommender.recommend(3)) == 3
        len(self.recommender.recommend(5)) == 5
        len(self.recommender.recommend(8)) == 8
