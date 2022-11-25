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
        assert len(self.recommender.data) == 8

    def test_recommend(self, init_recommender):
        assert len(self.recommender.recommend(1)) == 1
        assert len(self.recommender.recommend(3)) == 3
        assert len(self.recommender.recommend(5)) == 5
        assert len(self.recommender.recommend(8)) == 8

    def test_evaluate(self, init_recommender):
        # the evaluation values are 0 <= v <= 5
        recs = self.recommender.recommend(5, 0.0, 5.0)
        assert recs["values"].min() >= 0.0
        assert recs["values"].max() <= 5.0
        # the evaluation values are 1 <= v <= 20
        recs = self.recommender.recommend(5, 1.0, 20.0)
        assert recs["values"].min() >= 1.0
        assert recs["values"].max() <= 20.0
