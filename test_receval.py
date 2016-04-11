import pandas as pd

import receval


def test_data_loading():
    df = receval.download.load_movielens()
    df = receval.preprocessing.aggregate_ratings(df, method='min')
    receval.preprocessing.validate_dataframe(df)


def test_random_splitter():
    df = receval.download.load_movielens()
    df = receval.preprocessing.aggregate_ratings(df, method='max')
    for test_size in [0.1, 0.4, 0.9]:
        splitter = receval.splitter.RandomSplitter(
            test_size=test_size, per_user=True)
        train, test = splitter.split(df)

        test_count = test.groupby('user').count()['rating']
        train_count = train.groupby('user').count()['rating']

        assert (test_count / (test_count + train_count)).apply(lambda v: abs(v - test_size) < 0.05).all(), "test/train ratio is off"


def test_temporal_splitter():
    df = receval.download.load_movielens()
    for test_size in [0.1, 0.9]:
        splitter = receval.splitter.TemporalSplitter(
            test_size=test_size, per_user=True)
        train, test = splitter.split(df)

        test_count = test.groupby('user').count()['rating']
        train_count = train.groupby('user').count()['rating']

        assert (test_count / (test_count + train_count)).apply(lambda v: abs(v - test_size) < 0.05).all(), "test/train ratio is off"

    # TODO : do a more specific test (that tests the split is in fact on time,
    # not just random)


def test_recommender():
    df = receval.download.load_movielens()

    dummy_cmd = receval.recommender.DummyCommandRecommender()
    dummy_obj = receval.recommender.DummyRecommender()

    splitter = receval.splitter.RandomSplitter(0.5, per_user=True)
    print("* Splitting...")
    train, test = splitter.split(df)
    test_users = test.user.unique()

    print(dummy_cmd.recommend(train, test_users))
    print("* Dummy object...")
    print(dummy_obj.recommend(train, test_users))

    rec = receval.recommender.BaselineRecommender()
    recommendations = rec.recommend(train, test_users)
    print(recommendations)


def test_simple_preprocessing_recommender():
    df = pd.DataFrame(dict(user=[0, 0, 0, 1, 1, 2, 2, 2],
                           item=[0, 1, 3, 0, 0, 0, 1, 1],
                           rating=[1, 0.01, 0.9, 0.05, 0.85, 0.95, 1., 0.9]))
    df = df[['user', 'item', 'rating']]

    def threshold_and_dedup_func(ratings):
        ratings = ratings.copy()
        ratings['rating'] = ratings['rating'].apply(
            lambda v: 1 if v > 0.8 else 0)
        ratings = ratings.drop_duplicates(subset=['user', 'item'])
        return ratings

    splitter = receval.splitter.RandomSplitter(0.5)
    train, test = splitter.split(df)

    class ModifiedTestRecommender(receval.recommender.BaselineRecommender):

        def _recommend(self, train_ratings, test_users):
            assert train_ratings.rating.isin(
                [0, 1]).all(), "The ratings weren't thresholded like expected"
            return super(ModifiedTestRecommender, self)._recommend(train_ratings, test_users)

    recommender = ModifiedTestRecommender(
        preprocessing_func=threshold_and_dedup_func)
    recommender.recommend(train, test.user.unique())


def test_evaluation_instance():
    ratings = receval.download.load_movielens()
    splitter = receval.splitter.TemporalSplitter(test_size=0.3, per_user=True)
    train, test = splitter.split(ratings)
    rec = receval.recommender.BaselineRecommender(
        num_recommendations=20)
    recommendations = rec.recommend(train, test.user.unique())

    receval.evaluation.Evaluation(recommendations, test)


def test_word2vec_class():
    ratings = receval.download.load_movielens()
    splitter = receval.splitter.TemporalSplitter(test_size=0.3, per_user=True)
    train, test = splitter.split(ratings)
    rec = receval.recommender.Word2VecRecommender(num_recommendations=50)

    rec.recommend(train, test.user.unique())
