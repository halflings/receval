import receval

def test_splitter():
    df = receval.download.load_movielens()
    df = receval.preprocessing.aggregate_ratings(df, method='max')
    for test_size in [0.1, 0.4, 0.9]:
        splitter = receval.splitter.RandomSplitter(test_size=test_size, per_user=True)
        train, test = splitter.split(df)

        test_count = test.groupby('user').count()
        train_count = train.groupby('user').count()

        assert (test_count / (test_count + train_count))['rating'].apply(lambda v : abs(v - test_size) < 0.05).all(), "test/train ratio is off"

def test_recommender():
    df = receval.download.load_movielens()

    dummy_cmd = receval.recommender.DummyCommandRecommender()
    dummy_obj = receval.recommender.DummyRecommender()

    splitter = receval.splitter.RandomSplitter(0.5, per_user=True)
    print("* Splitting...")
    train, test = splitter.split(df)

    print(dummy_cmd.recommend(train, test))
    print("* Dummy object...")
    print(dummy_obj.recommend(train, test))

    rec = receval.recommender.AverageBaselineRecommender()
    recommendations = rec.recommend(train, test)
    print(recommendations)