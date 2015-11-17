import receval

def test_data_loading():
    df = receval.download.load_movielens()
    df = receval.preprocessing.aggregate_ratings(df, method='min')
    receval.preprocessing.validate_dataframe(df)

def test_random_splitter():
    df = receval.download.load_movielens()
    df = receval.preprocessing.aggregate_ratings(df, method='max')
    for test_size in [0.1, 0.4, 0.9]:
        splitter = receval.splitter.RandomSplitter(test_size=test_size, per_user=True)
        train, test = splitter.split(df)

        test_count = test.groupby('user').count()['rating']
        train_count = train.groupby('user').count()['rating']

        assert (test_count / (test_count + train_count)).apply(lambda v : abs(v - test_size) < 0.05).all(), "test/train ratio is off"

def test_temporal_splitter():
    df = receval.download.load_movielens()
    for test_size in [0.1, 0.9]:
        splitter = receval.splitter.TemporalSplitter(test_size=test_size, per_user=True)
        train, test = splitter.split(df)

        test_count = test.groupby('user').count()['rating']
        train_count = train.groupby('user').count()['rating']

        assert (test_count / (test_count + train_count)).apply(lambda v : abs(v - test_size) < 0.05).all(), "test/train ratio is off"

    # TODO : do a more specific test (that tests the split is in fact on time, not just random)

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
