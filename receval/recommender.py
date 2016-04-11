from uuid import uuid4
import os
import subprocess

import pandas as pd


def baseline_recommendation(train_ratings, users, num_recommendations=None):
    train_users = train_ratings['user'].unique()
    avg_ratings = train_ratings[['item', 'rating']].groupby('item', as_index=False).sum()
    avg_ratings['rating'] = avg_ratings['rating'] / len(train_users)
    if num_recommendations:
        avg_ratings = avg_ratings.sort_values('rating', ascending=False)[:num_recommendations]
    users_df = pd.DataFrame(dict(user=list(users)))

    # Cartesian product between the users dataframe and the average ratings
    avg_ratings.set_index([[0]*len(avg_ratings)], inplace=True)
    users_df.set_index([[0]*len(users_df)], inplace=True)
    recommendations = users_df.join(avg_ratings, how='outer')
    return recommendations


class Recommender(object):
    """
        Base class for recommender classes.
        Recommenders should implement a `_recommend` function that takes training ratings
        and test ratings (both dataframes) and returns a dataframe containin the predicted
        ratings for the (user, item) pairs present in the test set.
    """
    def __init__(self, preprocessing_func=None, baseline_fallback=False, num_recommendations=None, min_occurrences=None):
        self.preprocessing_func = preprocessing_func
        self.baseline_fallback = baseline_fallback
        self.num_recommendations = num_recommendations
        self.min_occurrences = min_occurrences

    def _validate_recommendations(self, predicted, train_ratings, users):
        predicted_users = set(predicted.user)
        test_users = set(users)
        missing_users = test_users - predicted_users
        if missing_users:
            raise ValueError("Recommendations missing for {} user(s) of the test "
                             "set: {}...".format(len(missing_users), repr(list(missing_users)[:3])))
        extra_users = predicted_users - test_users
        if extra_users:
            raise ValueError("Recommendations present for {} user(s) that were not in "
                             "the test set.".format(len(extra_users), repr(list(extra_users)[:3])))
        train_items = set(train_ratings.item)
        unknown_items = [item for item in predicted.item.unique() if item not in train_items]
        if unknown_items:
            raise ValueError("{} recommended items are not present in the training set. "
                             "5 first unknown items: {}...".format(len(unknown_items), unknown_items[:5]))

        duplicated = predicted.duplicated(subset=['user', 'item'])
        if any(duplicated):
            duplicated_pairs = predicted[duplicated].head(5)[['user', 'item']]
            raise ValueError("Found {} duplicated pairs: {}...".format(sum(duplicated), duplicated_pairs.values))

    def recommend(self, train_ratings, users):
        train_ratings = train_ratings.copy()
        # If we have a filter on the minimum number of occurences, we remove them before pre-processing
        # or recommending
        if self.min_occurrences:
            item_count = train_ratings.groupby('item', as_index=False).agg(dict(user='count'))
            train_ratings = item_count[item_count['user'] > self.min_occurrences][['item']].merge(train_ratings)

        if self.preprocessing_func:
            train_ratings = self.preprocessing_func(train_ratings)
        # TODO: some validation for the training data

        # Calling the recommender's recommendation function
        recommendations = self._recommend(train_ratings, users)

        # Baseline fallback for missing users:
        missing_users = set(users) - set(recommendations.user)
        if missing_users and self.baseline_fallback:
            print("Missing {} users: {}...; Using the baseline as a fallback "
                  "for these users.".format(len(missing_users), list(missing_users)[:5]))
            fallback_recs = baseline_recommendation(train_ratings, missing_users)
            recommendations = recommendations.append(fallback_recs)
        # Recommendations validation (for missing users, non-existant users or items, etc.)
        self._validate_recommendations(recommendations, train_ratings, users)
        # Sorting each user's recommendations by rating
        recommendations.sort_values(['user', 'rating'], ascending=False, inplace=True)
        # Reseting the index
        recommendations.reset_index(drop=True, inplace=True)
        return recommendations

    def _recommend(self, train_ratings, users):
        raise NotImplementedError("The `_recommend` method has to be implemented by all recommenders.")


class CommandRecommender(Recommender):
    """
        Class for external recommender systems that need to be called
        through a shell command.
    """
    def __init__(self, recommender_cmd, data_dir='/tmp/', *args, **kwargs):
        """
            `recommender_cmd` is the shell command to use, and can include the following arguments:
            `train_path`, `test_path` and `rec_path`.

            Example:
            >> CommandRecommender(recommender_cmd='cat "{train_path}" > /dev/null')
         """
        super(CommandRecommender, self).__init__(*args, **kwargs)
        self.recommender_cmd = recommender_cmd
        if not os.path.isdir(data_dir):
            raise ValueError("Provided data_dir does not exist or is a file.")
        self.data_dir = data_dir

    def _recommend(self, train_ratings, users):
        evaluation_id = str(uuid4())

        train_path = os.path.join(self.data_dir, '{}_test_users'.format(evaluation_id))
        test_users_path = os.path.join(self.data_dir, '{}_train'.format(evaluation_id))
        rec_path = os.path.join(self.data_dir, '{}_recs'.format(evaluation_id))

        train_ratings.to_csv(train_path, index=False)
        pd.DataFrame(dict(user=users)).to_csv(test_users_path, index=False)

        formatted_cmd = self.recommender_cmd.format(train_path=train_path, test_users_path=test_users_path,
                                                    rec_path=rec_path)
        print("* Executing '{}'...".format(formatted_cmd))
        cmd_output = subprocess.check_output(formatted_cmd, shell=True)
        print("* Command executed!")
        if cmd_output:
            print("- Output: {}".format(cmd_output))

        recommendations = pd.read_csv(rec_path)
        return recommendations


class DummyRecommender(Recommender):
    """A recommender that simply returns random ratings for the test users"""
    def _recommend(self, train_ratings, users):
        return train_ratings[['user', 'item', 'rating']][train_ratings.user.isin(users)].copy().reset_index(drop=True)


class DummyCommandRecommender(CommandRecommender):
    """Similar to DummyRecommender, but through calling a command. Mostly used to test command calling works correctly."""
    DUMMY_EVALUATOR_CMD = 'cat "{test_users_path}" > /dev/null; cat "{train_path}" > "{rec_path}"'

    def __init__(self, *args, **kwargs):
        super(DummyCommandRecommender, self).__init__(self.DUMMY_EVALUATOR_CMD, *args, **kwargs)


class BaselineRecommender(Recommender):
    """
        Useful as a baseline, this recommender rates each item
        with its average rating in the training set.
    """
    def __init__(self, *args, **kwargs):
        super(BaselineRecommender, self).__init__(*args, **kwargs)

    def _recommend(self, train_ratings, users):
        return baseline_recommendation(train_ratings, users)


class Word2VecRecommender(Recommender):
    """
        A recommender using `word2vec` to find similar songs to those a user listened to.
        We use the word2vec implementation in the `gensim` library.
        When no recommendations are found, the average baseline is used instead.
    """
    DEFAULT_WORD2VEC_PARAMETERS = dict(size=100, window=5, min_count=2, workers=8)

    def __init__(self, word2vec_params=None, *args, **kwargs):
        # NOTE: `gensim` is imported here because it's only used for this class
        from gensim.models import Word2Vec
        super(Word2VecRecommender, self).__init__(*args, **kwargs)
        word2vec_params = word2vec_params or dict()
        self.word2vec_params = dict(Word2VecRecommender.DEFAULT_WORD2VEC_PARAMETERS)
        if word2vec_params:
            self.word2vec_params.update(word2vec_params)

        self.word2vec = Word2Vec(**self.word2vec_params)

    def _train_word2vec(self, per_user_items):
        trim_rule = self.word2vec_params.get('trim_rule', None)
        self.word2vec.build_vocab(per_user_items, trim_rule=trim_rule)
        self.word2vec.train(per_user_items)
        self.word2vec.init_sims(replace=True)

    def _recommend(self, train_ratings, users):
        # Setting a fallback value for num_recommendations, as it's necessary with gensim
        self.num_recommendations = self.num_recommendations or len(train_ratings.item.unique())
        # Training the model
        train_ratings['item'] = train_ratings['item'].astype(str)
        per_user_items = train_ratings[['user', 'item']].groupby('user').agg(lambda items: tuple(items))
        self._train_word2vec(per_user_items['item'].tolist())

        # Getting recommendations for every user
        rec_data = []
        for user in users:
            # Getting the most similar items from the word2vec model
            user_items = []
            if user in per_user_items.index:
                user_items = per_user_items.loc[user]['item']
            user_items = [item for item in user_items if item in self.word2vec.vocab]
            most_similar_items = None
            if user_items:
                most_similar_items = self.word2vec.most_similar(positive=user_items, topn=self.num_recommendations)

            # Rarely, all of a user's items won't be in the vocabulary or he's not in the training set.
            # In these cases we skip the user (and can use a fallback like other recommenders)
            if not most_similar_items:
                continue

            for item, similarity in most_similar_items:
                rec_data.append([user, item, similarity])

        rec_df = pd.DataFrame(rec_data, columns=['user', 'item', 'rating'])
        self.rec_df = rec_df
        return rec_df


class SparkRecommender(Recommender):
    """
        A wrapper around Spark's ALS (Alternating Least Square) based recommender
    """
    DEFAULT_SPARK_ARGS = dict(rank=10, iterations=10)

    def __init__(self, spark_context, implicit=False, num_recommendations=None, spark_args=None, *args, **kwargs):
        # TODO: implement "num_recommendations", probably in the base "Recommender" class
        # TODO: implement a "baseline_fallback" parameter, again: probably in the base "Recommender class"
        super(SparkRecommender, self).__init__(*args, **kwargs)
        self.sc = spark_context
        self.implicit = implicit
        self.num_recommendations = num_recommendations
        self.spark_args = dict(SparkRecommender.DEFAULT_SPARK_ARGS)
        if spark_args:
            self.spark_context = spark_context

    def _recommend(self, train_ratings, users):
        from pyspark.mllib.recommendation import ALS, Rating

        # Preparing the user/item mapping as integers, since Spark's ALS implementation only works with integer values
        train_ratings['user'] = train_ratings['user'].astype('category')
        train_ratings['item'] = train_ratings['item'].astype('category')
        user_cat, item_cat = train_ratings['user'].cat, train_ratings['item'].cat
        self.user_cat = user_cat
        self.item_cat = item_cat
        self.train_ratings = train_ratings

        # Training the model
        self.ratings = self.sc.parallelize(Rating(u, i, rating) for u, i, rating in zip(user_cat.codes, item_cat.codes, train_ratings.rating))
        if self.implicit:
            model = ALS.trainImplicit(self.ratings, **self.spark_args)
        else:
            model = ALS.train(self.ratings, **self.spark_args)

        # Getting predictions from the model
        self.ratings_to_predict = self.sc.parallelize((user, item) for user in users for item in item_cat.codes.unique())
        self.predictions = model.predictAll(self.ratings_to_predict).collect()
        # Presenting the recommendations as a DataFrame
        self.predictions = [(user_cat.categories[p.user], item_cat.categories[p.product], p.rating) for p in self.predictions]
        self.predictions_df = pd.DataFrame(self.predictions, columns=['user', 'item', 'rating'])
        return self.predictions_df
