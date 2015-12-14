from uuid import uuid4
import os
import subprocess

import pandas as pd

class Recommender(object):
    """
        Base class for recommender classes.
        Recommenders should implement a `_recommend` function that takes training ratings
        and test ratings (both dataframes) and returns a dataframe containin the predicted
        ratings for the (user, item) pairs present in the test set.
    """
    def __init__(self, preprocessing_func=None):
        self.preprocessing_func = preprocessing_func

    def _validate_recommendations(self, predicted, train_ratings, users):
        predicted_users = set(predicted.user)
        test_users = set(users)
        missing_users = test_users - predicted_users
        if missing_users:
            raise ValueError("Recommendations missing for {} user(s) of the test set.".format(len(missing_users)))
        extra_users = predicted_users - test_users
        if extra_users:
            raise ValueError("Recommendations present for {} user(s) that were not in the test set.".format(len(extra_users)))
        train_items = set(train_ratings.item)
        unknown_items = [item for item in predicted.item.unique() if item not in train_items]
        if unknown_items:
            raise ValueError("{} recommended items are not present in the training set. 5 first unknown items: {}...".format(len(unknown_items), unknown_items[:5]))

    def recommend(self, train_ratings, users):
        train_ratings = train_ratings.copy()
        if self.preprocessing_func:
            train_ratings = self.preprocessing_func(train_ratings)
        # TODO: some validation for the training data
        recommendations = self._recommend(train_ratings, users)
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

        formatted_cmd = self.recommender_cmd.format(train_path=train_path, test_users_path=test_users_path, rec_path=rec_path)
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

class AverageBaselineRecommender(Recommender):
    """
        Useful as a baseline, this recommender rates each item
        with its average rating in the training set.
    """
    def __init__(self, num_recommendations=None, *args, **kwargs):
        super(AverageBaselineRecommender, self).__init__(*args, **kwargs)
        self.num_recommendations = num_recommendations

    def _recommend(self, train_ratings, users):
        train_users = train_ratings[['user']].drop_duplicates().user
        avg_ratings = train_ratings[['item', 'rating']].groupby('item', as_index=False).sum()
        avg_ratings['rating'] = avg_ratings['rating'] / len(train_users)
        if self.num_recommendations:
            avg_ratings = avg_ratings.sort_values('rating', ascending=False)[:self.num_recommendations]
        users_df = pd.DataFrame(dict(user=users))

        # Cartesian product between the users dataframe and the average ratings
        avg_ratings.set_index([[0]*len(avg_ratings)], inplace=True)
        users_df.set_index([[0]*len(users_df)], inplace=True)
        recommendations = users_df.join(avg_ratings, how='outer')

        return recommendations

class Word2VecRecommender(Recommender):
    """
        A recommender using `word2vec` to find similar songs to those a user listened to.
        This relies on `word2rec`, a wrapper around the word2vec module in `gensim` focused
        on recommendation.
        When no recommendations are found, the average baseline is used instead.
    """
    DEFAULT_WORD2VEC_PARAMETERS = dict(size=100, window=5, min_count=2, workers=8)
    def __init__(self, num_recommendations=50, word2vec_params=None, *args, **kwargs):
        # NOTE: `gensim` is imported here because it's only used for this class
        from gensim.models import Word2Vec

        super(Word2VecRecommender, self).__init__(*args, **kwargs)
        self.num_recommendations = num_recommendations
        word2vec_params = word2vec_params or dict()
        self.word2vec_params = Word2VecRecommender.DEFAULT_WORD2VEC_PARAMETERS
        self.word2vec_params.update(word2vec_params)

        self.word2vec = Word2Vec(**self.word2vec_params)
        self.baseline = AverageBaselineRecommender(num_recommendations=self.num_recommendations)
        self._baseline_recommendations = None

    def _train_word2vec(self, per_user_items):
        trim_rule = self.word2vec_params.get('trim_rule', None)
        self.word2vec.build_vocab(per_user_items, trim_rule=trim_rule)
        self.word2vec.train(per_user_items)
        self.word2vec.init_sims(replace=True)

    def baseline_recommendations(self, train_ratings, users):
        return self.baseline.recommend(train_ratings, users)[['item', 'rating']].drop_duplicates()

    def _recommend(self, train_ratings, users):
        # Training the model
        train_ratings['item'] = train_ratings['item'].astype(str)
        per_user_items = train_ratings[['user', 'item']].groupby('user').agg(lambda items : tuple(items))
        self._train_word2vec(per_user_items['item'].tolist())

        # Getting recommendations for every user
        baseline_recs = None
        rec_data = []
        for user in users:
            # Getting the most similar items from the word2vec model
            user_items = []
            if user in per_user_items.index:
                user_items = per_user_items.loc[user]['item']
            user_items = [item for item in user_items if item in self.word2vec.vocab]
            if user_items:
                most_similar_items = self.word2vec.most_similar(positive=user_items, topn=self.num_recommendations)

            # Rarely, all of a user's items won't be in the vocabulary or he's not in the training set.
            # In these cases we fallback to the average baseline:
            if not most_similar_items:
                if baseline_recs is None:
                    baseline_recs = self.baseline_recommendations(train_ratings, users)
                most_similar_items = [(r['item'], r['rating']) for _, r in baseline_recs.iterrows()]

            for item, similarity in most_similar_items:
                rec_data.append([user, item, similarity])

        rec_df = pd.DataFrame(rec_data, columns=['user', 'item', 'rating'])
        return rec_df
