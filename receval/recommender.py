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
        if self.preprocessing_func:
            train_ratings = self.preprocessing_func(train_ratings)
        # TODO: some validation for the training data
        recommendations = self._recommend(train_ratings, users)
        self._validate_recommendations(recommendations, train_ratings, users)
        # Sorting each user's recommendations by rating
        recommendations.sort_values(['user', 'rating'], ascending=False, inplace=True)
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
