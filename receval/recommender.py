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
    def __init__(self, data_dir='/tmp/'):
        if not os.path.isdir(data_dir):
            raise ValueError("Provided data_dir does not exist or is a file.")
        self.data_dir = data_dir

    def _validate_recommendations(self, predicted, test):
        predicted_users = set(predicted.user)
        test_users = set(test.user)
        missing_users = test_users - predicted_users
        if missing_users:
            raise ValueError("Recommendations missing for {} user(s) of the test set.".format(len(missing_users)))
        extra_users = predicted_users - test_users
        if extra_users:
            raise ValueError("Recommendations present for {} user(s) that were not in the test set.".format(len(extra_users)))

    def recommend(self, train_ratings, test_ratings):
        # TODO: some validation for the train and test data
        recommendations = self._recommend(train_ratings, test_ratings)
        self._validate_recommendations(recommendations, test_ratings)
        return recommendations

    def _recommend(self, train_ratings, test_ratings):
        raise NotImplementedError("The `_recommend` method has to be implemented by all recommenders.")

class CommandRecommender(Recommender):
    """
        Class for external recommender systems that need to be called
        through a shell command.
    """
    def __init__(self, recommender_cmd, *args, **kwargs):
        """
            `recommender_cmd` is the shell command to use, and can include the following arguments:
            `train_path`, `test_path` and `rec_path`.

            Example:
            >> CommandRecommender(recommender_cmd='cat "{train_path}" > /dev/null')
         """
        super(CommandRecommender, self).__init__(*args, **kwargs)
        self.recommender_cmd = recommender_cmd

    def _recommend(self, train_ratings, test_ratings):
        evaluation_id = str(uuid4())

        train_path = os.path.join(self.data_dir, '{}_test'.format(evaluation_id))
        test_path = os.path.join(self.data_dir, '{}_train'.format(evaluation_id))
        rec_path = os.path.join(self.data_dir, '{}_recs'.format(evaluation_id))

        train_ratings.to_csv(train_path, index=False)
        test_ratings.to_csv(test_path, index=False)

        formatted_cmd = self.recommender_cmd.format(train_path=train_path, test_path=test_path, rec_path=rec_path)
        print("* Executing '{}'...".format(formatted_cmd))
        cmd_output = subprocess.check_output(formatted_cmd, shell=True)
        print("* Command executed!")
        if cmd_output:
            print("- Output: {}".format(cmd_output))

        recommendations = pd.read_csv(rec_path)
        return recommendations

class DummyRecommender(Recommender):
    """A recommender that simply returns an exact copy of the expected ratings"""
    def _recommend(self, train_ratings, test_ratings):
        return test_ratings.copy().reset_index(drop=True)

class DummyCommandRecommender(CommandRecommender):
    """Similar to DummyRecommender, but through calling a command. Mostly used to test command calling works correctly."""
    DUMMY_EVALUATOR_CMD = 'cat "{train_path}" > /dev/null; cat "{test_path}" > "{rec_path}"'

    def __init__(self, *args, **kwargs):
        super(DummyCommandRecommender, self).__init__(self.DUMMY_EVALUATOR_CMD, *args, **kwargs)

class AverageBaselineRecommender(Recommender):
    """
        Useful as a baseline, this recommender rates each item
        with its average rating in the training set.
    """
    def _recommend(self, train_ratings, test_ratings):
        avg_ratings = train_ratings[['item', 'rating']].groupby('item', as_index=False).mean()
        users_df = train_ratings[['user']].drop_duplicates()

        # Cartesian product between the users dataframe and the average ratings
        avg_ratings.set_index([[0]*len(avg_ratings)], inplace=True)
        users_df.set_index([[0]*len(users_df)], inplace=True)
        recommendations = users_df.join(avg_ratings, how='outer')

        return recommendations

def main():
    from splitter import RandomSplitter
    from download import load_movielens

    dummy_cmd = DummyCommandRecommender()
    dummy_obj = DummyRecommender()
    df = load_movielens()
    splitter = RandomSplitter(0.5, per_user=True)
    print("* Splitting...")
    train, test = splitter.split(df)

    print(dummy_cmd.recommend(train, test))
    print("* Dummy object...")
    print(dummy_obj.recommend(train, test))

    rec = AverageBaselineRecommender()
    recommendations = rec.recommend(train, test)
    print(recommendations)

if __name__ == '__main__':
    main()
