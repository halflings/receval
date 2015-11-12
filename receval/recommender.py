from uuid import uuid4
import os
import subprocess

import pandas as pd

class Recommender(object):
    """
        Base class for recommender classes.
        Recommenders should implement a recommend function that takes training ratings
        and test ratings (both dataframes) and returns a dataframe containin the predicted
        ratings for the (user, item) pairs present in the test set.
    """
    def __init__(self, data_dir='/tmp/'):
        if not os.path.isdir(data_dir):
            raise ValueError("Provided data_dir does not exist or is a file.")
        self.data_dir = data_dir

    def recommend(self, train_ratings, test_ratings):
        return NotImplemented

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

    def recommend(self, train_ratings, test_ratings):
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
    def recommend(self, train_ratings, test_ratings):
        return test_ratings.copy().reset_index(drop=True)

class DummyCommandRecommender(CommandRecommender):
    """Similar to DummyRecommender, but through calling a command. Mostly used to test command calling works correctly."""
    DUMMY_EVALUATOR_CMD = 'cat "{train_path}" > /dev/null; cat "{test_path}" > "{rec_path}"'

    def __init__(self, *args, **kwargs):
        super(DummyCommandRecommender, self).__init__(self.DUMMY_EVALUATOR_CMD, *args, **kwargs)


def main():
    import pandas as pd

    from splitter import RandomSplitter

    dummy_cmd = DummyCommandRecommender()
    dummy_obj = DummyRecommender()
    df = pd.read_csv('data/tiny.csv')
    splitter = RandomSplitter(0.5, per_user=True)
    train, test = splitter.split(df)

    print(dummy_cmd.recommend(train, test))
    print(dummy_obj.recommend(train, test))

if __name__ == '__main__':
    main()
