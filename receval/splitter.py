import math

import pandas as pd

def _split_df(df, test_size, random_state=None):
    test = df.sample(math.ceil(len(df) * test_size), random_state=random_state)
    train = df[~ df.index.isin(test.index)]
    return train, test

class RandomSplitter(object):
    def __init__(self, test_size, per_user=False, random_state=None):
        self.test_size = test_size
        self.per_user = per_user
        self.random_state = random_state

    def split(self, ratings):
        if not self.per_user:
            return _split_df(ratings, self.test_size, self.random_state)

        train_dfs = []
        test_dfs = []
        for user in ratings['user'].unique():
            user_ratings = ratings[ratings.user == user]
            train, test = _split_df(user_ratings, self.test_size, self.random_state)
            train_dfs.append(train)
            test_dfs.append(test)

        train_ratings = pd.concat(train_dfs)
        test_ratings = pd.concat(test_dfs)

        return train_ratings, test_ratings

if __name__ == '__main__':
    df = pd.read_csv('data/tiny.csv')
    splitter = RandomSplitter(0.3, per_user=True, random_state=42)
    print("* Input dataframe:")
    print(df)
    print()
    train, test = splitter.split(df)

    print("* Training ratings:")
    print(train)
    print("* Testing ratings:")
    print(test)
