import math
import random

import pandas as pd

def _split_df(df, test_size, random_state=None):
    test = df.sample(math.floor(len(df) * test_size), random_state=random_state)
    train = df[~ df.index.isin(test.index)]
    return train, test

class RandomSplitter(object):
    def __init__(self, test_size, per_user=False, per_item=False, random_state=None):
        if per_user and per_item:
            raise ValueError("Only one of 'per_user' and 'per_item' can be set to True.")
        self.test_size = test_size
        self.per_user = per_user
        self.per_item = per_item
        self.random_state = random_state

    def split(self, ratings):
        if not (self.per_user or self.per_item):
            return _split_df(ratings, self.test_size, self.random_state)

        split_column = 'user' if self.per_user else 'item'
        test_index = []
        for key in ratings[split_column].unique():
            ratings_subset = ratings[ratings[split_column] == key]
            num_test_ratings = int(math.floor(len(ratings_subset) * self.test_size))
            test_index += random.sample(ratings_subset.index.tolist(), num_test_ratings)
        train_ratings = ratings[~ratings.index.isin(test_index)]
        test_ratings = ratings.loc[test_index]

        return train_ratings, test_ratings

if __name__ == '__main__':
    df = pd.read_csv('data/tiny.csv')
    splitter = RandomSplitter(0.5, per_user=True, random_state=42)
    print("* Input dataframe:")
    print(df)
    print()
    train, test = splitter.split(df)

    print("* Training ratings:")
    print(train)
    print("* Testing ratings:")
    print(test)
