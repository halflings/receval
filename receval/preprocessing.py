"""Functions related to loading data and pre-processing it"""

import pandas as pd
import numpy as np

DEFAULT_COLUMNS = ['user', 'item', 'rating']

def _numeric_type(dtype):
    return dtype.type == np.float64 or dtype.type == np.int64

def validate_dataframe(df):
    """Normalizes a given dataframe"""
    if len(df.columns) < 3:
        raise ValueError("Dataframe should have at least 3 columns: user, item and ratings.")

    if any(c not in df for c in DEFAULT_COLUMNS):
        raise ValueError("Dataframe is missing compulsory columns, and no columns' mapping was provided.")

    if not _numeric_type(df['rating'].dtype):
        raise TypeError("Expected numeric ratings, got type '{}'".format(ratings_type))

    return True

AGGREGATION_METHODS = {'max', 'min', 'first', 'last'}

def aggregate_ratings(df, method='max'):
    """
        Aggregates (user, item) ratings using different methods.
        Available aggregation methods: {max, min, first, last}.
    """
    if method in {'first', 'last'}:
        return df.drop_duplicates(subset=['user', 'item'], method=method)
    if method in {'max', 'min'}:
        agg_func = max if method == 'max' else min
        return df.groupby(['user', 'item']).agg(agg_func).reset_index()
    raise ValueError("Ratings' aggregation method '{}' isn't supported.".format(method))

def main():
    import pandas as pd

    df = pd.read_csv('data/movielens-100k.tsv', sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    validate_dataframe(df)
    print("* Validated movielens with columns: {}".format(df.columns.values))

    df = pd.read_csv('data/tiny.csv')
    agg_df = aggregate_ratings(df, method='min')
    print("* De-duplicated ratings by picking the minimum value:"
          " from {} ratings to {} ratings".format(len(df), len(agg_df)))

if __name__ == '__main__':
    main()