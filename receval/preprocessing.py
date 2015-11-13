"""Functions related to loading data and pre-processing it"""

import numpy as np

from . import download

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
        raise TypeError("Expected numeric ratings, got type '{}'".format(df['rating'].dtype))

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
        return df.groupby(['user', 'item'], as_index=False).agg(agg_func)
    raise ValueError("Ratings' aggregation method '{}' isn't supported.".format(method))
