"""Functions to fetch certain test datasets remotely"""

import os
import zipfile
import urllib

import pandas as pd

MOVIELENS_100K_URL = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'

RECEVAL_PATH = os.path.dirname(os.path.realpath(__file__))
MOVIELENS_PATH = os.path.join(RECEVAL_PATH, 'data', 'movielens-100k.tsv')

def download_movielens_100k(url=MOVIELENS_100K_URL, destination=MOVIELENS_PATH):
    zipfile_data, _ = urllib.urlretrieve(url)

    with zipfile.ZipFile(zipfile_data) as zf:
        with zf.open('ml-100k/u.data') as ratings_file:
            with open(destination, 'w') as destination_file:
                destination_file.write(ratings_file.read())

    print("MovieLens 100K succesfuly downloaded to '{}'".format(destination))

def load_movielens(path=MOVIELENS_PATH):
    if not os.path.exists(MOVIELENS_PATH):
        raise ValueError("The file '{}' does not exist.\n"
                         "If you haven't downloaded the movielens dataset yet, you can do so by running:\n"
                         "python -m receval.download".format(MOVIELENS_PATH))
    return pd.read_csv(path, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])

if __name__ == '__main__':
    download_movielens_100k()
