"""Functions to fetch certain test datasets remotely"""

import os
import zipfile
import urllib

MOVIELENS_100K_URL = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'

def download_movielens_100k(url=MOVIELENS_100K_URL, destination=None):
    if destination is None:
        receval_path = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(receval_path, 'data')
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        destination = os.path.join(data_dir, 'movielens-100k.tsv')

    zipfile_data, _ = urllib.urlretrieve(url)

    with zipfile.ZipFile(zipfile_data) as zf:
        with zf.open('ml-100k/u.data') as ratings_file:
            with open(destination, 'w') as destination_file:
                destination_file.write(ratings_file.read())

    print("MovieLens 100K succesfuly downloaded to '{}'".format(destination))

if __name__ == '__main__':
    download_movielens_100k()
