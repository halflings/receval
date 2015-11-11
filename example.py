import pandas as pd

import receval

def main():
    df = pd.read_csv('receval/data/movielens-100k.tsv', sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    df = receval.preprocessing.aggregate_ratings(df, method='max')
    splitter = receval.splitter.RandomSplitter(test_size=0.4, per_user=True)
    train, test = splitter.split(df)
    print(df.head(5))
    print(train.head(5))
    print(test.head(5))

if __name__ == '__main__':
    main()
