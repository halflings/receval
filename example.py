import pandas as pd

import receval

def main():
    df = receval.download.load_movielens()
    df = receval.preprocessing.aggregate_ratings(df, method='max')
    splitter = receval.splitter.RandomSplitter(test_size=0.4, per_user=True)
    train, test = splitter.split(df)
    print(df.head(5))
    print(train.head(5))
    print(test.head(5))

if __name__ == '__main__':
    main()
