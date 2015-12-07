# RecEval

The **evaluation of a recommender system** requires many steps: pre-processing and splitting of the data, evaluation of the results, etc.
Every recommender framework does these steps differently, so it's hard to compare the results you get from different recommender systems.

**RecEval** is a **simple framework** to evaluate the performance of recommender systems using various metrics. This is done by putting in common all the steps that are usually

# Example
```python
import pandas as pd
import receval

# RecEval relies heavily on pandas for organizing the data
# The ratings should be in a dataframe with three compulsory columns:
# 'user', 'item' and 'rating'
df = pd.read_csv('ratings.csv', names=['user', 'item', 'rating'])

# You can put a pre-processing step as a part of the recommender
# you want to evaluate.
def threshold_preprocessing(ratings):
    ratings['rating'] = ratings['rating'].clip(0, 1)
    return ratings

# A set of useful recommender are already defined in RecEval
# but it's easy to write your own or call an external recommender
# via a subprocess.
recommender = receval.recommender.AverageBaselineRecommender()

# Different splitters are provided (random, temporal, ...)
splitter = receval.splitter.RandomSplitter(0.5, per_user=True)
train, test = splitter.split(df)
test_users = test.user.unique()

# Recommendations is a pandas DataFrame similar to the train dataframe
# that contains ratings for different items for every user in the test set
recommendations = recommender.recommend(train, test_users)

# The Evaluation object lets you calculate multiple metrics from a set
# of recommendations
evaluation = receval.evaluation.Evaluation(recommendations, test)

print(evaluation.mean_reciprocal_rank())
print(evaluation.ncdc_at_k(k=10))
```
