import pandas as pd

from . import metrics


class Evaluation(object):
    DEFAULT_K_RANGE = [5, 10, 20]

    def __init__(self, predicted, test, k_range=None):
        self.k_range = k_range or Evaluation.DEFAULT_K_RANGE

        # Making sure predictions are sorted by decreasing rating for every user
        predicted.sort_values(['user', 'rating'], inplace=True)

        # Crossing the predicted/expected
        predicted = predicted.set_index(['user', 'item'])[['rating']]
        test = test.set_index(['user', 'item'])[['rating']]
        predicted['relevance'] = predicted.index.isin(test.index).astype(int)

        predicted['real_rating'] = test.loc[predicted.index].dropna()
        predicted['real_rating'].fillna(0, inplace=True)

        self.predicted = predicted
        self.per_user = self.predicted.groupby(level='user').agg(lambda r_values: tuple(r_values))

        # Calculating per-user evaluation metrics
        self.user_metrics = pd.DataFrame(index=self.per_user.index)
        self.user_metrics['reciprocal_rank'] = self.per_user['relevance'].apply(metrics.reciprocal_rank)
        self.user_metrics['average_precision'] = self.per_user['relevance'].apply(metrics.average_precision)
        self.user_metrics['r_precision'] = self.per_user['relevance'].apply(metrics.r_precision)

        for k in self.k_range:
            self.user_metrics['precision_at_{}'.format(k)] = self.per_user['relevance'].apply(lambda r: metrics.precision_at_k(r, k))
        for k in self.k_range:
            self.user_metrics['ndcg_at_{}'.format(k)] = self.per_user['real_rating'].apply(lambda r: metrics.ndcg_at_k(r, k))

    def mean_reciprocal_rank(self):
        return self.user_metrics['reciprocal_rank'].mean()

    def mean_average_precision(self):
        return self.user_metrics['average_precision'].mean()

    def mean_ndcg_at_k(self, k):
        return self.per_user['real_rating'].apply(lambda r: metrics.ndcg_at_k(r, k)).mean()


class ComparativeEvaluation(object):
    """
        A class that compares a list of recommenders using the same input data and data splitting method.
    """
    def __init__(self, ratings, splitter, recommenders):
        if not isinstance(recommenders, dict):
            raise TypeError("`recommenders` must be a dictionnary of the shape: `{'rec1': recommender1, 'rec2': ... }`")
        self.ratings = ratings
        self.splitter = splitter
        self.recommenders = recommenders

        train, test = self.splitter.split(self.ratings)
        test_users = test.user.unique()
        self.evaluations = {rec_name: Evaluation(rec.recommend(train, test_users), test)
                            for rec_name, rec in self.recommenders.items()}

        # Aggregating the metrics per recommender
        index = list(self.recommenders.keys())
        columns = self.evaluations[index[0]].user_metrics.columns
        self.metrics_df = pd.DataFrame(index=index, columns=columns)
        for rec_name, evaluation in self.evaluations.items():
            self.metrics_df.loc[rec_name] = evaluation.user_metrics.mean()
        self.metrics_df.sort_index(inplace=True)
