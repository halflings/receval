from . import metrics

class Evaluation(object):
    def __init__(self, predicted, test):
        predicted = predicted.set_index(['user', 'item'])[['rating']]
        test = test.set_index(['user', 'item'])[['rating']]

        predicted['relevance'] = predicted.index.isin(test.index).astype(int)
        predicted['real_rating'] = test.loc[predicted.index].dropna()
        predicted['real_rating'].dropna(inplace=True)

        # Setting unpredicted values to 0
        # predicted = predicted.append(test.loc[~ test.index.isin(predicted.index)])
        # predicted.loc[test.index, 'rating'] = 0
        # predicted.sort_index(ascending=False, inplace=True)
        # test.sort_index(ascending=False, inplace=True)

        self.predicted = predicted
        self.test = test

    def aggregate_per_user(self, column):
        return self.predicted[[column]].groupby(level='user').agg(lambda r_values : tuple(r_values))[column]

    def mean_reciprocal_rank(self):
        relevance = self.aggregate_per_user('relevance')
        return metrics.mean_reciprocal_rank(relevance)

    def mean_average_precision(self):
        relevance = self.aggregate_per_user('relevance')
        return metrics.mean_average_precision(relevance)

    def ndcg_at_k(self):
        # TODO : implement this
        pass
