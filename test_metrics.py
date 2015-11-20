import pytest

from receval.metrics import *


def test_mean_reciprocal_rank():
    rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    assert mean_reciprocal_rank(rs) == 0.61111111111111105

    rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    assert mean_reciprocal_rank(rs) == 0.5

    rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    assert mean_reciprocal_rank(rs) == 0.75


def test_r_precision():
    r = [0, 0, 1]
    assert r_precision(r) == 0.33333333333333331
    r = [0, 1, 0]
    assert r_precision(r) == 0.5
    r = [1, 0, 0]
    assert r_precision(r) == 1.0


def test_precision_at_k():
    r = [0, 0, 1]
    assert precision_at_k(r, 1) == 0.0
    assert precision_at_k(r, 2) == 0.0
    assert precision_at_k(r, 3) == 0.33333333333333331
    with pytest.raises(ValueError):
        precision_at_k(r, 4)

def test_average_precision():
    r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    delta_r = 1. / sum(r)
    calculated_avg_precision = sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    assert calculated_avg_precision == 0.7833333333333333
    assert average_precision(r) == 0.78333333333333333


def test_mean_average_precision():
    rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    assert mean_average_precision(rs) == 0.78333333333333333
    rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    assert mean_average_precision(rs) == 0.39166666666666666


def test_dcg_at_k():
    r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    assert dcg_at_k(r, 1) == 3.0
    assert dcg_at_k(r, 1, method=1) == 3.0
    assert dcg_at_k(r, 2) == 5.0
    assert dcg_at_k(r, 2, method=1) == 4.2618595071429155
    assert dcg_at_k(r, 10) == 9.6051177391888114
    assert dcg_at_k(r, 11) == 9.6051177391888114


def test_ndcg_at_k():
    r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    assert ndcg_at_k(r, 1) == 1.0
    r = [2, 1, 2, 0]
    assert ndcg_at_k(r, 4) == 0.9203032077642922
    assert ndcg_at_k(r, 4, method=1) == 0.96519546960144276
    assert ndcg_at_k([0], 1) == 0.0
    assert ndcg_at_k([1], 2) == 1.0
