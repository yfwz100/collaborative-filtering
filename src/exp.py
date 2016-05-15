#!/usr/bin/env python3
# coding: utf-8

'''
The experiments.
'''

from functools import lru_cache

from numpy import *
import pandas as pd

from cf import *


def diffratio_similarity(a, b):
    ''' The diffratio similarity
    '''
    assert a.shape == b.shape
    sim = pearson_similarity(a, b)
    if sim != 0:
        overlap = (a > 0) & (b > 0)
        union = (a > 0) | (b > 0)
        contrib = count_nonzero(overlap) / count_nonzero(union)
        return contrib * sim
    return sim


def diffratiolog_similarity(a, b):
    assert a.shape == b.shape
    sim = pearson_similarity(a, b)
    if sim != 0:
        overlap = count_nonzero((a > 0) & (b > 0))
        union = count_nonzero((a > 0) | (b > 0))
        contrib = overlap / union * log(overlap) / log(union)
        return contrib * sim
    return sim
    

def diffratio_one_similarity(a, b):
    assert a.shape == b.shape
    sim = pearson_one_similarity(a, b)
    if sim != 0:
        overlap = (a > 0) & (b > 0)
        union = (a > 0) | (b > 0)
        contrib = count_nonzero(overlap) / count_nonzero(union)
        return contrib * sim
    return sim


def diffratiolog_one_similarity(a, b):
    assert a.shape == b.shape
    sim = pearson_one_similarity(a, b)
    if sim != 0:
        overlap = count_nonzero((a > 0) & (b > 0))
        union = count_nonzero((a > 0) | (b > 0))
        contrib = overlap / union * log(overlap) / log(union)
        return contrib * sim
    return sim


@lru_cache()
def load_ratings_matrix(train_data_file, test_data_file):

    def load_ratings_dataframe(data_file):
        data = pd.read_csv(data_file, delimiter='\t', header=None)
        data.columns = ['user_id', 'item_id', 'rating', 'timestamp']
        return data

    def transform_rating_matrix(df, s):
        m = zeros(s)
        for _, r in df.iterrows():
            m[r.user_id, r.item_id] = r.rating
        return m

    train_data = load_ratings_dataframe(train_data_file)
    test_data = load_ratings_dataframe(test_data_file)

    users = max(train_data.user_id.max(), test_data.user_id.max()) + 1
    items = max(train_data.item_id.max(), test_data.item_id.max()) + 1

    s = (users, items)

    train_matrix = transform_rating_matrix(train_data, s)
    test_matrix = transform_rating_matrix(test_data, s)

    assert train_matrix.shape == test_matrix.shape

    return train_matrix, test_matrix


def run_item_solution(fnames, measures, aggs, metrics):
    """ Run item-based collaborative filtering.

    :param fname: the list of file name prefix.
    :param measures: the list of similarity measure.
    :metrics: the list of metrics type.
    :return : the metrics result.
    """
    result = []

    for fname in fnames:
        train_matrix, test_matrix = load_ratings_matrix(
            '%s.base' % fname, '%s.test' % fname
        )
        train_matrix = train_matrix.T
        test_matrix = test_matrix.T

        pairs = transpose(nonzero(test_matrix))

        for name, measure in measures:
            ii_sim = build_similarity_matrix(train_matrix, measure)

            for k, agg in aggs:
                prediction = cf(
                    pairs,
                    train_matrix,
                    ii_sim,
                    agg
                )

                result.append([name, fname, k] + [
                    m(test_matrix[nonzero(test_matrix)], prediction) for m in metrics
                ])

    return result


def item_solution_evaluate():
    data = ['u1', 'u2', 'u3', 'u4', 'u5']
    similarity_measures = [
        ('PCC', pearson_one_similarity),
        ('DR', diffratio_one_similarity),
        ('DRL', diffratiolog_one_similarity)
    ]
    aggs = [(k, agg_topk(k)) for k in [5, 10, 20, 30, 40, 50, 60,
                                       70, 80, 90, 100, 1000]] + [('all', agg_all)]
    metrics = [mae, rmse, hmae(3)]
    return run_item_solution(data, similarity_measures, aggs, metrics)
