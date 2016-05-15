# coding: utf-8

"""
Collaborative Filtering model.

main usage:

>>> prediction = cf(
>>>     pairs,
>>>     train_matrix,
>>>     ii_sim,
>>>     agg_topk(k)
>>> )

The prediction is indexed same as pairs.

author: yfwz100
"""

from numpy import *

## Evaluation Metrics

# Mean Absolute Error (MAE)
def mae(p, r):
    return mean(abs(r - p))

# Root Mean Square Error (RMSE)
def rmse(p, r):
    return sqrt(mean(pow(r - p, 2)))

# High Score MAE
def hmae(k=3):
    def metric(p, r):
        ri = r >= k
        return mean(abs(r[ri] - p[ri]))
    return metric


### Algorithm Framework

## Similarity builder
def build_similarity_matrix(ratings, measure):
    """ Build similarity matrix from the given rating matrix. by default, it
    generate the similiarity matrix by he first dimension, so the rating matrix
    should be properly transposed to be given as input.

    :param ratings: the rating matrix.
    :param measure: the similarity measure.
    :return : the similarity matrix.
    """
    dlen = ratings.shape[0]
    sim = zeros((dlen,) * 2)
    for i in range(dlen):
        for j in range(dlen):
            sim[i, j] = sim[j, i] = measure(ratings[i, :], ratings[j, :])
    return sim


def pearson_similarity(a, b):
    assert a.shape == b.shape
    overlap = (a > 0) & (b > 0)
    if any(overlap):
        va = a[overlap]
        vb = b[overlap]
        ma = mean(va)
        mb = mean(vb)
        den = sqrt(sum(pow(va - ma, 2))) * sqrt(sum(pow(vb - mb, 2)))
        if den != 0:
            return round(sum((va - ma) * (vb - mb)) / den, 6)
    return 0


def pearson_one_similarity(a, b):
    assert a.shape == b.shape
    overlap = (a > 0) & (b > 0)
    if any(overlap):
        va = a[overlap]
        vb = b[overlap]
        ma = mean(va)
        mb = mean(vb)
        den = sqrt(sum(pow(va - ma, 2))) * sqrt(sum(pow(vb - mb, 2)))
        if den != 0:
            return round(sum((va - ma) * (vb - mb)) / den, 6) * 0.5 + 0.5
    return 0

## Collaborative Filtering Framework.

# the main framework.
def cf(pairs, ratings, similarity, agg):
    """ The collaborative filtering algorithm.

    :param pairs: the given pairs of (user, item).
    :param ratings: the ratings matrix.
    :param similarity: the similarity matrix.
    :param agg: the aggregation function.
    :return : the prediction.
    """
    result = []
    for u, i in pairs:
        p = similarity[u, :]
        r = ratings[:, i]
        ri = r > 0
        rating = agg(p[ri], r[ri])
        if isnan(rating):
            rating = mean(ratings[u, :])
        result.append(rating)
    return array(result)

## Aggregation Metrics


def agg_positive(p, r):
    """ Aggregate the ratings from positive similar users/items.

    :param p: the parameters.
    :param r: the ratings
    :return : the prediction rating.
    """
    pi = p > 0
    return sum(p[pi] * r[pi]) / sum(p)


def agg_all(p, r):
    """ Aggregate all ratings.

    :param p: the parameters.
    :param r: the ratings
    :return : the prediction rating.
    """
    return sum(p * r) / sum(abs(p))


def agg_topk(k):
    """ Generate a function that aggregate top k ratings.

    :param k: the top k paramter.
    :return : a function that aggregate top k ratings.
    """
    def topk_agg(p, r):
        """ Aggregate top k ratings.

        :param p: the paramters.
        :param r: the ratings.
        :return : the prediction rating.
        """
        pi = argsort(p)
        sp = p[pi][-k:]
        sr = r[pi][-k:]
        return sum(sp * sr) / sum(abs(sp))

    return topk_agg
