# coding: utf-8

"""
Matrix Factorization Techniques.

By given ratings matrix (user, item => rating), the plain usage:

>>> p, q = sgd_mf(ratings)

The `p` and `q` is the factorized matrix.

author: yfwz100
"""

from numpy import *


def sgd_mf(ratings, p=None, q=None, factors=40, g=1e-2, l=1e-6, s=1.0, max_iters=100):
    """ Stochastic Gradient Descent for Matrix Factorization.
    
    :param ratings: the ratings matrix.
    :param p: (optional) the P matrix for the first dimension of ratings matrix.
    :param q: (optional) the Q matrix for the second dimension of ratings matrix.
    :param factors: (optional) the number of latent factors.
    :param g: (optional) the learning rate.
    :param l: (optional) the regularized coefficient.
    :param s: (optional) the number of samples that used to calculate the matrix.
    :param max_iters: (optional) the maximum number of iterations.
    :return : (optional) the tuple of (P, Q) matrix.
    """
    rows, cols = ratings.shape
    nz = transpose(nonzero(ratings))
    sn = int(len(nz) * s)
    p = p or random.random_sample((rows, factors)) * 0.1
    q = q or random.random_sample((cols, factors)) * 0.1
    for it in range(max_iters):
        snz = random.choice(len(nz), sn, replace=False)
        for n, (u, i) in enumerate(nz[snz]):
            pu = p[u].copy()
            qi = q[i].copy()
            e = ratings[u, i] - pu @ qi.T
            p[u] = pu + g * (e * qi - l * pu)
            assert not any(isnan(p[u]) | isinf(p[u])), '%d p Nan/inf: %d %d %d %f' % (n, e, u, i, pu @ qi.T)
            q[i] = qi + g * (e * pu - l * qi)
            assert not any(isnan(q[i]) | isinf(q[i])), '%d q Nan/inf: %d %d %d %f' % (n, e, u, i, pu @ qi.T)
    return p, q


def sgd_uimf(ratings, p=None, q=None, ub=None, ib=None, factors=40, g=1e-2, l=1e-6, s=1.0, max_iters=100):
    """ Stochastic Gradient Descent for Matrix Factorization, added bias for user and item.
    To generate the original matrix, use `((p @ q.T + ib).T + ub).T`.
    
    :param ratings: the ratings matrix.
    :param p: (optional) the P matrix for the first dimension of ratings matrix.
    :param q: (optional) the Q matrix for the second dimension of ratings matrix.
    :param factors: (optional) the number of latent factors.
    :param g: (optional) the learning rate.
    :param l: (optional) the regularized coefficient.
    :param s: (optional) the number of samples that used to calculate the matrix.
    :param max_iters: (optional) the maximum number of iterations.
    :return : (optional) the tuple of (P, Q) matrix.
    """
    rows, cols = ratings.shape
    nz = transpose(nonzero(ratings))
    sn = int(len(nz) * s)
    p = p or random.random_sample((rows, factors)) * 0.1
    q = q or random.random_sample((cols, factors)) * 0.1
    ub = ub or random.random_sample((rows, 1)) * 0.1
    ib = ib or random.random_sample((cols, 1)) * 0.1
    for it in range(max_iters):
        snz = random.choice(len(nz), sn, replace=False)
        for n, (u, i) in enumerate(nz[snz]):
            pu = p[u].copy()
            qi = q[i].copy()
            ubu = ub[u].copy()
            ibi = ib[i].copy()
            e = ratings[u, i] - ubu - ibi - pu @ qi.T
            p[u] = pu + g * (e * qi - l * pu)
            assert not any(isnan(p[u]) | isinf(p[u])), '%d p Nan/inf: %d %d %d %f' % (n, e, u, i, pu @ qi.T)
            q[i] = qi + g * (e * pu - l * qi)
            assert not any(isnan(q[i]) | isinf(q[i])), '%d q Nan/inf: %d %d %d %f' % (n, e, u, i, pu @ qi.T)
            ub[u] = ubu + g * (e - l * ubu)
            assert not any(isnan(ub[u]) | isinf(ub[u])), '%d ub Nan/inf: %d %d %d %f' % (n, e, ub[u])
            ib[i] = ibi + g * (e - l * ibi)
            assert not any(isnan(ib[i]) | isinf(ib[i])), '%d ib Nan/inf: %d %d %d %f' % (n, e, ib[i])
    return p, q, ub, ib