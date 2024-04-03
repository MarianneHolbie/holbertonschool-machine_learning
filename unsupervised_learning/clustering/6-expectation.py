#!/usr/bin/env python3
"""
    Clustering : EM algo for a GMM
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
        calculate the expectation step in the EM algorithm for a GMM

    :param X: ndarray, shape(n,d) data set
    :param pi: ndarray, shape(k,) priors for each cluster
    :param m: ndarray, shape(k,d) centroid means for each cluster
    :param S: ndarray, shape(k,d,d) covariance matrix for each cluster

    :return: g, l or None, None on failure
             g: ndarray, shape(k,n) posterior proba for each data
             point in each cluster
             likelihood: total log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 2:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if X.shape[1] != m.shape[1] or X.shape[1] != S.shape[1]:
        return None, None
    if S.shape[1] != S.shape[2]:
        return None, None
    if pi.shape[0] != m.shape[0] or pi.shape[0] != S.shape[0]:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    # initialize g and l
    g = np.zeros((k, n))
    likelihood = 0.0

    # calculate posterior proba (each cluster)
    for i in range(k):
        P = pdf(X, m[i], S[i])
        g[i] = pi[i] * P
    # normalize posterior proba
    marginal = np.sum(g, axis=0)
    g = g / marginal

    # likelihood
    likelihood += np.sum(np.log(marginal))

    return g, likelihood
