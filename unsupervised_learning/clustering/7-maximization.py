#!/usr/bin/env python3
"""
    Clustering : Maximization step in EM algo for a GMM
"""
import numpy as np


def maximization(X, g):
    """
        calculate the maximization step in the EM algo for a GMM

    :param X: ndarray, shape(n,d) data set
    :param g: ndarray, shape(k,n) posterior proba for each data point
        in each cluster

    :return: pi, m, S or None, None, None
        pi: ndarray, shape(k,) containing updated priors for each cluster
        m: ndarray, shape(k,d) containing updated centroid means
         for each cluster
        S: ndarray, shape(k,d,d) containing updated covariance matrices
         for each cluster
"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    n, d = X.shape
    if n != g.shape[1]:
        return None, None, None
    k, _ = g.shape

    # sum posterior proba for each point
    sum_gi = np.sum(g, axis=0)
    val_n = np.sum(sum_gi)
    # test if sum posterior proba != total number of data
    if val_n != n:
        return None, None, None

    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        # new prior
        pi[i] = 1 / n * np.sum(g[i])
        # new centroid mean
        m[i] = np.matmul(g[i], X) / np.sum(g[i])
        X_mean = X - m[i]
        # new cov
        S[i] = np.matmul(np.multiply(g[i], X_mean.T), X_mean) / np.sum(g[i])

    return pi, m, S
