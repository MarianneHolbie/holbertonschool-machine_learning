#!/usr/bin/env python3
"""
    Clustering : GMM
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
        initializes variables for a Gaussian Mixture Model

    :param X: ndarray, shape(n,d) data set
    :param k: int, number of clusters

    :return: pi, m, S or None, None, None on failure
            pi: ndarray, shape(k,) priors for each cluster, initialize evenly
            m: ndarray, shape(k,d) centroid means for each cluster,
            initialize with K-means
            S: ndarray, shape(k,d,d) covariance matrix for each cluster,
             initialized as identity matrices
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None

    n, d = X.shape

    # init prior: equal prior prob
    pi = np.full((k,), fill_value=1/k)

    # centroid mean
    m, _ = kmeans(X, k)

    # repeat id matrix size(d,d) k times along the first axis
    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
