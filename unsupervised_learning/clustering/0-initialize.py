#!/usr/bin/env python3
"""
    Clustering
"""

import numpy as np


def initialize(X, k):
    """
        initializes cluster centroids for K-means

    :param X: ndarray, shape(n,d) containing dataset for K-means clustering
        n: number of data points
        d: number of dimension for each data point
    :param k: positive integer containing the number of cluster

    :return: ndarray, shape(k,d) containing initialized centroids for each
            clusert of None on failure
    """

    if not isinstance(k, int) or k < 0:
        return None

    n, d = X.shape

    low = np.argmin(X)
    high = np.argmax(X)

    centroids = np.random.uniform(low=low, high=high, size=(k, d))

    return centroids
