#!/usr/bin/env python3
"""
    Clustering : optimize k
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
        tests for the optimum number of clusters by variance

    :param X: ndarray, shape(n,d) data set
    :param kmin: int, minimum number of clusters to check for
    :param kmax: int, maximum number of clusters to check for
    :param iterations: int, maximum number of iterations for K-means

    :return: results, d_vars, or None, None on failure
            results: list containing outputs of K-means for each cluster size
            d_vars: list containing difference in variance for the smallest
                cluster size for each cluster size
    """
    if not isinstance(X, np.ndarray):
        return None, None

    if not isinstance(kmin, int) or not isinstance(kmax, int):
        return None, None
    if kmin <= 0 or kmax <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    var = []
    results = []
    for i in range(kmin, kmax + 1):
        centroids, clss = kmeans(X, k=i, iterations=iterations)
        if centroids is not None:
            results.append((centroids, clss))
        variances = variance(X, centroids)
        var.append(variances)

    d_var = []
    for v in var:
        d_var.append(var[0] - v)

    return results, d_var
