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
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if kmax is not None and (type(kmax) is not int or kmax <= 0):
        return None, None
    if type(kmax) is not int or kmax <= 0:
        return None, None
    if kmin >= kmax:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    var = []
    results = []

    # minimal value
    centroids, clss = kmeans(X, k=kmin, iterations=iterations)
    minimal_var = variance(X, centroids)
    results.append((centroids, clss))
    var = [minimal_var]

    d_var = []
    for i in range(kmin + 1, kmax + 1):
        centroids, clss = kmeans(X, k=i, iterations=iterations)
        results.append((centroids, clss))
        variances = variance(X, centroids)
        var.append(variances)

    for v in var:
        d_var.append(var[0] - v)

    return results, d_var
