#!/usr/bin/env python3
"""
    Clustering : variance
"""
import numpy as np


def variance(X, C):
    """
        calculate the total intra-cluster variance for a data set

    :param X: ndarray, shape(n,d) data set
    :param C: ndarray, shape(k,d) centroid means for each cluster

    :return: var or None if failure
        var: total variance
    """

    distance = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    min_dist = np.min(distance, axis=1)
    var = np.sum(min_dist ** 2)

    return var
