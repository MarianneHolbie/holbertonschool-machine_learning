#!/usr/bin/env python3
"""
    Clustering
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
        performs K-means on a dataset

    :param X: ndarray, shape(n,d) dataset
        n: number of data points
        d: number of dimensions for each data point
    :param k: int, number of clusters
    :param iterations: positiv int, maximum number of iterations

    :return: C, clss or None, None on failure
            C: ndarray, shape(k,d) centroid means for each cluster
            clss: ndarray, shape(n,) index of cluster in c that
                each data point belongs to
    """

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    # first define centroid with multivariate uniform distribution
    centroids = np.random.uniform(low=low, high=high, size=(k, d))
    new_centroids = np.empty((k, d), dtype=X.dtype)

    # K-means algo
    for i in range(iterations):
        # distances between datapoints and centroids
        distances = np.sqrt(np.sum((X - centroids[:, np.newaxis]) ** 2),
                            axis=-1)
        clss = np.argmin(distances, axis=0)

        # Update centroids
        for j in range(k):
            mask = (clss == j)
            if np.any(mask):
                new_centroids[j] = X[mask].mean(axis=0)
            else:
                new_centroids[j] = (
                    np.random.uniform(low=low, high=high, size=(1, d)))

        # Check convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids.copy()

    return centroids, clss
