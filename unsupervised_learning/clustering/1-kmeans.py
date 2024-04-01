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

    # K-means algo
    for i in range(iterations):
        # distances between datapoints and centroids
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.empty((k, d), dtype=X.dtype)
        for j in range(k):
            mask = (clss == j)
            if np.any(mask):
                new_centroids[j] = np.mean(X[mask], axis=0)
            else:
                new_centroids[j] = X[np.random.choice(n)]

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, clss
