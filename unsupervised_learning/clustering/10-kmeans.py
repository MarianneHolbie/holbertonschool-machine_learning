#!/usr/bin/env python3
"""
    Clustering : K-means with scikit learn
"""
import sklearn.cluster


def kmeans(X, k):
    """
        performs K-means on dataset with scikit-learn

    :param X: ndarray, shape(n,d) dataset
    :param k: number of cluster

    :return: C, clss
        C: ndarray, shape(k,d) containing centroid means for each cluster
        clss: ndarray, shape(n,) containing index of cluster in C
            that each data point belongs to
    """
    k_means = sklearn.cluster.KMeans(n_clusters=k).fit(X)

    C = k_means.cluster_centers_
    clss = k_means.labels_

    return C, clss
