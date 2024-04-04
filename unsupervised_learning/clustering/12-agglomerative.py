#!/usr/bin/env python3
"""
    Clustering : agglomerative  with scipy
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
        performs agglomerative clustering on a dataset

    :param X: ndarray, shape(n,d) dataset
    :param dist: maximum cophenetic distance with Ward linkage

    :return: clss: ndarray, shape(n,) cluster indices
        for each data point
    """

    # clustering
    linkage_matrix = scipy.cluster.hierarchy.linkage(X, method='ward')

    # plot dendogramme
    plt.figure(figsize=(10, 5))
    scipy.cluster.hierarchy.dendrogram(linkage_matrix,
                                       color_threshold=100)
    plt.show()

    # clss
    clss = scipy.cluster.hierarchy.fcluster(linkage_matrix,
                                            dist,
                                            criterion='distance')

    return clss
