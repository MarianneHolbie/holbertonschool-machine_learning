#!/usr/bin/env python3
"""
    PCA on a dataset
"""
import numpy as np


def pca(X, var=0.95):
    """
    Function that performs PCA on a dataset

    :param X: numpy.ndarray of shape (n, d) where:
        - n is the number of data points
        - d is the number of dimensions in each point
        - all dimensions have a mean of 0 across all data points
    :param var: the  fraction of the variance that the PCA transformation
        should maintain

    :return: the weights matrix, W,
        that maintains var fraction of X‘s original variance
    """

    # covariance matrix
    cov_mat = np.cov(X, rowvar=False)

    # eigenvalue, eigenvector
    eigen_val, eigen_vector = np.linalg.eig(cov_mat)

    # sort eigenvalue, eigenvector
    sorted_index = np.argsort(eigen_val)[::-1]
    sorted_eigen_vector = eigen_vector[:, sorted_index]

    # fraction of variance
    variance_kept = np.cumsum(eigen_val[sorted_index]) / np.sum(eigen_val)

    # components to be conserved
    nb_comp = np.argmax(variance_kept >= var) + 1

    # select corresponding vector = matrix W
    W = sorted_eigen_vector[:, :nb_comp]

    return W
