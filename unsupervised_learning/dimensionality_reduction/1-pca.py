#!/usr/bin/env python3
"""
    PCA on a dataset
"""
import numpy as np


def pca(X, ndim):
    """
    Function that performs PCA on a dataset

    :param X: numpy.ndarray of shape (n, d) where:
        - n is the number of data points
        - d is the number of dimensions in each point
    :param ndim: new dimensionality of the transformed X

    :return: T, a numpy.ndarray of shape (n, ndim)
        containing the transformed version of X
    """

    # Calculate the SVD of input data
    U, S, V = np.linalg.svd(X, full_matrices=False)

    # select first ndim
    W = V[:ndim].T

    # Apply the transformation to X
    T = X @ W

    return T
