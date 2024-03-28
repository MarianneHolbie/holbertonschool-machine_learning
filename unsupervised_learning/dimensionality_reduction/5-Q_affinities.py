#!/usr/bin/env python3
"""
    Q affinities (t-SNE)
"""
import numpy as np


def Q_affinities(Y):
    """
        calculate the Q affinities

    :param Y: ndarray, shape(n, ndim) containing low dimensional
        transformation of x
            n: number of points
            ndim: new dimensional representation of x

    :return: Q, num
        Q: ndarray, shape(n,n) containing Q affinities
        num: ndarray, shape(n,n) containing numerator of the Q affinities
    """

    n, ndim = Y.shape
    Q = np.zeros((n, n))

    # calcul of Euclidean distance between every pair of points
    Yi = np.sum((Y[None, :] - Y[:, None]) ** 2, 2)

    # calculate numerator of Q affinities
    num = 1 / (1 + Yi)

    # calculate Q affinities
    Q = num / np.sum(num)

    return Q, num
