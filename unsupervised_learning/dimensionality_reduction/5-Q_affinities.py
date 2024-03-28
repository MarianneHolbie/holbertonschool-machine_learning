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

    # calcul of Euclidean distance between every pair of points
    Yi = np.sum((Y[None, :] - Y[:, None]) ** 2, 2)
    np.fill_diagonal(Yi, 0)

    # calculate numerator of Q affinities
    num = 1 / (1 + Yi)
    np.fill_diagonal(num, 0)

    # calculate Q affinities
    Q = num / np.sum(num)

    return Q, num
