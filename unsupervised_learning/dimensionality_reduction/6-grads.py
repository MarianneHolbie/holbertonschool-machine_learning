#!/usr/bin/env python3
"""
    Gradients (t-SNE)
"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
        calculate the gradients of Y

    :param Y: ndarray, shape(n, ndim) low dimensional transformation of X
    :param P: ndarray, shape(n,n) P affinities of X

    :return: (dY, Q)
        dY: ndarray, shape(n,ndim) containing gradients of Y
        Q: ndarray, shape(n,n) containing Q affinities for Y
    """
    n, ndim = Y.shape
    # initialize dY
    dY = np.zeros((n, ndim))

    # calculate Qij and numerator
    Q, num = Q_affinities(Y)

    # Pij = Qij
    PQ = P - Q

    # computing gradient
    for i in range(n):
        # difference between P and Q weighted by their numerator
        PQij_num = (PQ[i, :] * num[i, :])
        # sumup resulting vector multiplied by differences in Y coordinates
        dY[i, :] = - np.dot(PQij_num.T, (Y - Y[i, :]))

    return dY, Q
