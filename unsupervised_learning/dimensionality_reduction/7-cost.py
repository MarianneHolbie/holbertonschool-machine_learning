#!/usr/bin/env python3
"""
    Cost (t-SNE)
"""
import numpy as np


def cost(P, Q):
    """
        calculates the cost of the t-SNE transformation

    :param Y: ndarray, shape(n,n) containing P affinities
    :param P: ndarray, shape(n,n) containing Q affinities

    :return: c, the cost of the transformation
    """

    # avoid division by 0
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)

    c = np.sum(P * np.log(P / Q))

    return c
