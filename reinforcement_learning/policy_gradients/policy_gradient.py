#!/usr/bin/env python3
"""
    Policy Gradient
"""

import numpy as np


def policy(matrix, weight):
    """
        function that computes to policy with a weight of a matrix

    :param matrix: matrix, state
    :param weight: ndarray, weight to apply in policy

    :return: matrix of proba for each possible action
    """
    # matrix product: score for each possible action
    z = matrix @ weight

    # Softmax: normalize exp scores = distribution proba of action
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    softmax = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    return softmax
