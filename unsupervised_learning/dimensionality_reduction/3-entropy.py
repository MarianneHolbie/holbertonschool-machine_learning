#!/usr/bin/env python3
"""
    Entropy (t-SNE)
"""
import numpy as np


def HP(Di, beta):
    """
        calculates the Shannon entropy and P affinities relative
        to data point

    :param Di: ndarray, shape(n-1,) pairwise distances between a data point
            and other points except itself
                n: number of data points
    :param beta: ndarray, shape(1,) beta value for Gaussian distribution

    :return: (Hi, Pi)
        Hi: Shannon entropy of the points
        Pi: ndarray, shape(n-1,) P affinities of the points
    """
    # Calcul Pi affinities, don't divide by 2beta^2 because Di: distance one to
    # other points
    exp_val = np.exp(-Di / beta)
    sum_exp_val = np.sum(exp_val)
    Pi = exp_val / sum_exp_val

    # remove 0 from Pi to avoid log(0)
    # replace 0 by the smallest possible value
    new_Pi = np.maximum(Pi, np.finfo(float).eps)

    # calcul of Shannon entropy
    Hi = - np.sum(new_Pi * np.log2(new_Pi))

    return Hi, new_Pi
