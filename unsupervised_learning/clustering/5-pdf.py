#!/usr/bin/env python3
"""
    Clustering : PDF
"""
import numpy as np


def pdf(X, m, S):
    """
        calculates the probability density function of a Gaussian distribution

    :param X: ndarray, shape(n,d) data points whose PDF should be evaluated
    :param m: ndarray, shape(d,) mean of the distribution
    :param S: ndarray, shape(d,d) covariance of the distribution

    :return: P or None on failure
        P: ndarray, shape(n,) PDF values for each data point

    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    d = X.shape[1]

    constant = 1 / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(S))
    inv_cov = np.linalg.inv(S)
    expo_term = np.exp(-0.5 * np.sum((X - m) @ inv_cov * (X - m), axis=1))
    P = constant * expo_term

    P = np.maximum(P, 1e-300)

    return P
