#!/usr/bin/env python3
"""
    initialize T-Distributed Stochastic Neighbor Embedding (t-SNE)
"""
import numpy as np


def P_init(X, perplexity):
    """
        function that initializes all variables required to calculate the P affinities
        in t-SNE

    :param X: ndarray, shape(n,d), dataset to be transformed by t-SNE
                n: number of data points
                d: number of dimensions in each point
    :param perplexity: perplexity that all Gaussian distribution should have
    :return: (D, P, betas, H)
        D: ndarray, shape(n,n) calculates squared pairwise distance between
            two data points (diag of D = 0s)
        P: ndarray, shape(n,n), initialized to all 0's that will contain
            P affinities
        betas: ndarray, shape(n,1), initialized to all 1's that will contain
            all the beta values
        H: Shanon entropy for perplexity with a base of 2
    """

    # calcul of Euclidean distance between every pair of points
    D = np.sum((X[None, :] - X[:, None])**2, 2)

    # initialization P, shape(n,n)
    n, d = X.shape
    P = np.zeros((n, n))

    # initialization betas, shape(n,1)
    betas = np.ones((n, 1))

    # calcul of H
    H = np.log2(perplexity)

    return D, P, betas, H
