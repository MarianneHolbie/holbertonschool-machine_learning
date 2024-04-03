#!/usr/bin/env python3
"""
    Clustering : expectation maximization for a GMM
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
        performs expectation maximization for a GMM

    :param X: ndarray, shape(n,d), data set
    :param k: int, number of clusters
    :param iterations: int, max number of iterations for algo
    :param tol: non-neg float, tolerance of the likelihood (for early stopping)
    :param verbose: boolean, print information or not

    :return: pi, m, S, g, l or None, None, None, None, None
        pi: ndarray, shape(k,) priors for each cluster
        m: ndarray, shape(k,n) centroid means for each cluster
        S: ndarray, shape(k,d,d) covariance matrices for each cluster
        g: ndarray, shape(k,n) proba for each data point in each cluster
        l: log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <=0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    n, d = X.shape

    # initialize
    pi, m, S = initialize(X, k)

    likelihood_prev = 0

    for i in range(iterations):
        # expectation
        g, likelihood = expectation(X, pi, m, S)

        # maximization
        pi, m, S = maximization(X, g)

        # verbose
        if verbose:
            if i == 0 or i % 10 == 0:
                print("Log Likelihood after {} iterations: {}".format(i, likelihood.round(5)))

        diff = np.abs(likelihood - likelihood_prev)

        if diff <= tol:
            if verbose:
                print("Log Likelihood after {} iterations: {}".format(i, likelihood.round(5)))
            break

        likelihood_prev = likelihood

    return pi, m, S, g, likelihood
