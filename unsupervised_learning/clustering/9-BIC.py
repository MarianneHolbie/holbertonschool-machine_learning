#!/usr/bin/env python3
"""
    Clustering : a GMM using the Bayesian Information Criterion
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
        Finds the best number of clusters for a GMM using
        the Bayesian Information Criterion

    :param X: ndarray, shape(n,d) data set
    :param kmin: int, min number of cluster to check
    :param kmax: int, max number of cluster to check
    :param iterations: int, max number of iterations for EM
    :param tol: float, tolerance for EM algo
    :param verbose: bool, print the EM algo standard output

    :return: best_k, best_result, l, b, or None, None, None, None on failure
        best_k: best value for k based on its BIC
        best_result: tuple pi, m, S
            pi: ndarray, shape(k,) cluster priors for best k
            m: ndarray, shape(k,d) centroid means for best k
            S: ndarray, shape(K,d,d) cov matrices for best k
        l: ndarray, shape(kmax-kmin+1) log likelihood
            for each cluster size tested
        b: ndarray, shape(kmax-kmin+1) BIC value for each cluster size tested
            Use BIC=p*ln(n)-2*l
            p: number parameters required for the model
            n: number data points used to create model
            l: log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax < kmin:
        return None, None, None, None
    if kmax is not None and (kmax <= kmin):
        return None, None, None, None

    n, d = X.shape

    # kmax set to maximum number of clusters possible
    if kmax is None:
        kmax = n

    lh = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)

    best_k = None
    best_result = None
    best_bic = float('inf')

    # test all value between kmin and kmax
    for k in range(kmin, kmax + 1):
        pi, m, S, g, likelihood \
            = expectation_maximization(X, k, iterations, tol, verbose)

        lh[k - kmin] = likelihood

        # number of param (-1 correction)
        p = k * (d + 1) + k * d * (d + 1) / 2 - 1
        # BIC value
        b[k - kmin] = p * np.log(n) - 2 * likelihood

        if b[k - kmin] < best_bic:
            best_bic = b[k - kmin]
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, lh, b
