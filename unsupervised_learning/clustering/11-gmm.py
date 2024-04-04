#!/usr/bin/env python3
"""
    Clustering : GMM  with scikit learn
"""
import sklearn.mixture


def gmm(X, k):
    """
        performs GMM on dataset with scikit-learn

    :param X: ndarray, shape(n,d) dataset
    :param k: number of cluster

    :return: pi, m, S, clss, bic
        pi: ndarray, shape(k,) cluster prior
        m: ndarray, shape(k,d) centroids means
        S: ndarray, shape(k, d, d) covariance matrices
        clss: ndarray, shape(n,)  cluster indices for each data point
        bic: ndarray, shape(kmax - kmin + 1)  BIC value
            for each cluster size tested
    """
    gaus_mixt = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

    pi = gaus_mixt.weights_
    m = gaus_mixt.means_
    S = gaus_mixt.covariances_
    clss = gaus_mixt.predict(X)
    bic = gaus_mixt.bic(X)

    return pi, m, S, clss, bic
