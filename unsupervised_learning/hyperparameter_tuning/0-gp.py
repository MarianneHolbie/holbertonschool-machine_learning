#!/usr/bin/env python3
"""
    This module contains the implementation of the Gaussian Process
"""
import numpy as np


class GaussianProcess:
    """
        Noiseless 1D Gaussian process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
            class constructor

        :param X_init: ndarray, shape(t,1) inputs already sampled with the
            black-box function
            t : number of initial samples
        :param Y_init: ndarray, shape(t,1) outputs of the black-box function
            for each input in X_init
        :param l: length parameter for the kernel
        :param sigma_f: standard deviation given to the output of the black-box
            function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.Y)

    def kernel(self, X1, X2):
        """
            calculates covariance kernel matrix between two matrices
            Kernel use the Radial Basis Function(RBF)

        :param X1: ndarray, shape(m,1)
        :param X2: ndarray, shape(n,1)

        :return: cov kernel matrix
                ndarray, shape(m,n)
        """

        dist_mtx = (np.sum(X1 ** 2, 1).reshape(-1, 1)
                    + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T))
        K = self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * dist_mtx)

        return K
