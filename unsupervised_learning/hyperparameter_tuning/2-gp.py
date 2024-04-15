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
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
            calculates covariance kernel matrix between two matrices
            Kernel use the Radial Basis Function(RBF)

        :param X1: ndarray, shape(m,1)
        :param X2: ndarray, shape(n,1)

        :return: cov kernel matrix
                ndarray, shape(m,n)
        """

        dist_mtx = np.sum(X1 ** 2, 1).reshape(-1, 1) + \
            np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        K = self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * dist_mtx)

        return K

    def predict(self, X_s):
        """
            predicts the mean and standard deviation of points
            in a Gaussian process

        :param X_s: ndarray, shape(s,1) all of the points whose mean and
            standard deviation should be calculated
            s: number of sample points

        :return: mu, sigma
            mu: ndarray, shape(s,) mean for each point in X_s
            sigma: ndarray, shape(s,) variance for each point in X_s
        """

        # cov matrix between the sample points X_s
        K_ss = self.kernel(X_s, X_s)
        # cross-cov between training points and sample point
        K_s = self.kernel(self.X, X_s)
        # inv of cov matrix K of training points
        K_inv = np.linalg.inv(self.K)

        # mu = dot product of K_s.T and K_inv multiply by self.Y
        mu = np.dot(K_s.T, np.dot(K_inv, self.Y)).reshape(-1)
        # sigma = diag of cov matrix K_ss minus the dot
        # product of K_s.T, K_inv and K_s)
        sigma = np.diag(K_ss - np.dot(K_s.T, np.dot(K_inv, K_s)))

        return mu, sigma

    def update(self, X_new, Y_new):
        """
            update Gaussian Process: public X, Y and K

        :param X_new: ndarray, shape(1,) new sample point
        :param Y_new: ndarray, shape(1,) new sample function value
        """
        # np.row_stack : concatenation along first axis
        self.X = np.row_stack((self.X, X_new))
        self.Y = np.row_stack((self.Y, Y_new))
        self.K = self.kernel(self.X, self.X)
