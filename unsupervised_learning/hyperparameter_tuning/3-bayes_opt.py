#!/usr/bin/env python3
"""
    This module contains the implementation
    of the Bayesian Optimization Process
"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
        performs Bayesian optimization on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
            class construtor

        :param f: black-box function
        :param X_init: ndarray, shape(t,1), inputs already sampled with the
            black-box function
        :param Y_init: ndarray, shape(t,1), outputs of the black-box function
            for each input in X_init
        :param bounds: tuple, (min,max) bounds of the space in which to look
            for optimal point
        :param ac_samples: number of samples that should be analyzed during
            acquisition
        :param l: length parameter for the kernel
        :param sigma_f: standard deviation given to the output of the black-box
            function
        :param xsi: exploration-exploitation factor for acquisition
        :param minimize: bool, True: for minimization, False: for maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
