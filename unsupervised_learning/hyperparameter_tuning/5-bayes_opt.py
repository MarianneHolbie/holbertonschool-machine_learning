#!/usr/bin/env python3
"""
    This module contains the implementation
    of the Bayesian Optimization Process
"""
import numpy as np
from scipy.stats import norm

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
        self.X = X_init
        self.Y = Y_init
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
            calculates the next best sample location
            Uses the Expected Improvement acquisition function

        :return: X_next, EI
            X_next: ndarray, shape(1,) next best sample point
            EI: ndarray, (ac_samples,) expected improvement of each
                potential sample
        """

        y_pred, y_std = self.gp.predict(self.X_s)

        # Minimize objective function
        if self.minimize:
            best_idx = np.argmin(self.gp.Y)
            best_y = self.gp.Y[best_idx]

            z = (best_y - y_pred - self.xsi) / y_std
            ei = (best_y - y_pred - self.xsi) * norm.cdf(z) + \
                y_std * norm.pdf(z)
        # maximize objective function
        else:
            best_idx = np.argmax(self.gp.Y)
            best_y = self.gp.Y[best_idx]

            z = (y_pred - best_y - self.xsi) / y_std
            ei = (y_pred - best_y - self.xsi) * norm.cdf(z) + \
                y_std * norm.pdf(z)

        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei

    def optimize(self, iterations=100):
        """
            optimizes the black-box function
            - stop process if proposed point is one that has already
            been sampled (early stopping)

        :param iterations: maximum number of iterations to perform

        :return: X_opt, Y_opt
            X_opt: ndarray, shape(1,) optimal point
            Y_opt: ndarray, shape(1,) optimal function value
        """

        for i in range(iterations):
            X_next, ei = self.acquisition()

            # check if X_next has already been sampled
            if np.any(np.isclose(X_next, self.gp.X)):
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        # Get optimal point and value
        best_idx = np.argmin(self.gp.Y) if self.minimize \
            else np.argmax(self.gp.Y)
        self.gp.X = self.gp.X[:-1, :]
        X_opt = self.gp.X[best_idx]
        Y_opt = self.gp.Y[best_idx]

        return X_opt, Y_opt
