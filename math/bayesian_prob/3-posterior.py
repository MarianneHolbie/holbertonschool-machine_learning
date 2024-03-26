#!/usr/bin/env python3
"""
    Bayesian Probability :Likelihood, Intersection, Marginal Probability,
                        and Posterior
"""

import numpy as np


def likelihood(x, n, P):
    """
        function calculates the likelihood of obtaining this data given
        various hypothetical probabilities of developing severe side effects.

    :param x: number of patients that develop severe side effects
    :param n: total number of patients observed
    :param P: 1D ndarray, various hypothetical proba of developing
     severe side effects

    :return: 1D ndarray, containing likelihood of obtaining data, x and n
        for each probability in P, respectively

        ***** BINOMIALE ****
        P(X = x) = coeff binomial * p^x * (1 - p)^(n - x)

        coeff binomial = n! / ( x! * (n - x)!)
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise (ValueError
               ("x must be an integer that is greater than or equal to 0"))

    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # define coefficient binomial
    coeff_bin = (np.math.factorial(n)
                 / (np.math.factorial(x) * np.math.factorial(n - x)))

    # calculate on all value of P
    proba = coeff_bin * P ** x * (1 - P) ** (n - x)

    return proba


def intersection(x, n, P, Pr):
    """
        function that calculates the intersection of obtaining this data with
        the various hypothetical probabilities

    :param x: number of patients that develop severe side effects
    :param n: total number of patients observed
    :param P: 1D ndarray, various hypothetical probabilities of
     developing severe side effects
    :param Pr: 1d ndarray, prior beliefs of P

    :return:
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise (ValueError
               ("x must be an integer that is greater than or equal to 0"))

    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P > 1) | np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) | np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if np.abs(np.sum(Pr) - 1) > 1e-10:
        raise ValueError("Pr must sum to 1")

    # calculate likelihood
    val_likelihood = likelihood(x, n, P)

    # calculate intersection
    inter = val_likelihood * Pr

    return inter


def marginal(x, n, P, Pr):
    """
        function that calculates the marginal probability of obtaining the data

    :param x: number of patients that develop severe side effects
    :param n: total number of patients observed
    :param P: 1D ndarray, various hypothetical probabilities of
     developing severe side effects
    :param Pr: 1d ndarray, prior beliefs of P

    :return: marginal probability of obtaining x and n
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise (ValueError
               ("x must be an integer that is greater than or equal to 0"))

    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P > 1) | np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) | np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if np.abs(np.sum(Pr) - 1) > 1e-10:
        raise ValueError("Pr must sum to 1")

    # calculate likelihood
    val_likelihood = likelihood(x, n, P)

    # calculate marginal
    marginal_prob = np.sum(val_likelihood * Pr)

    return marginal_prob

def posterior(x, n, P, Pr):
    """
        function that calculates the marginal probability of obtaining the data

    :param x: number of patients that develop severe side effects
    :param n: total number of patients observed
    :param P: 1D ndarray, various hypothetical probabilities of
     developing severe side effects
    :param Pr: 1d ndarray, prior beliefs of P

    :return: the posterior probability of each probability in P given
        x and n, respectively
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise (ValueError
               ("x must be an integer that is greater than or equal to 0"))

    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P > 1) | np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) | np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if np.abs(np.sum(Pr) - 1) > 1e-10:
        raise ValueError("Pr must sum to 1")

    # calculate likelihood
    val_likelihood = likelihood(x, n, P)

    # calculate marginal
    val_marginal = marginal(x, n, P, Pr)

    # calculate posterior
    val_posterior = np.multiply(Pr, val_likelihood) / val_marginal

    return val_posterior
