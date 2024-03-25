#!/usr/bin/env python3
"""
    Likelihood
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
