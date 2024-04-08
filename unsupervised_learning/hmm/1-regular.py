#!/usr/bin/env python3
"""
    HMM : Hidden Markov Models
        regular chains
"""
import numpy as np


def regular(P):
    """
        determines the steady state probabilities of a regular markov chain

    :param P: ndarray, shape(n,n) transition matrix
        P[i, j] proba of transition from state i to state j
        n: number of states in the markov chain

    :return: ndarray, shape(1,n) steady state probabilities
        or None on failure
    """

    if not isinstance(P, np.ndarray) or P.shape[0] != P.shape[1]:
        return None
    if len(P.shape) != 2:
        return None
    # Check if the sum of each row is 1
    if not np.allclose(P.sum(axis=1), 1):
        return None

    n = P.shape[0]

    # Create a matrix for the system of linear equations
    A = P.T - np.eye(n)
    b = np.ones(n)

    # Solve the system of linear equations (Ax = b)
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    # Normalize the solution vector to ensure the sum is 1
    x = x / x.sum()

    return x.reshape(1, n)
