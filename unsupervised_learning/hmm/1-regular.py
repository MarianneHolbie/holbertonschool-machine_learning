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
    sums_P = np.sum(P, axis=1)
    for e in sums_P:
        if not np.isclose(e, 1):
            return None

    n = P.shape[0]

    # identity matrix
    Id_P = np.identity(n)

    # P - I
    PI = P - Id_P

    # vector
    ones = np.ones(n)

    # construct augmented matrix (concatened PI + ones) shape(n, n+1)
    q = np.c_[PI, ones]
    # q * q.T : shape(n,n)
    QTQ = np.dot(q, q.T)
    # q.T * qx = b
    bQT = np.ones(n)

    if np.linalg.det(PI) == 0:
        return None

    # resolve
    steady_state = np.linalg.solve(QTQ, bQT)

    return np.array([np.round(steady_state, 8)]).tolist()
