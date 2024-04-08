#!/usr/bin/env python3
"""
    HMM : Hidden Markov Models
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
        determines the probability of a markov chain being in a particular
        state after a specified number of iterations

    :param P: ndarray, shape(n,n), transition matrix
        P[i, j] proba of transitioning from state i to state j
        n: number of states in the markov chain
    :param s: ndarray, shape(1,n) proba of starting in each state
    :param t: number of iterations that the markov chain has been through

    :return: ndarray, shape(1,n) proba of being in a specific state
            after t iterations, or None on failure
    """

    current_state = s

    for i in range(t):
        next_state = np.matmul(current_state, P)
        current_state = next_state

    return current_state
