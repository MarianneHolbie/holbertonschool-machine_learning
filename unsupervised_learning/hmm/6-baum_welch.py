#!/usr/bin/env python3
"""
    HMM : Hidden Markov Models
        The Baum Welch Algorithm
"""
import numpy as np
forward = __import__('3-forward').forward
backward = __import__('5-backward').backward


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
        perform Baum-Welch algo for HMM

    :param Observations: ndarray, shape(T,) idx of obs
        T: number of observations
    :param Transition: ndarrayn shape(M,M) initialized transition proba
        M: number hidden states
    :param Emission: ndarray, shape(M,N) initialized emission proba
        N: number of output states
    :param Initial: ndarray, shape(M,1) initialized starting proba
    :param iterations: number of times expectation-maximisation should
        be performed

    :return: converged Transition, Emission or None, None on failure
    """
    if not isinstance(Observations, np.ndarray) or len(Observations.shape) != 1:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2 \
            or Transition.shape[0] != Transition.shape[1]:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2 \
            or Emission.shape[0] != Transition.shape[0]:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2 \
            or Initial.shape[0] != Emission.shape[0]:
        return None, None

    T = Observations.shape[0]
    M, N = Emission.shape

    for idx in range(iterations):

        P, b = backward(Observations, Emission, Transition, Initial)
        P_f, f = forward(Observations, Emission, Transition, Initial)

        x_i = np.zeros((T, M, M))
        gamma = np.zeros((T, M))

        for t in range(T - 1):
            for i in range(M):
                for j in range(M):
                    x_i[t, i, j] = (f[i, t] * Transition[i, j]
                                    * Emission[j, Observations[t+1]]
                                    * b[j, t + 1]) / P

        for t in range(T):
            gamma[t, :] = np.sum(x_i[t, :, :], axis=1)

        Initial_new = np.zeros(Initial.shape)
        for i in range(M):
            Initial_new[i] = gamma[0, i]

        Transition_new = np.zeros(Transition.shape)
        for i in range(M):
            for j in range(M):
                Transition_new[i, j] = (np.sum(x_i[:, i, j])
                                        / np.sum(gamma[:, i]))

        Emission_new = np.zeros(Emission.shape)
        for j in range(M):
            for k in range(N):
                indices = Observations == k
                Emission_new[j, k] = (np.sum(gamma[indices, j])
                                      / np.sum(gamma[:, j]))

    return Transition_new, Emission_new
