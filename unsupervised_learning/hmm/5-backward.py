#!/usr/bin/env python3
"""
    HMM : Hidden Markov Models
        The Backward Algorithm
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
        performs the backward algorithm for hidden markov model

    :param Observation: ndarray, shape(T,) idx of observation
        T: number of observations
    :param Emission: ndarray, shape(N,M), emission proba of a
        specific observation given a hidden state
        Emission[i][j]: proba observing j given hidden state i
        N: number of hidden states
        M: number of all possible observations
    :param Transition: ndarray, shape(N,N) transition proba
        Transition[i][j]: proba of transitioning from hidden
        state i to j
    :param Initial: ndarray, shape(N,1) proba of starting in
        a particular hidden state

    :return: P,B or None, None on failure
        P: likelihood of the observations given the model
        B: ndarray, shape(N,T) containing the backward path
        proba
            B[i,j]: proba generating the future observations
            form hidden state i at time j
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape
    if N != Transition.shape[0] or N != Transition.shape[1]:
        return None, None
    if (not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2
            or Initial.shape[0] != N or Initial.shape[1] != 1):
        return None, None

    # Forward algo
    alpha = np.zeros((N, T))
    # First observation using initial state
    alpha[:, 0] = Initial.reshape(-1) * Emission[:, Observation[0]]

    # For other observations
    for t in range(1, T):
        # for each hidden state
        for j in range(N):
            obs_idx = Observation[t]
            if obs_idx >= M:
                alpha[j, t] = 0  # Handle out-of-bounds observation index
            else:
                alpha[j, t] = np.dot(alpha[:, t - 1],
                                     Transition[:, j]) * Emission[j, obs_idx]

    # init
    backward_mat = np.zeros((N, T))
    backward_mat[:, T - 1] = 1

    for t in reversed(range(T - 1)):
        for i in range(N):
            backward_mat[i, t] = np.sum(Transition[:, i]
                                        * Emission[:, Observation[t]]
                                        * backward_mat[:, t + 1])

    P = np.sum(alpha[:, T - 1])

    return P, backward_mat
