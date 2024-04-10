#!/usr/bin/env python3
"""
    HMM : Hidden Markov Models
        The Forward Algorithm
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
        Calculate the most likely sequence of hidden states
         for a hidden markov model

    :param Observation: ndarray, shape(T,) idx of the observation
        T: number of observations
    :param Emission: ndarray, shape(N,M) emission proba of a
        specific observation given a hidden state
            Emission[i,j] proba observing j given hidden state i
            N: number hidden states
            M: number of all possible observations
    :param Transition: ndarray, shape(N,N) transition proba
        Transition[i,j] proba of transitioning from hidden state
            i to j
    :param Initial: ndarray, shape(N,1) proba of starting in a
        particular hidden state

    :return: path, P or None, None on failure
        path: list of length T containing the most likely sequence
         of hidden states
        P: proba of obtaining the path sequence
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

    # create verbati mat and backtrack for found best path
    viterbi_mat = np.zeros((T, N))
    backtrack_mat = np.zeros((T, N), dtype=int)

    # initialization time 0
    viterbi_mat[0, :] = Initial.reshape(-1) * Emission[:, Observation[0]]

    # recurrent algo
    for t in range(1, T):
        for j in range(N):
            viterbi_mat[t, j] = (
                    np.max(viterbi_mat[t - 1, :] * Transition[:, j])
                    * Emission[j, Observation[t]])
            backtrack_mat[t, j] = np.argmax(viterbi_mat[t-1, :]
                                            * Transition[:, j])

    # found best path
    path = [0] * T
    path[T - 1] = np.argmax(viterbi_mat[T - 1, :])
    P = viterbi_mat[T - 1, path[T - 1]]

    # found best path through backtrack matrix
    for t in range(T - 2, -1, -1):
        path[t] = backtrack_mat[t + 1, path[t + 1]]

    return path, P
