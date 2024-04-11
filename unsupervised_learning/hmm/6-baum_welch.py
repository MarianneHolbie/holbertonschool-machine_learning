#!/usr/bin/env python3
"""
    HMM : Hidden Markov Models
        The Baum Welch Algorithm
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
        performs the forward algorithm for a hidden markov model

    :param Observation: ndarray, shape(T,) index of observation
        T: number of observation
    :param Emission: ndarray, shape(N,M) emission proba of specific
        observation given a hidden state
            Emission[i,j] proba of observing j given the hidden state i
            N: number of hidden state
            M: number of all possible observations
    :param Transition: ndarray, shape(N,N) transition proba
        transition[i,j] proba transitioning from hidden state i to j
    :param Initial: ndarray, shape(N,1) proba of starting in a particular
        hidden state

    :return: P, F or None, None on failure
        P: likelihood of observations given model
        F: ndarray, shape(N,T) forward path proba
            F[i,j] proba being in hidden state i at time j given the
                previous observations
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

    P = np.sum(alpha[:, T - 1])

    return P, alpha


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

    # init
    backward_mat = np.zeros((N, T))
    backward_mat[:, T - 1] = 1

    for t in reversed(range(T - 1)):
        for i in range(N):
            backward_mat[i, t] = np.sum(Transition[i, :]
                                        * Emission[:, Observation[t + 1]]
                                        * backward_mat[:, t + 1])

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]]
               * backward_mat[:, 0])

    return P, backward_mat


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
    if (not isinstance(Observations, np.ndarray)
            or len(Observations.shape) != 1):
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

    Initial_new = np.zeros(Initial.shape)
    Transition_new = np.zeros(Transition.shape)
    Emission_new = np.zeros(Emission.shape)

    # loop of Baum-Welch algo
    for idx in range(iterations):

        # compute backward and forward proba and likelihood
        P_b, b = backward(Observations, Emission, Transition, Initial)
        P_f, f = forward(Observations, Emission, Transition, Initial)

        # x_i joint proba hidden state at time t + transition
        x_i = np.zeros((T, M, M))
        # marginal proba of being particular state at each time step
        gamma = np.zeros((T, M))

        # compute joint proba
        for t in range(T - 1):
            for i in range(M):
                for j in range(M):
                    x_i[t, i, j] = (f[i, t] * Transition[i, j]
                                    * Emission[j, Observations[t +1]]
                                    * b[j, t + 1]) / P_f

        # compute marginal proba : sum joint proba
        for t in range(T):
            gamma[t, :] = np.sum(x_i[t, :, :], axis=1)

        # update initial state proba
        Initial_new = gamma[0, :]

        # update transition proba using joint proba and marginal proba
        for i in range(M):
            for j in range(M):
                Transition_new[i, j] += (np.sum(x_i[:, i, j])
                                         / np.sum(gamma[:, i]))

        # update estimate emission proba
        for j in range(M):
            for k in range(N):
                indices = np.where(Observations == k)[0]
                Emission_new[j, k] += (np.sum(gamma[indices, j])
                                       / np.sum(gamma[:, j]))

    # normalize (for sum to 1)
    Transition_new /= iterations
    Emission_new /= iterations

    return Transition_new, Emission_new
