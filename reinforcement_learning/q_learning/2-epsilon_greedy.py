#!/usr/bin/env python3
"""
    Module to implement Epsilon Greedy
"""
import gym
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
        function that uses epsilon-greedy to determine the next action

        :param Q: ndarray, Q-table
        :param state: current state
        :param epsilon: epsilon use for the calculation

        using random uniform distribution to set if algo explore or exploit
        using random.randint to determine exploration action

        :return: next action index
    """
    # determine if algo explore or exploit
    p = np.random.uniform(0, 1)

    # exploration : random exploration
    if p < epsilon:
        action = np.random.randint(0, Q.shape[1])
    # exploitation (p > epsilon)
    else:
        action = np.argmax(Q[state, :])

    return action
