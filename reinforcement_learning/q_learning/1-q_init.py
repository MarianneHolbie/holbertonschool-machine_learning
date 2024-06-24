#!/usr/bin/env python3
"""
    Module to initialize Q-table
"""
import gym
import numpy as np


def q_init(env):
    """
        function that initializes the Q-table

        :param env: FrozenLakeEnv instance

        :return: Q-table as a numpy.ndarray of zeros
    """
    # possibles obs
    obs = env.observation_space.n
    # possibles actions
    act = env.action_space.n

    # init Q-table
    q_table = np.zeros((obs, act))

    return q_table
