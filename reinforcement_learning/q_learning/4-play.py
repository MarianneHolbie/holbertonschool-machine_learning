#!/usr/bin/env python3
"""
    Module to play FrozenLake
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
        function that has the trained agent play an episode

    :param env: FrozenLakeEnv instance
    :param Q: ndarray containing Q-table
    :param max_steps: max number of steps in the episode

    :return: total rewards for the episode
    """

    # intitial state
    state = env.reset()

    total_rewards = 0

    for step in range(max_steps):
        # actual state
        env.render()

        # select best action in Q-table
        action = np.argmax(Q[state, :])

        # apply action
        new_state, reward, done, info = env.step(action)

        # update reward
        total_rewards += reward

        # next state
        state = new_state

        if done:
            break

    return total_rewards
