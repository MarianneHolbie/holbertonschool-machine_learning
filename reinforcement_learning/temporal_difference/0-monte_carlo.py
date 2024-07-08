#!/usr/bin/env python3
"""
    Monte Carlo algorithm
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
        Function that perform Monte Carlo algorithm

    :param env: openAI env instance
    :param V: ndarray, shape(s,) value estimate
    :param policy: function that takes in state and return next action
    :param episodes: total number of episodes to train over
    :param max_steps: max number of steps per episode
    :param alpha: learning rate
    :param gamma: discount rate

    :return: V, updated value estimate
    """
    for ep in range(episodes):
        # start new episode
        state = env.reset()

        episode_data = []

        for step in range(max_steps):
            # determine action based on policy
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            # store tuple state reward
            episode_data.append((state, reward))

            if done or step > max_steps:
                break

            state = next_state

        # convert episode_data in ndarray for efficiency
        episode_data = np.array(episode_data, dtype=int)

        G = 0
        # calculate return for each state in episode
        for s, r in episode_data[::-1]:
            # update return
            G = gamma * G + r
            # first visit
            if s not in episode_data[:ep, 0]:
                V[s] = V[s] + alpha * (G - V[s])

    # precision option to have exactly same value as expected
    np.set_printoptions(precision=4, suppress=True)

    return V.round(4)
