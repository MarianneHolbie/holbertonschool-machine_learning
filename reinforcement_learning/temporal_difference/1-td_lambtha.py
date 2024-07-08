#!/usr/bin/env python3
"""
    TD(λ) algorithm
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha,
               episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
        Function that perform TD(λ) algorithm

    :param env: openAI env instance
    :param V: ndarray, shape(s,) value estimate
    :param policy: function that takes in state and return next action
    :param lambtha: eligibility trace factor
    :param episodes: total number of episodes to train over
    :param max_steps: max number of steps per episode
    :param alpha: learning rate
    :param gamma: discount rate

    :return: V, updated value estimate
    """

    for ep in range(episodes):
        # start new episode
        state = env.reset()
        eligibility = np.zeros_like(V)

        for step in range(max_steps):
            # determine action based on policy
            action = policy(state)
            next_state, reward, done, _ = env.step(action)

            # TD error
            delta = reward + (gamma * V[next_state]) - V[state]

            # update eligibilities
            eligibility *= lambtha * gamma
            eligibility[state] += 1

            # Update value function
            V = V + alpha * delta * eligibility

            if done:
                break

            state = next_state

        return V
