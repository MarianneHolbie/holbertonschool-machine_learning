#!/usr/bin/env python3
"""
    Module to implement Q-learning
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
        function to performs Q-learning

    :param env: FrozenLake instance
    :param Q: ndarray, Q-table
    :param episodes: total number of episode to train over
    :param max_steps: max number of steps per episode
    :param alpha: learning rate
    :param gamma: discount rate
    :param epsilon: initial threshold for epsilon greedy
    :param min_epsilon: min value that epsilon should decay to
    :param epsilon_decay: decay rate for updating epsilon between episodes

    :return: Q, total_rewards
        Q: updated Q-table
        total_rewards = list containing the rewards per episode
    """

    total_rewards = []

    for episode in range(episodes):
        episode_rewards = 0

        state = env.reset()
        for step in range(max_steps):
            # determined action
            action = epsilon_greedy(Q, state, epsilon)
            # set new_state and associated reward
            next_state, reward, done, info = env.step(action)

            # update reward if hole
            if done and reward == 0:
                reward = -1

            episode_rewards += reward

            # update Q-table
            next_value = np.max(Q[next_state])
            Q[state, action] *= 1 - alpha
            Q[state, action] += alpha * (reward + gamma * next_value)

            # set new begin state
            state = next_state

            if done:
                break

        # update epsilon
        epsilon = (min_epsilon + (1 - min_epsilon)
                   * np.exp(-epsilon_decay * episode))

        total_rewards.append(episode_rewards)

    return Q, total_rewards
