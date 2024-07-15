#!/usr/bin/env python3
"""
    Policy Gradient training
"""
import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
        function implements a full training

    :param env: initial env
    :param nb_episodes: number of episodes used for training
    :param alpha: learning rate
    :param gamma: discount factor

    :return: sum of all rewards during one episode loop
    """
    # Initialize the weight
    weight = np.random.rand(*env.observation_space.shape, env.action_space.n)

    # Initialize the scores
    scores = []

    for episode in range(1, nb_episodes + 1):
        state = env.reset()[None, :]
        grad = 0
        score = 0
        done = False

        while not done:
            # Compute the action and the gradient
            action, delta_grad = policy_gradient(state, weight)

            # Take a step in the environment
            new_state, reward, done, info = env.step(action)
            new_state = new_state[None, :]

            # Update the score
            score += reward

            # Compute the gradient
            grad += delta_grad

            # Update the weight
            weight += (alpha * grad
                       * (reward + gamma * np.max(new_state.dot(weight))
                          * (not done) - state.dot(weight)[0, action]))

            # Update the state
            state = new_state

        # Store the score
        scores.append(score)

        # Print the current episode number and the score
        print("Episode: {}, Score: {}".format(
            episode, score), end="\r", flush=False)

        # Render the environment every 1000 episodes
        if show_result and episode % 1000 == 0:
            env.render()

    return scores
