# Policy Gradients

## Introduction

Policy gradients are a class of reinforcement learning algorithms that are used to learn the policy of an agent. The policy is a function that maps states to actions. The goal of the agent is to learn a policy that maximizes the expected return. The expected return is the sum of rewards that the agent receives over time. Policy gradients are used to learn the policy by directly optimizing the expected return.

## Policy Gradient Theorem

The policy gradient theorem is a fundamental result in reinforcement learning that provides a way to compute the gradient of the expected return with respect to the policy parameters. The policy gradient theorem states that the gradient of the expected return with respect to the policy parameters is equal to the expected value of the gradient of the log-probability of the action multiplied by the return.

## Policy Gradient Algorithms

There are several policy gradient algorithms that are used to learn the policy of an agent. In this project we use `REINFORCE`

## TASKS

| Task                                                         | Description                                                                                                                      |
|--------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| [Simple Policy Function](./policy_gradients.py)              | Function `def policy(matrix, weight)` that computes to policy with a weight of a matrix                                          |
| [Compute Monte-Carlo policy Gradient](./policy_gradients.py) | Function `def policy_gradient(state, weight)` that computes the Monte-Carlo policy gradient based on a state and a weight matrix |
| [Implement training](./train.py)                             | Function `def train(env, nb_episodes, alpha=0.000045, gamma=0.98)` that implements a full training                               |
| [Animate iteration](./train.py)                             | Function `def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=True)` that displays the training results          |
