# Reinforcement learning

## Deep Q Learning

### Introduction

Deep Q Learning is a reinforcement learning algorithm that combines Q-Learning with deep neural networks to learn the optimal policy. The algorithm was introduced by DeepMind in 2013 and has been used to achieve superhuman performance

### Algorithm

The algorithm uses a neural network to approximate the Q function. The Q function is a function that takes the state and action as input and returns the expected reward. The neural network is trained using the Bellman equation, which is a recursive equation that defines the optimal policy.

### Implementation

The algorithm is implemented using Python and TensorFlow. The code is available on GitHub and can be run on a CPU or GPU.

### TASKS

| Task                   | Description                                                                              |
|------------------------|------------------------------------------------------------------------------------------|
| [train.py](./train.py) | utilizes `keras`, `keras-rl`, and `gym` to train an agent that can play Atari’s Breakout |
| [play.py](./play.py)   | display a game played by the agent trained by `train.py`                                 |