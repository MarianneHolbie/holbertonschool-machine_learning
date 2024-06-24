# Q LEARNING

## Introduction

Q-learning is a model-free reinforcement learning algorithm to learn quality of actions telling an agent what action to take under what circumstances. It does not require a model of the environment, and it can handle problems with stochastic transitions and rewards, without requiring adaptations.

## Requirements

- Python 3.5
- Numpy 1.15
- gym 0.13
- pycodestyle 2.4

## Tasks

| Task                                       | Description                                                                                                                                                            |
|--------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [0. Load the Environment](./0-load_env.py) | Write a function `load_frozen_lake(desc=None, map_name=None, is_slippery=False)` that loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym.                  |
| [1. Initialize Q-table](./1-q_init.py)     | Write a function `q_init(env)` that initializes the Q-table.                                                                                                           |
| [2. Epsilon Greedy](./2-epsilon_greedy.py) | Write a function `epsilon_greedy(Q, state, epsilon)` that uses epsilon-greedy to determine the next action.                                                            |
| [3. Q-learning](./3-q_learning.py)         | Write the function `q_learning(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05)` that performs Q-learning. |
| [4. Play](./4-play.py)                     | Write a function `play(env, Q, max_steps=100)` that has the trained agent play an episode.                                                                             |

