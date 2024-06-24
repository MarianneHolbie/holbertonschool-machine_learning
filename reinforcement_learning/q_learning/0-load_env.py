#!/usr/bin/env python3
"""
    Module to load environment frozen-lake from gym
"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
        load the premade FrozenLakeEnv env from OpenAI's gym

        :param desc: None or list custom description of the map to load
        :param map_name: None or string pre-made map to load
        :param is_slippery: bool, determine if the ice is slippery

        :return: the env
    """
    env = gym.make('FrozenLake-v0',
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)

    return env
