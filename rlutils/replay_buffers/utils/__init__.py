import gym
import numpy as np


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def get_data_spec_from_env(env, memory_efficient=False):
    data_spec = {
        'obs': None if memory_efficient else env.observation_space,
        'act': env.action_space,
        'next_obs': None if memory_efficient else env.observation_space,
        'rew': gym.spaces.Space(shape=None, dtype=np.float32),
        'done': gym.spaces.Space(shape=None, dtype=np.float32)
    }
    return data_spec


def get_data_spec_from_vec_env(vec_env, memory_efficient=False):
    data_spec = {
        'obs': None if memory_efficient else vec_env.single_observation_space,
        'act': vec_env.single_action_space,
        'next_obs': None if memory_efficient else vec_env.single_observation_space,
        'rew': gym.spaces.Space(shape=None, dtype=np.float32),
        'done': gym.spaces.Space(shape=None, dtype=np.float32)
    }
    return data_spec
