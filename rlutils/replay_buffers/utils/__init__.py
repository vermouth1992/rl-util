from typing import Dict

import gym
import numpy as np

from . import segtree


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def get_data_spec_from_dataset(dataset: Dict[str, np.ndarray], obj_keys=None):
    if obj_keys is None:
        obj_keys = set()
    data_spec = {}
    data_size = None
    for key, data in dataset.items():
        if key in obj_keys:
            print(f'Store key {key} as object')
            data_spec[key] = None
        else:
            data_spec[key] = gym.spaces.Space(shape=data.shape[1:], dtype=data.dtype)

        if data_size is None:
            data_size = data.shape[0]
        else:
            assert data_size == data.shape[0]
    return data_spec, data_size


def get_data_spec_from_env(env, memory_efficient=False):
    if memory_efficient:
        obs_spec = None
    else:
        obs_spec = env.observation_space

    act_spec = env.action_space

    data_spec = {
        'obs': obs_spec,
        'act': act_spec,
        'next_obs': obs_spec,
        'rew': gym.spaces.Space(shape=None, dtype=np.float32),
        'done': gym.spaces.Space(shape=None, dtype=np.float32),
        'gamma': gym.spaces.Space(shape=None, dtype=np.float32)
    }
    return data_spec
