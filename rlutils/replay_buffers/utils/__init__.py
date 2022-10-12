import gym
import numpy as np

from . import segtree


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


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
        'done': gym.spaces.Space(shape=None, dtype=np.float32)
    }
    return data_spec
