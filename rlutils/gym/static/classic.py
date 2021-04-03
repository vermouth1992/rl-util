import numpy as np

from .base import ModelBasedStaticFn


class PendulumFn(ModelBasedStaticFn):
    env_name = ['Pendulum-v0']
    terminate = True
    reward = True

    @staticmethod
    def reward_fn_tf_batch(states, actions, next_states):
        import tensorflow as tf
        cos_th, sin_th, thdot = states[:, 0], states[:, 1], states[:, 2]
        th = tf.atan2(sin_th, cos_th)

        costs = th ** 2 + .1 * thdot ** 2 + .001 * (actions[:, 0] ** 2)
        return -costs

    @staticmethod
    def reward_fn_numpy_batch(states, actions, next_states):
        cos_th, sin_th, thdot = states[:, 0], states[:, 1], states[:, 2]
        th = np.arctan2(sin_th, cos_th)

        costs = th ** 2 + .1 * thdot ** 2 + .001 * (actions[:, 0] ** 2)
        return -costs

    @staticmethod
    def reward_fn_torch_batch(states, actions, next_states):
        import torch
        cos_th, sin_th, thdot = states[:, 0], states[:, 1], states[:, 2]
        th = torch.atan2(sin_th, cos_th)
        costs = th ** 2 + .1 * thdot ** 2 + .001 * (actions[:, 0] ** 2)
        return -costs


import sys
import inspect

model_based_wrapper_dict = {}


def register():
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            for name in obj().env_name:
                model_based_wrapper_dict[name] = obj


if len(model_based_wrapper_dict) == 0:
    register()
