import inspect
import sys

import numpy as np

from .base import ModelBasedStaticFn

model_based_wrapper_dict = {}


class ReacherFn(ModelBasedStaticFn):
    reward = False
    terminate = True
    env_name = ['Reacher-v2']


class HopperFn(ModelBasedStaticFn):
    reward = False
    terminate = True
    env_name = ['Hopper-v2']

    @staticmethod
    def terminate_fn_tf_batch(states, actions, next_states):
        import tensorflow as tf
        height = next_states[:, 0]
        angle = next_states[:, 1]
        t1 = tf.reduce_all(next_states[:, 1:] < 100., axis=-1)
        t2 = height > 0.7
        t3 = tf.abs(angle) < 0.2
        not_done = tf.logical_and(tf.logical_and(t1, t2), t3)
        return tf.logical_not(not_done)

    @staticmethod
    def terminate_fn_torch_batch(states, actions, next_states):
        import torch
        height = next_states[:, 0]
        angle = next_states[:, 1]
        t1 = torch.all(next_states[:, 1:] < 100., dim=-1)
        t2 = height > 0.7
        t3 = torch.abs(angle) < 0.2
        not_done = t1 & t2 & t3
        return torch.logical_not(not_done)

    @staticmethod
    def terminate_fn_numpy_batch(states, actions, next_states):
        assert len(states.shape) == len(next_states.shape) == len(actions.shape) == 2

        height = next_states[:, 0]
        angle = next_states[:, 1]
        not_done = np.isfinite(next_states).all(axis=-1) \
                   * np.abs(next_states[:, 1:] < 100).all(axis=-1) \
                   * (height > .7) \
                   * (np.abs(angle) < .2)

        done = ~not_done
        return done


class Walker2dFn(ModelBasedStaticFn):
    reward = False
    terminate = True
    env_name = ['Walker2d-v2']

    @staticmethod
    def terminate_fn_tf_batch(states, actions, next_states):
        import tensorflow as tf
        height = next_states[:, 0]
        angle = next_states[:, 1]
        t1 = tf.logical_and(height > 0.8, height < 2.0)
        t2 = tf.logical_and(angle > -1.0, angle < 1.0)
        not_done = tf.logical_and(t1, t2)
        return tf.logical_not(not_done)

    @staticmethod
    def terminate_fn_torch_batch(states, actions, next_states):
        import torch
        height = next_states[:, 0]
        angle = next_states[:, 1]
        t1 = height > 0.8 & height < 2.0
        t2 = angle > -1.0 & angle < 1.0
        not_done = t1 & t2
        return torch.logical_not(not_done)

    @staticmethod
    def terminate_fn_numpy_batch(states, actions, next_states):
        assert len(states.shape) == len(next_states.shape) == len(actions.shape) == 2

        height = next_states[:, 0]
        angle = next_states[:, 1]
        not_done = (height > 0.8) \
                   * (height < 2.0) \
                   * (angle > -1.0) \
                   * (angle < 1.0)
        done = ~not_done
        return done


class HalfCheetahFn(ModelBasedStaticFn):
    reward = False
    terminate = True
    env_name = ['HalfCheetah-v2']


class AntFn(ModelBasedStaticFn):
    reward = False
    terminate = True
    env_name = ['Ant-v2', 'AntTruncatedObs-v2']

    @staticmethod
    def terminate_fn_numpy_batch(states, actions, next_states):
        assert len(states.shape) == len(next_states.shape) == len(actions.shape) == 2

        x = next_states[:, 0]
        not_done = np.isfinite(next_states).all(axis=-1) \
                   * (x >= 0.2) \
                   * (x <= 1.0)

        done = ~not_done
        return done

    @staticmethod
    def terminate_fn_torch_batch(states, actions, next_states):
        import torch
        x = next_states[:, 0]
        not_done = torch.logical_and(x >= 0.2, x <= 1.0)
        return torch.logical_not(not_done)

    @staticmethod
    def terminate_fn_tf_batch(states, actions, next_states):
        import tensorflow as tf
        x = next_states[:, 0]
        not_done = tf.logical_and(x >= 0.2, x <= 1.0)
        return tf.logical_not(not_done)


class HumanoidFn(ModelBasedStaticFn):
    reward = False
    terminate = True
    env_name = ['Humanoid-v2']

    @staticmethod
    def terminate_fn_numpy_batch(states, actions, next_states):
        assert len(states.shape) == len(next_states.shape) == len(actions.shape) == 2

        z = next_states[:, 0]
        done = (z < 1.0) + (z > 2.0)

        return done

    @staticmethod
    def terminate_fn_torch_batch(states, actions, next_states):
        z = next_states[:, 0]
        done = (z < 1.0) | (z > 2.0)
        return done

    @staticmethod
    def terminate_fn_tf_batch(states, actions, next_states):
        import tensorflow as tf
        z = next_states[:, 0]
        done = tf.logical_or(z < 1.0, z > 2.0)
        return done


def register():
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            for name in obj().env_name:
                model_based_wrapper_dict[name] = obj


if len(model_based_wrapper_dict) == 0:
    register()
