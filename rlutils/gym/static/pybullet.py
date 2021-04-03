import numpy as np

from .base import ModelBasedStaticFn


class InvertedPendulumBulletEnvFn(ModelBasedStaticFn):
    env_name = ['InvertedPendulumBulletEnv-v0']
    terminate = True
    reward = True

    @staticmethod
    def terminate_fn_numpy_batch(states, actions, next_states):
        cos_th, sin_th = next_states[:, 2], next_states[:, 3]
        theta = np.arctan2(sin_th, cos_th)
        return np.abs(theta) > .2

    @staticmethod
    def terminate_fn_torch_batch(states, actions, next_states):
        import torch
        cos_th, sin_th = next_states[:, 2], next_states[:, 3]
        theta = torch.atan2(sin_th, cos_th)
        return torch.abs(theta) > .2

    @staticmethod
    def terminate_fn_tf_batch(states, actions, next_states):
        import tensorflow as tf
        cos_th, sin_th = next_states[:, 2], next_states[:, 3]
        theta = tf.atan2(sin_th, cos_th)
        return tf.abs(theta) > .2

    @staticmethod
    def reward_fn_tf_batch(states, actions, next_states):
        import tensorflow as tf
        cos_th, sin_th = next_states[:, 2], next_states[:, 3]
        theta = tf.atan2(sin_th, cos_th)
        cost = tf.cast(tf.abs(theta), tf.float32)
        return -cost

    @staticmethod
    def reward_fn_torch_batch(states, actions, next_states):
        import torch
        cos_th, sin_th = next_states[:, 2], next_states[:, 3]
        theta = torch.atan2(sin_th, cos_th)
        cost = torch.abs(theta).float()
        return -cost

    @staticmethod
    def reward_fn_numpy_batch(states, actions, next_states):
        cos_th, sin_th = next_states[:, 2], next_states[:, 3]
        theta = np.arctan2(sin_th, cos_th)
        cost = np.abs(theta).astype(np.float32)
        return -cost


class InvertedPendulumSwingupBulletEnvFn(ModelBasedStaticFn):
    env_name = ['InvertedPendulumSwingupBulletEnv-v0']
    reward = True
    terminate = True

    @staticmethod
    def reward_fn_tf_batch(states, actions, next_states):
        return -next_states[:, 2]

    @staticmethod
    def reward_fn_numpy_batch(states, actions, next_states):
        return -next_states[:, 2]

    @staticmethod
    def reward_fn_torch_batch(states, actions, next_states):
        return -next_states[:, 2]


class ReacherBulletEnvFn(ModelBasedStaticFn):
    env_name = ['ReacherBulletEnv-v0']
    terminate = True
    reward = True

    def reward_fn_tf_batch(states, actions, next_states):
        import tensorflow as tf
        old_to_target_vec = states[:, 2:4]
        to_target_vec = next_states[:, 2:4]
        theta_dot = next_states[:, 6]
        gamma = next_states[:, 7]
        gamma_dot = next_states[:, 8]

        old_potential = 100 * tf.sqrt(tf.reduce_sum(old_to_target_vec ** 2, axis=-1))
        potential = 100 * tf.sqrt(tf.reduce_sum(to_target_vec ** 2, axis=-1))

        electricity_cost = (
                0.10 * (tf.abs(actions[:, 0] * theta_dot) + tf.abs(actions[:, 1] * gamma_dot))
                + 0.01 * (tf.abs(actions[:, 0]) + tf.abs(actions[:, 1]))
        )

        stuck_joint_cost = 0.1 * tf.cast((tf.abs(tf.abs(gamma) - 1) < 0.01), dtype=tf.float32)

        cost = potential - old_potential + electricity_cost + stuck_joint_cost
        return -cost

    def reward_fn_numpy_batch(states, actions, next_states):
        old_to_target_vec = states[:, 2:4]
        to_target_vec = next_states[:, 2:4]
        theta_dot = next_states[:, 6]
        gamma = next_states[:, 7]
        gamma_dot = next_states[:, 8]

        old_potential = 100 * np.sqrt(np.sum(old_to_target_vec ** 2, axis=-1))
        potential = 100 * np.sqrt(np.sum(to_target_vec ** 2, axis=-1))

        electricity_cost = (
                0.10 * (np.abs(actions[:, 0] * theta_dot) + np.abs(actions[:, 1] * gamma_dot))
                + 0.01 * (np.abs(actions[:, 0]) + np.abs(actions[:, 1]))
        )

        stuck_joint_cost = 0.1 * (np.abs(np.abs(gamma) - 1) < 0.01).astype(np.float32)

        cost = potential - old_potential + electricity_cost + stuck_joint_cost
        return -cost

    def reward_fn_torch_batch(states, actions, next_states):
        import torch
        old_to_target_vec = states[:, 2:4]
        to_target_vec = next_states[:, 2:4]
        theta_dot = next_states[:, 6]
        gamma = next_states[:, 7]
        gamma_dot = next_states[:, 8]

        old_potential = 100 * torch.sqrt(torch.sum(old_to_target_vec ** 2, dim=-1))
        potential = 100 * torch.sqrt(torch.sum(to_target_vec ** 2, dim=-1))

        electricity_cost = (
                0.10 * (torch.abs(actions[:, 0] * theta_dot) + torch.abs(actions[:, 1] * gamma_dot))
                + 0.01 * (torch.abs(actions[:, 0]) + torch.abs(actions[:, 1]))
        )

        stuck_joint_cost = 0.1 * (torch.abs(torch.abs(gamma) - 1) < 0.01).float()

        cost = potential - old_potential + electricity_cost + stuck_joint_cost
        return -cost


class HopperBulletEnvFn(ModelBasedStaticFn):
    env_name = ['HopperBulletEnv-v0']
    terminate = True
    reward = False

    # the initial_z is 1.25
    @staticmethod
    def terminate_fn_tf_batch(states, actions, next_states):
        import tensorflow as tf
        # +1 if z > 0.8 and abs(pitch) < 1.0 else -1
        z = next_states[:, 0]
        p = next_states[:, 7]
        return tf.logical_or(z <= -0.45, tf.abs(p) >= 1.0)

    @staticmethod
    def terminate_fn_numpy_batch(states, actions, next_states):
        # +1 if z > 0.8 and abs(pitch) < 1.0 else -1
        z = next_states[:, 0]
        p = next_states[:, 7]
        return np.logical_or(z <= -0.45, np.abs(p) >= 1.0)

    @staticmethod
    def terminate_fn_torch_batch(states, actions, next_states):
        import torch
        # +1 if z > 0.8 and abs(pitch) < 1.0 else -1
        z = next_states[:, 0]
        p = next_states[:, 7]
        return torch.abs(p) >= 1.0 | z <= -0.45


class Walker2DBulletEnvFn(HopperBulletEnvFn):
    env_name = ['Walker2DBulletEnv-v0']


class HalfCheetahBulletEnvFn(HopperBulletEnvFn):
    env_name = ['HalfCheetahBulletEnv-v0']
    terminate = True
    reward = False


class AntBulletEnvFn(HopperBulletEnvFn):
    env_name = ['AntBulletEnv-v0']
    terminate = True
    reward = False

    @staticmethod
    def terminate_fn_tf_batch(states, actions, next_states):
        # +1 if z > 0.26 else -1
        z = next_states[:, 0]
        return z <= -0.49

    @staticmethod
    def terminate_fn_numpy_batch(states, actions, next_states):
        # +1 if z > 0.8 and abs(pitch) < 1.0 else -1
        z = next_states[:, 0]
        return z <= -0.49

    @staticmethod
    def terminate_fn_torch_batch(states, actions, next_states):
        # +1 if z > 0.8 and abs(pitch) < 1.0 else -1
        z = next_states[:, 0]
        return z <= -0.49


# retrieve all the class

import sys, inspect

model_based_wrapper_dict = {}


def register():
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            for name in obj().env_name:
                model_based_wrapper_dict[name] = obj


if len(model_based_wrapper_dict) == 0:
    register()
