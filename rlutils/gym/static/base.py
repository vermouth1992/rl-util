"""
Static functions for environments
"""

import numpy as np


class ModelBasedStaticFn(object):
    env_name = 'Base'
    terminate = False
    reward = False
    device = 'cpu'

    def set_device(self, device):
        self.device = device

    @staticmethod
    def terminate_fn_numpy_batch(states, actions, next_states):
        return np.zeros(shape=(states.shape[0]), dtype=np.bool)

    @staticmethod
    def terminate_fn_torch_batch(states, actions, next_states):
        import torch
        return torch.zeros(states.shape[0], dtype=torch.bool, device=self.device)

    @staticmethod
    def terminate_fn_tf_batch(states, actions, next_states):
        import tensorflow as tf
        return tf.zeros(shape=(states.shape[0]), dtype=tf.bool)

    @staticmethod
    def reward_fn_numpy_batch(states, actions, next_states):
        raise NotImplementedError

    @staticmethod
    def reward_fn_torch_batch(states, actions, next_states):
        raise NotImplementedError

    @staticmethod
    def reward_fn_tf_batch(states, actions, next_states):
        raise NotImplementedError
