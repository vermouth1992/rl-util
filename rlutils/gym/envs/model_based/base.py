import abc

import gym
import numpy as np


class ModelBasedEnv(gym.Env):

    @abc.abstractmethod
    def terminate_fn_numpy_batch(self, obs, action, next_obs):
        return np.zeros(shape=(obs.shape[0]), dtype=np.bool)

    @abc.abstractmethod
    def terminate_fn_torch_batch(self, obs, action, next_obs):
        import torch
        return torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)

    @abc.abstractmethod
    def reward_fn_numpy_batch(self, obs, action, next_obs):
        raise NotImplementedError

    @abc.abstractmethod
    def reward_fn_torch_batch(self, obs, action, next_obs):
        raise NotImplementedError
