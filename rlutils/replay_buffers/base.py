"""
Abstract class for replay buffers
1. Uniform sampling dataset.
2. Prioritized replay
3. n-step return
4. Frame stack
5. Trajectory-based replay buffer for on-policy methods
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseReplayBuffer(ABC):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def add(self, data, priority=1.0):
        raise NotImplementedError

    @property
    @abstractmethod
    def capacity(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    def load(self, data):
        pass


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class PyUniformParallelEnvReplayBuffer(BaseReplayBuffer):
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, capacity, batch_size, num_parallel_env):
        assert capacity % num_parallel_env == 0
        assert batch_size % num_parallel_env == 0
        self.num_parallel_env = num_parallel_env
        self.per_env_capacity = capacity // num_parallel_env
        self.per_env_batch_size = batch_size // num_parallel_env
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_buf = np.zeros(combined_shape(self.per_env_capacity, (num_parallel_env, obs_dim)), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(self.per_env_capacity, (num_parallel_env, obs_dim)), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(self.per_env_capacity, (num_parallel_env, act_dim)), dtype=np.float32)
        self.rew_buf = np.zeros(shape=(self.per_env_capacity, num_parallel_env), dtype=np.float32)
        self.done_buf = np.zeros(shape=(self.per_env_capacity, num_parallel_env), dtype=np.float32)
        self.batch_size = batch_size
        self.ptr, self.per_env_size = 0, 0

    def __len__(self):
        return self.per_env_size * self.num_parallel_env

    @property
    def capacity(self):
        return self.per_env_capacity * self.num_parallel_env

    def add(self, data, priority=None):
        assert priority is None, 'Uniform Replay Buffer'
        obs, act, rew, next_obs, done = data
        self.obs_buf[self.ptr, :] = obs
        self.obs2_buf[self.ptr, :] = next_obs
        self.act_buf[self.ptr, :] = act
        self.rew_buf[self.ptr, :] = rew
        self.done_buf[self.ptr, :] = done
        self.ptr = (self.ptr + 1) % self.per_env_capacity
        self.per_env_size = min(self.per_env_size + 1, self.per_env_capacity)

    def sample(self):
        idxs = np.random.randint(0, self.per_env_size, size=self.per_env_batch_size)
        batch = dict(obs=self.obs_buf[idxs].reshape(-1, self.obs_dim),
                     obs2=self.obs2_buf[idxs].reshape(-1, self.obs_dim),
                     act=self.act_buf[idxs].reshape(-1, self.act_dim),
                     rew=self.rew_buf[idxs].reshape(-1),
                     done=self.done_buf[idxs].reshape(-1))
        return batch
