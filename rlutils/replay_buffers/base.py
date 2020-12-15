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
        self.max_size = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_buf = np.zeros(combined_shape(self.capacity, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(self.capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(self.capacity, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(shape=(self.capacity), dtype=np.float32)
        self.done_buf = np.zeros(shape=(self.capacity), dtype=np.float32)
        self.batch_size = batch_size
        self.ptr, self.size = 0, 0

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        """ Make it compatible with Pytorch data loaders """
        return dict(obs=self.obs_buf[item],
                    next_obs=self.obs2_buf[item],
                    act=self.act_buf[item],
                    rew=self.rew_buf[item],
                    done=self.done_buf[item])

    @property
    def capacity(self):
        return self.max_size

    def add(self, data, priority=None):
        assert priority is None, 'Uniform Replay Buffer'
        obs = data['obs']
        act = data['act']
        rew = data['rew']
        next_obs = data['next_obs']
        done = data['done']
        self.obs_buf[self.ptr:self.ptr + self.num_parallel_env, :] = obs
        self.obs2_buf[self.ptr:self.ptr + self.num_parallel_env, :] = next_obs
        self.act_buf[self.ptr:self.ptr + self.num_parallel_env, :] = act
        self.rew_buf[self.ptr:self.ptr + self.num_parallel_env] = rew
        self.done_buf[self.ptr:self.ptr + self.num_parallel_env] = done
        self.ptr = (self.ptr + self.num_parallel_env) % self.capacity
        self.size = min(self.size + self.num_parallel_env, self.capacity)

    def sample(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     next_obs=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return batch
