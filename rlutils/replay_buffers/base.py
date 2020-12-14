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


class PyUniformReplayBuffer(BaseReplayBuffer):
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, batch_size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.batch_size = batch_size
        self.ptr, self.size, self.max_size = 0, 0, size

    def __len__(self):
        return self.size

    @property
    def capacity(self):
        return self.max_size

    def load(self, data):
        obs = data['obs']
        act = data['act']
        next_obs = data['next_obs']
        rew = data['rew']
        done = data['done']
        self.size = obs.shape[0]
        assert self.size <= self.max_size
        self.obs_buf = obs
        self.act_buf = act
        self.obs2_buf = next_obs
        self.rew_buf = rew
        self.done_buf = done
        self.ptr = (self.size + 1) % self.max_size

    def add(self, data, priority=None):
        assert priority is None, 'Uniform Replay Buffer'
        obs, act, rew, next_obs, done = data
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return batch
