"""
Abstract class for replay buffers
1. Uniform sampling dataset.
2. Prioritized replay
3. n-step return
4. Frame stack
5. Trajectory-based replay buffer for on-policy methods
"""

from abc import ABC, abstractmethod
from typing import Dict

import gym.spaces
import numpy as np
from gym.utils import seeding
from rlutils.np.functional import shuffle_dict_data

from .utils import combined_shape


class BaseReplayBuffer(ABC):
    def __init__(self, seed=None):
        self.set_seed(seed)

    def reset(self):
        pass

    def set_seed(self, seed=None):
        self.np_random, self.seed = seeding.np_random(seed)

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def add(self, data):
        raise NotImplementedError

    @property
    @abstractmethod
    def capacity(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    def load(self, data):
        raise NotImplementedError

    def append(self, data):
        raise NotImplementedError

    def is_full(self):
        return len(self) == self.capacity

    def is_empty(self):
        return len(self) <= 0


class PyReplayBuffer(BaseReplayBuffer):
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self,
                 data_spec: Dict[str, gym.spaces.Space],
                 capacity,
                 batch_size,
                 seed=None,
                 **kwargs):
        super(PyReplayBuffer, self).__init__(seed=seed)
        self.max_size = capacity
        self.data_spec = data_spec
        self.storage = {key: np.zeros(combined_shape(self.capacity, item.shape), dtype=item.dtype)
                        for key, item in data_spec.items()}
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.ptr, self.size = 0, 0

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        """ Make it compatible with Pytorch data loaders """
        return {key: data[item] for key, data in self.storage.items()}

    @property
    def capacity(self):
        return self.max_size

    @classmethod
    def from_data_dict(cls, data: Dict[str, np.ndarray], batch_size, shuffle=False, seed=None, **kwargs):
        if shuffle:
            data = shuffle_dict_data(data)
        data_spec = {key: gym.spaces.Space(shape=item.shape[1:], dtype=item.dtype) for key, item in data.items()}
        capacity = list(data.values())[0].shape[0]
        replay_buffer = cls(data_spec=data_spec, capacity=capacity, batch_size=batch_size, seed=seed, **kwargs)
        replay_buffer.append(data=data)
        assert replay_buffer.is_full()
        return replay_buffer

    @classmethod
    def from_vec_env(cls, vec_env, capacity, batch_size, seed=None, **kwargs):
        data_spec = {
            'obs': vec_env.single_observation_space,
            'act': vec_env.single_action_space,
            'next_obs': vec_env.single_observation_space,
            'rew': gym.spaces.Space(shape=None, dtype=np.float32),
            'done': gym.spaces.Space(shape=None, dtype=np.float32)
        }
        return cls(data_spec=data_spec, capacity=capacity, batch_size=batch_size, seed=seed, **kwargs)

    @classmethod
    def from_env(cls, env, capacity, batch_size, seed=None, **kwargs):
        data_spec = {
            'obs': env.observation_space,
            'act': env.action_space,
            'next_obs': env.observation_space,
            'rew': gym.spaces.Space(shape=None, dtype=np.float32),
            'done': gym.spaces.Space(shape=None, dtype=np.float32)
        }
        return cls(data_spec=data_spec, capacity=capacity, batch_size=batch_size, seed=seed, **kwargs)

    def append(self, data: Dict[str, np.ndarray]):
        batch_size = list(data.values())[0].shape[0]
        for key, item in data.items():
            assert batch_size == item.shape[0], 'Mismatch batch size in the dataset'

        if self.ptr + batch_size > self.capacity:
            print(f'Truncated dataset due to limited capacity. Original size {batch_size}. '
                  f'Truncated size {self.capacity - self.ptr}')
        for key, item in data.items():
            self.storage[key][self.ptr:self.ptr + batch_size] = item

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def get(self):
        idxs = np.arange(self.size)
        return self.__getitem__(idxs)

    def add(self, data: Dict[str, np.ndarray]):
        batch_size = list(data.values())[0].shape[0]
        for key, item in data.items():
            assert batch_size == item.shape[0], 'The batch size in the data is not consistent'
            if self.ptr + batch_size > self.max_size:
                print('Reaches the end of the replay buffer')
                self.storage[key][self.ptr:] = item[:self.max_size - self.ptr]
                self.storage[key][:batch_size - (self.max_size - self.ptr)] = item[self.max_size - self.ptr:]
            else:
                self.storage[key][self.ptr:self.ptr + batch_size] = item
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
