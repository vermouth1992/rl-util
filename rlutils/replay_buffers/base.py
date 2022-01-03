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

import gym
import numpy as np
from gym.utils import seeding

from rlutils.np.functional import shuffle_dict_data
from .storage import PyDictStorage, Storage
from .utils import get_data_spec_from_env, get_data_spec_from_vec_env


class BaseReplayBuffer(ABC):
    def __init__(self, capacity, seed=None):
        self.set_seed(seed)
        self.storage = self._create_storage(capacity)
        self.reset()

    def _create_storage(self, capacity) -> Storage:
        raise NotImplementedError

    def reset(self):
        self.storage.reset()

    def set_seed(self, seed=None):
        self.np_random, self.seed = seeding.np_random(seed)

    def __len__(self):
        return len(self.storage)

    def add(self, data):
        self.storage.add(data)

    @property
    def capacity(self):
        return self.storage.capacity

    @abstractmethod
    def sample(self, batch_size):
        raise NotImplementedError

    def is_full(self):
        return self.storage.is_full()

    def is_empty(self):
        return self.storage.is_empty()

    def post_process(self, info):
        pass


class PyDictReplayBuffer(BaseReplayBuffer):
    def __init__(self, data_spec, capacity, seed=None):
        self.data_spec = data_spec
        super(PyDictReplayBuffer, self).__init__(capacity, seed)

    def _create_storage(self, capacity):
        return PyDictStorage(self.data_spec, capacity)

    @classmethod
    def from_data_dict(cls, data: Dict[str, np.ndarray], shuffle=False, seed=None):
        if shuffle:
            data = shuffle_dict_data(data)
        data_spec = {key: gym.spaces.Space(shape=item.shape[1:], dtype=item.dtype) for key, item in data.items()}
        capacity = list(data.values())[0].shape[0]
        replay_buffer = cls(data_spec=data_spec, capacity=capacity, seed=seed)
        replay_buffer.add(data=data)
        assert replay_buffer.is_full()
        return replay_buffer

    @classmethod
    def from_vec_env(cls, vec_env, capacity, seed=None):
        data_spec = get_data_spec_from_vec_env(vec_env)
        return cls(data_spec=data_spec, capacity=capacity, seed=seed)

    @classmethod
    def from_env(cls, env, capacity, seed=None):
        data_spec = get_data_spec_from_env(env)
        return cls(data_spec=data_spec, capacity=capacity, seed=seed)


class MemoryEfficientDictReplayBuffer(PyDictReplayBuffer):
    def _create_storage(self, capacity):
        return MemoryEfficientDictReplayBuffer(self.data_spec, capacity)

    @classmethod
    def from_vec_env(cls, vec_env, capacity, seed=None):
        data_spec = get_data_spec_from_vec_env(vec_env, memory_efficient=True)
        return cls(data_spec=data_spec, capacity=capacity, seed=seed)

    @classmethod
    def from_env(cls, env, capacity, seed=None):
        data_spec = get_data_spec_from_env(env, memory_efficient=True)
        return cls(data_spec=data_spec, capacity=capacity, seed=seed)
