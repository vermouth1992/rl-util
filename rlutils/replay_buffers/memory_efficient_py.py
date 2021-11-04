"""
Memory efficient replay buffer. Instead of storing everything in explict numpy array, the observation
is stored in LazyFrames, which stores the points to the actual data to avoid duplication
"""

from typing import Dict

import gym.spaces
import numpy as np

from .base import DictReplayBuffer
from .utils import combined_shape


class PyMemoryEfficientReplayBuffer(DictReplayBuffer):
    def _create_storage(self):
        storage = {}
        self.np_key = []
        self.obj_key = []
        for key, item in self.data_spec.items():
            if isinstance(item, gym.spaces.Space):
                storage[key] = np.zeros(combined_shape(self.capacity, item.shape), dtype=item.dtype)
                self.np_key.append(key)
            else:
                print(f"Store key {key} as an object")
                storage[key] = np.zeros(self.capacity, dtype=object)
                self.obj_key.append(key)
        return storage

    def __getitem__(self, item):
        data = {key: self.storage[key][item] for key in self.np_key}
        for key in self.obj_key:
            output = []
            for idx in item:
                output.append(self.storage[key][idx])
            data[key] = np.array(output)
        return data

    def add(self, data: Dict[str, np.ndarray]):
        batch_size = data[self.np_key[0]].shape[0]
        for key, item in data.items():
            if isinstance(item, np.ndarray):
                if self.ptr + batch_size > self.max_size:
                    print('Reaches the end of the replay buffer')
                    self.storage[key][self.ptr:] = item[:self.max_size - self.ptr]
                    self.storage[key][:batch_size - (self.max_size - self.ptr)] = item[self.max_size - self.ptr:]
                else:
                    self.storage[key][self.ptr:self.ptr + batch_size] = item
            elif isinstance(item, list):
                for i in range(batch_size):
                    self.storage[key][(self.ptr + i) % self.max_size] = item[i]
            else:
                raise ValueError(f'Unknown type {type(item)}')

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self):
        assert not self.is_empty()
        idxs = self.np_random.randint(0, self.size, size=self.batch_size)
        return self.__getitem__(idxs)

    @classmethod
    def from_env(cls, env, capacity, batch_size, seed=None, **kwargs):
        data_spec = {
            'obs': None,
            'act': env.action_space,
            'next_obs': None,
            'rew': gym.spaces.Space(shape=None, dtype=np.float32),
            'done': gym.spaces.Space(shape=None, dtype=np.float32)
        }
        return cls(data_spec=data_spec, capacity=capacity, batch_size=batch_size, seed=seed, **kwargs)

    @classmethod
    def from_vec_env(cls, vec_env, capacity, batch_size, seed=None, **kwargs):
        data_spec = {
            'obs': None,
            'act': vec_env.single_action_space,
            'next_obs': None,
            'rew': gym.spaces.Space(shape=None, dtype=np.float32),
            'done': gym.spaces.Space(shape=None, dtype=np.float32)
        }
        return cls(data_spec=data_spec, capacity=capacity, batch_size=batch_size, seed=seed, **kwargs)
