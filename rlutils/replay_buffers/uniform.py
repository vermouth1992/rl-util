import threading
from typing import Dict

import numpy as np
from gym.utils import seeding

from . import storage, utils


class UniformReplayBuffer(object):
    def __init__(self, capacity, data_spec, seed=None):
        self.data_spec = data_spec
        self.storage = storage.PyDictStorage(self.data_spec, capacity)
        self.set_seed(seed)

        self.lock = threading.Lock()

        self.storage.reset()

    def __len__(self):
        with self.lock:
            return len(self.storage)

    def add(self, data):
        with self.lock:
            self.storage.add(data)

    @property
    def capacity(self):
        return self.storage.capacity

    def is_full(self):
        with self.lock:
            return self.storage.is_full()

    def is_empty(self):
        with self.lock:
            return self.storage.is_empty()

    def set_seed(self, seed=None):
        self.np_random, self.seed = seeding.np_random(seed)

    def sample(self, batch_size):
        assert not self.is_empty()
        with self.lock:
            idxs = self.np_random.integers(0, len(self.storage), size=batch_size)
            data = self.storage[idxs]
            for key in self.storage.obj_key:
                data[key] = np.array(data[key])
            return data

    @classmethod
    def from_env(cls, env, memory_efficient, **kwargs):
        data_spec = utils.get_data_spec_from_env(env, memory_efficient=memory_efficient)
        return cls(data_spec=data_spec, **kwargs)

    @classmethod
    def from_dataset(cls, dataset: Dict[str, np.ndarray], obj_keys=None, **kwargs):
        # sanity check
        if obj_keys is None:
            obj_keys = set()
        data_spec, capacity = utils.get_data_spec_from_dataset(dataset, obj_keys=obj_keys)
        replay_buffer = cls(data_spec=data_spec, capacity=capacity, **kwargs)
        replay_buffer.add(dataset)
        assert replay_buffer.is_full()
        return replay_buffer
