import threading

import numpy as np
from gym.utils import seeding

from . import storage, utils


class UniformReplayBuffer(object):
    def __init__(self, capacity, data_spec, memory_efficient, seed=None):
        self.data_spec = data_spec
        self.memory_efficient = memory_efficient
        if self.memory_efficient:
            self.storage = storage.MemoryEfficientPyDictStorage(data_spec=self.data_spec, capacity=capacity)
        else:
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
            if self.memory_efficient:
                for key in self.storage.obj_key:
                    data[key] = np.array(data[key])
            return data

    @classmethod
    def from_env(cls, env, is_vec_env, memory_efficient, **kwargs):
        data_spec = utils.get_data_spec_from_env(env, is_vec_env=is_vec_env,
                                                 memory_efficient=memory_efficient)
        return cls(data_spec=data_spec, memory_efficient=memory_efficient, **kwargs)
