"""
Adapted from https://github.com/thu-ml/tianshou/
"""

from typing import Dict

import numpy as np

EPS = np.finfo(np.float32).eps.item()

from gym.utils import seeding
import threading

from . import utils, storage


class PrioritizedReplayBuffer(object):
    def __init__(self, data_spec, capacity, memory_efficient, alpha=0.6, beta=0.4, eviction=None, seed=None):
        self.eviction = eviction
        if eviction is None:
            print('Using FIFO eviction policy')
            self.eviction_tree = None
        else:
            assert eviction < 0.
            print(f'Using prioritized eviction policy with alpha_evict={eviction}')
            self.eviction_tree = utils.segtree.SumTree(size=capacity)
        if memory_efficient:
            self.storage = storage.MemoryEfficientPyDictStorage(data_spec=data_spec, capacity=capacity)
        else:
            self.storage = storage.PyDictStorage(data_spec=data_spec, capacity=capacity)
        self.storage.reset()
        self.alpha = alpha
        self.beta = beta
        self.max_tree = utils.segtree.MaxTree(size=capacity)
        self.min_tree = utils.segtree.MinTree(size=capacity)
        self.sum_tree = utils.segtree.SumTree(size=capacity)
        self.lock = threading.Lock()
        self.set_seed(seed=seed)

        self.sampled_idx = []
        self.sampled_mask = []

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

    def __len__(self):
        with self.lock:
            return len(self.storage)

    def add(self, data: Dict[str, np.ndarray], priority: np.ndarray = None):
        batch_size = data[list(data.keys())[0]].shape[0]
        if priority is None:
            if len(self) == 0:
                max_priority = 1.0
            else:
                max_priority = self.max_tree.reduce()
            priority = np.ones(shape=(batch_size,), dtype=np.float32) * max_priority
        with self.lock:
            priority = np.abs(priority) + EPS
            # assert np.all(priority > 0.), f'Priority must be all greater than zero. Got {priority}'
            if self.eviction_tree is None or (not self.storage.is_full()):
                idx = self.storage.add(data)
            else:
                scalar = self.np_random.random(batch_size) * self.eviction_tree.reduce()
                eviction_idx = self.eviction_tree.get_prefix_sum_idx(scalar)
                idx = self.storage.add(data, index=eviction_idx)
                assert np.all(idx == eviction_idx)

            self.sum_tree[idx] = priority ** self.alpha
            self.max_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha
            if self.eviction_tree is not None:
                self.eviction_tree[idx] = priority ** self.eviction

            for i in range(len(self.sampled_idx)):
                sampled_idx = self.sampled_idx[i]
                mask = np.in1d(sampled_idx, idx, invert=True)  # False if in the idx
                old_mask = self.sampled_mask[i]
                self.sampled_mask[i] = np.logical_and(mask, old_mask)

            return idx

    def sample(self, batch_size, beta=None):
        if beta is None:
            beta = self.beta

        with self.lock:
            # assert self.idx is None
            scalar = self.np_random.random(batch_size) * self.sum_tree.reduce()
            idx = self.sum_tree.get_prefix_sum_idx(scalar)
            # get data
            data = self.storage[idx]
            # get weights
            weights = (self.sum_tree[idx] / self.min_tree.reduce()) ** (-beta)
            data['weights'] = weights
            # create mask
            self.sampled_idx.append(idx)
            self.sampled_mask.append(np.ones(shape=(batch_size,), dtype=bool))

        for key, item in data.items():
            if not isinstance(item, np.ndarray):
                data[key] = np.array(item)
        return data

    def update_priorities(self, priorities, min_priority=None, max_priority=None):
        with self.lock:
            idx = self.sampled_idx.pop(0)
            mask = self.sampled_mask.pop(0)
            # assert len(self.sampled_idx) == 0

            # only update valid entries
            idx = idx[mask]
            priorities = priorities[mask]

            assert idx.shape == priorities.shape
            priorities = np.abs(priorities) + EPS
            if min_priority is not None or max_priority is not None:
                priorities = np.clip(priorities, a_min=min_priority, a_max=max_priority)
            self.sum_tree[idx] = priorities ** self.alpha
            self.max_tree[idx] = priorities ** self.alpha
            self.min_tree[idx] = priorities ** self.alpha

            if self.eviction_tree is not None:
                self.eviction_tree[idx] = priorities ** self.eviction

    @classmethod
    def from_env(cls, env, **kwargs):
        data_spec = utils.get_data_spec_from_env(env)
        return cls(data_spec=data_spec, **kwargs)
