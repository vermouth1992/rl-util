"""
Adapted from https://github.com/thu-ml/tianshou/

Design methodology:
- We do not want to return the sampled indices because it may be overriden by new data, and we shouldn't update the
priorities of those data.
- The priority update and sampled indices may not be in order when there are multiple learners. Thus, we maintain a
transaction id for each sampled batch.
"""

from typing import Dict

import numpy as np

EPS = np.finfo(np.float32).eps.item()

from gym.utils import seeding
import threading

from . import utils, storage


class PrioritizedReplayBuffer(object):
    def __init__(self, data_spec, capacity, alpha=0.6, beta=0.4, eviction=None, seed=None):
        self.eviction = eviction
        if eviction is None:
            print('Using FIFO eviction policy')
            self.eviction_tree = None
        else:
            assert eviction < 0.
            print(f'Using prioritized eviction policy with alpha_evict={eviction}')
            self.eviction_tree = utils.segtree.SumTree(size=capacity)

        self.storage = storage.PyDictStorage(data_spec=data_spec, capacity=capacity)
        self.storage.reset()
        self.alpha = alpha
        self.beta = beta
        self.max_tree = utils.segtree.MaxTree(size=capacity)
        self.min_tree = utils.segtree.MinTree(size=capacity)
        self.sum_tree = utils.segtree.SumTree(size=capacity)
        self.lock = threading.Lock()
        self.set_seed(seed=seed)

        self.sampled_idx_mask = {}  # map from transaction_id to (sampled_idx, sampled_mask)

        self.transaction_id = 0
        self.max_transaction_id = 1000

    def get_available_transaction_id(self):
        transaction_id = None
        for _ in range(self.max_transaction_id):
            if self.transaction_id not in self.sampled_idx_mask:
                transaction_id = self.transaction_id
                self.transaction_id = (self.transaction_id + 1) % self.max_transaction_id
                break
            else:
                self.transaction_id += 1
        assert transaction_id is not None, f'Fail to find valid transaction id. Slowdown sampling. ' \
                                           f'Current size {len(self.sampled_idx_mask)}'
        return transaction_id

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
        batch_size = len(data[list(data.keys())[0]])
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

            for transaction_id in self.sampled_idx_mask:
                sampled_idx, old_mask = self.sampled_idx_mask[transaction_id]
                mask = np.in1d(sampled_idx, idx, invert=True)  # False if in the idx
                self.sampled_idx_mask[transaction_id] = (sampled_idx, np.logical_and(mask, old_mask))

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
            transaction_id = self.get_available_transaction_id()
            self.sampled_idx_mask[transaction_id] = (idx, np.ones(shape=(batch_size,), dtype=np.bool))

        for key, item in data.items():
            if not isinstance(item, np.ndarray):
                data[key] = np.array(item)
        return transaction_id, data

    def update_priorities(self, transaction_id, priorities, min_priority=None, max_priority=None):
        with self.lock:
            assert transaction_id in self.sampled_idx_mask
            idx, mask = self.sampled_idx_mask.pop(transaction_id)
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
    def from_env(cls, env, memory_efficient, **kwargs):
        data_spec = utils.get_data_spec_from_env(env, memory_efficient=memory_efficient)
        return cls(data_spec=data_spec, **kwargs)
