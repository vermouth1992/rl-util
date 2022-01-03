"""
Adapted from https://github.com/thu-ml/tianshou/
"""

from typing import Dict

import gym.spaces
import numpy as np

from .base import BaseReplayBuffer, PyDictReplayBuffer, MemoryEfficientDictReplayBuffer
from .utils import segtree, get_data_spec_from_env, get_data_spec_from_vec_env

EPS = np.finfo(np.float32).eps.item()


class PrioritizedReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity, alpha=0.6, seed=None, **kwargs):
        assert alpha < 1.0, f"alpha={alpha}"
        self.alpha = alpha
        self.max_tree = segtree.MaxTree(size=capacity)
        self.sum_tree = segtree.SumTree(size=capacity)
        self.idx = None
        super(PrioritizedReplayBuffer, self).__init__(capacity=capacity, seed=seed)

    def add(self, data: Dict[str, np.ndarray], priority=None):
        batch_size = len(list(data.values())[0])
        if priority is None:
            if len(self) == 0:
                priority = np.ones(shape=(batch_size,), dtype=np.float32)
            else:
                priority = np.ones(shape=(batch_size,), dtype=np.float32) * self.max_tree.reduce()
        assert np.all(priority > 0.), f'Priority must be all greater than zero. Got {priority}'
        idx = self.storage.add(data)
        self.sum_tree[idx] = priority ** self.alpha
        self.max_tree[idx] = priority ** self.alpha

    def sample(self, batch_size, beta=0.4):
        assert self.idx is None
        scalar = self.np_random.rand(batch_size) * self.sum_tree.reduce()
        idx = self.sum_tree.get_prefix_sum_idx(scalar)
        data = self.storage[idx]
        total = np.array([len(self)], dtype=np.float32)
        weights = (total * self.sum_tree[idx] / self.sum_tree.reduce()) ** (-beta)
        weights = weights / np.max(weights)
        data['weights'] = weights
        self.idx = idx
        return data

    def update_priorities(self, priorities, min_priority=None, max_priority=None):
        assert self.idx is not None
        assert self.idx.shape == priorities.shape
        priorities = np.abs(priorities) + EPS
        if min_priority is not None or max_priority is not None:
            priorities = np.clip(priorities, a_min=min_priority, a_max=max_priority)
        self.sum_tree[self.idx] = priorities ** self.alpha
        self.max_tree[self.idx] = priorities ** self.alpha
        self.idx = None

    def post_process(self, info):
        new_priorities = info['TDError'].cpu().numpy()
        self.update_priorities(new_priorities)


class PrioritizedPyDictReplayBuffer(PyDictReplayBuffer, PrioritizedReplayBuffer):
    """
    A simple implementation of PER based on pure numpy. No advanced data structure is used.
    """

    def __init__(self, data_spec: Dict[str, gym.spaces.Space], capacity, alpha=0.6, seed=None):
        PyDictReplayBuffer.__init__(self, data_spec=data_spec, capacity=capacity, seed=seed)
        PrioritizedReplayBuffer.__init__(self, capacity=capacity, seed=seed, alpha=alpha)

    @classmethod
    def from_env(cls, env, capacity, seed=None, alpha=0.6):
        data_spec = get_data_spec_from_env(env)
        return cls(data_spec=data_spec, capacity=capacity, seed=seed, alpha=alpha)

    @classmethod
    def from_vec_env(cls, vec_env, capacity, seed=None, alpha=0.6):
        data_spec = get_data_spec_from_vec_env(vec_env)
        return cls(data_spec=data_spec, capacity=capacity, seed=seed, alpha=alpha)


class PrioritizedMemoryEfficientPyDictReplayBuffer(MemoryEfficientDictReplayBuffer, PrioritizedReplayBuffer):
    def __init__(self, data_spec: Dict[str, gym.spaces.Space], capacity, alpha=0.6, seed=None):
        print(alpha)
        MemoryEfficientDictReplayBuffer.__init__(self, data_spec=data_spec, capacity=capacity, seed=seed)
        PrioritizedReplayBuffer.__init__(self, capacity=capacity, seed=seed, alpha=alpha)

    @classmethod
    def from_vec_env(cls, vec_env, capacity, seed=None, alpha=0.6):
        data_spec = get_data_spec_from_vec_env(vec_env, memory_efficient=True)
        return cls(data_spec=data_spec, capacity=capacity, seed=seed, alpha=alpha)

    @classmethod
    def from_env(cls, env, capacity, seed=None, alpha=0.6):
        data_spec = get_data_spec_from_env(env, memory_efficient=True)
        return cls(data_spec=data_spec, capacity=capacity, seed=seed, alpha=alpha)
