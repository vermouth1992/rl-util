"""
Adapted from https://github.com/thu-ml/tianshou/
"""

from typing import Dict

import gym.spaces
import numpy as np

from .base import PyReplayBuffer
from .utils import segtree

EPS = np.finfo(np.float32).eps.item()


class PyPrioritizedReplayBuffer(PyReplayBuffer):
    """
    A simple implementation of PER based on pure numpy. No advanced data structure is used.
    """

    def __init__(self, data_spec: Dict[str, gym.spaces.Space], capacity, batch_size, alpha=0.6, seed=None):
        super(PyPrioritizedReplayBuffer, self).__init__(data_spec=data_spec,
                                                        capacity=capacity,
                                                        batch_size=batch_size,
                                                        seed=seed)
        self.alpha = alpha
        self.max_priority = 1.0
        self.min_priority = 1.0
        self.segtree = segtree.SegmentTree(size=capacity)

    def add(self, data: Dict[str, np.ndarray], priority=None):
        batch_size = list(data.values())[0].shape[0]
        if priority is None:
            priority = np.ones(shape=(batch_size,), dtype=np.float32) * self.max_priority
        assert np.all(priority > 0.), f'Priority must be all greater than zero. Got {priority}'
        idx = np.arange(self.ptr, self.ptr + batch_size)
        self.segtree[idx] = priority ** self.alpha
        self.max_priority = max(self.max_priority, np.max(priority))
        self.min_priority = min(self.min_priority, np.min(priority))
        super(PyPrioritizedReplayBuffer, self).add(data=data)

    def sample(self, beta=0.4):
        scalar = self.np_random.rand(self.batch_size) * self.segtree.reduce()
        idx = self.segtree.get_prefix_sum_idx(scalar)
        data = self.__getitem__(idx)
        # important sampling weight calculation
        # original formula: ((p_j/p_sum*N)**(-beta))/((p_min/p_sum*N)**(-beta))
        # simplified formula: (p_j/p_min)**(-beta)
        data['weights'] = (self.segtree[idx].astype(np.float32) / self.min_priority) ** (-beta)
        return data, idx

    def update_priorities(self, idx, priorities, min_priority=None, max_priority=None):
        assert idx.shape == priorities.shape
        priorities = np.abs(priorities) + EPS
        if min_priority is not None or max_priority is not None:
            priorities = np.clip(priorities, a_min=min_priority, a_max=max_priority)
        self.segtree[idx] = priorities ** self.alpha
        self.max_priority = max(self.max_priority, np.max(priorities))
        self.min_priority = min(self.min_priority, np.min(priorities))

    @classmethod
    def from_data_dict(cls, alpha=0.6, **kwargs):
        return super(PyPrioritizedReplayBuffer, cls).from_data_dict(alpha=alpha, **kwargs)

    @classmethod
    def from_vec_env(cls, alpha=0.6, **kwargs):
        return super(PyPrioritizedReplayBuffer, cls).from_vec_env(alpha=alpha, **kwargs)

    @classmethod
    def from_env(cls, alpha=0.6, **kwargs):
        return super(PyPrioritizedReplayBuffer, cls).from_env(alpha=alpha, **kwargs)
