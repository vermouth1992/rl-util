"""
Adapted from https://github.com/thu-ml/tianshou/
"""

from typing import Dict

import gym.spaces
import numpy as np
import torch

import rlutils.pytorch.utils as ptu
from .base import DictReplayBuffer
from .utils import segtree_torch, combined_shape

EPS = np.finfo(np.float32).eps.item()


class DictPrioritizedReplayBufferTorch(DictReplayBuffer):
    """
    A simple implementation of PER based on pure numpy. No advanced data structure is used.
    """

    def __init__(self, data_spec: Dict[str, gym.spaces.Space], capacity, batch_size, alpha=0.6, seed=None,
                 device=ptu.device):
        self.device = device
        super(DictPrioritizedReplayBufferTorch, self).__init__(data_spec=data_spec,
                                                               capacity=capacity,
                                                               batch_size=batch_size,
                                                               seed=seed)
        self.alpha = alpha
        self.max_priority = torch.tensor([1.0], device=self.device)
        self.min_priority = torch.tensor([1.0], device=self.device)
        self.segtree = segtree_torch.SegmentTree(size=capacity, device=device)

    def get(self):
        idxs = torch.arange(self.size)
        return self.__getitem__(idxs)

    def _create_storage(self):
        data = {key: torch.as_tensor(np.zeros(combined_shape(self.capacity, item.shape), dtype=item.dtype),
                                     device=self.device)
                for key, item in self.data_spec.items()}
        return data

    def add(self, data: Dict[str, torch.Tensor], priority=None):
        batch_size = list(data.values())[0].shape[0]
        if priority is None:
            priority = torch.ones(size=(batch_size,), dtype=torch.float32, device=self.device) * self.max_priority
        idx = torch.arange(self.ptr, self.ptr + batch_size, device=self.device)
        self.segtree[idx] = priority ** self.alpha
        self.max_priority = torch.maximum(self.max_priority, torch.max(priority))
        self.min_priority = torch.minimum(self.min_priority, torch.min(priority))
        super(DictPrioritizedReplayBufferTorch, self).add(data=data)

    def sample(self, beta=0.4):
        scalar = torch.rand(size=(self.batch_size,)) * self.segtree.reduce()
        idx = self.segtree.get_prefix_sum_idx(scalar)
        data = self.__getitem__(idx)
        # important sampling weight calculation
        # original formula: ((p_j/p_sum*N)**(-beta))/((p_min/p_sum*N)**(-beta))
        # simplified formula: (p_j/p_min)**(-beta)
        data['weights'] = (self.segtree[idx] / self.min_priority) ** (-beta)
        return data, idx

    def update_priorities(self, idx, priorities, min_priority=None, max_priority=None):
        assert idx.shape == priorities.shape
        priorities = torch.abs(priorities) + EPS
        if min_priority is not None or max_priority is not None:
            priorities = torch.clamp(priorities, min=min_priority, max=max_priority)
        self.segtree[idx] = priorities ** self.alpha
        self.max_priority = torch.maximum(self.max_priority, torch.max(priorities))
        self.min_priority = torch.minimum(self.min_priority, torch.min(priorities))

    @classmethod
    def from_data_dict(cls, alpha=0.6, **kwargs):
        return super(DictPrioritizedReplayBufferTorch, cls).from_data_dict(alpha=alpha, **kwargs)

    @classmethod
    def from_vec_env(cls, alpha=0.6, **kwargs):
        return super(DictPrioritizedReplayBufferTorch, cls).from_vec_env(alpha=alpha, **kwargs)

    @classmethod
    def from_env(cls, alpha=0.6, **kwargs):
        return super(DictPrioritizedReplayBufferTorch, cls).from_env(alpha=alpha, **kwargs)
