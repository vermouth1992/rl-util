from typing import Dict

import gym.spaces
import numpy as np

from .uniform_py import PyUniformReplayBuffer
from .utils import segtree

import gym.envs.mujoco.hopper
import gym.envs.mujoco.hopper_v3

class PyPrioritizedReplayBuffer(PyUniformReplayBuffer):
    """
    A simple implementation of PER based on pure numpy. No advanced data structure is used.
    """

    def __init__(self, data_spec: Dict[str, gym.spaces.Space], capacity, batch_size, alpha=0.6):
        super(PyPrioritizedReplayBuffer, self).__init__(data_spec=data_spec,
                                                        capacity=capacity,
                                                        batch_size=batch_size)
        self.alpha = alpha
        self.default_priority = 1.0
        self.priority = segtree.SegmentTree(size=capacity)

    def add(self, data: Dict[str, np.ndarray], priority=None):
        batch_size = list(data.values())[0].shape[0]
        if priority is None:
            if self.is_empty():
                max_priority = self.default_priority
            else:
                max_priority = np.max(self.storage['priority'][:len(self)])
            data['priority'] = np.ones(shape=(batch_size,), dtype=np.float32) * max_priority
        else:
            data['priority'] = priority
        super(PyPrioritizedReplayBuffer, self).add(data=data)

    def sample(self, beta=0.4):
        total_num = len(self)
        p = self.storage['priority'][:total_num] ** self.alpha
        p = p / np.sum(p)
        weights = (1. / p / np.asarray(total_num, dtype=np.float32)) ** beta
        weights = weights / np.max(weights)
        idx = np.random.choice(total_num, size=self.batch_size, replace=True, p=p)
        weights = weights[idx]
        data = self.__getitem__(idx)
        data['weights'] = weights
        # assert not np.any(np.isnan(weights)), f'NAN weights: {weights}'
        return data, idx

    def update_priorities(self, idx, priorities, min_priority=None, max_priority=None):
        self.storage['priority'][idx] = np.clip(priorities, a_min=min_priority, a_max=max_priority)
