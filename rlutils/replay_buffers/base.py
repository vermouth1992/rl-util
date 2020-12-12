"""
Abstract class for replay buffers
1. Uniform sampling dataset.
2. Prioritized replay
3. n-step return
4. Frame stack
5. Trajectory-based replay buffer for on-policy methods
"""

from abc import ABC, abstractmethod


class BaseReplayBuffer(ABC):
    @abstractmethod
    def add(self, data, priority=1.0):
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        raise NotImplementedError
