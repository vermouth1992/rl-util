import numpy as np

from .base import BaseReplayBuffer, PyDictReplayBuffer, MemoryEfficientDictReplayBuffer


class UniformReplayBuffer(BaseReplayBuffer):
    def sample(self, batch_size):
        assert not self.is_empty()
        idxs = self.np_random.randint(0, len(self), size=batch_size)
        return self.storage[idxs]


class UniformPyDictReplayBuffer(PyDictReplayBuffer, UniformReplayBuffer):
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    pass


class UniformMemoryEfficientPyDictReplayBuffer(MemoryEfficientDictReplayBuffer, UniformReplayBuffer):
    def sample(self, batch_size):
        data = super(UniformMemoryEfficientPyDictReplayBuffer, self).sample(batch_size)
        for key in self.storage.obj_key:
            data[key] = np.array(data[key])
        return data
