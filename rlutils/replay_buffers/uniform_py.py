import numpy as np

from .base import DictReplayBuffer
from .utils import combined_shape


class PyDictReplayBuffer(DictReplayBuffer):
    def get(self):
        idxs = np.arange(self.size)
        return self.__getitem__(idxs)

    def _create_storage(self):
        return {key: np.zeros(combined_shape(self.capacity, item.shape), dtype=item.dtype)
                for key, item in self.data_spec.items()}


class PyUniformReplayBuffer(PyDictReplayBuffer):
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def sample(self):
        assert not self.is_empty()
        idxs = self.np_random.randint(0, self.size, size=self.batch_size)
        return self.__getitem__(idxs)
