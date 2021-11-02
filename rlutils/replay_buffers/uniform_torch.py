from .base import DictReplayBuffer
from .utils import combined_shape

import torch
import torch.nn as nn

from typing import Dict

import numpy as np


class PytorchDictReplayBuffer(nn.Module, DictReplayBuffer):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        DictReplayBuffer.__init__(self, *args, **kwargs)

    def get(self):
        idxs = torch.arange(self.size)
        return self.__getitem__(idxs)

    def _create_storage(self):
        parameters = nn.ParameterDict()
        for key, item in self.data_spec.items():
            data = torch.zeros(combined_shape(self.capacity, item.shape), dtype=getattr(torch, item.dtype.name))
            parameters.update({key: nn.Parameter(data, requires_grad=False)})
        return parameters

    def add(self, data: Dict[str, np.ndarray]):
        data = {key: torch.as_tensor(item, device=self.storage[key].device) for key, item in data.items()}
        super(PytorchDictReplayBuffer, self).add(data)


class PytorchUniformReplayBuffer(PytorchDictReplayBuffer):
    def sample(self):
        assert not self.is_empty()
        idxs = torch.randint(self.size, size=(self.batch_size,))
        return self.__getitem__(idxs)
