try:
    import cpprb
except:
    print('cpprb is not installed. To use cpprb, run "pip install cpprb"')

from .base import DictReplayBuffer
import numpy as np
from typing import Dict


class CPPRBUniformReplayBuffer(DictReplayBuffer):
    def _create_storage(self):
        self.env_dict = {}
        for key, item in self.data_spec.items():
            self.env_dict[key] = {"dtype": item.dtype}
            if item.shape is not None:
                self.env_dict[key]["shape"] = item.shape
        return cpprb.ReplayBuffer(size=self.capacity,
                                  env_dict=self.env_dict)

    def sample(self):
        data = self.storage.sample(self.batch_size)
        for key, item in data.items():
            if 'shape' not in self.env_dict[key]:
                data[key] = np.squeeze(item, axis=-1)
        return data

    def add(self, data: Dict[str, np.ndarray]):
        self.storage.add(**data)

    def reset(self):
        self.storage.clear()

    def __len__(self):
        return self.storage.get_stored_size()

    def __getitem__(self, item):
        raise NotImplementedError

    def append(self, data: Dict[str, np.ndarray]):
        self.storage.add(**data)

    def get(self):
        return self.storage.get_all_transitions()
