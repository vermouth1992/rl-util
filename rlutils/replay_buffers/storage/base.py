"""
Base class for a storage class
"""


class Storage(object):
    def reset(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    @property
    def capacity(self):
        raise NotImplementedError

    def add(self, data):
        raise NotImplementedError

    def append(self, data):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

    def is_full(self):
        return len(self) == self.capacity

    def is_empty(self):
        return len(self) <= 0


