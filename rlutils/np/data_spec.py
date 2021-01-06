class DataSpec(object):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def __repr__(self):
        return f'shape={self.shape}, dtype={self.dtype}'
