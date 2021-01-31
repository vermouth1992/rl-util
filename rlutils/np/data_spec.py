class DataSpec(object):
    def __init__(self, shape, dtype, minval=None, maxval=None):
        self.shape = shape
        self.dtype = dtype
        self.minval = minval
        self.maxval = maxval

    def __repr__(self):
        return f'shape={self.shape}, dtype={self.dtype}, minval={self.minval}, maxval={self.maxval}'
