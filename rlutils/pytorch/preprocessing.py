import torch
from torch import nn

EPS = 1e-8


class StandardScaler(nn.Module):
    def __init__(self, input_shape):
        super(StandardScaler, self).__init__()
        self.mean = nn.Parameter(data=torch.zeros(size=[1] + list(input_shape), dtype=torch.float32),
                                 requires_grad=False)
        self.std = nn.Parameter(data=torch.ones(size=[1] + list(input_shape), dtype=torch.float32),
                                requires_grad=False)

    def adapt(self, data):
        self.mean.data = torch.mean(data, dim=0, keepdim=True)
        self.std.data = torch.std(data, dim=0, keepdim=True)
        self.std.data[self.std.data < EPS] = 1.

    def forward(self, data, inverse=False):
        if inverse:
            return data * self.std + self.mean
        else:
            return (data - self.mean) / self.std
