import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlutils.np.functional import inverse_softplus
from rlutils.pytorch.functional import clip_by_value_preserve_gradient


class EnsembleBatchNorm1d(nn.Module):
    def __init__(self, num_ensembles, num_features, **kwargs):
        super(EnsembleBatchNorm1d, self).__init__()
        self.num_ensembles = num_ensembles
        self.num_features = num_features
        self.batch_norm_layer = nn.BatchNorm1d(num_features=num_features * num_ensembles, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """

        Args:
            input: shape (num_ensembles, None, num_features)

        Returns:

        """
        batch_size = input.shape[1]
        input = input.permute(1, 0, 2)  # (None, num_ensembles, num_features)
        input = input.reshape(batch_size, self.num_ensembles * self.num_features)
        output = self.batch_norm_layer(input)  # (None, num_ensembles, num_features)
        output = output.view(batch_size, self.num_ensembles, self.num_features)
        output = output.permute(1, 0, 2)  # (num_ensembles, None, num_features)
        return output


class EnsembleDense(nn.Module):
    __constants__ = ['num_ensembles', 'in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, num_ensembles: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super(EnsembleDense, self).__init__()
        self.num_ensembles = num_ensembles
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(num_ensembles, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_ensembles, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan = self.in_features
        gain = nn.init.calculate_gain('leaky_relu', param=math.sqrt(5))
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            nn.init.uniform_(self.weight, -bound, bound)

        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.bmm(input, self.weight) + self.bias

    def extra_repr(self) -> str:
        return 'num_ensembles={}, in_features={}, out_features={}, bias={}'.format(
            self.num_ensembles, self.in_features, self.out_features, self.bias is not None
        )


class SqueezeLayer(nn.Module):
    def __init__(self, dim=-1):
        super(SqueezeLayer, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.squeeze(inputs, dim=self.dim)


class LagrangeLayer(nn.Module):
    def __init__(self, initial_value=0., min_value=None, max_value=10000.):
        super(LagrangeLayer, self).__init__()
        self.log_alpha = nn.Parameter(data=torch.as_tensor(inverse_softplus(initial_value), dtype=torch.float32))
        self.min_value = min_value
        self.max_value = max_value

    def forward(self):
        alpha = F.softplus(self.log_alpha)
        return clip_by_value_preserve_gradient(alpha, clip_value_min=self.min_value, clip_value_max=self.max_value)


class LambdaLayer(nn.Module):
    def __init__(self, function):
        super(LambdaLayer, self).__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)
