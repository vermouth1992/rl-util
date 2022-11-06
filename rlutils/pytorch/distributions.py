import numpy as np

import torch
import torch.nn.functional as F
from torch import distributions as td
from torch import nn


def _compute_rank(tensor):
    return len(tensor.shape)


def uniform(size, low, high):
    return (high - low) * torch.rand(*size) + low


def apply_squash_log_prob(raw_log_prob, x):
    """ Compute the log probability after applying tanh on raw_actions
    Args:
        log_prob: (None,)
        raw_actions: (None, act_dim)
    Returns:
    """
    log_det_jacobian = 2. * (np.log(2.) - x - F.softplus(-2. * x))
    num_reduce_dim = _compute_rank(x) - _compute_rank(raw_log_prob)
    log_det_jacobian = torch.sum(log_det_jacobian, dim=list(range(-num_reduce_dim, 0)))
    log_prob = raw_log_prob - log_det_jacobian
    return log_prob


def make_independent_normal_from_params(params, min_log_scale=None, max_log_scale=None):
    loc_params, scale_params = torch.split(params, params.shape[-1] // 2, dim=-1)
    if min_log_scale is not None or max_log_scale is not None:
        scale_params = torch.clip(scale_params, min=min_log_scale, max=max_log_scale)
    scale_params = F.softplus(scale_params)
    pi_distribution = make_independent_normal(loc_params, scale_params, ndims=1)
    return pi_distribution


def make_independent_normal(loc, scale, ndims=1):
    distribution = td.Independent(base_distribution=td.Normal(loc=loc, scale=scale),
                                  reinterpreted_batch_ndims=ndims)
    return distribution


def make_independent_categorical_from_params(params, ndims=1):
    return td.Independent(td.Categorical(logits=params), reinterpreted_batch_ndims=ndims)


class IndependentNormalWithFixedVar(nn.Module):
    def __init__(self, var_shape, reinterpreted_batch_ndims):
        super(IndependentNormalWithFixedVar, self).__init__()
        self.var_shape = var_shape
        self.scale = nn.Parameter(data=torch.ones(size=var_shape, dtype=torch.float32), requires_grad=True)
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def forward(self, mean):
        # check shape
        assert mean.shape[1:] == self.var_shape
        return make_independent_normal(loc=mean, scale=torch.unsqueeze(self.scale, dim=0),
                                       ndims=self.reinterpreted_batch_ndims)
