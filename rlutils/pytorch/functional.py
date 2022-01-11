from typing import Iterable

import torch
import torch.nn as nn


def gather_q_values(q_values, action):
    assert q_values.shape[0] == action.shape[0]
    assert len(q_values.shape) == 2 and len(action.shape) == 1
    return q_values.gather(1, action.unsqueeze(1)).squeeze(1)


def model_averaging(global_model: nn.Module, local_models: Iterable[nn.Module]):
    global_weights = list(global_model.parameters())
    trainable_weights = [[] for _ in range(len(global_weights))]
    for model in local_models:
        for i, param in enumerate(model.parameters()):
            trainable_weights[i].append(param.data)

    for target_param, param in zip(global_model.parameters(), trainable_weights):
        param = torch.mean(torch.stack(param, dim=0), dim=0)
        target_param.data.copy_(param.to(target_param.data.device))

    for local_q_network in local_models:
        hard_update(local_q_network, global_model)


def soft_update(target: nn.Module, source: nn.Module, tau):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data.to(target_param.data.device) * tau)


def hard_update(target: nn.Module, source: nn.Module):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data.to(target_param.data.device))


def compute_target_value(reward, gamma, done, next_q):
    q_target = reward + gamma * (1.0 - done) * next_q
    return q_target


def clip_by_value_preserve_gradient(t, clip_value_min=None, clip_value_max=None):
    clip_t = torch.clip(t, min=clip_value_min, max=clip_value_max)
    return t + (clip_t - t).detach()

# def to_numpy_or_python_type(tensors):
#     """Converts a structure of `Tensor`s to `NumPy` arrays or Python scalar types.
#
#     For each tensor, it calls `tensor.numpy()`. If the result is a scalar value,
#     it converts it to a Python type, such as a float or int, by calling
#     `result.item()`.
#
#     Numpy scalars are converted, as Python types are often more convenient to deal
#     with. This is especially useful for bfloat16 Numpy scalars, which don't
#     support as many operations as other Numpy values.
#
#     Args:
#       tensors: A structure of tensors.
#
#     Returns:
#       `tensors`, but scalar tensors are converted to Python types and non-scalar
#       tensors are converted to Numpy arrays.
#     """
#
#     def _to_single_numpy_or_python_type(t):
#         if isinstance(t, torch.Tensor):
#             x = t.detach().cpu().numpy()
#             return x.item() if np.ndim(x) == 0 else x
#         return t  # Don't turn ragged or sparse tensors to NumPy.
#
#     import tensorflow as tf
#     return tf.nest.map_structure(_to_single_numpy_or_python_type, tensors)
