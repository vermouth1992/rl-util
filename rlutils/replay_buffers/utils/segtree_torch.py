"""
Code copied from https://github.com/thu-ml/tianshou/blob/master/tianshou/data/utils/segtree.py
"""

from typing import Union, Optional

import torch
import torch.nn as nn


class SegmentTree(nn.Module):
    """Implementation of Segment Tree.
    The segment tree stores an array ``arr`` with size ``n``. It supports value
    update and fast query of the sum for the interval ``[left, right)`` in
    O(log n) time. The detailed procedure is as follows:
    1. Pad the array to have length of power of 2, so that leaf nodes in the \
    segment tree have the same depth.
    2. Store the segment tree in a binary heap.
    :param int size: the size of segment tree.
    """

    def __init__(self, size: int, device) -> None:
        super(SegmentTree, self).__init__()
        bound = 1
        while bound < size:
            bound *= 2
        self._size = size
        self._bound = bound
        self._device = device
        self._value = torch.zeros(size=[bound * 2], device=self._device)

    def __len__(self) -> int:
        return self._size

    def __getitem__(
            self, index: Union[int, torch.Tensor]
    ) -> Union[float, torch.Tensor]:
        """Return self[index]."""
        return self._value[index + self._bound]

    def __setitem__(
            self, index: Union[int, torch.Tensor], value: Union[float, torch.Tensor]
    ) -> None:
        if isinstance(index, int):
            index, value = torch.tensor([index], device=self._device), torch.tensor([value], device=self._device)
        _setitem(self._value, index + self._bound, value)

    def reduce(self, start: int = 0, end: Optional[int] = None) -> float:
        """Return operation(value[start:end])."""
        if start == 0 and end is None:
            return self._value[1]
        if end is None:
            end = self._size
        if end < 0:
            end += self._size
        return _reduce(self._value, start + self._bound - 1, end + self._bound)

    def get_prefix_sum_idx(
            self, value: Union[float, torch.Tensor]
    ) -> Union[int, torch.Tensor]:
        r"""Find the index with given value.
        Return the minimum index for each ``v`` in ``value`` so that
        :math:`v \le \mathrm{sums}_i`, where
        :math:`\mathrm{sums}_i = \sum_{j = 0}^{i} \mathrm{arr}_j`.
        .. warning::
            Please make sure all of the values inside the segment tree are
            non-negative when using this function.
        """
        single = False
        if not isinstance(value, torch.Tensor):
            value = torch.tensor([value], device=self._device)
            single = True
        index = _get_prefix_sum_idx(value, self._bound, self._value)
        return index.item() if single else index


def _setitem(tree: torch.Tensor, index: torch.Tensor, value: torch.Tensor) -> None:
    """Numba version, 4x faster: 0.1 -> 0.024."""
    tree[index] = value
    while index[0] > 1:
        index = torch.div(index, 2, rounding_mode='floor')
        tree[index] = tree[index * 2] + tree[index * 2 + 1]


def _reduce(tree: torch.Tensor, start: int, end: int) -> float:
    """Numba version, 2x faster: 0.009 -> 0.005."""
    # nodes in (start, end) should be aggregated
    result = 0.0
    while end - start > 1:  # (start, end) interval is not empty
        if start % 2 == 0:
            result += tree[start + 1]
        start //= 2
        if end % 2 == 1:
            result += tree[end - 1]
        end //= 2
    return result


def _get_prefix_sum_idx(value: torch.Tensor, bound: int, sums: torch.Tensor) -> torch.Tensor:
    """Numba version (v0.51), 5x speed up with size=100000 and bsz=64.
    vectorized np: 0.0923 (numpy best) -> 0.024 (now)
    for-loop: 0.2914 -> 0.019 (but not so stable)
    """
    index = torch.ones_like(value, dtype=torch.int64)
    while index[0] < bound:
        index *= 2
        lsons = sums[index]
        direct = lsons < value
        value -= lsons * direct
        index += direct
    index -= bound
    return index
