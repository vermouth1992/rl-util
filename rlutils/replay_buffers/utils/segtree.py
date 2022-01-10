"""
Code copied from https://github.com/thu-ml/tianshou/blob/master/tianshou/data/utils/segtree.py
"""

from typing import Union, Optional

import numpy as np
from numba import njit


class SegmentTree:
    """Implementation of Segment Tree.
    The segment tree stores an array ``arr`` with size ``n``. It supports value
    update and fast query of the sum for the interval ``[left, right)`` in
    O(log n) time. The detailed procedure is as follows:
    1. Pad the array to have length of power of 2, so that leaf nodes in the \
    segment tree have the same depth.
    2. Store the segment tree in a binary heap.
    :param int size: the size of segment tree.
    """

    def __init__(self, size: int) -> None:
        bound = 1
        while bound < size:
            bound *= 2
        self._size = size
        self._bound = bound
        self._value = np.full(shape=[bound * 2], fill_value=self._init_value, dtype=np.float64)
        self._compile()

    @property
    def _init_value(self):
        raise NotImplementedError

    def _setitem(self, tree: np.ndarray, index: np.ndarray, value: np.ndarray):
        raise NotImplementedError

    def _reduce(self, tree: np.ndarray, start: int, end: int):
        raise NotImplementedError

    def __len__(self) -> int:
        return self._size

    def __getitem__(
            self, index: Union[int, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Return self[index]."""
        return self._value[index + self._bound]

    def __setitem__(
            self, index: Union[int, np.ndarray], value: Union[float, np.ndarray]
    ) -> None:
        """Update values in segment tree.
        Duplicate values in ``index`` are handled by numpy: later index
        overwrites previous ones.
        ::
            >>> a = np.array([1, 2, 3, 4])
            >>> a[[0, 1, 0, 1]] = [4, 5, 6, 7]
            >>> print(a)
            [6 7 3 4]
        """
        if isinstance(index, int):
            index, value = np.array([index]), np.array([value])
        assert np.all(0 <= index) and np.all(index < self._size)
        assert np.all(np.isfinite(value)), f"value={value}"
        self._setitem(self._value, index + self._bound, value)

    def reduce(self, start: int = 0, end: Optional[int] = None) -> float:
        """Return operation(value[start:end])."""
        if start == 0 and end is None:
            return self._value[1]
        if end is None:
            end = self._size
        if end < 0:
            end += self._size
        return self._reduce(self._value, start + self._bound - 1, end + self._bound)

    def _compile(self) -> None:
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        i64 = np.array([0, 1], dtype=np.int64)
        self._setitem(f64, i64, f64)
        self._setitem(f64, i64, f32)
        self._reduce(f64, 0, 1)


class SumTree(SegmentTree):
    @property
    def _init_value(self):
        return 0.

    def _setitem(self, tree: np.ndarray, index: np.ndarray, value: np.ndarray):
        _setitem_add(tree, index, value)

    def _reduce(self, tree: np.ndarray, start: int, end: int):
        return _reduce_add(tree, start, end)

    def get_prefix_sum_idx(
            self, value: Union[float, np.ndarray]
    ) -> Union[int, np.ndarray]:
        r"""Find the index with given value.
        Return the minimum index for each ``v`` in ``value`` so that
        :math:`v \le \mathrm{sums}_i`, where
        :math:`\mathrm{sums}_i = \sum_{j = 0}^{i} \mathrm{arr}_j`.
        .. warning::
            Please make sure all of the values inside the segment tree are
            non-negative when using this function.
        """
        assert np.all(value >= 0.0) and np.all(value < self._value[1]), f"value={value}, sum_value={self._value[1]}"
        single = False
        if not isinstance(value, np.ndarray):
            value = np.array([value])
            single = True
        index = _get_prefix_sum_idx(value, self._bound, self._value)
        return index.item() if single else index

    def _compile(self) -> None:
        super(SumTree, self)._compile()
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        _get_prefix_sum_idx(f64, 1, f64)
        _get_prefix_sum_idx(f32, 1, f64)


class MaxTree(SegmentTree):
    @property
    def _init_value(self):
        return -np.inf

    def _setitem(self, tree: np.ndarray, index: np.ndarray, value: np.ndarray):
        _setitem_max(tree, index, value)

    def _reduce(self, tree: np.ndarray, start: int, end: int):
        return _reduce_max(tree, start, end)


class MinTree(SegmentTree):
    @property
    def _init_value(self):
        return np.inf

    def _setitem(self, tree: np.ndarray, index: np.ndarray, value: np.ndarray):
        _setitem_min(tree, index, value)

    def _reduce(self, tree: np.ndarray, start: int, end: int):
        return _reduce_min(tree, start, end)


@njit
def _setitem_add(tree: np.ndarray, index: np.ndarray, value: np.ndarray) -> None:
    """Numba version, 4x faster: 0.1 -> 0.024."""
    tree[index] = value
    while index[0] > 1:
        index //= 2
        tree[index] = np.add(tree[index * 2], tree[index * 2 + 1])


@njit
def _setitem_max(tree: np.ndarray, index: np.ndarray, value: np.ndarray) -> None:
    """Numba version, 4x faster: 0.1 -> 0.024."""
    tree[index] = value
    while index[0] > 1:
        index //= 2
        tree[index] = np.maximum(tree[index * 2], tree[index * 2 + 1])


@njit
def _setitem_min(tree: np.ndarray, index: np.ndarray, value: np.ndarray) -> None:
    """Numba version, 4x faster: 0.1 -> 0.024."""
    tree[index] = value
    while index[0] > 1:
        index //= 2
        tree[index] = np.minimum(tree[index * 2], tree[index * 2 + 1])


@njit
def _reduce_add(tree: np.ndarray, start: int, end: int) -> float:
    """Numba version, 2x faster: 0.009 -> 0.005."""

    # nodes in (start, end) should be aggregated
    result = 0.0
    while end - start > 1:  # (start, end) interval is not empty
        if start % 2 == 0:
            result = np.add(result, tree[start + 1])
        start //= 2
        if end % 2 == 1:
            result = np.add(result, tree[end - 1])
        end //= 2
    return result


@njit
def _reduce_max(tree: np.ndarray, start: int, end: int) -> float:
    """Numba version, 2x faster: 0.009 -> 0.005."""

    # nodes in (start, end) should be aggregated
    result = -np.inf
    while end - start > 1:  # (start, end) interval is not empty
        if start % 2 == 0:
            result = np.maximum(result, tree[start + 1])
        start //= 2
        if end % 2 == 1:
            result = np.maximum(result, tree[end - 1])
        end //= 2
    return result


@njit
def _reduce_min(tree: np.ndarray, start: int, end: int) -> float:
    """Numba version, 2x faster: 0.009 -> 0.005."""

    # nodes in (start, end) should be aggregated
    result = np.inf
    while end - start > 1:  # (start, end) interval is not empty
        if start % 2 == 0:
            result = np.minimum(result, tree[start + 1])
        start //= 2
        if end % 2 == 1:
            result = np.minimum(result, tree[end - 1])
        end //= 2
    return result


@njit
def _get_prefix_sum_idx(
        value: np.ndarray, bound: int, sums: np.ndarray
) -> np.ndarray:
    """Numba version (v0.51), 5x speed up with size=100000 and bsz=64.
    vectorized np: 0.0923 (numpy best) -> 0.024 (now)
    for-loop: 0.2914 -> 0.019 (but not so stable)
    """
    index = np.ones(value.shape, dtype=np.int64)
    while index[0] < bound:
        index *= 2
        lsons = sums[index]
        direct = lsons < value
        value -= lsons * direct
        index += direct
    index -= bound
    return index
