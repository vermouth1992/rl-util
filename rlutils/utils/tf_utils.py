import tensorflow as tf
import numpy as np

ALLOW_GROWTH = False


def set_tf_allow_growth(enable=True):
    global ALLOW_GROWTH
    if enable != ALLOW_GROWTH:
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, enable)
        ALLOW_GROWTH = enable


def get_tf_func(instance, fun):
    func = getattr(instance, fun.__name__, None)
    if func is None:
        setattr(instance, fun.__name__, fun)
        func = getattr(instance, fun.__name__, None)
    return func


def to_numpy_or_python_type(tensors):
    """Converts a structure of `Tensor`s to `NumPy` arrays or Python scalar types.

    For each tensor, it calls `tensor.numpy()`. If the result is a scalar value,
    it converts it to a Python type, such as a float or int, by calling
    `result.item()`.

    Numpy scalars are converted, as Python types are often more convenient to deal
    with. This is especially useful for bfloat16 Numpy scalars, which don't
    support as many operations as other Numpy values.

    Args:
      tensors: A structure of tensors.

    Returns:
      `tensors`, but scalar tensors are converted to Python types and non-scalar
      tensors are converted to Numpy arrays.
    """

    def _to_single_numpy_or_python_type(t):
        if isinstance(t, tf.Tensor):
            x = t.numpy()
            return x.item() if np.ndim(x) == 0 else x
        return t  # Don't turn ragged or sparse tensors to NumPy.

    return tf.nest.map_structure(_to_single_numpy_or_python_type, tensors)
