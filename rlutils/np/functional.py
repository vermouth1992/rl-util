import numpy as np
import scipy.signal

EPS = 1e-6


def inverse_softplus(x, beta=1.):
    assert x > 0, 'x must be positive'
    if x < 20:
        return np.log(np.exp(x * beta) - 1.) / beta
    else:
        return x


def flatten_leading_dims(array, n_dims):
    """ Flatten the leading n dims of a numpy array """
    if n_dims <= 1:
        return array
    newshape = [-1] + list(array.shape[n_dims:])
    return np.reshape(array, newshape=newshape)


def clip_arctanh(x):
    return np.arctanh(np.clip(x, a_min=-1. + EPS, a_max=1. - EPS))


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
