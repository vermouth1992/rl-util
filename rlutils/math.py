import numpy as np


def inverse_softplus(x, beta=1.):
    return np.log(np.exp(x * beta) - 1.) / beta


def flatten_leading_dims(array, n_dims):
    """ Flatten the leading n dims of a numpy array """
    if n_dims <= 1:
        return array
    newshape = [-1] + list(array.shape[n_dims:])
    return np.reshape(array, newshape=newshape)
