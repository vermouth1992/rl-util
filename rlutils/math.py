import numpy as np

def inverse_softplus(x, beta=1.):
    return np.log(np.exp(x * beta) - 1.) / beta