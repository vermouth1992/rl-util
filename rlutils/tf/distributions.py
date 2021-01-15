import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def apply_squash_log_prob(raw_log_prob, x):
    """ Compute the log probability after applying tanh on raw_actions
    Args:
        log_prob: (None,)
        raw_actions: (None, act_dim)
    Returns:
    """
    log_det_jacobian = 2. * (np.log(2.) - x - tf.math.softplus(-2. * x))
    num_reduce_dim = tf.rank(x) - tf.rank(raw_log_prob)
    log_det_jacobian = tf.reduce_sum(log_det_jacobian, axis=tf.range(-num_reduce_dim, 0))
    log_prob = raw_log_prob - log_det_jacobian
    return log_prob


def make_independent_normal_from_params(params, min_log_scale=None, max_log_scale=None):
    loc_params, scale_params = tf.split(params, 2, axis=-1)
    if min_log_scale is not None:
        scale_params = tf.maximum(scale_params, min_log_scale)
    if max_log_scale is not None:
        scale_params = tf.minimum(scale_params, max_log_scale)
    scale_params = tf.math.softplus(scale_params)
    distribution = make_independent_normal(loc_params, scale_params, ndims=1)
    return distribution


def make_independent_normal(loc, scale, ndims=1):
    distribution = tfd.Independent(distribution=tfd.Normal(loc=loc, scale=scale),
                                   reinterpreted_batch_ndims=ndims)
    return distribution


def make_independent_truncated_normal(loc_params, scale_params, ndims=1):
    pi_distribution = tfd.Independent(distribution=tfd.TruncatedNormal(loc=loc_params, scale=scale_params,
                                                                       low=-1., high=1.),
                                      reinterpreted_batch_ndims=ndims)
    return pi_distribution


def make_independent_beta_from_params(params):
    params = tf.math.softplus(params)
    c1, c2 = tf.split(params, 2, axis=-1)
    distribution = make_independent_beta(c1, c2, ndims=1)
    return distribution


def make_independent_beta(c1, c2, ndims=1):
    beta_distribution = tfd.Beta(concentration1=c1, concentration0=c2, validate_args=False, allow_nan_stats=False)
    distribution = tfd.Independent(beta_distribution,
                                   reinterpreted_batch_ndims=ndims)
    return distribution
