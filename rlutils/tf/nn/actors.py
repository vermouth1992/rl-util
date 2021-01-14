import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from rlutils.tf.distributions import make_independent_normal_from_params, apply_squash_log_prob, \
    make_independent_beta_from_params, make_independent_truncated_normal, make_independent_normal

from .functional import build_mlp

tfd = tfp.distributions
LOG_STD_RANGE = (-20., 5.)
EPS = 1e-5


@tf.function
def get_pi_action(deterministic, pi_distribution):
    print(f'Tracing get_pi_action with deterministic={deterministic}')
    return tf.cond(pred=deterministic, true_fn=lambda: pi_distribution.mean(),
                   false_fn=lambda: pi_distribution.sample())


@tf.function
def get_pi_action_categorical(deterministic, pi_distribution):
    print(f'Tracing get_pi_action with deterministic={deterministic}')
    return tf.cond(pred=deterministic,
                   true_fn=lambda: tf.argmax(pi_distribution.probs_parameter(), axis=-1, output_type=tf.int32),
                   false_fn=lambda: pi_distribution.sample())


class CategoricalActor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, mlp_hidden):
        super(CategoricalActor, self).__init__()
        self.net = build_mlp(input_dim=obs_dim, output_dim=act_dim, mlp_hidden=mlp_hidden)
        self.pi_dist_layer = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Categorical(logits=t)
        )

    def call(self, inputs, **kwargs):
        inputs, deterministic = inputs
        params = self.net(inputs)
        pi_distribution = self.pi_dist_layer(params)
        pi_action = get_pi_action_categorical(deterministic, pi_distribution)
        logp_pi = pi_distribution.log_prob(pi_action)
        pi_action_final = pi_action
        return pi_action_final, logp_pi, pi_action, pi_distribution


class NormalActor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, mlp_hidden, global_std=True):
        super(NormalActor, self).__init__()
        self.global_std = global_std
        if self.global_std:
            self.net = build_mlp(input_dim=obs_dim, output_dim=act_dim, mlp_hidden=mlp_hidden)
            self.log_std = tf.Variable(initial_value=-0.5 * tf.ones(act_dim))
        else:
            self.net = build_mlp(input_dim=obs_dim, output_dim=act_dim * 2, mlp_hidden=mlp_hidden)
            self.log_std = None
        self.pi_dist_layer = self._get_pi_dist_layer()

    def _get_pi_dist_layer(self):
        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: make_independent_normal(t[0], t[1]))

    def call(self, inputs, **kwargs):
        inputs, deterministic = inputs
        params = self.net(inputs)
        if self.global_std:
            pi_distribution = self.pi_dist_layer((params, tf.math.softplus(self.log_std)))
        else:
            mean, log_std = tf.split(params, 2, axis=-1)
            pi_distribution = self.pi_dist_layer((tf.tanh(mean), tf.math.softplus(log_std)))

        pi_action = get_pi_action(deterministic, pi_distribution)
        logp_pi = pi_distribution.log_prob(pi_action)
        pi_action_final = pi_action
        return pi_action_final, logp_pi, pi_action, pi_distribution


class TruncatedNormalActor(NormalActor):
    def _get_pi_dist_layer(self):
        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: make_independent_truncated_normal(t[0], t[1]))


class CenteredBetaMLPActor(tf.keras.Model):
    """ Note that Beta distribution is 2x slower than SquashedGaussian"""

    def __init__(self, ob_dim, ac_dim, mlp_hidden):
        super(CenteredBetaMLPActor, self).__init__()
        print('Warning! This actor is not tested')
        self.net = build_mlp(ob_dim, ac_dim * 2, mlp_hidden)
        self.ac_dim = ac_dim
        self.pi_dist_layer = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: make_independent_beta_from_params(t))
        self.build(input_shape=[(None, ob_dim), ()])

    def call(self, inputs, **kwargs):
        inputs, deterministic = inputs
        # print(f'Tracing call with inputs={inputs}, deterministic={deterministic}')
        params = self.net(inputs)
        pi_distribution = self.pi_dist_layer(params)
        pi_action = get_pi_action(deterministic, pi_distribution)
        pi_action = tf.clip_by_value(pi_action, -1 + EPS, 1. - EPS)
        logp_pi = pi_distribution.log_prob(pi_action)
        pi_action_final = (pi_action - 0.5) * 2.
        logp_pi = logp_pi - np.log(2.)
        return pi_action_final, logp_pi, pi_action, pi_distribution


class SquashedGaussianMLPActor(tf.keras.Model):
    def __init__(self, ob_dim, ac_dim, mlp_hidden):
        super(SquashedGaussianMLPActor, self).__init__()
        self.net = build_mlp(ob_dim, ac_dim * 2, mlp_hidden)
        self.ac_dim = ac_dim
        self.pi_dist_layer = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: make_independent_normal_from_params(t, min_log_scale=LOG_STD_RANGE[0],
                                                                               max_log_scale=LOG_STD_RANGE[1]))
        self.build(input_shape=[(None, ob_dim), (None,)])

    def call(self, inputs, **kwargs):
        inputs, deterministic = inputs
        params = self.net(inputs)
        pi_distribution = self.pi_dist_layer(params)
        pi_action = get_pi_action(deterministic, pi_distribution)
        logp_pi = pi_distribution.log_prob(pi_action)
        logp_pi = apply_squash_log_prob(logp_pi, pi_action)
        pi_action_final = tf.tanh(pi_action)
        return pi_action_final, logp_pi, pi_action, pi_distribution
