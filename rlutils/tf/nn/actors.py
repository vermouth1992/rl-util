import tensorflow as tf
import tensorflow_probability as tfp
from rlutils.tf.distributions import make_independent_normal_from_params, apply_squash_log_prob

from .utils import build_mlp

LOG_STD_RANGE = (-20., 5.)


class SquashedGaussianMLPActor(tf.keras.Model):
    def __init__(self, ob_dim, ac_dim, mlp_hidden):
        super(SquashedGaussianMLPActor, self).__init__()
        self.net = build_mlp(ob_dim, ac_dim * 2, mlp_hidden)
        self.ac_dim = ac_dim
        self.pi_dist_layer = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: make_independent_normal_from_params(t, min_log_scale=LOG_STD_RANGE[0],
                                                                               max_log_scale=LOG_STD_RANGE[1]))
        self.build(input_shape=[(None, ob_dim), (None,)])

    def call(self, inputs):
        inputs, deterministic = inputs
        # print(f'Tracing call with inputs={inputs}, deterministic={deterministic}')
        params = self.net(inputs)
        pi_distribution = self.pi_dist_layer(params)
        pi_action = self.get_pi_action(deterministic, pi_distribution)
        logp_pi = pi_distribution.log_prob(pi_action)
        logp_pi = apply_squash_log_prob(logp_pi, pi_action)
        pi_action_final = tf.tanh(pi_action)
        return pi_action_final, logp_pi, pi_action, pi_distribution

    @tf.function
    def get_pi_action(self, deterministic, pi_distribution):
        print(f'Tracing get_pi_action with deterministic={deterministic}')
        return tf.cond(pred=deterministic, true_fn=lambda: pi_distribution.mean(),
                       false_fn=lambda: pi_distribution.sample())
