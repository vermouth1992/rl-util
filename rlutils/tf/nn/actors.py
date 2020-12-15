import tensorflow as tf
import tensorflow_probability as tfp

from rlutils.tf.distributions import make_independent_normal_from_params
from .utils import build_mlp


class SquashedGaussianMLPActor(tf.keras.Model):
    def __init__(self, ob_dim, ac_dim, mlp_hidden):
        super(SquashedGaussianMLPActor, self).__init__()
        self.net = build_mlp(ob_dim, ac_dim * 2, mlp_hidden)
        self.ac_dim = ac_dim
        self.pi_dist_layer = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: make_independent_normal_from_params(t, min_log_scale=-10, max_log_scale=5.))
        self.call = tf.function(func=self.call, input_signature=[
            (tf.TensorSpec(shape=[None, ob_dim], dtype=tf.float32),
             tf.TensorSpec(shape=(), dtype=tf.bool))
        ])

    def call(self, inputs):
        inputs, deterministic = inputs
        # print(f'Tracing call with inputs={inputs}, deterministic={deterministic}')
        params = self.net(inputs)
        pi_distribution = self.pi_dist_layer(params)
        pi_action = tf.cond(pred=deterministic, true_fn=lambda: pi_distribution.mean(),
                            false_fn=lambda: pi_distribution.sample())
        logp_pi = pi_distribution.log_prob(pi_action)
        logp_pi -= tf.reduce_sum(2. * (tf.math.log(2.) - pi_action - tf.math.softplus(-2. * pi_action)), axis=-1)
        pi_action_final = tf.tanh(pi_action)
        return pi_action_final, logp_pi, pi_action, pi_distribution
