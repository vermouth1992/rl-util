import tensorflow as tf
import tensorflow_probability as tfp

from rlutils.tf.distributions import make_independent_normal_from_params, apply_squash_log_prob
from .functional import build_mlp

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


class AtariQNetworkDeepMind(tf.keras.layers.Layer):
    def __init__(self, act_dim, frame_stack=4, dueling=False, data_format='channels_first', scale_input=True):
        super(AtariQNetworkDeepMind, self).__init__()
        if data_format == 'channels_first':
            self.batch_input_shape = (None, frame_stack, 84, 84)
        else:
            self.batch_input_shape = (None, 84, 84, frame_stack)
        self.features = tf.keras.Sequential([
            tf.keras.layers.InputLayer(batch_input_shape=self.batch_input_shape),
            tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, padding='same', activation='relu',
                                   data_format=data_format),
            tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu',
                                   data_format=data_format),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                                   data_format=data_format),
            tf.keras.layers.Flatten()
        ])
        self.dueling = dueling
        self.scale_input = scale_input
        self.q_feature = tf.keras.layers.Dense(units=512, activation='relu')
        self.adv_fc = tf.keras.layers.Dense(units=act_dim)
        if self.dueling:
            self.value_fc = tf.keras.layers.Dense(units=1)
        else:
            self.value_fc = None
        self.build(input_shape=self.batch_input_shape)

    def call(self, inputs, training=None):
        if self.scale_input:
            # this assumes the inputs is in image format (None, frame_stack, 84, 84)
            inputs = tf.cast(inputs, dtype=tf.float32) / 255.
        features = self.features(inputs, training=training)
        q_value = self.q_feature(features, training=training)
        adv = self.adv_fc(q_value)  # (None, act_dim)
        if self.dueling:
            adv = adv - tf.reduce_mean(adv, axis=-1, keepdims=True)
            value = self.value_fc(q_value)
            q_value = value + adv
        else:
            q_value = adv
        return q_value
