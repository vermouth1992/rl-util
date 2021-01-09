import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from .base import ConditionalBetaVAE

tfl = tfp.layers

from rlutils.tf.nn.functional import build_mlp
from rlutils.tf.distributions import make_independent_normal_from_params


class BehaviorPolicy(ConditionalBetaVAE):
    def __init__(self, obs_dim, act_dim, mlp_hidden=256, lr=1e-3):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.mlp_hidden = mlp_hidden
        super(BehaviorPolicy, self).__init__(latent_dim=self.act_dim * 2, beta=1.0, lr=lr)

    def _make_encoder(self) -> tf.keras.Model:
        obs_input = tf.keras.Input(shape=(self.obs_dim,), dtype=tf.float32)
        act_input = tf.keras.Input(shape=(self.act_dim,), dtype=tf.float32)
        input = tf.concat((act_input, obs_input), axis=-1)
        encoder = build_mlp(input_dim=self.obs_dim + self.act_dim,
                            output_dim=self.latent_dim * 2,
                            mlp_hidden=self.mlp_hidden,
                            num_layers=3)
        encoder.add(tfl.DistributionLambda(
            make_distribution_fn=lambda t: make_independent_normal_from_params(t, min_log_scale=-10.,
                                                                               max_log_scale=5.)))
        output = encoder(input)
        model = tf.keras.Model(inputs=[act_input, obs_input], outputs=output)
        return model

    def _make_decoder(self) -> tf.keras.Model:
        obs_input = tf.keras.Input(shape=(self.obs_dim,), dtype=tf.float32)
        latent_input = tf.keras.Input(shape=(self.latent_dim,), dtype=tf.float32)
        input = tf.concat((latent_input, obs_input), axis=-1)
        decoder = build_mlp(input_dim=self.obs_dim + self.latent_dim,
                            output_dim=self.act_dim * 2,
                            mlp_hidden=self.mlp_hidden,
                            num_layers=3)
        decoder.add(tfl.DistributionLambda(
            make_distribution_fn=lambda t: make_independent_normal_from_params(t, min_log_scale=-10.,
                                                                               max_log_scale=5.)))
        output = decoder(input)
        model = tf.keras.Model(inputs=[latent_input, obs_input], outputs=output)
        return model

    def call_n(self, inputs, training=None, n=5):
        x, cond = inputs
        batch_size = tf.shape(x)[0]
        print(f'Tracing call_n with x={x}, cond={cond}')
        posterior = self.encoder(inputs=(x, cond), training=training)
        encode_sample = posterior.sample(n)  # (n, None, z_dim)
        encode_sample = tf.reshape(encode_sample, shape=(n * batch_size, self.latent_dim))
        cond = tf.tile(cond, multiples=(n, 1))
        out = self.decoder((encode_sample, cond), training=training)
        posterior_kld = tfd.kl_divergence(posterior, self.prior)  # (None,)
        posterior_kld = tf.tile(posterior_kld, multiples=(n,))
        return out, posterior_kld

    @tf.function
    def act_batch(self, obs, deterministic=tf.convert_to_tensor(True)):
        print(f'Tracing vae act_batch with obs {obs}')
        pi_final = self.sample(cond=obs, full_path=tf.logical_not(deterministic))
        pi_final = tf.tanh(pi_final)
        return pi_final
