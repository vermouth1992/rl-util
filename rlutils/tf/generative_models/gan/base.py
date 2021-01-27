import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class GAN(tf.keras.Model):
    def __init__(self, noise_dim=128, lr=1e-4):
        super(GAN, self).__init__()
        self.noise_dim = noise_dim
        self.generator = self._make_generator()
        self.discriminator = self._make_discriminator()
        self.prior = self._make_prior()
        self.generator_optimizer = tf.keras.optimizers.Adam(lr=lr, beta_1=0.5, beta_2=0.9)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(lr=lr, beta_1=0.5, beta_2=0.9)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.logger = None
        self.compile()

    @tf.function
    def generate(self, z):
        return self.generator(z, training=False)

    def _make_prior(self):
        return tfd.Independent(tfd.Normal(loc=tf.zeros(self.noise_dim), scale=tf.ones(self.noise_dim)),
                               reinterpreted_batch_ndims=1)

    def _make_generator(self) -> tf.keras.Model:
        raise NotImplementedError

    def _make_discriminator(self) -> tf.keras.Model:
        raise NotImplementedError

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('GenLoss', average_only=True)
        self.logger.log_tabular('DiscLoss', average_only=True)

    def sample(self, n):
        print(f'Tracing sample with n={n}')
        noise = self.prior.sample(n)
        outputs = self.generator(noise, training=False)
        return outputs

    def predict_real_fake(self, x):
        print(f'Tracing predict_real_fake with x={x}')
        return tf.sigmoid(self.discriminator(x, training=False))

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def _generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def _train_generator(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = self.prior.sample(batch_size)
        with tf.GradientTape() as tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = self._generator_loss(fake_output)
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        return gen_loss

    def _train_discriminator(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = self.prior.sample(batch_size)
        with tf.GradientTape() as tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            disc_loss = self._discriminator_loss(real_output, fake_output)
        grads = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return disc_loss

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        gen_loss = self._train_generator(x)
        disc_loss = self._train_discriminator(x)
        return {
            'gen_loss': gen_loss,
            'disc_loss': disc_loss
        }
