import tensorflow as tf


class GAN(tf.keras.Model):
    def __init__(self, noise_dim=5, lr=1e-3):
        super(GAN, self).__init__()
        self.noise_dim = noise_dim
        self.generator = self._make_generator()
        self.discriminator = self._make_discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(lr=lr)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(lr=lr)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.logger = None
        self.compile()

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
        noise = tf.random.normal([n, self.noise_dim])
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

    def call(self, inputs, training=None, mask=None):
        batch_size = tf.shape(inputs)[0]
        noise = tf.random.normal([batch_size, self.noise_dim])
        generated_images = self.generator(noise, training=training)

        real_output = self.discriminator(inputs, training=training)
        fake_output = self.discriminator(generated_images, training=training)

        gen_loss = self._generator_loss(fake_output)
        disc_loss = self._discriminator_loss(real_output, fake_output)
        return gen_loss, disc_loss

    def train_step(self, data):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_loss, disc_loss = self(data, training=True)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                         self.discriminator.trainable_variables))
        return {
            'gen_loss': gen_loss,
            'disc_loss': disc_loss
        }
