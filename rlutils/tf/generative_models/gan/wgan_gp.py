"""
Improved Training of Wasserstein GANs
"""

import tensorflow as tf
from tqdm.auto import tqdm
from .base import GAN


class WGAN_GP(GAN):
    def __init__(self, n_critic=5, gp_weight=10, *args, **kwargs):
        self.n_critic = n_critic
        self.gp_weight = gp_weight
        super(WGAN_GP, self).__init__(*args, **kwargs)

    def _discriminator_loss(self, real_output, fake_output):
        loss = tf.reduce_mean(fake_output, axis=0) - tf.reduce_mean(real_output, axis=0)
        return loss

    def _generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output, axis=0)

    def _compute_gp(self, real_images, fake_images, training):
        batch_size = tf.shape(real_images)[0]
        alpha = tf.random.uniform(shape=[batch_size], minval=0., maxval=1.)
        for _ in range(len(real_images.shape) - 1):
            alpha = tf.expand_dims(alpha, axis=-1)
        interpolate = real_images * alpha + fake_images * (1 - alpha)
        with tf.GradientTape() as tape:
            tape.watch(interpolate)
            prediction = self.discriminator(interpolate, training=training)
        grads = tape.gradient(prediction, interpolate)
        grads = tf.reshape(grads, shape=(batch_size, -1))
        grads = tf.square(tf.norm(grads, axis=-1) - 1)
        return tf.reduce_mean(grads, axis=0)

    @tf.function
    def _train_discriminator(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = self.prior.sample(batch_size)
        with tf.GradientTape() as tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            disc_loss = self._discriminator_loss(real_output, fake_output)
            gp_loss = self._compute_gp(real_images, generated_images, training=True)
            disc_loss = disc_loss + gp_loss * self.gp_weight
        grads = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return disc_loss

    def train(self,
              x=None,
              batch_size=None,
              epochs=1,
              callbacks=None):
        for callback in callbacks:
            callback.set_model(self)
        t = 0
        dataset = tf.data.Dataset.from_tensor_slices(x).shuffle(buffer_size=x.shape[0]).batch(batch_size)
        for i in range(1, epochs + 1):
            bar = tqdm(total=-(-x.shape[0] // batch_size))
            gen_loss = 0
            for images in dataset:
                disc_loss = self._train_discriminator(images)
                if (t == self.n_critic - 1):
                    gen_loss = self._train_generator(images)
                t = (t + 1) % self.n_critic
                bar.update(1)
                bar.set_description(f'Epoch {i}/{epochs}, disc_loss: {disc_loss:.4f}, gen_loss: {gen_loss:.4f}')
            bar.close()

            for callback in callbacks:
                callback.on_epoch_end(i)
