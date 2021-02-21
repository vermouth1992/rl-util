"""
Convolutional conditional VAE. Not directly relevant to this repo. But may be used for image-based RL.
Test on MNIST and Cifar10
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfl = tfp.layers
tfd = tfp.distributions

from rlutils.tf.generative_models.vae.base import ConditionalBetaVAE
from rlutils.tf.distributions import make_independent_normal_from_params
from rlutils.future.optimizer import get_adam_optimizer
from rlutils.infra.runner import run_func_as_main
from tensorboardX import SummaryWriter
import torchvision
import torch


class ConvolutionalCVAE(ConditionalBetaVAE):
    def __init__(self, image_size, num_classes, latent_dim=5):
        self.channel, self.height, self.width = image_size
        assert self.height % 4 == 0 and self.width % 4 == 0
        self.data_format = 'channels_first'
        self.num_classes = num_classes
        super(ConvolutionalCVAE, self).__init__(latent_dim=latent_dim, beta=1.0)

    def _make_encoder(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(self.channel, self.height, self.width), dtype=tf.float32)
        cond_inputs = tf.keras.Input(shape=(), dtype=tf.int32)
        cond = tf.one_hot(indices=cond_inputs, depth=self.num_classes)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same',
                                   data_format=self.data_format, activation='relu')(inputs)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same',
                                   data_format=self.data_format, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Concatenate(axis=-1)([x, cond])
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(self.latent_dim * 2)(x)
        x = tfl.DistributionLambda(make_distribution_fn=lambda t: make_independent_normal_from_params(t))(x)
        model = tf.keras.Model(inputs=[inputs, cond_inputs], outputs=x)
        return model

    def _make_decoder(self) -> tf.keras.Model:
        latent = tf.keras.Input(shape=(self.latent_dim,), dtype=tf.float32)
        cond_inputs = tf.keras.Input(shape=(), dtype=tf.int32)
        cond = tf.one_hot(indices=cond_inputs, depth=self.num_classes)
        z = tf.keras.layers.Concatenate(axis=-1)([latent, cond])
        z = tf.keras.layers.Dense(16 * self.width // 4 * self.height // 4, activation='relu')(z)
        z = tf.keras.layers.Reshape(target_shape=[16, self.height // 4, self.width // 4])(z)
        z = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',
                                            data_format=self.data_format, activation='relu')(z)
        z = tf.keras.layers.Conv2DTranspose(filters=self.channel, kernel_size=3, strides=2, padding='same',
                                            data_format=self.data_format)(z)
        out_lambda = lambda t: tfd.Independent(distribution=tfd.Bernoulli(logits=t), reinterpreted_batch_ndims=3)
        z = tfl.DistributionLambda(make_distribution_fn=out_lambda)(z)
        model = tf.keras.Model(inputs=[latent, cond_inputs], outputs=z)
        return model


def prepare_dataset(dataset):
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    else:
        raise NotImplementedError
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    x_train = x_train.astype(np.float32) / 255.
    y_train = y_train.astype(np.int32)
    x_test = x_test.astype(np.float32) / 255.
    y_test = y_test.astype(np.int32)
    return (x_train, y_train), (x_test, y_test)


class ShowImage(tf.keras.callbacks.Callback):
    def __init__(self, logdir, model):
        super(ShowImage, self).__init__()
        self.writer = SummaryWriter(logdir=logdir)
        self.x = model.prior.sample(model.num_classes * 10)
        self.classes = tf.convert_to_tensor([[i for _ in range(10)] for i in range(model.num_classes)],
                                            dtype=tf.int32)
        self.classes = tf.reshape(self.classes, shape=(model.num_classes * 10))

    def on_epoch_end(self, epoch, logs=None):
        out_images = torch.as_tensor(
            self.model.decode_mean(z=(self.x, self.classes)).numpy())  # (num_classes * 10, C, H, W)
        images = torchvision.utils.make_grid(out_images, nrow=self.model.num_classes)
        self.writer.add_image(tag='data/image', img_tensor=images, global_step=epoch)


def cnn_cvae(dataset,
             latent_dim=10,
             lr=1e-4,
             epochs=100,
             batch_size=256):
    (x_train, y_train), (x_test, y_test) = prepare_dataset(dataset)
    num_classes = np.unique(y_train).shape[0]
    print(f'x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}, '
          f'num_classes: {num_classes}')
    image_size = x_train.shape[1:]
    model = ConvolutionalCVAE(image_size=image_size, num_classes=num_classes, latent_dim=latent_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr))
    model.compile(optimizer=get_adam_optimizer(lr=lr))
    callback = ShowImage(logdir=f'data/{dataset}', model=model)
    model.fit(x=[x_train, y_train], validation_data=((x_test, y_test),), epochs=epochs, batch_size=batch_size,
              callbacks=[callback])


if __name__ == '__main__':
    run_func_as_main(cnn_cvae)
