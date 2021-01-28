import numpy as np
import tensorflow as tf

from rlutils.runner import run_func_as_main
from tensorboardX import SummaryWriter
import torchvision
import torch

from rlutils.tf.generative_models.gan.wgan_gp import WGAN_GP


class CNN_WGAN_GP(WGAN_GP):
    def __init__(self, image_size, *args, **kwargs):
        self.channel, self.height, self.width = image_size
        assert self.height % 8 == 0 and self.width % 8 == 0
        self.data_format = 'channels_first'
        super(CNN_WGAN_GP, self).__init__(*args, **kwargs)

    def _make_generator(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.noise_dim,)),
            tf.keras.layers.Dense(self.height // 8 * self.width // 8 * 64),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape(target_shape=(64, self.height // 8, self.width // 8))
        ])
        # 4 x 4 x 64
        for _ in range(3):
            model.add(tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2,
                                                      data_format=self.data_format, padding='same'))
            model.add(tf.keras.layers.BatchNormalization(axis=-1))
            model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Conv2DTranspose(filters=self.channel, kernel_size=3, strides=1,
                                                  data_format=self.data_format, padding='same', activation='tanh'))
        return model

    def _make_discriminator(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.channel, self.height, self.width)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, data_format=self.data_format, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, data_format=self.data_format, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, data_format=self.data_format, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, data_format=self.data_format, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, data_format=self.data_format, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, data_format=self.data_format, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda(function=lambda x: tf.squeeze(x, axis=-1))
        ])
        return model


def prepare_dataset(dataset):
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), mode='edge')
        x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), mode='edge')
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    else:
        raise NotImplementedError
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    y_train = y_train.astype(np.int32)
    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    y_test = y_test.astype(np.int32)
    return (x_train, y_train), (x_test, y_test)


class ShowImage(tf.keras.callbacks.Callback):
    def __init__(self, logdir):
        super(ShowImage, self).__init__()
        self.writer = SummaryWriter(logdir=logdir)
        self.x = None

    def on_epoch_end(self, epoch, logs=None):
        if self.x is None:
            self.x = self.model.prior.sample(100)
        out_images = torch.as_tensor(self.model.generate(self.x).numpy())
        images = torchvision.utils.make_grid(out_images, nrow=10)
        self.writer.add_image(tag='data/image', img_tensor=images, global_step=epoch)


def cnn_cvae(dataset,
             epochs=100,
             batch_size=64):
    (x_train, y_train), (x_test, y_test) = prepare_dataset(dataset)
    num_classes = np.unique(y_train).shape[0]
    print(f'x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}, '
          f'num_classes: {num_classes}')
    image_size = x_train.shape[1:]
    model = CNN_WGAN_GP(image_size=image_size, lr=2e-4)
    callback = ShowImage(logdir=f'data/{dataset}_cnn_wgan_gp')
    model.train(x=x_train, epochs=epochs, batch_size=batch_size, callbacks=[callback])


if __name__ == '__main__':
    run_func_as_main(cnn_cvae)
