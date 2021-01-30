import os

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import torch
import torchvision
from rlutils.runner import run_func_as_main
from rlutils.tf.generative_models.gan import ACWassersteinGANGradientPenalty
from tensorboardX import SummaryWriter


class CNNACWassersteinGANGradientPenalty(ACWassersteinGANGradientPenalty):
    def __init__(self, image_size, *args, **kwargs):
        self.image_size = image_size
        self.data_format = 'channels_last'
        if self.data_format == 'channels_first':
            self.data_format_ = 'NCHW'
            self.channel, self.height, self.width = image_size
        else:
            self.data_format_ = 'NHWC'
            self.height, self.width, self.channel = image_size
        assert self.height % 8 == 0 and self.width % 8 == 0
        super(CNNACWassersteinGANGradientPenalty, self).__init__(*args, **kwargs)

    def _make_generator(self) -> tf.keras.Model:
        noise_input = tf.keras.Input(shape=(self.noise_dim,), dtype=tf.float32)
        label_input = tf.keras.Input(shape=(), dtype=tf.int32)
        label_embedding = tf.keras.layers.Embedding(input_dim=self.num_classes, output_dim=self.num_classes)(
            label_input)
        inputs = tf.concat((noise_input, label_embedding), axis=-1)
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.noise_dim + self.num_classes,)),
            tf.keras.layers.Dense(self.height // 8 * self.width // 8 * 64),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.LeakyReLU(alpha=0.1),

        ])

        if self.data_format == 'channels_first':
            model.add(tf.keras.layers.Reshape(target_shape=(64, self.height // 8, self.width // 8)))
        else:
            model.add(tf.keras.layers.Reshape(target_shape=(self.height // 8, self.width // 8, 64)))

        # 4 x 4 x 64
        for i in [1, 2, 4]:
            model.add(tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(
                filters=512 // i, kernel_size=3, strides=1, data_format=self.data_format, padding='same')))
            model.add(tf.keras.layers.BatchNormalization(axis=-1))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, block_size=2,
                                                                            data_format=self.data_format_)))
        model.add(tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(
            filters=self.channel, kernel_size=3, strides=1, data_format=self.data_format,
            padding='same', activation='tanh')))

        x = inputs
        for layer in model.layers:
            x = layer(x)
        outputs = x
        final_model = tf.keras.Model(inputs=[noise_input, label_input], outputs=outputs)
        return final_model

    def _make_discriminator(self) -> tf.keras.Model:
        model = tf.keras.Sequential()
        for i in [1, 2, 4]:
            model.add(tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(
                filters=64 * i, kernel_size=3, strides=1, data_format=self.data_format, padding='same')))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(
                filters=64 * i, kernel_size=3, strides=1, data_format=self.data_format, padding='same')))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            model.add(tf.keras.layers.Lambda(lambda x: tf.nn.space_to_depth(x, block_size=2,
                                                                            data_format=self.data_format_)))
        model.add(tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(
            filters=64 * 8, kernel_size=3, strides=1, data_format=self.data_format, padding='same'
        )))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
        model.add(tf.keras.layers.Flatten())

        validity_linear = tf.keras.layers.Dense(units=1)
        class_linear = tf.keras.layers.Dense(units=self.num_classes)
        inputs = tf.keras.Input(shape=self.image_size)
        x = inputs
        for layer in model.layers:
            x = layer(x)
        features = x
        validity_out = tf.squeeze(validity_linear(features), axis=-1)
        class_out = class_linear(features)
        model = tf.keras.Model(inputs=inputs, outputs=[validity_out, class_out])
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
        y_train = np.squeeze(y_train, axis=-1)
        y_test = np.squeeze(y_test, axis=-1)
    else:
        raise NotImplementedError
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
            self.classes = tf.convert_to_tensor([[i for _ in range(10)] for i in range(self.model.num_classes)],
                                                dtype=tf.int32)
            self.classes = tf.reshape(self.classes, shape=(self.model.num_classes * 10))
            self.x = (self.model.prior.sample(self.model.num_classes * 10), self.classes)
        images = self.model.generate(self.x).numpy()
        images = np.transpose(images, (0, 3, 1, 2))
        out_images = torch.as_tensor(images)
        images = torchvision.utils.make_grid(out_images, nrow=10, normalize=True)
        self.writer.add_image(tag='data/image', img_tensor=images, global_step=epoch)


def cnn_cvae(dataset,
             lr=1e-4,
             class_loss_weight=5,
             epochs=200,
             batch_size=128,
             n_critics=2,
             seed=1):
    import shutil
    tf.random.set_seed(seed=seed)
    (x_train, y_train), (x_test, y_test) = prepare_dataset(dataset)
    num_classes = np.unique(y_train).shape[0]
    print(f'x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}, '
          f'num_classes: {num_classes}')
    image_size = x_train.shape[1:]
    model = CNNACWassersteinGANGradientPenalty(image_size=image_size, num_classes=num_classes,
                                               class_loss_weight=class_loss_weight, n_critics=n_critics)
    print(model.discriminator.summary())
    print(model.generator.summary())
    logdir = f'data/{dataset}_cnn_wgan_gp/seed{seed}'
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    callbacks = [ShowImage(logdir=logdir)]
    model.compile(generator_optimizer=tf.keras.optimizers.Adam(lr=lr, beta_1=0., beta_2=0.9),
                  discriminator_optimizer=tf.keras.optimizers.Adam(lr=lr, beta_1=0., beta_2=0.9))

    # maybe image augmentation?

    # fitting
    model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, shuffle=True,
              validation_data=(x_test, y_test), verbose=True)


if __name__ == '__main__':
    run_func_as_main(cnn_cvae)
