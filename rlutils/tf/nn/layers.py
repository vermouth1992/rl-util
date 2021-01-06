import tensorflow as tf

from rlutils.np.functional import inverse_softplus


class SqueezeLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1):
        super(SqueezeLayer, self).__init__()
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.squeeze(inputs, axis=self.axis)


class EnsembleDense(tf.keras.layers.Dense):
    def __init__(self, num_ensembles, units, **kwargs):
        super(EnsembleDense, self).__init__(units=units, **kwargs)
        self.num_ensembles = num_ensembles

    def build(self, input_shape):
        last_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            'kernel',
            shape=[self.num_ensembles, last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.num_ensembles, 1, self.units],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        outputs = tf.linalg.matmul(inputs, self.kernel)  # (num_ensembles, None, units)
        if self.use_bias:
            outputs = outputs + self.bias
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


class LagrangeLayer(tf.keras.layers.Layer):
    def __init__(self, initial_value=1.0):
        super(LagrangeLayer, self).__init__()
        self.log_value = inverse_softplus(initial_value)
        self.build(input_shape=None)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(self.log_value)
        )
        self.built = True

    def __call__(self, *args, **kwargs):
        return super(LagrangeLayer, self).__call__(tf.random.normal(shape=(), dtype=tf.float32))

    def call(self, inputs, **kwargs):
        return tf.nn.softplus(self.kernel)

    def assign(self, value):
        self.kernel.assign(value)
