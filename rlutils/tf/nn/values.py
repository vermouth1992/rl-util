import tensorflow as tf

from rlutils.tf.nn.functional import build_mlp


class EnsembleMinQNet(tf.keras.layers.Layer):
    def __init__(self, ob_dim, ac_dim, mlp_hidden, num_ensembles=2, num_layers=3):
        super(EnsembleMinQNet, self).__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.mlp_hidden = mlp_hidden
        self.num_ensembles = num_ensembles
        self.num_layers = num_layers
        self.q_net = build_mlp(input_dim=self.ob_dim + self.ac_dim,
                               output_dim=1,
                               mlp_hidden=self.mlp_hidden,
                               num_ensembles=self.num_ensembles,
                               num_layers=num_layers,
                               squeeze=True)
        self.build(input_shape=[(None, ob_dim), (None, ac_dim)])

    def get_config(self):
        config = super(EnsembleMinQNet, self).get_config()
        config.update({
            'ob_dim': self.ob_dim,
            'ac_dim': self.ac_dim,
            'mlp_hidden': self.mlp_hidden,
            'num_ensembles': self.num_ensembles,
            'num_layers': self.num_layers
        })
        return config

    def call(self, inputs, training=None, mask=None):
        obs, act = inputs
        inputs = tf.concat((obs, act), axis=-1)
        inputs = tf.tile(tf.expand_dims(inputs, axis=0), (self.num_ensembles, 1, 1))
        q = self.q_net(inputs)  # (num_ensembles, None)
        if training:
            return q
        else:
            return tf.reduce_min(q, axis=0)
