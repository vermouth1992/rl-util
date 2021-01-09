import tensorflow as tf
from tensorflow.keras.regularizers import l2

from .layers import EnsembleDense, SqueezeLayer


def build_mlp(input_dim, output_dim, mlp_hidden, num_ensembles=None, num_layers=3,
              activation='relu', out_activation=None, squeeze=False, dropout=None,
              batch_norm=False, layer_norm=False, regularization=None, out_regularization=None):
    assert not (batch_norm and layer_norm), "batch_norm and layer_norm can't be True simultaneously"
    assert squeeze and output_dim == 1, "squeeze must have output_dim=1"
    model = tf.keras.Sequential()
    regularizer = l2(regularization) if regularization is not None else None
    out_regularizer = l2(out_regularization) if out_regularization is not None else None
    # input layer
    if num_ensembles is None:
        model.add(tf.keras.layers.InputLayer(batch_input_shape=(None, input_dim)))
    else:
        model.add(tf.keras.layers.InputLayer(batch_input_shape=(num_ensembles, None, input_dim)))
    # intermediate layers: Dense + normalization layer (optional) + activation + dropout (optional)
    for _ in range(num_layers - 1):
        if num_ensembles is None:
            model.add(tf.keras.layers.Dense(mlp_hidden, kernel_regularizer=regularizer))
            if batch_norm:
                model.add(tf.keras.layers.BatchNormalization(axis=-1))
            if layer_norm:
                model.add(tf.keras.layers.LayerNormalization(axis=-1))
        else:
            model.add(EnsembleDense(num_ensembles, mlp_hidden, kernel_regularizer=regularizer))
            if batch_norm:
                model.add(tf.keras.layers.BatchNormalization(axis=[0, -1]))
            if layer_norm:
                model.add(tf.keras.layers.LayerNormalization(axis=[0, -1]))
        model.add(tf.keras.layers.Activation(activation=activation))
        if dropout is not None:
            model.add(tf.keras.layers.Dropout(rate=dropout))
    # final layer
    if num_ensembles is None:
        model.add(tf.keras.layers.Dense(output_dim, activation=out_activation, kernel_regularizer=out_regularizer))
    else:
        model.add(EnsembleDense(num_ensembles, output_dim, activation=out_activation,
                                kernel_regularizer=out_regularizer))
    if output_dim == 1 and squeeze is True:
        model.add(SqueezeLayer(axis=-1))
    return model
