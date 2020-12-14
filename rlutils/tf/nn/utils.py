import tensorflow as tf
from .layers import EnsembleDense, SqueezeLayer
from tensorflow.keras.regularizers import l2


def build_mlp(input_dim, output_dim, mlp_hidden, num_ensembles=None, num_layers=3,
              activation='relu', out_activation=None, squeeze=False, dropout=None,
              batch_norm=False, regularization=None, out_regularization=None):
    model = tf.keras.Sequential()
    regularizer = l2(regularization) if regularization is not None else None
    out_regularizer = l2(out_regularization) if out_regularization is not None else None
    if num_ensembles is None:
        model.add(tf.keras.layers.InputLayer(batch_input_shape=(None, input_dim)))
    else:
        model.add(tf.keras.layers.InputLayer(batch_input_shape=(num_ensembles, None, input_dim)))
    for _ in range(num_layers - 1):
        if num_ensembles is None:
            model.add(tf.keras.layers.Dense(mlp_hidden, kernel_regularizer=regularizer))
            if batch_norm:
                model.add(tf.keras.layers.BatchNormalization(axis=-1))
            model.add(tf.keras.layers.Activation(activation=activation))
        else:
            model.add(EnsembleDense(num_ensembles, mlp_hidden, kernel_regularizer=regularizer))
            if batch_norm:
                model.add(tf.keras.layers.BatchNormalization(axis=[0, -1]))
            model.add(tf.keras.layers.Activation(activation=activation))
        if dropout is not None:
            model.add(tf.keras.layers.Dropout(rate=dropout))
    if num_ensembles is None:
        model.add(tf.keras.layers.Dense(mlp_hidden, activation=out_activation, kernel_regularizer=out_regularizer))
    else:
        model.add(EnsembleDense(num_ensembles, mlp_hidden, activation=out_activation,
                                kernel_regularizer=out_regularizer))
    if output_dim == 1 and squeeze is True:
        model.add(SqueezeLayer(axis=-1))
    return model
