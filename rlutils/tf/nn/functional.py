import tensorflow as tf
from tensorflow.keras.regularizers import l2

from .layers import EnsembleDense, SqueezeLayer


def build_mlp(input_dim, output_dim, mlp_hidden, num_ensembles=None, num_layers=3,
              activation='relu', out_activation=None, squeeze=False, dropout=None,
              batch_norm=False, layer_norm=False, regularization=None, out_regularization=None,
              kernel_initializer='glorot_uniform', bias_initializer='zeros',
              out_kernel_initializer='glorot_uniform', out_bias_initializer='zeros'):
    """

    Args:
        input_dim: input dimension
        output_dim: output dimension
        mlp_hidden: hidden size. int or a list of integers
        num_ensembles: number of ensembles
        num_layers: number of layers. Must be compatible with mlp_hidden
        activation: activation after each hidden layer
        out_activation: activation after the output layer
        squeeze: whether squeeze the output
        dropout: apply dropout
        batch_norm: apply batch normalization
        layer_norm: apply layer normalization
        regularization: hidden kernel regularization
        out_regularization: output kernel regularization
        kernel_initializer: hidden kernel initializer
        bias_initializer: bias initializer
        out_kernel_initializer: The range of the output kernel is set to small number.
        out_bias_initializer: output bias initializer

    Returns:

    """
    assert not (batch_norm and layer_norm), "batch_norm and layer_norm can't be True simultaneously"
    if squeeze:
        assert output_dim == 1, "if squeeze, output_dim must have size 1"
    if isinstance(mlp_hidden, int):
        mlp_hidden = [mlp_hidden] * (num_layers - 1)
    elif isinstance(mlp_hidden, list) or isinstance(mlp_hidden, tuple):
        assert len(mlp_hidden) == num_layers - 1, 'len(mlp_hidden) must equal to num_layers - 1.'
    else:
        raise ValueError(f'Unknown type mlp_hidden. Got {type(mlp_hidden)}')

    model = tf.keras.Sequential()
    regularizer = l2(regularization) if regularization is not None else None
    out_regularizer = l2(out_regularization) if out_regularization is not None else None
    # input layer
    if num_ensembles is None:
        model.add(tf.keras.layers.InputLayer(batch_input_shape=(None, input_dim)))
    else:
        model.add(tf.keras.layers.InputLayer(batch_input_shape=(num_ensembles, None, input_dim)))
    # intermediate layers: Dense + normalization layer (optional) + activation + dropout (optional)
    for i in range(num_layers - 1):
        if num_ensembles is None:
            model.add(tf.keras.layers.Dense(mlp_hidden[i], kernel_regularizer=regularizer,
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer))
            if batch_norm:
                model.add(tf.keras.layers.BatchNormalization(axis=-1))
            if layer_norm:
                model.add(tf.keras.layers.LayerNormalization(axis=-1))
        else:
            model.add(EnsembleDense(num_ensembles, mlp_hidden[i], kernel_regularizer=regularizer,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer))
            if batch_norm:
                model.add(tf.keras.layers.BatchNormalization(axis=[0, -1]))
            if layer_norm:
                model.add(tf.keras.layers.LayerNormalization(axis=[0, -1]))
        model.add(tf.keras.layers.Activation(activation=activation))
        if dropout is not None:
            model.add(tf.keras.layers.Dropout(rate=dropout))
    # final layer
    if num_ensembles is None:
        model.add(tf.keras.layers.Dense(output_dim, activation=out_activation, kernel_regularizer=out_regularizer,
                                        kernel_initializer=out_kernel_initializer,
                                        bias_initializer=out_bias_initializer))
    else:
        model.add(EnsembleDense(num_ensembles, output_dim, activation=out_activation,
                                kernel_regularizer=out_regularizer,
                                kernel_initializer=out_kernel_initializer,
                                bias_initializer=out_bias_initializer))
    if output_dim == 1 and squeeze is True:
        model.add(SqueezeLayer(axis=-1))
    return model
