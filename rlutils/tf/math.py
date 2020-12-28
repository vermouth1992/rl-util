import tensorflow as tf


def flatten_leading_dims(tensor, n_dims):
    if n_dims <= 1:
        return tensor
    newshape = [tf.math.reduce_prod(tf.shape(tensor)[:n_dims])] + tf.TensorShape(tf.shape(tensor)[n_dims:])
    return tf.reshape(tensor, shape=newshape)
