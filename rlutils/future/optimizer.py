import tensorflow as tf


def get_adam_optimizer(lr, **kwargs):
    """ This optimizer can be saved and loaded as a normal tf.keras.Model """
    if isinstance(lr, float):
        lr = tf.Variable(initial_value=lr)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        beta_1=tf.Variable(0.9),
        beta_2=tf.Variable(0.999),
        epsilon=tf.Variable(1e-7),
        **kwargs
    )
    _ = optimizer.iterations  # this access will invoke optimizer._iterations method and create optimizer.iter attribute
    optimizer.decay = tf.Variable(optimizer.decay)
    return optimizer
