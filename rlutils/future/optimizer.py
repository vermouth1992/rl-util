import tensorflow as tf


def get_adam_optimizer(lr, **kwargs):
    """ This optimizer can be saved and loaded as a normal tf.keras.Model """
    if isinstance(lr, float):
        lr = tf.Variable(initial_value=lr, trainable=False)
    elif isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
        pass
    elif isinstance(lr, tf.Variable):
        pass
    else:
        raise ValueError(f'Unknown type lr. Got {type(lr)}')
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        beta_1=tf.Variable(0.9, trainable=False),
        beta_2=tf.Variable(0.999, trainable=False),
        epsilon=tf.Variable(1e-7, trainable=False),
        **kwargs
    )
    _ = optimizer.iterations  # this access will invoke optimizer._iterations method and create optimizer.iter attribute
    optimizer.decay = tf.Variable(optimizer.decay, trainable=False)
    return optimizer


def minimize(loss, tape, model, optimizer=None):
    grads = tape.gradient(loss, model.trainable_variables)
    if optimizer is None:
        optimizer = model.optimizer
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return grads
