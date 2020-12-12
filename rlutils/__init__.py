import tensorflow as tf


def hard_update(target: tf.keras.Model, source: tf.keras.Model):
    target.set_weights(source.get_weights())


def soft_update(target: tf.keras.Model, source: tf.keras.Model, tau):
    """ This is slow because there are potential data transfer between CPU and GPU. """
    new_weights = []
    for target_weights, source_weights in zip(target.get_weights(), source.get_weights()):
        new_weights.append(target_weights * (1. - tau) + source_weights * tau)
    target.set_weights(new_weights)


@tf.function
def soft_update_tf(target: tf.keras.Model, source: tf.keras.Model, tau):
    print('Tracing soft_update_tf')
    for target_param, source_param in zip(target.variables, source.variables):
        target_param.assign(target_param * (1. - tau) + source_param * tau)


@tf.function
def hard_update_tf(target: tf.keras.Model, source: tf.keras.Model):
    print('Tracing hard_update_tf')
    for target_param, source_param in zip(target.variables, source.variables):
        target_param.assign(source_param)
