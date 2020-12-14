import tensorflow as tf


@tf.function
def soft_update(target: tf.keras.Model, source: tf.keras.Model, tau):
    print('Tracing soft_update_tf')
    for target_param, source_param in zip(target.variables, source.variables):
        target_param.assign(target_param * (1. - tau) + source_param * tau)


@tf.function
def hard_update(target: tf.keras.Model, source: tf.keras.Model):
    print('Tracing hard_update_tf')
    for target_param, source_param in zip(target.variables, source.variables):
        target_param.assign(source_param)
