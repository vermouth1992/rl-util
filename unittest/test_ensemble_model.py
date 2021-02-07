import unittest

import numpy as np
import rlutils.tf as rlu
import tensorflow as tf


class TestEnsembleModel(unittest.TestCase):
    def test_EnsembleMinQNet(self):
        num_ensembles = 5
        model = rlu.nn.build_mlp(input_dim=10, output_dim=3, mlp_hidden=20, num_ensembles=num_ensembles)
        inputs = tf.random.normal(shape=[100, 10])
        target = tf.random.normal(shape=[100, 3])

        for i in range(num_ensembles):
            with tf.GradientTape() as tape:
                output = model(inputs=inputs)  # (ensemble, None, out_dim)
                loss = tf.reduce_mean(tf.square(target - output[i]))
            grad = tape.gradient(loss, model.trainable_variables)
            for g in grad:
                for j in range(num_ensembles):
                    if j != i:
                        # the grad must be all zero for all others
                        assert np.allclose(g.numpy()[j], 0.)


if __name__ == '__main__':
    unittest.main()
