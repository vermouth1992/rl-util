import os

import tensorflow as tf

from .base import BaseRunner, OffPolicyRunner, OnPolicyRunner


class TFRunner(BaseRunner):
    def setup_global_seed(self):
        super(TFRunner, self).setup_global_seed()
        tf.random.set_seed(seed=self.seeder.generate_seed())
        os.environ['TF_DETERMINISTIC_OPS'] = '1'


class TFOffPolicyRunner(OffPolicyRunner, TFRunner):
    pass


class TFOnPolicyRunner(OnPolicyRunner, TFRunner):
    pass
