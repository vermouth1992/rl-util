from gym.utils import seeding


class Seeder(object):
    """
    A seeder generate a random number between [0, max_seed). We first seed np and use it to generate seeds for others
    """

    def __init__(self, seed):
        self.max_seed = 2 ** 31 - 1
        self.seed = seed
        self.reset()

    def reset(self):
        self.np_random, _ = seeding.np_random(self.seed)  # won't be interfered by the global numpy random

    def generate_seed(self):
        return self.np_random.randint(self.max_seed)

    def setup_random_global_seed(self):
        import random
        global_random_seed = self.generate_seed()
        random.seed(global_random_seed)

    def setup_np_global_seed(self):
        import numpy as np
        global_np_seed = self.generate_seed()
        np.random.seed(global_np_seed)

    def setup_tf_global_seed(self):
        import tensorflow as tf
        import os
        tf.random.set_seed(seed=self.generate_seed())
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

    def setup_torch_global_seed(self):
        import torch
        torch.random.manual_seed(self.generate_seed())
        torch.cuda.manual_seed_all(self.generate_seed())
        torch.backends.cudnn.benchmark = True
