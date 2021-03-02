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
