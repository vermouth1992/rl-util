from abc import ABC, abstractmethod

from gym.vector import VectorEnv
from rlutils.interface.logging import LogUser


class Sampler(LogUser, ABC):
    def __init__(self, env: VectorEnv, seed=None):
        super(Sampler, self).__init__()
        self.env = env
        self.seed = seed

    def reset(self):
        pass

    @abstractmethod
    def sample(self, num_steps, collect_fn, replay_buffer):
        pass

    @property
    @abstractmethod
    def total_env_steps(self):
        pass
