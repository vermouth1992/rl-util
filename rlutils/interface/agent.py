from abc import ABC, abstractmethod

from .logging import LogUser


class Agent(LogUser, ABC):
    @abstractmethod
    def act_batch_test(self, obs):
        pass

    @abstractmethod
    def act_batch_explore(self, obs, global_steps):
        pass
