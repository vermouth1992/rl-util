from abc import ABC, abstractmethod

from .logging import LogUser


class Agent(LogUser, ABC):
    @abstractmethod
    def act_batch_test(self, obs):
        pass

    @abstractmethod
    def act_batch_explore(self, obs, global_steps):
        pass


class OffPolicyAgent(Agent):
    def update_target(self):
        pass

    def sync_target(self):
        return self.update_target()

    @abstractmethod
    def train_on_batch(self, data, **kwargs):
        pass
