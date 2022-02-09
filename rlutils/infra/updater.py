"""
An updater updates the agent from the replay buffer. It also maintains statistics of the update.
"""

from abc import ABC, abstractmethod


class PolicyUpdater(ABC):
    def __init__(self, agent, replay_buffer):
        super(PolicyUpdater, self).__init__()
        self.agent = agent
        self.replay_buffer = replay_buffer

    @abstractmethod
    def update(self, global_step):
        pass


class OnPolicyUpdater(PolicyUpdater):
    def update(self, global_step):
        data = self.replay_buffer.get()
        self.agent.train_on_batch(**data)


class OffPolicyUpdater(PolicyUpdater):
    def __init__(self, agent, replay_buffer, update_per_step, update_every, update_after, batch_size):
        super(OffPolicyUpdater, self).__init__(agent=agent, replay_buffer=replay_buffer)
        self.update_per_step = update_per_step
        self.update_every = update_every
        self.update_after = update_after
        self.batch_size = batch_size

    def update(self, global_step):
        if global_step > self.update_after:
            if global_step % self.update_every == 0:
                for _ in range(int(self.update_per_step * self.update_every)):
                    batch = self.replay_buffer.sample(self.batch_size)
                    info = self.agent.train_on_batch(data=batch)
