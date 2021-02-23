"""
An updater updates the agent from the replay buffer. It also maintains statistics of the update.
"""

from abc import ABC, abstractmethod


class PolicyUpdater(ABC):
    def __init__(self, agent, replay_buffer):
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.logger = None

    @property
    def num_policy_updates(self):
        return self.policy_updates

    def reset(self):
        self.policy_updates = 0

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.agent.log_tabular()
        self.logger.log_tabular('PolicyUpdates', self.num_policy_updates)

    @abstractmethod
    def update(self, global_step):
        pass


class OnPolicyUpdater(PolicyUpdater):
    def update(self, global_step):
        data = self.replay_buffer.get()
        self.agent.train_on_batch(**data)
        self.policy_updates += 1


class OffPolicyUpdater(PolicyUpdater):
    def __init__(self, agent, replay_buffer, policy_delay, update_per_step, update_every):
        super(OffPolicyUpdater, self).__init__(agent=agent, replay_buffer=replay_buffer)
        self.policy_delay = policy_delay
        self.update_per_step = update_per_step
        self.update_every = update_every

    def update(self, global_step):
        if global_step % self.update_every == 0:
            for _ in range(self.update_per_step * self.update_every):
                batch = self.replay_buffer.sample()
                batch['update_target'] = ((self.policy_updates + 1) % self.policy_delay == 0)
                self.agent.train_on_batch(data=batch)
                self.policy_updates += 1
