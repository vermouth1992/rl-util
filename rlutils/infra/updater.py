"""
An updater updates the agent from the replay buffer. It also maintains statistics of the update.
"""


class OffPolicyUpdater(object):
    def __init__(self, agent, replay_buffer, policy_delay, update_per_step):
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.policy_delay = policy_delay
        self.update_per_step = update_per_step
        self.logger = None

    def reset(self):
        self.policy_updates = 0

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.agent.log_tabular()

    def update(self):
        for _ in range(self.update_per_step):
            batch = self.replay_buffer.sample()
            batch['update_target'] = ((self.policy_updates + 1) % self.policy_delay == 0)
            self.agent.train_on_batch(data=batch)
            self.policy_updates += 1
