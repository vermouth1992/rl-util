"""
The prioritized replay buffer is integrated with the learner
"""

import torch.nn as nn

from rlutils.interface.agent import Agent


class Learner(object):
    def __init__(self, agent_fn, make_global_buffer,
                 stats_send_freq=100,
                 terminal_check_freq=100,
                 weight_sync_freq=1000,
                 batch_size=256,
                 update_after=5000,
                 num_fetch_per_update=1
                 ):
        self.agent = agent_fn()
        self.global_buffer = make_global_buffer()
        self.terminal_check_freq = terminal_check_freq
        self.weight_sync_freq = weight_sync_freq
        self.stats_send_freq = stats_send_freq
        self.batch_size = batch_size
        self.update_after = update_after
        self.num_fetch_per_update = num_fetch_per_update
        self.infos = []
        assert isinstance(self.agent, Agent)
        assert isinstance(self.agent, nn.Module)

    def is_terminate(self) -> bool:
        raise NotImplementedError

    def send_weights(self):
        raise NotImplementedError

    def send_stats(self, stats):
        raise NotImplementedError

    def fetch_data(self, blocking=True):
        raise NotImplementedError

    def run(self):
        self.send_weights()
        local_steps = 0
        while True:
            local_steps += 1
            if local_steps % self.terminal_check_freq == 0:
                if self.is_terminate():
                    break

            for _ in range(self.num_fetch_per_update):
                self.fetch_data()

            if self.global_buffer.size() >= self.update_after:
                data = self.global_buffer.sample(self.batch_size)
                info, new_priorities = self.agent.train_on_batch(data)
                self.infos.append(info)
                self.global_buffer.update_priority(new_priorities)

            if local_steps % self.weight_sync_freq:
                self.send_weights()

            if local_steps % self.stats_send_freq:
                self.send_stats({
                    'infos': self.infos,
                    'PolicyUpdates': local_steps
                })
                self.infos = []


class SingleMachineLearner(Learner):
    def __init__(self, **kwargs):
        super(SingleMachineLearner, self).__init__(**kwargs)

    def is_terminate(self) -> bool:
        pass

    def send_stats(self, stats):
        pass

    def send_weights(self):
        pass

    def fetch_data(self, blocking=True):
        pass
