import time
from typing import Dict

import numpy as np


class ReplayManager(object):
    def __init__(self, make_replay_fn,
                 update_after=10000):
        self.replay_buffer = make_replay_fn()
        self.update_after = update_after
        self.total_env_interactions = 0
        self.start_time = None
        self.last_total_env_interactions = 0

    def get_stats(self):
        current_time = time.time()
        stats = {}
        stats['TotalEnvInteracts'] = self.total_env_interactions
        stats['Samples/s'] = (self.total_env_interactions - self.last_total_env_interactions) / (
                current_time - self.start_time)

        self.start_time = current_time
        self.last_total_env_interactions = self.total_env_interactions
        return stats

    def sample(self, batch_size):
        data = self.replay_buffer.sample(batch_size)
        return data

    def update_priorities(self, transaction_id, priorities):
        self.replay_buffer.update_priorities(transaction_id, priorities)

    def ready(self):
        return len(self.replay_buffer) >= self.update_after, len(self.replay_buffer)

    def add(self, data: Dict[str, np.ndarray], priority=None):
        if self.start_time is None:
            self.start_time = time.time()
        self.replay_buffer.add(data, priority)
        self.total_env_interactions += len(list(data.values())[0])
