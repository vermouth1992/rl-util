import collections
import threading
import time

import ray
from ray.util import queue

from rlutils import logx


class Learner(object):
    def __init__(self,
                 make_agent_fn,
                 replay_manager,
                 set_thread_fn,
                 get_weights_fn,
                 testing_queue: queue.Queue,
                 logging_freq=1000,
                 weight_push_freq=10,
                 batch_size=256,
                 prefetch=10,
                 prefetch_rate_limit=0.99,
                 num_threads=1
                 ):
        set_thread_fn(num_threads)

        self.agent = make_agent_fn()
        self.get_weights_fn = get_weights_fn
        self.replay_manager = replay_manager
        self.batch_size = batch_size
        self.weight_push_freq = weight_push_freq
        self.testing_queue = testing_queue
        self.logging_freq = logging_freq
        self.policy_updates = 0

        from rlutils.logx import EpochLogger
        self.logger = EpochLogger()
        self.agent.set_logger(self.logger)

        self.prefetch_queue = collections.deque(maxlen=prefetch)
        self.prefetch_times = 0
        self.prefetch_interval = 0.1
        self.prefetch_rate_limit = prefetch_rate_limit

        self.store_weights()

    def get_weights(self):
        return self.weights_id

    def store_weights(self):
        state_dict = self.get_weights_fn(self.agent)
        self.weights_id = ray.put(state_dict)

    def get_stats(self):
        stats = self.logger.get_epoch_dict()
        self.logger.clear_epoch_dict()
        if self.policy_updates == 0:
            assert self.prefetch_times == 0
            prefetch_rate = 0.
        else:
            prefetch_rate = self.prefetch_times / self.policy_updates
        stats['PrefetchRate'] = prefetch_rate

        if stats['PrefetchRate'] < self.prefetch_rate_limit:
            # at least sleep for 1ms.
            self.prefetch_interval = max(self.prefetch_interval / 2., 0.01)
        return stats

    def run(self):
        background_thread = threading.Thread(target=self.train, daemon=True)
        background_thread.start()

    def sample_from_remote(self):
        replay_manager = self.replay_manager
        return ray.get(replay_manager.sample.remote(self.batch_size))

    def prefetch(self):
        while True:
            if len(self.prefetch_queue) < self.prefetch_queue.maxlen:
                data = self.sample_from_remote()
                self.prefetch_queue.append(data)
            else:
                time.sleep(self.prefetch_interval)

    def get_data(self):
        if len(self.prefetch_queue) != 0:
            transaction_id, data = self.prefetch_queue.popleft()
            self.prefetch_times += 1
        else:
            transaction_id, data = self.sample_from_remote()

        return transaction_id, data

    def train(self):
        # check replay buffer.
        logx.log('Waiting for replay buffer to fill', color='green')
        while True:
            ready, current_size = ray.get(self.replay_manager.ready.remote())
            if ready:
                # If ready, start to prefetch
                background_thread = threading.Thread(target=self.prefetch, daemon=True)
                background_thread.start()
                break
            else:
                time.sleep(0.5)

        logx.log('Start training', color='green')
        start = time.time()
        while True:
            self.policy_updates += 1
            transaction_id, data = self.get_data()
            info = self.agent.train_on_batch(data)
            self.replay_manager.update_priorities.remote(transaction_id, info['TDErrorNumpy'])

            if self.policy_updates % self.weight_push_freq:
                self.store_weights()

            if self.policy_updates % self.logging_freq == 0:
                # perform testing and logging
                try:
                    self.testing_queue.put((self.get_weights_fn(self.agent),
                                            self.policy_updates,
                                            self.policy_updates / (time.time() - start)),
                                           block=False)
                except queue.Full:
                    logx.log('Testing queue is full. Skip this epoch', color='red')
