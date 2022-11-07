import collections
import threading
import time
from typing import List

import ray
from ray.util import queue

from rlutils import logx


class Learner(object):
    def __init__(self,
                 make_agent_fn,
                 replay_manager,
                 receive_queue: queue.Queue,
                 push_queue: queue.Queue,
                 weight_push_freq=10,
                 batch_size=256,
                 prefetch=10,
                 prefetch_rate_limit=0.99,
                 num_threads=1,
                 # field for the main learner
                 learner_push_queue: queue.Queue = None,
                 learner_receive_queues: List[queue.Queue] = None,
                 testing_queue: queue.Queue = None,
                 sync_freq=None,
                 target_update_freq=None,
                 logging_freq=None,
                 main_learner=False
                 ):
        import torch
        torch.set_num_threads(num_threads)

        self.agent = make_agent_fn()
        self.replay_manager = replay_manager
        self.batch_size = batch_size
        self.weight_push_freq = weight_push_freq
        self.receive_queue = receive_queue
        self.push_queue = push_queue
        self.policy_updates = 0

        from rlutils.logx import EpochLogger
        self.logger = EpochLogger()
        self.agent.set_logger(self.logger)

        self.sampled_data = []

        self.prefetch_queue = collections.deque(maxlen=prefetch)
        self.prefetch_times = 0
        self.prefetch_interval = 0.1
        self.prefetch_rate_limit = prefetch_rate_limit

        self.store_weights()

        self.learner_push_queue = learner_push_queue
        self.learner_receive_queues = learner_receive_queues
        self.testing_queue = testing_queue
        self.sync_freq = sync_freq
        self.logging_freq = logging_freq
        self.target_update_freq = target_update_freq

        self.main = main_learner
        if self.main:
            self.local_agent = make_agent_fn()
            assert self.receive_queue is None and self.push_queue is None
        else:
            self.local_agent = None

    def get_weights(self):
        return self.weights_id

    def store_weights(self):
        state_dict = rlu.nn.functional.get_state_dict(self.agent)
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
        background_thread = threading.Thread(target=self.prefetch, daemon=True)
        background_thread.start()
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

    def local_train(self, local_gradient_steps):
        for _ in range(local_gradient_steps):
            self.policy_updates += 1
            transaction_id, data = self.get_data()
            info = self.agent.train_on_batch(data, update_target=None)
            self.replay_manager.update_priorities.remote(transaction_id,
                                                         info['TDError'].cpu().numpy())

            if self.policy_updates % self.weight_push_freq:
                self.store_weights()

    def train(self):
        if self.main:
            num_learners = len(self.learner_receive_queues) + 1
            logx.log(f'Number of learners: {num_learners}')
        else:
            num_learners = None
        start = time.time()
        while True:

            if not self.main:
                # step 1: get weights from queue
                weights, local_gradient_steps = self.receive_queue.get()
                # step 2: run for local_gradient_steps training
                self.agent.load_state_dict(weights)
                self.local_train(local_gradient_steps)
                # step 3: push the new weights to a queue
                self.push_queue.put(rlu.nn.functional.get_state_dict(self.agent))

            else:
                # step 1: push the weights to the queue
                if num_learners > 1:
                    state_dict = rlu.nn.functional.get_state_dict(self.agent)
                    for q in self.learner_receive_queues:
                        q.put((state_dict, self.sync_freq))

                # perform updates
                self.local_train(self.sync_freq)

                # wait for other learners
                for i in range(num_learners - 1):
                    weights = self.learner_push_queue.get()
                    # Process partial data when available
                    self.local_agent.load_state_dict(weights)
                    for target_param, param in zip(self.agent.q_network.parameters(),
                                                   self.local_agent.q_network.parameters()):
                        param = param.to(target_param.data.device)
                        if i == 0:
                            target_param.data = target_param.data / num_learners

                        target_param.data += param.data / num_learners

                if num_learners > 1:
                    if self.policy_updates % self.target_update_freq == 0 or self.sync_freq >= self.target_update_freq:
                        self.agent.update_target()

                if self.policy_updates % self.logging_freq == 0:
                    # perform testing and logging
                    try:
                        self.testing_queue.put((rlu.nn.functional.get_state_dict(self.agent),
                                                self.policy_updates,
                                                self.policy_updates / (time.time() - start)),
                                               block=False)
                    except queue.Full:
                        logx.log('Testing queue is full. Skip this epoch', color='red')
