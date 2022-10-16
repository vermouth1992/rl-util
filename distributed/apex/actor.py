import threading

import ray


class Actor(object):
    def __init__(self,
                 make_agent_fn,
                 make_sampler_fn,
                 make_local_buffer_fn,
                 weight_server_lst,
                 replay_manager_lst,
                 set_thread_fn,
                 weight_update_freq=100,
                 num_threads=1,
                 ):
        set_thread_fn(num_threads)

        self.agent = make_agent_fn()
        self.agent.eval()
        self.weight_server_lst = weight_server_lst
        self.replay_manager_lst = replay_manager_lst
        self.weight_update_freq = weight_update_freq
        self.sampler = make_sampler_fn()
        self.local_buffer = make_local_buffer_fn()

        from rlutils.logx import EpochLogger
        self.logger = EpochLogger()
        self.sampler.set_logger(self.logger)

        self.current_data_index = 0
        self.current_weight_index = 0
        self.num_learners = len(self.weight_server_lst)

        assert len(self.weight_server_lst) == len(self.replay_manager_lst)

    def get_stats(self):
        stats = self.logger.get_epoch_dict()
        self.logger.clear_epoch_dict()
        return stats

    def run(self):
        background_thread = threading.Thread(target=self.train, daemon=True)
        background_thread.start()

    def add_local_to_global(self, data, priority):
        # pick a random replay manager
        replay_manager = self.replay_manager_lst[self.current_data_index]
        replay_manager.add.remote(data, priority)

        self.current_data_index = (self.current_data_index + 1) % self.num_learners

    def get_weights(self):
        weight_server = self.weight_server_lst[self.current_weight_index]
        weights_id = ray.get(weight_server.get_weights.remote())
        weights = ray.get(weights_id)
        self.current_weight_index = (self.current_weight_index + 1) % self.num_learners
        return weights

    def train(self):
        self.update_weights()
        self.sampler.reset()
        local_steps = 0
        while True:
            self.sampler.sample(num_steps=1, collect_fn=lambda o: self.agent.act_batch_explore(o, None),
                                replay_buffer=self.local_buffer)
            local_steps += 1
            if self.local_buffer.is_full():
                data = self.local_buffer.storage.get()
                priority = self.agent.compute_priority(data)
                self.add_local_to_global(data, priority)
                self.local_buffer.reset()

            if local_steps % self.weight_update_freq == 0:
                self.update_weights()

    def update_weights(self):
        weights = self.get_weights()
        self.agent.load_state_dict(weights)
