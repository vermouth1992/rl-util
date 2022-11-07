import numpy as np
import ray
from ray.util import queue


class Logger(object):
    def __init__(self,
                 receive_queue: queue.Queue,
                 set_weights_fn,
                 set_thread_fn,
                 logging_freq,
                 total_num_policy_updates,
                 actors,
                 local_learners,
                 replay_manager_lst,
                 num_cpus_tester,
                 make_test_agent_fn,
                 make_tester_fn,
                 num_test_episodes,
                 exp_name,
                 seed,
                 logger_path,
                 config):
        set_thread_fn(num_cpus_tester)
        # modules
        self.set_weights_fn = set_weights_fn
        self.receive_queue = receive_queue
        self.logging_freq = logging_freq
        self.actors = actors
        self.local_learners = local_learners
        self.replay_manager_lst = replay_manager_lst

        # create tester
        self.test_agent = make_test_agent_fn()
        self.tester = make_tester_fn()
        self.num_test_episodes = num_test_episodes
        self.total_num_policy_updates = total_num_policy_updates

        from rlutils.logx import EpochLogger, setup_logger_kwargs

        self.logger = EpochLogger(**setup_logger_kwargs(exp_name=exp_name, seed=seed, data_dir=logger_path))
        self.logger.save_config(config)

        self.tester.set_logger(self.logger)
        self.test_agent.set_logger(self.logger)

    def log(self):
        while True:
            weights, num_policy_updates, training_throughput = self.receive_queue.get()
            self.set_weights_fn(self.test_agent, weights)
            self.tester.test_agent(get_action=self.test_agent.act_batch_test,
                                   name=self.test_agent.__class__.__name__,
                                   num_test_episodes=self.num_test_episodes,
                                   max_episode_length=None,
                                   timeout=None,
                                   verbose=False)
            # actor stats
            for actor in self.actors:
                stats = ray.get(actor.get_stats.remote())
                self.logger.store(**stats)

            # learner states
            stats_lst = [dict(ray.get(learner.get_stats.remote())) for learner in self.local_learners]

            # replay stats
            replay_stats_lst = [dict(ray.get(replay_manager.get_stats.remote()))
                                for replay_manager in self.replay_manager_lst]

            # analyze stats
            prefetch_rates_lst = []
            for stats in stats_lst:
                prefetch_rates_lst.append(stats.pop('PrefetchRate'))
                self.logger.store(**stats)

            prefetch_rates = np.mean(prefetch_rates_lst)

            total_env_interactions = 0
            sampling_throughput = []
            for replay_stats in replay_stats_lst:
                total_env_interactions += replay_stats.pop('TotalEnvInteracts')
                sampling_throughput.append(replay_stats.pop('Samples/s'))
            sampling_throughput = np.sum(sampling_throughput)

            self.logger.log_tabular('Epoch', num_policy_updates // self.logging_freq)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('TotalEnvInteracts', total_env_interactions)
            self.logger.log_tabular('PolicyUpdates', num_policy_updates)
            self.logger.log_tabular('GradientSteps/s', training_throughput)
            self.logger.log_tabular('Samples/s', sampling_throughput)
            self.logger.log_tabular('PrefetchRate', prefetch_rates)
            self.logger.dump_tabular()

            if num_policy_updates >= self.total_num_policy_updates:
                break
