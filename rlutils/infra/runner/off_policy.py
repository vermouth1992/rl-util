import pprint

import numpy as np
import rlutils.infra as rl_infra
from rlutils.replay_buffers import UniformPyDictReplayBuffer as ReplayBuffer

from .base import BaseRunner


class OffPolicyRunner(BaseRunner):
    def setup_logger(self, config, tensorboard=False):
        super(OffPolicyRunner, self).setup_logger(config=config, tensorboard=tensorboard)
        self.sampler.set_logger(self.logger)
        self.tester.set_logger(self.logger)

    def setup_tester(self, num_test_episodes, **kwargs):
        test_env_seed = self.seeder.generate_seed()
        self.seeds_info['test_env'] = test_env_seed
        self.num_test_episodes = num_test_episodes
        self.tester = rl_infra.Tester(env_fn=self.env_fn, num_parallel_env=num_test_episodes,
                                      asynchronous=self.asynchronous, seed=test_env_seed)

    def setup_replay_buffer(self,
                            replay_size,
                            **kwargs):
        self.seeds_info['replay_buffer'] = self.seeder.generate_seed()
        self.replay_buffer = ReplayBuffer.from_vec_env(vec_env=self.env, capacity=replay_size,
                                                       seed=self.seeds_info['replay_buffer'])

    def setup_sampler(self, start_steps, **kwargs):
        self.start_steps = start_steps
        self.sampler = rl_infra.samplers.BatchSampler(env=self.env)

    def setup_updater(self, update_after, update_per_step, update_every, batch_size, **kwargs):
        self.updater = rl_infra.OffPolicyUpdater(agent=self.agent,
                                                 replay_buffer=self.replay_buffer,
                                                 update_per_step=update_per_step,
                                                 update_every=update_every,
                                                 update_after=update_after,
                                                 batch_size=batch_size)

    def run_one_step(self, t, **kwargs):
        if self.sampler.total_env_steps < self.start_steps:
            self.sampler.sample(num_steps=1,
                                collect_fn=lambda o: np.asarray(self.env.action_space.sample()),
                                replay_buffer=self.replay_buffer)
        else:
            self.sampler.sample(num_steps=1,
                                collect_fn=lambda obs: self.agent.act_batch_explore(obs, self.global_step),
                                replay_buffer=self.replay_buffer)
        # Update handling
        self.updater.update(self.global_step)

    def on_epoch_end(self, epoch, **kwargs):
        self.tester.test_agent(get_action=lambda obs: self.agent.act_batch_test(obs),
                               name=self.agent.__class__.__name__,
                               num_test_episodes=self.num_test_episodes)
        # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.logger.dump_tabular()

    def on_train_begin(self, **kwargs):
        self.sampler.reset()
        self.timer.start()

    @classmethod
    def main(cls,
             env_name,
             env_fn=None,
             exp_name=None,
             steps_per_epoch=10000,
             epochs=100,
             start_steps=10000,
             update_after=5000,
             update_every=1,
             update_per_step=1,
             batch_size=256,
             num_parallel_env=1,
             num_test_episodes=30,
             seed=1,
             # agent args
             agent_cls=None,
             agent_kwargs={},
             # replay
             replay_size=int(1e6),
             logger_path=None
             ):
        config = locals()

        runner = cls(seed=seed, steps_per_epoch=steps_per_epoch, epochs=epochs,
                     exp_name=exp_name, logger_path=logger_path)
        runner.setup_env(env_name=env_name, env_fn=env_fn, num_parallel_env=num_parallel_env,
                         asynchronous=False, num_test_episodes=num_test_episodes)
        runner.setup_agent(agent_cls=agent_cls, **agent_kwargs)
        runner.setup_replay_buffer(replay_size=replay_size)
        runner.setup_sampler(start_steps=start_steps)
        runner.setup_tester(num_test_episodes=num_test_episodes)
        runner.setup_updater(update_after=update_after,
                             update_per_step=update_per_step,
                             update_every=update_every,
                             batch_size=batch_size)
        runner.setup_logger(config=config, tensorboard=False)

        pprint.pprint(runner.seeds_info)

        runner.run()
