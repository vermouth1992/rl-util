"""
Common in the runner:
1. Setup environment
2. Setup logger
3. Setup agent
4. Run
"""

import os
import random
from abc import abstractmethod, ABC

import numpy as np
from tqdm.auto import trange

import rlutils.gym
import rlutils.infra as rl_infra
from rlutils.interface.agent import Agent
from rlutils.logx import EpochLogger, setup_logger_kwargs


class BaseRunner(ABC):
    def __init__(self, seed, steps_per_epoch, epochs, exp_name=None, logger_path='data'):
        self.exp_name = exp_name
        self.logger_path = logger_path
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.seed = seed
        self.global_step = 1
        self.total_steps = steps_per_epoch * epochs
        self.seeder = rl_infra.Seeder(seed=seed)
        self.timer = rl_infra.StopWatch()
        self.agent = None
        self.setup_global_seed()

    def setup_logger(self, config, tensorboard=False):
        if self.exp_name is None:
            self.exp_name = f'{self.env_name}_{self.agent.__class__.__name__}_test'
        assert self.exp_name is not None, 'Call setup_env before setup_logger if exp passed by the contructor is None.'
        logger_kwargs = setup_logger_kwargs(exp_name=self.exp_name, data_dir=self.logger_path, seed=self.seed)
        self.logger = EpochLogger(**logger_kwargs, tensorboard=tensorboard)
        self.logger.save_config(config)

        self.timer.set_logger(logger=self.logger)
        self.agent.set_logger(logger=self.logger)

    def setup_global_seed(self):
        self.seeds_info = {}
        # we set numpy seed first and use it to generate other seeds
        global_np_seed = self.seeder.generate_seed()
        global_random_seed = self.seeder.generate_seed()
        np.random.seed(global_np_seed)
        random.seed(global_random_seed)

        self.seeds_info['global_np'] = global_np_seed
        self.seeds_info['global_random'] = global_random_seed

    @property
    def seeds(self):
        return self.seeds_info

    @abstractmethod
    def run_one_step(self, t):
        raise NotImplementedError

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def setup_env(self,
                  env_name,
                  env_fn=None,
                  num_parallel_env=1,
                  asynchronous=False,
                  num_test_episodes=None):
        import gym
        self.env_name = env_name
        if env_fn is None:
            env_fn = lambda: gym.make(env_name)
        self.env_fn = rlutils.gym.utils.wrap_env_fn(env_fn, truncate_obs_dtype=True, normalize_action_space=True)
        self.dummy_env = self.env_fn()

        if num_parallel_env > 0:
            self.env = rlutils.gym.utils.create_vector_env(env_fn=env_fn,
                                                           normalize_action_space=True,
                                                           num_parallel_env=num_parallel_env,
                                                           asynchronous=asynchronous)
            env_seed = self.seeder.generate_seed()
            env_action_space_seed = self.seeder.generate_seed()
            self.env.seed(env_seed)
            self.env.action_space.seed(env_action_space_seed)
            self.seeds_info['env'] = env_seed
            self.seeds_info['env_action_space'] = env_action_space_seed
        else:
            # no training environment is used. Used in Offline RL
            self.env = None

        self.num_test_episodes = num_test_episodes
        self.asynchronous = asynchronous

    def setup_agent(self, agent_cls, **kwargs):
        self.agent = agent_cls(env=self.dummy_env, **kwargs)
        assert isinstance(self.agent, Agent), f'agent must be an Agent class. Got {type(self.agent)}'

    def run(self):
        self.on_train_begin()
        for i in range(1, self.epochs + 1):
            self.on_epoch_begin(i)
            for t in trange(self.steps_per_epoch, desc=f'Epoch {i}/{self.epochs}'):
                self.run_one_step(t)
                self.global_step += 1
            self.on_epoch_end(i)
        self.on_train_end()

    @classmethod
    def main(cls, *args, **kwargs):
        raise NotImplementedError

    def save_checkpoint(self, path=None):
        pass

    def load_checkpoint(self, path=None):
        pass

    def save_agent(self, path=None):
        if path is None:
            path = os.path.join(self.logger.output_dir, 'agent.tf')
        self.agent.save_weights(path)

    def load_agent(self, path=None):
        if path is None:
            path = os.path.join(self.logger.output_dir, 'agent.tf')
        self.agent.load_weights(path)
