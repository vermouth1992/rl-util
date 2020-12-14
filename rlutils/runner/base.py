"""
Common in the runner:
1. Setup environment
2. Setup logger
3. Setup agent
4. Run
"""

import random
from abc import abstractmethod, ABC

import gym
import numpy as np
import tensorflow as tf
from gym.wrappers import FrameStack
from tqdm.auto import trange

from rlutils.logx import EpochLogger
from .run_utils import setup_logger_kwargs


class BaseRunner(ABC):
    def __init__(self, env_name, seed, steps_per_epoch, epochs, num_parallel_env, wrappers=None,
                 num_test_episodes=None, asynchronous=False, max_ep_len=None, frame_stack=1,
                 exp_name='exp', logger_path='data', agent_args={}, replay_args={}, extra_args={}):
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.max_ep_len = max_ep_len
        self.num_parallel_env = num_parallel_env
        self.num_test_episodes = num_test_episodes
        self.frame_stack = frame_stack
        self.env_name = env_name
        frame_stack_wrapper = lambda env: FrameStack(env, num_stack=frame_stack)
        self.wrappers = wrappers
        self.seed = seed
        self.asynchronous = asynchronous
        self.setup_env()
        self.setup_seed(seed)
        logger_kwargs = setup_logger_kwargs(exp_name=exp_name, data_dir=logger_path, seed=seed)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        self.setup_agent(**agent_args)
        self.setup_extra(**extra_args)
        self.setup_replay_buffer(**replay_args)
        self.global_step = 0

    def setup_seed(self, seed):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def setup_extra(self, **kwargs):
        pass

    def setup_replay_buffer(self, **kwargs):
        pass

    @abstractmethod
    def setup_agent(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run_one_step(self, global_step):
        raise NotImplementedError

    @abstractmethod
    def on_end_epoch(self):
        raise NotImplementedError

    def setup_env(self):
        self.env = gym.vector.make(self.env_name, wrappers=self.wrappers, num_envs=self.num_parallel_env,
                                   asynchronous=self.asynchronous)
        self.env.seed(self.seed)
        if self.num_test_episodes is not None:
            self.test_env = gym.vector.make(self.env_name, wrappers=self.wrappers, num_envs=self.num_test_episodes,
                                            asynchronous=self.asynchronous)
            self.test_env.seed(self.seed + 10)
        else:
            self.test_env = None

    def run(self):
        for i in range(self.epochs):
            for _ in trange(self.steps_per_epoch, desc=f'Epoch {i + 1}/{self.epochs}'):
                self.global_step += 1
                self.run_one_step(global_step=self.global_step)
            self.on_end_epoch()

    def save_checkpoint(self, path=None):
        pass

    def load_checkpoint(self, path=None):
        pass
