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
from gym.wrappers import FrameStack
from tqdm.auto import trange

from rlutils.logx import EpochLogger
from .run_utils import setup_logger_kwargs


def _add_frame_stack(wrappers, frame_stack):
    if wrappers is None:
        if frame_stack is not None:
            frame_stack_wrapper = lambda env: FrameStack(env, num_stack=frame_stack)
            return frame_stack_wrapper
        else:
            return None
    else:
        if not isinstance(wrappers, list):
            wrappers = [wrappers]
        if frame_stack is not None:
            frame_stack_wrapper = lambda env: FrameStack(env, num_stack=frame_stack)
            return wrappers.append(frame_stack_wrapper)
        else:
            return wrappers


class BaseRunner(ABC):
    def __init__(self, seed, steps_per_epoch, epochs, exp_name='exp', logger_path='data'):
        self.exp_name = exp_name
        self.logger_path = logger_path
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.seed = seed
        self.global_step = 0

    def setup_logger(self, config):
        logger_kwargs = setup_logger_kwargs(exp_name=self.exp_name, data_dir=self.logger_path, seed=self.seed)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(config)

    def setup_seed(self, seed):
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
    def run_one_step(self):
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
                  num_parallel_env,
                  frame_stack=None,
                  wrappers=None,
                  asynchronous=False,
                  num_test_episodes=None):
        self.num_parallel_env = num_parallel_env
        self.num_test_episodes = num_test_episodes
        self.asynchronous = asynchronous
        self.frame_stack = frame_stack
        self.env_name = env_name
        self.wrappers = _add_frame_stack(wrappers, frame_stack)

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
        self.on_train_begin()
        for i in range(1, self.epochs + 1):
            self.on_epoch_begin(i)
            for _ in trange(self.steps_per_epoch, desc=f'Epoch {i}/{self.epochs}'):
                self.run_one_step()
                self.global_step += 1
            self.on_epoch_end(i)
        self.on_train_end()

    def save_checkpoint(self, path=None):
        pass

    def load_checkpoint(self, path=None):
        pass