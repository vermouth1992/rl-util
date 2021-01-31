"""
Common in the runner:
1. Setup environment
2. Setup logger
3. Setup agent
4. Run
"""

import random
import sys
import time
from abc import abstractmethod, ABC

import gym
import numpy as np
import rlutils.gym
from rlutils.logx import EpochLogger, setup_logger_kwargs
from rlutils.np import DataSpec
from rlutils.replay_buffers import PyUniformParallelEnvReplayBuffer
from tqdm.auto import trange, tqdm


class BaseRunner(ABC):
    def __init__(self, seed, steps_per_epoch, epochs, exp_name=None, logger_path='data'):
        self.exp_name = exp_name
        self.logger_path = logger_path
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.seed = seed
        self.global_step = 0
        self.max_seed = sys.maxsize
        self.setup_seed(seed)

    def setup_logger(self, config, tensorboard=False):
        assert self.exp_name is not None, 'Call setup_env before setup_logger if exp passed by the contructor is None.'
        logger_kwargs = setup_logger_kwargs(exp_name=self.exp_name, data_dir=self.logger_path, seed=self.seed)
        self.logger = EpochLogger(**logger_kwargs, tensorboard=tensorboard)
        self.logger.save_config(config)

    def get_action_batch_test(self, obs):
        raise NotImplementedError

    def get_action_batch_explore(self, obs):
        raise NotImplementedError

    def test_agent(self):
        o, d, ep_ret, ep_len = self.test_env.reset(), np.zeros(shape=self.num_test_episodes, dtype=np.bool), \
                               np.zeros(shape=self.num_test_episodes), \
                               np.zeros(shape=self.num_test_episodes, dtype=np.int64)
        t = tqdm(total=self.num_test_episodes, desc='Testing')
        while not np.all(d):
            a = self.get_action_batch_test(o)
            o, r, d_, _ = self.test_env.step(a)
            ep_ret = r * (1 - d) + ep_ret
            ep_len = np.ones(shape=self.num_test_episodes, dtype=np.int64) * (1 - d) + ep_len
            prev_d = d.copy()
            d = np.logical_or(d, d_)
            newly_finished = np.sum(d) - np.sum(prev_d)
            if newly_finished > 0:
                t.update(newly_finished)
        t.close()
        self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def setup_seed(self, seed):
        # we set numpy seed first and use it to generate other seeds
        np.random.seed(seed)
        random.seed(self.generate_seed())

    def generate_seed(self):
        return np.random.randint(self.max_seed)

    def setup_extra(self, **kwargs):
        pass

    def setup_replay_buffer(self, **kwargs):
        pass

    @abstractmethod
    def setup_agent(self, **kwargs):
        raise NotImplementedError

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
        """
        Environment is either created by env_name or env_fn. In addition, we apply Rescale action wrappers to
        run Pendulum-v0 using env_name from commandline.
        Other complicated wrappers should be included in env_fn.
        """

        if self.exp_name is None:
            self.exp_name = f'{env_name}_{self.__class__.__name__}_test'
        self.num_parallel_env = num_parallel_env
        self.num_test_episodes = num_test_episodes
        self.asynchronous = asynchronous
        self.env_name = env_name
        if env_fn is not None:
            self.env_fn = env_fn
        else:
            self.env_fn = lambda: gym.make(self.env_name)

        self.dummy_env = self.env_fn()

        if isinstance(self.dummy_env.action_space, gym.spaces.Box):
            high_all = np.all(self.dummy_env.action_space.high == 1)
            low_all = np.all(self.dummy_env.action_space.low == -1)
            if not (high_all and low_all):
                print('Rescale action space to [-1, 1]')
                fn = lambda env: gym.wrappers.RescaleAction(env, a=-1., b=1.)
                self.env_fn = lambda env: fn(self.env_fn(env))

        VecEnv = rlutils.gym.vector.AsyncVectorEnv if asynchronous else rlutils.gym.vector.SyncVectorEnv

        self.env = VecEnv([self.env_fn for _ in range(self.num_parallel_env)])

        self.is_discrete_env = isinstance(self.env.single_action_space, gym.spaces.Discrete)
        self.env.seed(self.generate_seed())
        self.env.action_space.seed(self.generate_seed())
        if self.num_test_episodes is not None:
            self.test_env = VecEnv([self.env_fn for _ in range(self.num_test_episodes)])
            self.test_env.seed(self.generate_seed())
            self.test_env.action_space.seed(self.generate_seed())
        else:
            self.test_env = None

    def run(self):
        self.on_train_begin()
        for i in range(1, self.epochs + 1):
            self.on_epoch_begin(i)
            for t in trange(self.steps_per_epoch, desc=f'Epoch {i}/{self.epochs}'):
                self.run_one_step(t)
                self.global_step += 1
            self.on_epoch_end(i)
        self.on_train_end()

    @property
    def obs_data_spec(self):
        if isinstance(self.env.single_observation_space, gym.spaces.Box):
            obs_data_spec = DataSpec(shape=self.env.single_observation_space.shape,
                                     dtype=np.float32)
        elif isinstance(self.env.single_observation_space, gym.spaces.Discrete):
            obs_data_spec = DataSpec(shape=None,
                                     dtype=np.int32)
        else:
            raise NotImplementedError
        return obs_data_spec

    @property
    def act_data_spec(self):
        if isinstance(self.env.single_action_space, gym.spaces.Box):
            act_data_spec = DataSpec(shape=self.env.single_action_space.shape,
                                     dtype=np.float32,
                                     minval=self.env.single_action_space.low,
                                     maxval=self.env.single_action_space.high)
        elif isinstance(self.env.single_action_space, gym.spaces.Discrete):
            act_data_spec = DataSpec(shape=None, dtype=np.int32, minval=0, maxval=self.env.single_action_space.n)
        else:
            raise NotImplementedError
        return act_data_spec

    def save_checkpoint(self, path=None):
        pass

    def load_checkpoint(self, path=None):
        pass


class OffPolicyRunner(BaseRunner):
    def setup_replay_buffer(self,
                            replay_size,
                            batch_size):

        data_spec = {
            'obs': self.obs_data_spec,
            'act': self.act_data_spec,
            'next_obs': self.obs_data_spec,
            'rew': DataSpec(shape=None, dtype=np.float32),
            'done': DataSpec(shape=None, dtype=np.float32)
        }
        self.replay_buffer = PyUniformParallelEnvReplayBuffer(data_spec=data_spec,
                                                              capacity=replay_size,
                                                              batch_size=batch_size,
                                                              num_parallel_env=self.num_parallel_env)

    def setup_agent(self, agent_cls, **kwargs):
        self.agent = agent_cls(obs_spec=self.obs_data_spec, act_spec=self.act_data_spec, **kwargs)
        self.agent.set_logger(self.logger)

    def setup_extra(self,
                    start_steps,
                    update_after,
                    update_every,
                    update_per_step,
                    policy_delay):
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.update_per_step = update_per_step
        self.policy_delay = policy_delay

    def run_one_step(self, t):
        global_env_steps = self.global_step * self.num_parallel_env
        if global_env_steps >= self.start_steps:
            a = self.get_action_batch_explore(self.o)
            assert not np.any(np.isnan(a)), f'NAN action: {a}'
        else:
            a = self.env.action_space.sample()

        # Step the env
        o2, r, d, info = self.env.step(a)
        self.ep_ret += r
        self.ep_len += 1

        timeouts = np.array([i.get('TimeLimit.truncated', False) for i in info], dtype=np.bool)
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        true_d = np.logical_and(d, np.logical_not(timeouts))

        # Store experience to replay buffer
        self.replay_buffer.add(data={
            'obs': self.o,
            'act': a,
            'rew': r,
            'next_obs': o2,
            'done': true_d
        })

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        self.o = o2

        # End of trajectory handling
        if np.any(d):
            self.logger.store(EpRet=self.ep_ret[d], EpLen=self.ep_len[d])
            self.ep_ret[d] = 0
            self.ep_len[d] = 0
            self.o = self.env.reset_done()

        # Update handling
        if global_env_steps >= self.update_after and global_env_steps % self.update_every == 0:
            for j in range(self.update_every * self.update_per_step):
                batch = self.replay_buffer.sample()
                self.agent.update(**batch, update_target=j % self.policy_delay == 0)

    def on_epoch_end(self, epoch):
        self.test_agent()

        # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TestEpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', self.global_step * self.num_parallel_env)
        self.agent.log_tabular()
        self.logger.log_tabular('Time', time.time() - self.start_time)
        self.logger.dump_tabular()

    def on_train_begin(self):
        self.start_time = time.time()
        self.o = self.env.reset()
        self.ep_ret = np.zeros(shape=self.num_parallel_env)
        self.ep_len = np.zeros(shape=self.num_parallel_env, dtype=np.int64)
