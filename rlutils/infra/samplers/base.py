from abc import ABC, abstractmethod

import numpy as np
import rlutils.np as rln
from rlutils.gym.vector import VectorEnv
from tqdm.auto import trange


class Sampler(ABC):
    def __init__(self, env: VectorEnv):
        self.env = env

    def reset(self):
        pass

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        pass

    @abstractmethod
    def sample(self, num_steps, collect_fn, replay_buffer):
        pass

    @property
    @abstractmethod
    def total_env_steps(self):
        pass


class TrajectorySampler(Sampler):
    def reset(self):
        self._global_env_step = 0

    def log_tabular(self):
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('VVals', with_min_and_max=True)
        self.logger.log_tabular('TotalEnvInteracts', self.total_env_steps)

    @property
    def total_env_steps(self):
        return self._global_env_step

    def sample(self, num_steps, collect_fn, replay_buffer):
        """ Only collect dataset. No computation """
        self.obs = self.env.reset()
        self.ep_ret = np.zeros(shape=self.env.num_envs, dtype=np.float32)
        self.ep_len = np.zeros(shape=self.env.num_envs, dtype=np.int32)
        actor_fn, value_fn = collect_fn
        for t in trange(num_steps, desc='Sampling'):
            act, logp, val = actor_fn(self.obs)
            if isinstance(act, np.float32):
                act_taken = np.clip(act, -1., 1.)
            else:
                act_taken = act
            obs2, rew, dones, infos = self.env.step(act_taken)
            replay_buffer.store(self.obs, act, rew, val, logp)
            self.logger.store(VVals=val)
            self.ep_ret += rew
            self.ep_len += 1

            # There are four cases there:
            # 1. if done is False. Bootstrap (truncated due to trajectory length)
            # 2. if done is True, if TimeLimit.truncated not in info. Don't bootstrap (didn't truncate)
            # 3. if done is True, if TimeLimit.truncated in info, if it is True, Bootstrap (true truncated)
            # 4. if done is True, if TimeLimit.truncated in info, if it is False. Don't bootstrap (same time)

            if t == num_steps - 1:
                time_truncated_dones = np.array([info.get('TimeLimit.truncated', False) for info in infos],
                                                dtype=np.bool_)
                # need to finish path for all the environments
                last_vals = value_fn(obs2)
                last_vals = last_vals * np.logical_or(np.logical_not(dones), time_truncated_dones)
                replay_buffer.finish_path(dones=np.ones(shape=self.env.num_envs, dtype=np.bool_),
                                          last_vals=last_vals)
                self.logger.store(EpRet=self.ep_ret[dones], EpLen=self.ep_len[dones])
                self.obs = None
            elif np.any(dones):
                time_truncated_dones = np.array([info.get('TimeLimit.truncated', False) for info in infos],
                                                dtype=np.bool_)
                last_vals = value_fn(obs2) * time_truncated_dones
                replay_buffer.finish_path(dones=dones,
                                          last_vals=last_vals)
                self.logger.store(EpRet=self.ep_ret[dones], EpLen=self.ep_len[dones])
                self.ep_ret[dones] = 0.
                self.ep_len[dones] = 0
                self.obs = self.env.reset_done()

            else:
                self.obs = obs2
        self._global_env_step += num_steps * self.env.num_envs


class BatchSampler(Sampler):
    @property
    def total_env_steps(self):
        return self._global_env_step

    def reset(self):
        self._global_env_step = 0
        self.o = self.env.reset()
        self.ep_ret = np.zeros(shape=self.env.num_envs)
        self.ep_len = np.zeros(shape=self.env.num_envs, dtype=np.int64)

    def log_tabular(self):
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', self._global_env_step)

    def sample(self, num_steps, collect_fn, replay_buffer):
        for _ in range(num_steps):
            a = collect_fn(self.o)
            assert not np.any(np.isnan(a)), f'NAN action: {a}'
            # Step the env
            o2, r, d, infos = self.env.step(a)
            self.ep_ret += r
            self.ep_len += 1

            timeouts = rln.gather_dict_key(infos=infos, key='TimeLimit.truncated', default=False, dtype=np.bool)
            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            true_d = np.logical_and(d, np.logical_not(timeouts))

            # Store experience to replay buffer
            replay_buffer.add(dict(
                obs=self.o,
                act=a,
                rew=r,
                next_obs=o2,
                done=true_d
            ))

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            self.o = o2

            # End of trajectory handling
            if np.any(d):
                self.logger.store(EpRet=self.ep_ret[d], EpLen=self.ep_len[d])
                self.ep_ret[d] = 0
                self.ep_len[d] = 0
                self.o = self.env.reset_done()

            self._global_env_step += self.env.num_envs
