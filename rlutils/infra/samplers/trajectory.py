from .base import Sampler
import numpy as np
from tqdm.auto import trange


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
            if act.dtype == np.float32 or act.dtype == np.float64:
                act_taken = np.clip(act, self.env.action_space.low, self.env.action_space.high)
            else:
                act_taken = act
            obs2, rew, dones, infos = self.env.step(act_taken)
            replay_buffer.store(self.obs, act, rew, val, logp)
            self.logger.store(VVals=val)
            self.ep_ret += rew
            self.ep_len += 1

            # TODO: retrieve next observation

            # according to the latest version of gym.vector, if done due to terminal state or timeout, terminal
            if np.any(dones):
                next_obs = np.copy(obs2)
                terminal_obs = np.asarray(infos['terminal_observation'][dones].tolist())
                next_obs[dones] = terminal_obs
            else:
                next_obs = obs2

            # TODO: retrieve truncated signal
            timeouts = np.zeros(shape=self.env.num_envs, dtype=np.bool_)
            keyword = 'TimeLimit.truncated'
            if keyword in infos:
                mask = infos['_' + keyword]
                timeouts[mask] = infos[keyword][mask]

            # There are four cases there:
            # 1. if done is False. Bootstrap (truncated due to trajectory length)
            # 2. if done is True, if TimeLimit.truncated not in info. Don't bootstrap (didn't truncate)
            # 3. if done is True, if TimeLimit.truncated in info, if it is True, Bootstrap (true truncated)
            # 4. if done is True, if TimeLimit.truncated in info, if it is False. Don't bootstrap (same time)

            if t == num_steps - 1:
                # need to finish path for all the environments
                last_vals = value_fn(next_obs)
                last_vals = last_vals * np.logical_or(np.logical_not(dones), timeouts)
                replay_buffer.finish_path(dones=np.ones(shape=self.env.num_envs, dtype=np.bool_),
                                          last_vals=last_vals)
                self.logger.store(EpRet=self.ep_ret[dones], EpLen=self.ep_len[dones])
            elif np.any(dones):
                last_vals = value_fn(next_obs) * timeouts
                replay_buffer.finish_path(dones=dones,
                                          last_vals=last_vals)
                self.logger.store(EpRet=self.ep_ret[dones], EpLen=self.ep_len[dones])
                self.ep_ret[dones] = 0.
                self.ep_len[dones] = 0

            self.obs = obs2

        self._global_env_step += num_steps * self.env.num_envs
