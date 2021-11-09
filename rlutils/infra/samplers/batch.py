from .base import Sampler
import numpy as np
import rlutils.np as rln


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
