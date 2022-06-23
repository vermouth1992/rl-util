import collections

import numpy as np

from .base import Sampler


class BatchSampler(Sampler):
    def __init__(self, n_steps, gamma, **kwargs):
        super(BatchSampler, self).__init__(**kwargs)
        self.n_steps = n_steps
        self.gamma_vector = gamma ** np.arange(self.n_steps)
        self.gamma_vector = np.expand_dims(self.gamma_vector, axis=0)  # (1, n_steps)
        self.oa_queue = collections.deque(maxlen=n_steps)
        self.rew_queue = collections.deque(maxlen=n_steps)

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

            timeouts = np.zeros(shape=self.env.num_envs, dtype=np.bool_)
            keyword = 'TimeLimit.truncated'
            if keyword in infos:
                mask = infos['_' + keyword]
                timeouts[mask] = infos[keyword][mask]

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            true_d = np.logical_and(d, np.logical_not(timeouts))

            # according to the latest version of gym.vector, if done due to terminal state or timeout, terminal
            if np.any(d):
                next_obs = np.copy(o2)
                terminal_obs = np.asarray(infos['terminal_observation'][d].tolist())
                next_obs[d] = terminal_obs
            else:
                next_obs = o2

            self.oa_queue.append((self.o, a))
            self.rew_queue.append(r)

            valid = self.ep_len >= self.n_steps

            if np.any(valid):
                last_o, last_a = self.oa_queue.popleft()
                last_r = np.sum(np.stack(self.rew_queue, axis=-1) * self.gamma_vector, axis=-1)  # (num_envs,)
                self.rew_queue.popleft()

                # Store experience to replay buffer
                replay_buffer.add(dict(
                    obs=last_o[valid],
                    act=last_a[valid],
                    rew=last_r[valid],
                    next_obs=next_obs[valid],
                    done=true_d[valid]
                ))

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            self.o = o2

            # End of trajectory handling
            if np.any(d):
                if self.logger is not None:
                    self.logger.store(EpRet=self.ep_ret[d], EpLen=self.ep_len[d])
                self.ep_ret[d] = 0
                self.ep_len[d] = 0

            self._global_env_step += self.env.num_envs
