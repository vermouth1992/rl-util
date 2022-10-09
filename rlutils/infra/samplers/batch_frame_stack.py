from collections import deque

import numpy as np
from gym.wrappers import LazyFrames

import rlutils.np as rln
from .batch import BatchSampler


class BatchFrameStackSampler(BatchSampler):
    """
    Stack frame. Using LazyFrames to store pointers instead of raw numpy array to save memory
    """

    def __init__(self, num_stack=4, *args, **kwargs):
        super(BatchFrameStackSampler, self).__init__(*args, **kwargs)
        self.num_stack = num_stack
        self.frames = None

    def reset(self):
        super(BatchFrameStackSampler, self).reset()
        self.num_envs = self.o.shape[0]
        self.frames = [deque(maxlen=self.num_stack) for _ in range(self.num_envs)]
        self.reset_frame(np.array([True for _ in range(self.num_envs)], dtype=bool))

    def reset_frame(self, done):
        # self.o should be (num_env, ...)
        for i in range(self.num_envs):
            if done[i]:
                # if environment is done,
                for _ in range(self.num_stack):
                    self.frames[i].append(self.o[i])

    def append_obs(self, obs):
        for i in range(self.num_envs):
            self.frames[i].append(obs[i])

    def get_obs(self):
        return np.array(self.frames)

    def get_lazy_frames(self):
        return [LazyFrames(list(f)) for f in self.frames]

    def sample(self, num_steps, collect_fn, replay_buffer):
        for _ in range(num_steps):
            # record current frame
            current_frame = self.get_lazy_frames()
            o = self.get_obs()
            a = collect_fn(o)
            assert not np.any(np.isnan(a)), f'NAN action: {a}'
            # Step the env
            o2, r, terminate, truncate, infos = self.env.step(a)
            self.ep_ret += r
            self.ep_len += 1

            d = np.logical_or(terminate, truncate)

            true_d = terminate  # affect value function boostrap

            # obtain next_obs
            if np.any(d):
                next_obs = np.copy(o2)
                terminal_obs = np.asarray(infos['final_observation'][d].tolist())
                next_obs[d] = terminal_obs
            else:
                next_obs = o2

            self.append_obs(next_obs)
            next_frame = self.get_lazy_frames()

            self.oa_queue.append((current_frame, a))
            self.rew_queue.append(r)

            valid = self.ep_len >= self.n_steps

            if np.any(valid):
                last_o, last_a = self.oa_queue.popleft()
                last_r = np.sum(np.stack(self.rew_queue, axis=-1) * self.gamma_vector, axis=-1)  # (num_envs,)
                self.rew_queue.popleft()

                valid_o = []
                valid_next_o = []
                for i in range(self.num_envs):
                    if valid[i]:
                        valid_o.append(last_o[i])
                        valid_next_o.append(next_frame[i])

                # Store experience to replay buffer
                replay_buffer.add(dict(
                    obs=valid_o,
                    act=last_a[valid],
                    rew=last_r[valid],
                    next_obs=valid_next_o,
                    done=true_d[valid]
                ))

            # End of trajectory handling
            if np.any(d):
                self.logger.store(EpRet=self.ep_ret[d], EpLen=self.ep_len[d])
                self.ep_ret[d] = 0
                self.ep_len[d] = 0
                # only reset frame of the environments with done
                self.o = o2
                self.reset_frame(d)

            self._global_env_step += self.env.num_envs
