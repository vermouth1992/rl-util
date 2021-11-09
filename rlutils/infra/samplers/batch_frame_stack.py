from gym.wrappers import LazyFrames
from collections import deque
from .batch import BatchSampler
import numpy as np
import rlutils.np as rln


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
            o2, r, d, infos = self.env.step(a)
            self.ep_ret += r
            self.ep_len += 1

            timeouts = rln.gather_dict_key(infos=infos, key='TimeLimit.truncated', default=False, dtype=np.bool)
            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            true_d = np.logical_and(d, np.logical_not(timeouts))

            self.append_obs(o2)
            next_frame = self.get_lazy_frames()

            # Store experience to replay buffer
            replay_buffer.add(dict(
                obs=current_frame,
                act=a,
                rew=r,
                next_obs=next_frame,
                done=true_d
            ))

            # End of trajectory handling
            if np.any(d):
                self.logger.store(EpRet=self.ep_ret[d], EpLen=self.ep_len[d])
                self.ep_ret[d] = 0
                self.ep_len[d] = 0
                self.o = self.env.reset_done()
                # only reset frame of the environments with done
                self.reset_frame(d)

            self._global_env_step += self.env.num_envs
