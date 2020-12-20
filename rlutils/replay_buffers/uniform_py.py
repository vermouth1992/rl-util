import numpy as np

from .base import BaseReplayBuffer


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class PyUniformParallelEnvReplayBuffer(BaseReplayBuffer):
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self,
                 obs_dim,
                 act_dim,
                 act_dtype,
                 capacity,
                 batch_size,
                 num_parallel_env):
        assert capacity % num_parallel_env == 0
        self.num_parallel_env = num_parallel_env
        self.max_size = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_buf = np.zeros(combined_shape(self.capacity, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(self.capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(self.capacity, act_dim), dtype=act_dtype)
        self.rew_buf = np.zeros(shape=(self.capacity), dtype=np.float32)
        self.done_buf = np.zeros(shape=(self.capacity), dtype=np.float32)
        self.batch_size = batch_size
        self.ptr, self.size = 0, 0

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        """ Make it compatible with Pytorch data loaders """
        return dict(obs=self.obs_buf[item],
                    next_obs=self.obs2_buf[item],
                    act=self.act_buf[item],
                    rew=self.rew_buf[item],
                    done=self.done_buf[item])

    @property
    def capacity(self):
        return self.max_size

    def add(self, data, priority=None):
        assert priority is None, 'Uniform Replay Buffer'
        obs = data['obs']
        act = data['act']
        rew = data['rew']
        next_obs = data['next_obs']
        done = data['done']
        self.obs_buf[self.ptr:self.ptr + self.num_parallel_env] = obs
        self.obs2_buf[self.ptr:self.ptr + self.num_parallel_env] = next_obs
        self.act_buf[self.ptr:self.ptr + self.num_parallel_env] = act
        self.rew_buf[self.ptr:self.ptr + self.num_parallel_env] = rew
        self.done_buf[self.ptr:self.ptr + self.num_parallel_env] = done
        self.ptr = (self.ptr + self.num_parallel_env) % self.capacity
        self.size = min(self.size + self.num_parallel_env, self.capacity)

    def sample(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     next_obs=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return batch


class PyUniformParallelEnvReplayBufferFrame(BaseReplayBuffer):
    def __init__(self,
                 num_parallel_env,
                 obs_spec,
                 act_spec,
                 replay_capacity,
                 batch_size,
                 gamma=0.99,
                 update_horizon=1,
                 frame_stack=1):
        """This is a memory efficient implementation of the replay buffer.
        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.
        For the typical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes
        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        assert update_horizon == 1, f'only support update_horizon=1, Got {update_horizon}'

        self.obs = np.empty([self.size, num_parallel_env] + list(obs_spec.shape), dtype=np.uint8)
        self.action = np.empty([self.size, num_parallel_env] + list(act_spec.shape), dtype=np.int32)
        self.reward = np.empty([self.size, num_parallel_env], dtype=np.float32)
        self.done = np.empty([self.size, num_parallel_env], dtype=np.bool)

        self.size = replay_capacity
        self.frame_history_len = frame_stack
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_horizon = update_horizon

        self.next_idx = 0
        self.num_in_buffer = 0

        self.obs = None
        self.action = None
        self.reward = None
        self.done = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        all_obs_batch = np.stack([self._encode_observation(idx + 1, self.frame_history_len + 1) for idx in idxes], 0)
        obs_batch = all_obs_batch[:, :, :, 0:self.frame_history_len]
        act_batch = self.action[idxes]
        rew_batch = self.reward[idxes]
        next_obs_batch = all_obs_batch[:, :, :, 1:self.frame_history_len + 1]
        done_mask = self.done[idxes].astype(np.float32)

        return dict(
            obs=obs_batch,
            act=act_batch,
            obs2=next_obs_batch,
            done=done_mask,
            rew=rew_batch,
        )

    def sample(self):
        """Sample `batch_size` different transitions.
        i-th sample transition is the following:
        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(self.batch_size)
        idxes = np.random.choice(self.num_in_buffer - 1, size=self.batch_size, replace=True)
        return self._encode_sample(idxes)

    def _encode_observation(self, idx, num_frames):
        end_idx = idx + 1  # make noninclusive
        start_idx = end_idx - num_frames
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = num_frames - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 2)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            return self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)

    def add(self, data, priority=1.0):
        obs = data['obs']
        act = data['act']
        rew = data['rew']
        done = data['done']
        self.obs[self.next_idx] = obs
        self.action[self.next_idx] = act
        self.reward[self.next_idx] = rew
        self.done[self.next_idx] = done
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
