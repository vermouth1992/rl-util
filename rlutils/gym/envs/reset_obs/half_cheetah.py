import gym.envs.mujoco.half_cheetah as half_cheetah
import numpy as np


class HalfCheetahEnv(half_cheetah.HalfCheetahEnv):
    def reset_obs(self, obs):
        state = np.insert(obs, 0, 0.)
        qpos = state[:self.model.nq]
        qvel = state[self.model.nq:]
        self.set_state(qpos, qvel)
        return self._get_obs()
