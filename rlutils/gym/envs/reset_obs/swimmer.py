import gym.envs.mujoco.swimmer as swimmer

import numpy as np


class SwimmerEnv(swimmer.SwimmerEnv):
    def reset_obs(self, obs):
        state = np.insert(obs, [0, 0], 0.)
        qpos = state[:self.model.nq]
        qvel = state[self.model.nq:]
        self.set_state(qpos, qvel)
        return self._get_obs()
