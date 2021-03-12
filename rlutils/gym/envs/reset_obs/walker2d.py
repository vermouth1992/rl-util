import gym.envs.mujoco.walker2d as walker2d
import numpy as np


class Walker2dEnv(walker2d.Walker2dEnv):
    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], qvel]).ravel()

    def reset_obs(self, obs):
        state = np.insert(obs, 0, 0.)
        qpos = state[:self.model.nq]
        qvel = state[self.model.nq:]
        self.set_state(qpos, qvel)
        return self._get_obs()
