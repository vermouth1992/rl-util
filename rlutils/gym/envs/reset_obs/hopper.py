import gym.envs.mujoco.hopper as hopper
import numpy as np


class HopperEnv(hopper.HopperEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_obs(self, obs):
        state = np.insert(obs, 0, 0.)
        qpos = state[:self.model.nq]
        qvel = state[self.model.nq:]
        self.set_state(qpos, qvel)
        return self._get_obs()
