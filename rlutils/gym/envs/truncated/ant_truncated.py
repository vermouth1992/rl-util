import numpy as np

import gym.envs.mujoco.ant_v4 as ant


class AntEnv(ant.AntEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])
