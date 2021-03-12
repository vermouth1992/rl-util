import gym.envs.classic_control.pendulum as pendulum
import numpy as np


class PendulumEnv(pendulum.PendulumEnv):
    def reset_obs(self, obs):
        cos_theta, sin_theta, thetadot = obs
        theta = np.arctan2(sin_theta, cos_theta)
        self.state = np.array([theta, thetadot], dtype=self.observation_space.dtype)
        self.last_u = None
        return self._get_obs()
