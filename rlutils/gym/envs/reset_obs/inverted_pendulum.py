import gym.envs.mujoco.inverted_pendulum as inverted_pendulum


class InvertedPendulumEnv(inverted_pendulum.InvertedPendulumEnv):
    def reset_obs(self, obs):
        state = obs
        qpos = state[:self.model.nq]
        qvel = state[self.model.nq:]
        self.set_state(qpos, qvel)
        return self._get_obs()
