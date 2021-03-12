from gym.vector.vector_env import VectorEnv as GymVectorEnv


class VectorEnv(GymVectorEnv):
    def step_async(self, actions, mask=None):
        pass

    def step(self, actions, mask=None):
        self.step_async(actions, mask)
        return self.step_wait()

    def reset_obs(self, obs, mask=None):
        self.reset_obs_async(obs, mask)
        return self.reset_obs_wait()

    def reset_obs_async(self, obs, mask=None):
        pass

    def reset_obs_wait(self, **kwargs):
        raise NotImplementedError

    def reset_done(self):
        self.reset_done_async()
        return self.reset_done_wait()

    def reset_done_wait(self):
        raise NotImplementedError

    def reset_done_async(self):
        pass
