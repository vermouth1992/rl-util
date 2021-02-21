from gym.vector.vector_env import VectorEnv as GymVectorEnv


class VectorEnv(GymVectorEnv):
    def step_async(self, actions, mask=None):
        pass

    def step(self, actions, mask=None):
        self.step_async(actions, mask)
        return self.step_wait()

    def reset_done_wait(self):
        raise NotImplementedError

    def reset_done_async(self):
        raise NotImplementedError

    def reset_done(self):
        raise NotImplementedError
