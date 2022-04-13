from gym import spaces
from gym.wrappers import TransformObservation
import gym


class TransformObservationDtype(TransformObservation):
    def __init__(self, env, dtype):
        super(TransformObservationDtype, self).__init__(env, f=lambda x: x.astype(dtype))
        self.observation_space = spaces.Box(low=self.env.observation_space.low,
                                            high=self.env.observation_space.high,
                                            shape=self.env.observation_space.shape,
                                            dtype=dtype)


class TransformActionDtype(gym.ActionWrapper):
    def __init__(self, env, dtype):
        super(TransformActionDtype, self).__init__(env)
        self.original_dtype = env.action_space.dtype
        self.action_space = spaces.Box(low=self.env.action_space.low,
                                       high=self.env.action_space.high,
                                       shape=self.env.action_space.shape,
                                       dtype=dtype)

    def action(self, action):
        return action.astype(self.original_dtype)
