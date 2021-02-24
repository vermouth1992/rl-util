from gym import spaces
from gym.wrappers import TransformObservation


class TransformObservationDtype(TransformObservation):
    def __init__(self, env, dtype):
        super(TransformObservationDtype, self).__init__(env, f=lambda x: x.astype(dtype))
        self.observation_space = spaces.Box(low=self.env.observation_space.low,
                                            high=self.env.observation_space.high,
                                            shape=self.env.observation_space.shape,
                                            dtype=dtype)
