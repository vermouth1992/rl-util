import gym
import numpy as np


class RandomAction(gym.ActionWrapper):
    def __init__(self, env, prob=None):
        super(RandomAction, self).__init__(env=env)
        self.prob = prob
        assert prob is not None and isinstance(prob, float)
        assert prob >= 0 and prob <= 1.

    def action(self, action):
        if np.random.rand() < self.prob:
            return self.action_space.sample()
        else:
            return action
