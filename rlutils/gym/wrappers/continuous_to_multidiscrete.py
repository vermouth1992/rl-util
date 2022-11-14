import gym
import numpy as np


class ContinuousToMultiDiscrete(gym.ActionWrapper):
    def __init__(self, env, bins_per_dim):
        super(ContinuousToMultiDiscrete, self).__init__(env=env)
        assert bins_per_dim > 1
        self.bins_per_dim = bins_per_dim
        assert isinstance(self.env.action_space, gym.spaces.Box)
        assert len(self.env.action_space.shape) == 1
        n_dim = self.env.action_space.shape[0]
        self.action_space = gym.spaces.MultiDiscrete(nvec=[bins_per_dim for _ in range(n_dim)])

    def action(self, action):
        """Convert the discrete action to continuous action

        Args:
            action:

        Returns:

        """
        high = self.env.action_space.high
        low = self.env.action_space.low
        action = action * (high - low) / (self.bins_per_dim - 1) + low
        return action
