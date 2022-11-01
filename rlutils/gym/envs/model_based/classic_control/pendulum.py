from gym.envs.classic_control import pendulum

from ..base import ModelBasedEnv


class PendulumEnv(ModelBasedEnv, pendulum.PendulumEnv):
    def terminate_fn_numpy_batch(self, obs, action, next_obs):
        import numpy as np
        return np.zeros(shape=(obs.shape[0]), dtype=np.bool)

    def terminate_fn_torch_batch(self, obs, action, next_obs):
        import torch
        return torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)

    def reward_fn_numpy_batch(self, obs, action, next_obs):
        import numpy as np
        cos_th, sin_th, thdot = obs[:, 0], obs[:, 1], obs[:, 2]
        th = np.arctan2(sin_th, cos_th)

        costs = th ** 2 + .1 * thdot ** 2 + .001 * (action[:, 0] ** 2)
        return -costs

    def reward_fn_torch_batch(self, obs, action, next_obs):
        import torch
        cos_th, sin_th, thdot = obs[:, 0], obs[:, 1], obs[:, 2]
        th = torch.atan2(sin_th, cos_th)
        costs = th ** 2 + .1 * thdot ** 2 + .001 * (action[:, 0] ** 2)
        return -costs
