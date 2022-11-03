import numpy as np
from gym.envs.mujoco import hopper_v4

from ..base import ModelBasedEnv


class HopperEnv(ModelBasedEnv, hopper_v4.HopperEnv):
    """
    A subclass of Hopper Environment that provides method to compute reward and terminal signal based on the states
    """

    def __init__(self, **kwargs):
        hopper_v4.HopperEnv.__init__(self, exclude_current_positions_from_observation=False, **kwargs)
        assert not self._exclude_current_positions_from_observation

    def healthy_reward_np_batch(self, obs):
        healthy_boolean = np.logical_or(self.is_healthy_np_batch(obs=obs), self._terminate_when_unhealthy)
        return healthy_boolean.astype(np.float) * self._healthy_reward

    def healthy_reward_torch_batch(self, obs):
        import torch
        terminate_when_unhealthy = torch.as_tensor(data=[self._terminate_when_unhealthy], device=obs.device)
        healthy_boolean = torch.logical_or(self.is_healthy_torch_batch(obs=obs), terminate_when_unhealthy)
        return healthy_boolean.float() * self._healthy_reward

    def state_vector_np_batch(self, qpos, qvel):
        """Return the position and velocity joint states of the model"""
        return np.concatenate([qpos, qvel], axis=-1)

    def state_vector_torch_batch(self, qpos, qvel):
        """Return the position and velocity joint states of the model"""
        import torch
        return torch.cat([qpos, qvel], dim=-1)

    def is_healthy_np_batch(self, obs):
        qpos = obs[:, :self.model.nq]
        qvel = obs[:, self.model.nq:]

        z = qpos[:, 1]
        angle = qpos[:, 2]
        state = self.state_vector_np_batch(qpos, qvel)[:, 2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state), axis=-1)
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = np.all(np.stack([healthy_state, healthy_z, healthy_angle], axis=-1), axis=-1)

        return is_healthy

    def is_healthy_torch_batch(self, obs):
        import torch
        qpos = obs[:, :self.model.nq]
        qvel = obs[:, self.model.nq:]

        z = qpos[:, 1]
        angle = qpos[:, 2]
        state = self.state_vector_torch_batch(qpos, qvel)[:, 2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = torch.all(torch.logical_and(min_state < state, state < max_state), dim=-1)
        # healthy_z = min_z < z < max_z
        healthy_z = torch.logical_and(min_z < z, z < max_z)
        # healthy_angle = min_angle < angle < max_angle
        healthy_angle = torch.logical_and(min_angle < angle, angle < max_angle)

        is_healthy = torch.all(torch.stack([healthy_state, healthy_z, healthy_angle], dim=-1), dim=-1)

        return is_healthy

    def terminate_fn_numpy_batch(self, obs, action, next_obs):
        if not self._terminate_when_unhealthy:
            return np.zeros(shape=(next_obs.shape[0]), dtype=np.bool)
        else:
            return np.logical_not(self.is_healthy_np_batch(next_obs))

    def terminate_fn_torch_batch(self, obs, action, next_obs):
        import torch
        if not self._terminate_when_unhealthy:
            return torch.zeros(size=(next_obs.shape[0]), dtype=torch.bool)
        else:
            return torch.logical_not(self.is_healthy_torch_batch(next_obs))

    def control_cost_np_batch(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action), axis=-1)
        return control_cost

    def control_cost_torch_batch(self, action):
        import torch
        control_cost = self._ctrl_cost_weight * torch.sum(torch.square(action), dim=-1)
        return control_cost

    def reward_fn_numpy_batch(self, obs, action, next_obs):
        ctrl_cost = self.control_cost_np_batch(action)
        qpos_prev = obs[:, :self.model.nq]
        qpos = next_obs[:, :self.model.nq]

        x_position_before = qpos_prev[:, 0]
        x_position_after = qpos[:, 0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward_np_batch(next_obs)
        rewards = forward_reward + healthy_reward
        costs = ctrl_cost
        reward = rewards - costs

        return reward

    def reward_fn_torch_batch(self, obs, action, next_obs):
        ctrl_cost = self.control_cost_torch_batch(action)
        qpos_prev = obs[:, :self.model.nq]
        qpos = next_obs[:, :self.model.nq]

        x_position_before = qpos_prev[:, 0]
        x_position_after = qpos[:, 0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward_torch_batch(next_obs)
        rewards = forward_reward + healthy_reward
        costs = ctrl_cost
        reward = rewards - costs
        return reward
