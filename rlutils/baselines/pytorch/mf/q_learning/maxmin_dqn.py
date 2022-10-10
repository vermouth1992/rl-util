import copy
import itertools

import numpy as np
import rlutils.infra as rl_infra
import rlutils.np as rln
import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
import torch.nn as nn
import torch.optim
from rlutils.interface.agent import OffPolicyAgent


def gather_q_values(q_values, action):
    return q_values.gather(1, action.unsqueeze(1)).squeeze(1)


class MaxMinDQN(OffPolicyAgent, nn.Module):
    def __init__(self,
                 env,
                 mlp_hidden=128,
                 double_q=True,
                 q_lr=1e-3,
                 gamma=0.99,
                 huber_delta=None,
                 grad_norm=None,
                 epsilon_greedy_steps=1000,
                 target_update_freq=100,
                 device=ptu.device,
                 ):
        OffPolicyAgent.__init__(self, env=env)
        nn.Module.__init__(self)
        self.grad_norm = grad_norm
        self.target_update_freq = target_update_freq
        self.gamma = gamma
        self.double_q = double_q
        self.q_lr = q_lr
        self.obs_spec = env.observation_space
        self.act_dim = env.action_space.n
        self.mlp_hidden = mlp_hidden
        self.epsilon_greedy_steps = epsilon_greedy_steps
        self.q_network_1 = self._create_q_network()
        self.q_network_2 = self._create_q_network()
        self.target_q_network_1 = copy.deepcopy(self.q_network_1)
        self.target_q_network_2 = copy.deepcopy(self.q_network_2)
        rlu.nn.functional.freeze(self.target_q_network_1)
        rlu.nn.functional.freeze(self.target_q_network_2)
        self.reset_optimizer()
        # define loss function
        reduction = 'none'
        if huber_delta is None:
            self.loss_fn = torch.nn.MSELoss(reduction=reduction)
        else:
            self.loss_fn = torch.nn.HuberLoss(delta=huber_delta, reduction=reduction)
        self.epsilon_greedy_scheduler = self._create_epsilon_greedy_scheduler()
        self.device = device

        self.to(self.device)

    def reset_optimizer(self):
        params = itertools.chain(self.q_network_1.parameters(), self.q_network_2.parameters())
        self.q_optimizer = torch.optim.Adam(params, lr=self.q_lr)

    def _create_q_network(self):
        if len(self.obs_spec.shape) == 1:  # 1D observation
            q_network = rlu.nn.build_mlp(input_dim=self.obs_spec.shape[0], output_dim=self.act_dim,
                                         mlp_hidden=self.mlp_hidden, num_layers=3)
        else:
            raise NotImplementedError
        return q_network

    def _create_epsilon_greedy_scheduler(self):
        return rln.schedulers.LinearSchedule(schedule_timesteps=self.epsilon_greedy_steps,
                                             final_p=0.1,
                                             initial_p=0.1)

    def log_tabular(self):
        super(MaxMinDQN, self).log_tabular()
        self.logger.log_tabular('Q1Vals', with_min_and_max=True)
        self.logger.log_tabular('Q2Vals', with_min_and_max=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('TDError', average_only=True)

    def update_target(self):
        rlu.functional.hard_update(self.target_q_network_1, self.q_network_1)
        rlu.functional.hard_update(self.target_q_network_2, self.q_network_2)

    def compute_priority(self, data):
        np_data = {}
        for key, d in data.items():
            if not isinstance(d, np.ndarray):
                d = np.array(d)
            np_data[key] = torch.as_tensor(d).to(self.device, non_blocking=True)
        return self.compute_priority_torch(**np_data).cpu().numpy()

    def compute_priority_torch(self, obs, act, next_obs, rew, done):
        with torch.no_grad():
            target_q_values = self.compute_target_values(next_obs, rew, done)
            q_values_1 = self.q_network_1(obs)
            q_values_2 = self.q_network_2(obs)
            q_values = torch.minimum(q_values_1, q_values_2)
            q_values = gather_q_values(q_values, act)
            abs_td_error = torch.abs(q_values - target_q_values)
            return abs_td_error

    def compute_target_values(self, next_obs, rew, done):
        with torch.no_grad():
            target_q_values_1 = self.target_q_network_1(next_obs)  # (None, act_dim)
            target_q_values_2 = self.target_q_network_2(next_obs)
            target_q_values = torch.minimum(target_q_values_1, target_q_values_2)
            target_q_values = torch.max(target_q_values, dim=-1)[0]
            target_q_values = rew + self.gamma * (1. - done) * target_q_values
            return target_q_values

    def _update_nets(self, obs, act, next_obs, rew, done, weights=None):
        target_q_values = self.compute_target_values(next_obs, rew, done)
        self.q_optimizer.zero_grad()
        q_values_1 = self.q_network_1(obs)
        q_values_2 = self.q_network_2(obs)
        q_values_1 = gather_q_values(q_values_1, act)
        q_values_2 = gather_q_values(q_values_2, act)
        loss = self.loss_fn(q_values_1, target_q_values) + self.loss_fn(q_values_2, target_q_values)
        if weights is not None:
            loss = loss * weights
        loss = torch.mean(loss, dim=0)
        loss.backward()
        if self.grad_norm is not None:
            params = itertools.chain(self.q_network_1.parameters(), self.q_network_2.parameters())
            torch.nn.utils.clip_grad_norm(params, max_norm=self.grad_norm)
        self.q_optimizer.step()
        info = dict(
            Q1Vals=q_values_1,
            Q2Vals=q_values_2,
            LossQ=loss
        )
        with torch.no_grad():
            abs_td_error = torch.abs(torch.minimum(q_values_1, q_values_2) - target_q_values).detach()
        info['TDError'] = abs_td_error
        return info

    def train_on_batch(self, data, **kwargs):
        tensor_data = ptu.convert_dict_to_tensor(data, device=self.device)

        info = self._update_nets(**tensor_data)

        self.policy_updates += 1
        if self.policy_updates % self.target_update_freq == 0:
            self.update_target()

        if self.logger is not None:
            self.logger.store(**info)

        return info

    def act_batch_explore(self, obs, global_steps):
        num_envs = obs.shape[0]
        actions = np.zeros(shape=(num_envs,), dtype=np.int64)
        epsilon = self.epsilon_greedy_scheduler.value(global_steps)
        for i in range(num_envs):
            if np.random.rand() < epsilon:
                actions[i] = np.random.randint(low=0, high=self.act_dim)
            else:
                actions[i:i + 1] = self.act_batch_test(obs[i:i + 1])
        return actions

    def act_batch_test(self, obs):
        obs = torch.as_tensor(obs, device=self.device)
        with torch.no_grad():
            q_values_1 = self.q_network_1(obs)
            q_values_2 = self.q_network_2(obs)
            q_values = torch.minimum(q_values_1, q_values_2)
            return torch.argmax(q_values, dim=-1).cpu().numpy()


class Runner(rl_infra.runner.PytorchOffPolicyRunner):
    @classmethod
    def main(cls,
             env_name: str,
             # agent args
             q_lr=3e-4,
             gamma=0.99,
             target_update_freq=500,
             **kwargs
             ):
        agent_kwargs = dict(
            q_lr=q_lr,
            gamma=gamma,
            target_update_freq=target_update_freq
        )
        super(Runner, cls).main(env_name=env_name,
                                agent_cls=MaxMinDQN,
                                agent_kwargs=agent_kwargs,
                                )


if __name__ == '__main__':
    rl_infra.runner.run_func_as_main(Runner.main)
