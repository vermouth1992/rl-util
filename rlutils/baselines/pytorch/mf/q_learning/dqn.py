import copy

import numpy as np
import torch.nn as nn
import torch.optim

import rlutils.np as rln
import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
from rlutils.interface.agent import OffPolicyAgent


def gather_q_values(q_values, action):
    return q_values.gather(1, action.unsqueeze(1)).squeeze(1)


class DQN(OffPolicyAgent, nn.Module):
    def __init__(self,
                 env,
                 make_q_net=lambda env: rlu.nn.build_mlp(input_dim=env.observation_space.shape[0],
                                                         output_dim=env.action_space.n,
                                                         mlp_hidden=256, num_layers=3),
                 double_q=True,
                 q_lr=1e-4,
                 gamma=0.99,
                 n_steps=1,
                 huber_delta=None,
                 grad_norm=None,
                 epsilon_greedy_scheduler=rln.schedulers.LinearSchedule(schedule_timesteps=1000,
                                                                        final_p=0.1,
                                                                        initial_p=0.1),
                 target_update_freq=500,
                 device=None,
                 ):
        OffPolicyAgent.__init__(self, env=env)
        nn.Module.__init__(self)
        self.grad_norm = grad_norm
        self.target_update_freq = target_update_freq
        self.gamma = gamma ** n_steps
        self.double_q = double_q
        self.q_lr = q_lr
        self.obs_spec = env.observation_space
        self.act_dim = env.action_space.n
        self.epsilon_greedy_scheduler = epsilon_greedy_scheduler
        self.q_network = make_q_net(env)
        self.target_q_network = copy.deepcopy(self.q_network)
        rlu.nn.functional.freeze(self.target_q_network)
        self.reset_optimizer()
        # define loss function
        reduction = 'none'
        if huber_delta is None:
            self.loss_fn = torch.nn.MSELoss(reduction=reduction)
        else:
            self.loss_fn = torch.nn.HuberLoss(delta=huber_delta, reduction=reduction)
        self.device = device

        self.policy_updates = 0
        self.to(self.device)

    def reset_optimizer(self):
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.q_lr)

    def log_tabular(self):
        super(DQN, self).log_tabular()
        self.logger.log_tabular('QVals', with_min_and_max=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('TDError', average_only=True)

    def update_target(self):
        rlu.functional.hard_update(self.target_q_network, self.q_network)

    def compute_target_values(self, next_obs, rew, done):
        with torch.no_grad():
            target_q_values = self.target_q_network(next_obs)  # (None, act_dim)
            if self.double_q:
                target_actions = torch.argmax(self.q_network(next_obs), dim=-1)  # (None,)
                target_q_values = gather_q_values(target_q_values, target_actions)
            else:
                target_q_values = torch.max(target_q_values, dim=-1)[0]
            target_q_values = rew + self.gamma * (1. - done) * target_q_values
            return target_q_values

    def train_on_batch_torch(self, obs, act, next_obs, rew, done, weights=None):
        target_q_values = self.compute_target_values(next_obs, rew, done)
        self.q_optimizer.zero_grad()
        q_values = self.q_network(obs)
        q_values = gather_q_values(q_values, act)
        loss = self.loss_fn(q_values, target_q_values)
        if weights is not None:
            loss = loss * weights
        loss = torch.mean(loss, dim=0)
        loss.backward()
        if self.grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.grad_norm)
        self.q_optimizer.step()
        info = dict(
            QVals=q_values,
            LossQ=loss
        )
        with torch.no_grad():
            abs_td_error = torch.abs(q_values - target_q_values).detach()
        info['TDError'] = abs_td_error
        return info

    def train_on_batch(self, data):
        tensor_data = ptu.convert_dict_to_tensor(data, device=self.device)

        info = self.train_on_batch_torch(**tensor_data)
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
            q_values = self.q_network(obs)
            return torch.argmax(q_values, dim=-1).cpu().numpy()


if __name__ == '__main__':
    from rlutils.baselines.trainer import run_offpolicy
    import rlutils.infra as rl_infra

    make_agent_fn = lambda env: DQN(env, device=ptu.get_cuda_device())
    rl_infra.runner.run_func_as_main(run_offpolicy, passed_args={
        'make_agent_fn': make_agent_fn,
        'backend': 'torch'
    })
