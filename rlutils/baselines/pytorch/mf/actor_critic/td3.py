"""
Twin Delayed DDPG. https://arxiv.org/abs/1802.09477.
To obtain DDPG, set target smooth to zero and Q network ensembles to 1.
"""

import copy

import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
import torch
import torch.nn as nn
from rlutils.gym.utils import verify_continuous_action_space
from rlutils.interface.agent import OffPolicyAgent


class TD3Agent(nn.Module, OffPolicyAgent):
    def __init__(self,
                 env,
                 num_q_ensembles=2,
                 make_policy_net=lambda env: rlu.nn.build_mlp(env.observation_space.shape[0],
                                                              env.action_space.shape[0],
                                                              mlp_hidden=256,
                                                              num_layers=3,
                                                              out_activation='tanh'),
                 policy_lr=3e-4,
                 policy_update_freq=2,
                 make_q_network=lambda env, num_q_ensembles: rlu.nn.EnsembleMinQNet(env.observation_space.shape[0],
                                                                                    env.action_space.shape[0],
                                                                                    mlp_hidden=256,
                                                                                    num_layers=3,
                                                                                    num_ensembles=num_q_ensembles),
                 q_lr=3e-4,
                 tau=5e-3,
                 gamma=0.99,
                 n_steps=1,
                 actor_noise=0.1,
                 target_noise=0.2,
                 noise_clip=0.5,
                 reward_scale=1.0,
                 device=ptu.device
                 ):
        nn.Module.__init__(self)
        OffPolicyAgent.__init__(self, env=env)

        self.obs_spec = env.observation_space
        self.act_spec = env.action_space
        self.act_dim = self.act_spec.shape[0]
        verify_continuous_action_space(self.act_spec)
        self.act_lim = self.act_spec.high[0]
        self.actor_noise = actor_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.tau = tau
        self.gamma = gamma ** n_steps
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.num_q_ensembles = num_q_ensembles
        self.device = device
        self.reward_scale = reward_scale
        self.policy_update_freq = policy_update_freq

        self.obs_dim = self.obs_spec.shape[0]
        self.policy_net = make_policy_net(env)
        self.target_policy_net = copy.deepcopy(self.policy_net)
        self.q_network = make_q_network(env, num_q_ensembles)
        self.target_q_network = copy.deepcopy(self.q_network)

        rlu.nn.functional.freeze(self.target_policy_net)
        rlu.nn.functional.freeze(self.target_q_network)

        self.reset_optimizer()

        self.to(self.device)

        self.policy_updates = 0

    def reset_optimizer(self):
        self.policy_optimizer = torch.optim.Adam(params=self.policy_net.parameters(), lr=self.policy_lr)
        self.q_optimizer = torch.optim.Adam(params=self.q_network.parameters(), lr=self.q_lr)

    def log_tabular(self):
        for i in range(self.num_q_ensembles):
            self.logger.log_tabular(f'Q{i + 1}Vals', with_min_and_max=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('TDError', average_only=True)
        super(TD3Agent, self).log_tabular()

    def update_target(self):
        rlu.functional.soft_update(self.target_q_network, self.q_network, self.tau)
        rlu.functional.soft_update(self.target_policy_net, self.policy_net, self.tau)

    def compute_next_obs_q_torch(self, next_obs):
        next_action = self.target_policy_net(next_obs)
        # Target policy smoothing
        epsilon = torch.randn_like(next_action) * self.target_noise
        epsilon = torch.clip(epsilon, -self.noise_clip, self.noise_clip)
        next_action = next_action + epsilon
        next_action = torch.clip(next_action, -self.act_lim, self.act_lim)
        next_q_value = self.target_q_network((next_obs, next_action), training=False)
        return next_q_value

    def train_q_network_on_batch_torch(self, obs, act, next_obs, done, rew, weights=None):
        # compute target q
        with torch.no_grad():
            next_q_value = self.compute_next_obs_q_torch(next_obs)
            q_target = rlu.functional.compute_target_value(rew / self.reward_scale, self.gamma, done, next_q_value)
        # q loss
        self.q_optimizer.zero_grad()
        q_values = self.q_network((obs, act), training=True)  # (num_ensembles, None)
        q_values_loss = 0.5 * torch.square(torch.unsqueeze(q_target, dim=0) - q_values)
        # (num_ensembles, None)
        q_values_loss = torch.sum(q_values_loss, dim=0)  # (None,)
        # apply importance weights
        if weights is not None:
            q_values_loss = q_values_loss * weights
        q_values_loss = torch.mean(q_values_loss)
        q_values_loss.backward()
        self.q_optimizer.step()

        with torch.no_grad():
            abs_td_error = torch.abs(torch.min(q_values, dim=0)[0] - q_target)

        info = dict(
            LossQ=q_values_loss.detach(),
            TDError=abs_td_error.detach()
        )
        for i in range(self.num_q_ensembles):
            info[f'Q{i + 1}Vals'] = q_values[i].detach()
        return info

    def train_actor_on_batch_torch(self, obs):
        # policy loss
        self.q_network.eval()
        self.policy_optimizer.zero_grad()
        a = self.policy_net(obs)
        q = self.q_network((obs, a), training=False)
        policy_loss = -torch.mean(q, dim=0)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.q_network.train()
        info = dict(
            LossPi=policy_loss.detach(),
        )
        return info

    def train_on_batch(self, data):
        new_data = ptu.convert_dict_to_tensor(data, device=self.device)
        info = self.train_q_network_on_batch_torch(**new_data)

        self.policy_updates += 1

        if self.policy_updates % self.policy_update_freq == 0:
            obs = new_data['obs']
            actor_info = self.train_actor_on_batch_torch(obs)
            info.update(actor_info)
            self.update_target()

        if self.logger is not None:
            self.logger.store(**info)

        return info

    def act_batch_test(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            result = self.policy_net(obs)
            return ptu.to_numpy(result)

    def act_batch_explore(self, obs, global_steps):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pi_final = self.policy_net(obs)
            noise = torch.randn_like(pi_final) * self.actor_noise
            pi_final = pi_final + noise
            pi_final = torch.clip(pi_final, -self.act_lim, self.act_lim)
            return ptu.to_numpy(pi_final)


if __name__ == '__main__':
    from rlutils.baselines.trainer import run_offpolicy
    from rlutils.infra.runner import run_func_as_main

    make_agent_fn = lambda env: TD3Agent(env, device=ptu.get_cuda_device())
    run_func_as_main(run_offpolicy, passed_args={
        'make_agent_fn': make_agent_fn,
        'backend': 'torch'
    })
