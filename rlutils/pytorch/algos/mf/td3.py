"""
Twin Delayed DDPG. https://arxiv.org/abs/1802.09477.
To obtain DDPG, set target smooth to zero and Q network ensembles to 1.
"""

import copy

import rlutils.pytorch.utils as ptu
import torch
import torch.nn as nn
from rlutils.gym.utils import verify_continuous_action_space
from rlutils.infra.runner import run_func_as_main, PytorchOffPolicyRunner
from rlutils.interface.agent import OffPolicyAgent
from rlutils.pytorch.functional import soft_update, compute_target_value, hard_update
from rlutils.pytorch.nn import EnsembleMinQNet
from rlutils.pytorch.nn.functional import build_mlp, freeze


class TD3Agent(nn.Module, OffPolicyAgent):
    def __init__(self,
                 env,
                 num_q_ensembles=2,
                 policy_mlp_hidden=128,
                 policy_lr=3e-4,
                 q_mlp_hidden=256,
                 q_lr=3e-4,
                 tau=5e-3,
                 gamma=0.99,
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
        self.gamma = gamma
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.num_q_ensembles = num_q_ensembles
        self.device = device
        self.reward_scale = reward_scale
        if len(self.obs_spec.shape) == 1:  # 1D observation
            self.obs_dim = self.obs_spec.shape[0]
            self.policy_net = build_mlp(self.obs_dim, self.act_dim, mlp_hidden=policy_mlp_hidden, num_layers=3,
                                        out_activation='tanh').to(self.device)
            self.target_policy_net = copy.deepcopy(self.policy_net).to(self.device)
            self.q_network = EnsembleMinQNet(self.obs_dim, self.act_dim, q_mlp_hidden,
                                             num_ensembles=num_q_ensembles).to(self.device)
            self.target_q_network = copy.deepcopy(self.q_network).to(self.device)

            freeze(self.target_policy_net)
            freeze(self.target_q_network)
        else:
            raise NotImplementedError

        self.reset_optimizer()

    def reset_optimizer(self):
        self.policy_optimizer = torch.optim.Adam(params=self.policy_net.parameters(), lr=self.policy_lr)
        self.q_optimizer = torch.optim.Adam(params=self.q_network.parameters(), lr=self.q_lr)

    def log_tabular(self):
        super(TD3Agent, self).log_tabular()
        for i in range(self.num_q_ensembles):
            self.logger.log_tabular(f'Q{i + 1}Vals', with_min_and_max=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('TDError', average_only=True)

    def update_target(self):
        soft_update(self.target_q_network, self.q_network, self.tau)
        soft_update(self.target_policy_net, self.policy_net, self.tau)

    def compute_priority(self, data):
        data_tensor = {}
        for key, d in data.items():
            data_tensor[key] = torch.as_tensor(d).to(self.device, non_blocking=True)
        return self.compute_priority_torch(**data_tensor).cpu().numpy()

    def compute_priority_torch(self, obs, act, next_obs, done, rew):
        with torch.no_grad():
            next_q_value = self._compute_next_obs_q(next_obs)
            q_target = compute_target_value(rew / self.reward_scale, self.gamma, done, next_q_value)
            q_values = self.q_network((obs, act), training=False)  # (None,)
            abs_td_error = torch.abs(q_values - q_target)
            return abs_td_error

    def _compute_next_obs_q(self, next_obs):
        next_action = self.target_policy_net(next_obs)
        # Target policy smoothing
        epsilon = torch.randn_like(next_action) * self.target_noise
        epsilon = torch.clip(epsilon, -self.noise_clip, self.noise_clip)
        next_action = next_action + epsilon
        next_action = torch.clip(next_action, -self.act_lim, self.act_lim)
        next_q_value = self.target_q_network((next_obs, next_action), training=False)
        return next_q_value

    def _update_nets(self, obs, act, next_obs, done, rew, weights=None):
        # compute target q
        with torch.no_grad():
            next_q_value = self._compute_next_obs_q(next_obs)
            q_target = compute_target_value(rew / self.reward_scale, self.gamma, done, next_q_value)
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

    def _update_actor(self, obs, weights=None):
        # policy loss
        self.q_network.eval()
        self.policy_optimizer.zero_grad()
        a = self.policy_net(obs)
        q = self.q_network((obs, a), training=False)
        if weights is not None:
            q = q * weights
        policy_loss = -torch.mean(q, dim=0)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.q_network.train()
        info = dict(
            LossPi=policy_loss.detach(),
        )
        return info

    def train_on_batch(self, data, **kwargs):
        new_data = {}
        for key, d in data.items():
            new_data[key] = torch.as_tensor(d).to(self.device, non_blocking=True)

        info = self._update_nets(**new_data)

        self.policy_updates += 1

        if self.policy_updates % 2 == 0:
            obs = new_data['obs']
            weights = new_data.get('weights', None)
            actor_info = self._update_actor(obs, weights)
            info.update(actor_info)

        if self.policy_updates % 2 == 0:
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


class TD3Agent_v2(TD3Agent):
    def __init__(self, tau=1000, q_lr=1e-3, policy_lr=1e-3, **kwargs):
        super(TD3Agent_v2, self).__init__(**kwargs, tau=tau, q_lr=q_lr, policy_lr=policy_lr)
        assert isinstance(tau, int), 'Tau must be an integer that represents the hard target update frequency'

    def update_target(self):
        hard_update(self.target_q_network, self.q_network)
        hard_update(self.target_policy_net, self.policy_net)

    def train_on_batch_q_network(self, data, **kwargs):
        new_data = {}
        for key, d in data.items():
            new_data[key] = torch.as_tensor(d).to(self.device, non_blocking=True)
        info = self._update_nets(**new_data)
        if self.logger is not None:
            self.logger.store(**info)
        return info

    def train_on_batch_actor_network(self, data, **kwargs):
        new_data = {}
        for key, d in data.items():
            new_data[key] = torch.as_tensor(d).to(self.device, non_blocking=True)
        obs = new_data['obs']
        info = self._update_actor(obs)
        if self.logger is not None:
            self.logger.store(**info)
        return info

    def train_on_batch(self, data, **kwargs):
        new_data = {}
        for key, d in data.items():
            new_data[key] = torch.as_tensor(d).to(self.device, non_blocking=True)
        update_target = kwargs.pop('update_target', None)

        info = self._update_nets(**new_data)

        self.policy_updates += 1

        obs = new_data['obs']
        actor_info = self._update_actor(obs)
        info.update(actor_info)

        if update_target is None:
            if self.policy_updates % self.tau == 0:
                self.update_target()
        else:
            if update_target:
                self.update_target()

        if self.logger is not None:
            self.logger.store(**info)

        return info


class Runner(PytorchOffPolicyRunner):
    @classmethod
    def main(cls,
             env_name,
             epochs=200,
             policy_mlp_hidden=256,
             policy_lr=1e-3,
             q_mlp_hidden=256,
             q_lr=1e-3,
             actor_noise=0.1,
             target_noise=0.2,
             noise_clip=0.5,
             tau=5e-3,
             gamma=0.99,
             seed=1,
             logger_path: str = None,
             **kwargs
             ):
        agent_kwargs = dict(
            policy_mlp_hidden=policy_mlp_hidden,
            policy_lr=policy_lr,
            q_mlp_hidden=q_mlp_hidden,
            q_lr=q_lr,
            tau=tau,
            gamma=gamma,
            actor_noise=actor_noise,
            target_noise=target_noise,
            noise_clip=noise_clip
        )

        super(Runner, cls).main(env_name=env_name,
                                epochs=epochs,
                                agent_cls=TD3Agent,
                                agent_kwargs=agent_kwargs,
                                seed=seed,
                                logger_path=logger_path,
                                **kwargs)


if __name__ == '__main__':
    ptu.set_device('cuda')
    run_func_as_main(Runner.main)
