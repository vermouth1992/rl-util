"""
Implement soft actor critic agent here
"""

import copy

import torch
from torch import nn

import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
from rlutils.interface.agent import OffPolicyAgent
from rlutils.gym.utils import verify_continuous_action_space


class SACAgent(OffPolicyAgent, nn.Module):
    def __init__(self,
                 env,
                 make_policy_net=lambda env: rlu.nn.SquashedGaussianMLPActor(env.observation_space.shape[0],
                                                                             env.action_space.shape[0],
                                                                             mlp_hidden=256),
                 policy_lr=3e-4,
                 num_q_ensembles=2,
                 make_q_network=lambda env, num_q_ensembles: rlu.nn.EnsembleMinQNet(env.observation_space.shape[0],
                                                                                    env.action_space.shape[0],
                                                                                    mlp_hidden=256,
                                                                                    num_layers=3,
                                                                                    num_ensembles=num_q_ensembles),
                 q_lr=3e-4,
                 alpha=1.0,
                 alpha_lr=1e-3,
                 tau=5e-3,
                 gamma=0.99,
                 target_entropy=None,
                 device=ptu.device
                 ):
        nn.Module.__init__(self)
        OffPolicyAgent.__init__(self, env=env)
        self.obs_spec = env.observation_space
        self.act_spec = env.action_space
        verify_continuous_action_space(self.act_spec)
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.alpha_lr = alpha_lr
        self.act_dim = self.act_spec.shape[0]
        self.policy_net = make_policy_net(env)
        self.num_q_ensembles = num_q_ensembles
        self.q_network = make_q_network(env, self.num_q_ensembles)
        self.target_q_network = copy.deepcopy(self.q_network)
        rlu.nn.functional.freeze(self.target_q_network)

        self.alpha_net = rlu.nn.LagrangeLayer(initial_value=alpha)

        self.target_entropy = -self.act_dim if target_entropy is None else target_entropy

        self.tau = tau
        self.gamma = gamma

        self.reset_optimizer()

        self.device = device
        self.to(device)

    def reset_optimizer(self):
        self.policy_optimizer = torch.optim.Adam(params=self.policy_net.parameters(), lr=self.policy_lr)
        self.q_optimizer = torch.optim.Adam(params=self.q_network.parameters(), lr=self.q_lr)
        self.alpha_optimizer = torch.optim.Adam(params=self.alpha_net.parameters(), lr=self.alpha_lr)

    def log_tabular(self):
        for i in range(self.num_q_ensembles):
            self.logger.log_tabular(f'Q{i + 1}Vals', with_min_and_max=True)
        self.logger.log_tabular('LogPi', average_only=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('Alpha', average_only=True)
        self.logger.log_tabular('LossAlpha', average_only=True)
        super(SACAgent, self).log_tabular()

    def update_target(self):
        rlu.functional.soft_update(self.target_q_network, self.q_network, self.tau)

    def train_on_batch_torch(self, obs, act, next_obs, done, rew):
        """ Sample a mini-batch from replay buffer and update the network

        Args:
            obs: (batch_size, ob_dim)
            actions: (batch_size, action_dim)
            next_obs: (batch_size, ob_dim)
            done: (batch_size,)
            reward: (batch_size,)

        Returns: None

        """
        with torch.no_grad():
            alpha = self.alpha_net()
            next_action, next_action_log_prob, _, _ = self.policy_net((next_obs, False))
            target_q_values = self.target_q_network((next_obs, next_action),
                                                    training=False) - alpha * next_action_log_prob
            q_target = rew + self.gamma * (1.0 - done) * target_q_values

        # q loss
        q_values = self.q_network((obs, act), training=True)  # (num_ensembles, None)
        q_values_loss = 0.5 * torch.square(torch.unsqueeze(q_target, dim=0) - q_values)
        # (num_ensembles, None)
        q_values_loss = torch.sum(q_values_loss, dim=0)  # (None,)
        # apply importance weights
        q_values_loss = torch.mean(q_values_loss)
        self.q_optimizer.zero_grad()
        q_values_loss.backward()
        self.q_optimizer.step()

        # policy loss
        action, log_prob, _, _ = self.policy_net((obs, False))
        q_values_pi_min = self.q_network((obs, action), training=False)
        policy_loss = torch.mean(log_prob * alpha - q_values_pi_min)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha = self.alpha_net()
        alpha_loss = -torch.mean(alpha * (log_prob.detach() + self.target_entropy))
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        info = dict(
            LogPi=log_prob,
            Alpha=alpha,
            LossQ=q_values_loss,
            LossAlpha=alpha_loss,
            LossPi=policy_loss,
        )

        for i in range(self.num_q_ensembles):
            info[f'Q{i + 1}Vals'] = q_values[i]

        return info

    def train_on_batch(self, data):
        data = ptu.convert_dict_to_tensor(data, device=self.device)
        info = self.train_on_batch_torch(**data)
        self.logger.store(**info)
        self.update_target()

    def act_batch_torch(self, obs, deterministic):
        with torch.no_grad():
            pi_final = self.policy_net.select_action((obs, deterministic))
            return pi_final

    def act_batch_explore(self, obs, global_steps):
        obs = torch.as_tensor(obs, device=ptu.device)
        return self.act_batch_torch(obs, deterministic=False).cpu().numpy()

    def act_batch_test(self, obs):
        obs = torch.as_tensor(obs, device=ptu.device)
        return self.act_batch_torch(obs, deterministic=True).cpu().numpy()


if __name__ == '__main__':
    from rlalgos.runner import run_offpolicy
    from rlutils.infra.runner import run_func_as_main

    make_agent_fn = lambda env: SACAgent(env, device=ptu.get_cuda_device())
    run_func_as_main(run_offpolicy, passed_args={
        'make_agent_fn': make_agent_fn,
        'backend': 'torch'
    })
