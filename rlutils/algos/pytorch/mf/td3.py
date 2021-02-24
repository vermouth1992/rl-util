"""
Twin Delayed DDPG. https://arxiv.org/abs/1802.09477.
To obtain DDPG, set target smooth to zero and Q network ensembles to 1.
"""

import copy

import rlutils.pytorch.utils as ptu
import torch
import torch.nn as nn
from rlutils.infra.runner import run_func_as_main, PytorchOffPolicyRunner
from rlutils.pytorch.functional import soft_update, compute_target_value, to_numpy_or_python_type
from rlutils.pytorch.nn import EnsembleMinQNet
from rlutils.pytorch.nn.functional import build_mlp


class TD3Agent(nn.Module):
    def __init__(self,
                 obs_spec,
                 act_spec,
                 num_q_ensembles=2,
                 policy_mlp_hidden=128,
                 policy_lr=3e-4,
                 q_mlp_hidden=256,
                 q_lr=3e-4,
                 tau=5e-3,
                 gamma=0.99,
                 actor_noise=0.1,
                 target_noise=0.2,
                 noise_clip=0.5
                 ):
        super(TD3Agent, self).__init__()
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.act_dim = self.act_spec.shape[0]
        self.act_lim = 1.
        self.actor_noise = actor_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.tau = tau
        self.gamma = gamma
        if len(self.obs_spec.shape) == 1:  # 1D observation
            self.obs_dim = self.obs_spec.shape[0]
            self.policy_net = build_mlp(self.obs_dim, self.act_dim, mlp_hidden=policy_mlp_hidden, num_layers=3,
                                        out_activation='tanh').to(ptu.device)
            self.target_policy_net = copy.deepcopy(self.policy_net).to(ptu.device)
            self.q_network = EnsembleMinQNet(self.obs_dim, self.act_dim, q_mlp_hidden,
                                             num_ensembles=num_q_ensembles).to(ptu.device)
            self.target_q_network = copy.deepcopy(self.q_network).to(ptu.device)
        else:
            raise NotImplementedError

        self.policy_optimizer = torch.optim.Adam(params=self.policy_net.parameters(), lr=policy_lr)
        self.q_optimizer = torch.optim.Adam(params=self.q_network.parameters(), lr=q_lr)

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('Q1Vals', with_min_and_max=True)
        self.logger.log_tabular('Q2Vals', with_min_and_max=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)

    def update_target(self):
        soft_update(self.target_q_network, self.q_network, self.tau)
        soft_update(self.target_policy_net, self.policy_net, self.tau)

    def _compute_next_obs_q(self, next_obs):
        next_action = self.target_policy_net(next_obs)
        # Target policy smoothing
        epsilon = torch.randn_like(next_action) * self.target_noise
        epsilon = torch.clip(epsilon, -self.noise_clip, self.noise_clip)
        next_action = next_action + epsilon
        next_action = torch.clip(next_action, -self.act_lim, self.act_lim)
        next_q_value = self.target_q_network((next_obs, next_action), training=False)
        return next_q_value

    def _update_nets(self, obs, actions, next_obs, done, reward):
        # compute target q
        with torch.no_grad():
            next_q_value = self._compute_next_obs_q(next_obs)
            q_target = compute_target_value(reward, self.gamma, done, next_q_value)
        # q loss
        self.q_optimizer.zero_grad()
        q_values = self.q_network((obs, actions), training=True)  # (num_ensembles, None)
        q_values_loss = 0.5 * torch.square(torch.unsqueeze(q_target, dim=0) - q_values)
        # (num_ensembles, None)
        q_values_loss = torch.sum(q_values_loss, dim=0)  # (None,)
        # apply importance weights
        q_values_loss = torch.mean(q_values_loss)
        q_values_loss.backward()
        self.q_optimizer.step()

        info = dict(
            Q1Vals=q_values[0],
            Q2Vals=q_values[1],
            LossQ=q_values_loss,
        )
        return info

    def _update_actor(self, obs):
        # policy loss
        self.policy_optimizer.zero_grad()
        a = self.policy_net(obs)
        q = self.q_network((obs, a), training=False)
        policy_loss = -torch.mean(q, dim=0)
        policy_loss.backward()
        self.policy_optimizer.step()
        info = dict(
            LossPi=policy_loss,
        )
        return info

    def train_on_batch(self, data, **kwargs):
        obs = data['obs']
        act = data['act']
        next_obs = data['next_obs']
        done = data['done']
        rew = data['rew']
        update_target = data['update_target']
        obs = torch.as_tensor(obs, dtype=torch.float32, device=ptu.device)
        act = torch.as_tensor(act, dtype=torch.float32, device=ptu.device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=ptu.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=ptu.device)
        rew = torch.as_tensor(rew, dtype=torch.float32, device=ptu.device)

        info = self._update_nets(obs, act, next_obs, done, rew)

        if update_target:
            actor_info = self._update_actor(obs)
            info.update(actor_info)
            self.update_target()

        self.logger.store(**to_numpy_or_python_type(info))

    def act_batch_test(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=ptu.device)
        with torch.no_grad():
            return self.policy_net(obs).cpu().numpy()

    def act_batch_explore(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=ptu.device)
        with torch.no_grad():
            pi_final = self.policy_net(obs)
            noise = torch.randn_like(pi_final) * self.actor_noise
            pi_final = pi_final + noise
            pi_final = torch.clip(pi_final, -self.act_lim, self.act_lim)
            return pi_final.cpu().numpy()


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
                                policy_delay=2,
                                agent_cls=TD3Agent,
                                agent_kwargs=agent_kwargs,
                                seed=seed,
                                logger_path=logger_path,
                                **kwargs)


if __name__ == '__main__':
    ptu.set_device('cuda')
    run_func_as_main(Runner.main)
