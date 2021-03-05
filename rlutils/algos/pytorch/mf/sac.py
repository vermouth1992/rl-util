"""
Implement soft actor critic agent here
"""

import copy

import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
import torch
from rlutils.infra.runner import PytorchOffPolicyRunner, run_func_as_main
from torch import nn


class SACAgent(nn.Module):
    def __init__(self,
                 obs_spec,
                 act_spec,
                 num_ensembles=2,
                 policy_mlp_hidden=128,
                 policy_lr=3e-4,
                 q_mlp_hidden=256,
                 q_lr=3e-4,
                 alpha=1.0,
                 alpha_lr=1e-3,
                 tau=5e-3,
                 gamma=0.99,
                 target_entropy=None,
                 ):
        super(SACAgent, self).__init__()
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.act_dim = self.act_spec.shape[0]
        if len(self.obs_spec.shape) == 1:  # 1D observation
            self.obs_dim = self.obs_spec.shape[0]
            self.policy_net = rlu.nn.SquashedGaussianMLPActor(self.obs_dim, self.act_dim, policy_mlp_hidden)
            self.q_network = rlu.nn.EnsembleMinQNet(self.obs_dim, self.act_dim, q_mlp_hidden,
                                                    num_ensembles=num_ensembles)
            self.target_q_network = copy.deepcopy(self.q_network)
        else:
            raise NotImplementedError

        self.alpha_net = rlu.nn.LagrangeLayer(initial_value=alpha)

        self.policy_optimizer = torch.optim.Adam(params=self.policy_net.parameters(), lr=policy_lr)
        self.q_optimizer = torch.optim.Adam(params=self.q_network.parameters(), lr=q_lr)
        self.alpha_optimizer = torch.optim.Adam(params=self.alpha_net.parameters(), lr=alpha_lr)
        self.target_entropy = -self.act_dim if target_entropy is None else target_entropy

        self.tau = tau
        self.gamma = gamma

        self.to(ptu.device)

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('Q1Vals', with_min_and_max=True)
        self.logger.log_tabular('Q2Vals', with_min_and_max=True)
        self.logger.log_tabular('LogPi', average_only=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('Alpha', average_only=True)
        self.logger.log_tabular('LossAlpha', average_only=True)

    def update_target(self):
        rlu.functional.soft_update(self.target_q_network, self.q_network, self.tau)

    def _update_nets(self, obs, act, next_obs, done, rew):
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
            Q1Vals=q_values[0],
            Q2Vals=q_values[1],
            LogPi=log_prob,
            Alpha=alpha,
            LossQ=q_values_loss,
            LossAlpha=alpha_loss,
            LossPi=policy_loss,
        )
        return info

    def train_on_batch(self, data, **kwargs):
        update_target = data.pop('update_target')
        data = {key: torch.as_tensor(value, device=ptu.device) for key, value in data.items()}
        info = self._update_nets(**data)
        for key, item in info.items():
            info[key] = item.detach().cpu().numpy()
        self.logger.store(**info)
        if update_target:
            self.update_target()

    def act_batch_torch(self, obs, deterministic):
        with torch.no_grad():
            pi_final = self.policy_net.select_action((obs, deterministic))
            return pi_final

    def act_batch_explore(self, obs):
        obs = torch.as_tensor(obs, device=ptu.device)
        return self.act_batch_torch(obs, deterministic=False).cpu().numpy()

    def act_batch_test(self, obs):
        obs = torch.as_tensor(obs, device=ptu.device)
        return self.act_batch_torch(obs, deterministic=True).cpu().numpy()


class Runner(PytorchOffPolicyRunner):
    @classmethod
    def main(cls,
             env_name,
             epochs=100,
             # sac args
             policy_mlp_hidden=256,
             policy_lr=3e-4,
             q_mlp_hidden=256,
             q_lr=3e-4,
             alpha=0.2,
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
            alpha=alpha,
            alpha_lr=q_lr,
            tau=tau,
            gamma=gamma,
            target_entropy=None
        )

        super(Runner, cls).main(
            env_name=env_name,
            epochs=epochs,
            agent_cls=SACAgent,
            agent_kwargs=agent_kwargs,
            policy_delay=1,
            seed=seed,
            logger_path=logger_path,
            **kwargs
        )


if __name__ == '__main__':
    run_func_as_main(Runner.main)
