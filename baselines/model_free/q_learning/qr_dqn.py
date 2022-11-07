"""
Quantile regression DQN
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
from baselines.model_free.q_learning.dqn import DQN


class QRDQN(DQN):
    def __init__(self,
                 make_q_net=lambda env, num_quantiles: rlu.nn.functional.build_mlp(
                     input_dim=env.observation_space.shape[0],
                     output_dim=env.action_space.n * num_quantiles,
                     mlp_hidden=256,
                     out_activation=lambda
                             x: torch.reshape(x, shape=(
                             -1, env.action_space.n,
                             num_quantiles))),
                 num_quantiles=32, **kwargs):
        assert num_quantiles > 1, 'The number of quantiles must be greater than 1'
        super(QRDQN, self).__init__(**kwargs, double_q=False, make_q_net=lambda env: make_q_net(env, num_quantiles))

        self.num_quantiles = num_quantiles
        tau = torch.linspace(0., 1., self.num_quantiles + 1)
        self.quantiles = nn.Parameter((tau[:-1] + tau[1:]) / 2, requires_grad=False)

        self.to(self.device)

    def compute_target_values(self, next_obs, rew, done, gamma):
        # double q doesn't perform very well.
        with torch.no_grad():
            batch_size = next_obs.shape[0]
            target_action_compute_q = self.q_network if self.double_q else self.target_q_network
            target_logits_action = target_action_compute_q(next_obs)  # (None, act_dim, num_quantiles)
            target_q_values = torch.mean(target_logits_action, dim=-1)  # (None, act_dim)
            target_actions = torch.argmax(target_q_values, dim=-1)  # (None,)
            if self.double_q:
                target_logits = self.target_q_network(next_obs)
            else:
                target_logits = target_logits_action
            target_logits = target_logits[np.arange(batch_size), target_actions]  # (None, num_quantiles)
            # atom values
            target_quantile = rew[:, None] + gamma[:, None] * (1. - done[:, None]) * target_logits
            # (None, num_quantiles)
            return target_quantile

    def train_on_batch_torch(self, obs, act, next_obs, rew, done, gamma, weights=None):
        batch_size = obs.shape[0]
        target_q_values = self.compute_target_values(next_obs, rew, done, gamma)  # (None, num_quantiles)
        self.q_optimizer.zero_grad()
        q_values = self.q_network(obs)  # (None, act_dim, num_quantiles)
        q_values = q_values[torch.arange(batch_size), act]  # (None, num_quantiles)
        u = F.smooth_l1_loss(torch.tile(q_values[:, :, None], (1, 1, self.num_quantiles)),
                             torch.tile(target_q_values[:, None, :], (1, self.num_quantiles, 1)),
                             reduction='none')
        with torch.no_grad():
            td_error = target_q_values[:, None, :] - q_values[:, :, None]  # (None, num_quantiles, num_quantiles)
            quantile_weight = torch.abs(self.quantiles[None, :, None] - torch.le(td_error, 0.).float())
        quantile_loss = quantile_weight * u
        quantile_loss = torch.mean(quantile_loss, dim=-1)  # expected over target j
        quantile_loss = torch.sum(quantile_loss, dim=-1)  # sum over i

        loss = quantile_loss
        if weights is not None:
            loss = loss * weights

        loss = torch.mean(loss, dim=0)  # average over bath
        loss.backward()
        self.q_optimizer.step()
        with torch.no_grad():
            q_values = torch.mean(q_values, dim=-1)
        info = dict(
            QVals=q_values,
            LossQ=loss,
            TDError=quantile_loss
        )
        return info

    def act_batch_test(self, obs):
        obs = torch.as_tensor(obs, device=self.device)
        with torch.no_grad():
            target_logits_action = self.q_network(obs)  # (None, act_dim, num_atoms)
            target_q_values = torch.mean(target_logits_action, dim=-1)  # (None, act_dim)
            target_actions = torch.argmax(target_q_values, dim=-1)  # (None,)
            return target_actions.cpu().numpy()


if __name__ == '__main__':
    from baselines.model_free.trainer import run_offpolicy
    import rlutils.infra as rl_infra

    make_agent_fn = lambda env: QRDQN(env=env, device=ptu.get_cuda_device())
    rl_infra.runner.run_func_as_main(run_offpolicy, passed_args={
        'make_agent_fn': make_agent_fn,
        'backend': 'torch'
    })
