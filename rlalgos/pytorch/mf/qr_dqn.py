"""
Quantile regression DQN
"""

from typing import Callable

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import rlutils.infra as rl_infra
import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
from rlalgos.pytorch.mf.dqn import DQN


class QRDQN(DQN):
    def __init__(self, num_quantiles=32, **kwargs):
        assert num_quantiles > 1, 'The number of quantiles must be greater than 1'
        super(QRDQN, self).__init__(**kwargs)
        self.num_quantiles = num_quantiles
        tau = torch.linspace(0., 1., self.num_quantiles + 1)
        self.quantiles = nn.Parameter((tau[:-1] + tau[1:]) / 2, requires_grad=False)
        self.to(ptu.device)

        if self.double_q:
            self.double_q = False
            print('Double q is set to False in QRDQN')

    def _create_q_network(self):
        out_activation = lambda x: torch.reshape(x, shape=(-1, self.act_dim, self.num_quantiles))
        model = rlu.nn.functional.build_mlp(input_dim=self.obs_spec.shape[0],
                                            output_dim=self.act_dim * self.num_quantiles,
                                            mlp_hidden=self.mlp_hidden,
                                            out_activation=out_activation)
        return model

    def compute_target_values(self, next_obs, rew, done):
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
            target_quantile = rew[:, None] + self.gamma * (1. - done[:, None]) * target_logits  # (None, num_quantiles)
            return target_quantile

    def _update_nets(self, obs, act, next_obs, rew, done):
        batch_size = obs.shape[0]
        target_q_values = self.compute_target_values(next_obs, rew, done)  # (None, num_quantiles)
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
        loss = torch.mean(quantile_loss, dim=0)  # average over bath
        loss.backward()
        self.q_optimizer.step()
        with torch.no_grad():
            q_values = torch.mean(q_values, dim=-1)
        info = dict(
            QVals=q_values,
            LossQ=loss
        )
        return info

    def act_batch_test(self, obs):
        obs = torch.as_tensor(obs, device=ptu.device)
        with torch.no_grad():
            target_logits_action = self.q_network(obs)  # (None, act_dim, num_atoms)
            target_q_values = torch.mean(target_logits_action, dim=-1)  # (None, act_dim)
            target_actions = torch.argmax(target_q_values, dim=-1)  # (None,)
            return target_actions.cpu().numpy()


class Runner(rl_infra.runner.PytorchOffPolicyRunner):
    @classmethod
    def main(cls,
             env_name,
             env_fn: Callable = None,
             exp_name: str = None,
             steps_per_epoch=10000,
             epochs=100,
             start_steps=10000,
             update_after=5000,
             update_every=1,
             update_per_step=1,
             policy_delay=1,
             num_parallel_env=1,
             num_test_episodes=10,
             seed=1,
             # agent args
             q_lr=1e-4,
             tau=5e-3,
             gamma=0.99,
             # replay
             replay_size=int(1e6),
             logger_path: str = None
             ):
        agent_kwargs = dict(
            q_lr=q_lr,
            tau=tau,
            gamma=gamma,
        )

        super(Runner, cls).main(env_name=env_name,
                                env_fn=None,
                                exp_name=exp_name,
                                steps_per_epoch=steps_per_epoch,
                                epochs=epochs,
                                start_steps=start_steps,
                                update_after=update_after,
                                update_every=update_every,
                                update_per_step=update_per_step,
                                policy_delay=1,
                                num_parallel_env=1,
                                num_test_episodes=num_test_episodes,
                                agent_cls=QRDQN,
                                agent_kwargs=agent_kwargs,
                                seed=seed,
                                logger_path=logger_path
                                )


if __name__ == '__main__':
    ptu.set_device('cuda')
    rl_infra.runner.run_func_as_main(Runner.main)
