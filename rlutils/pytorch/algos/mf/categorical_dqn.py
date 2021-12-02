import copy
from typing import Callable

import numpy as np
import torch.nn as nn
import torch.optim

import rlutils.infra as rl_infra
import rlutils.np as rln
import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
from rlutils.interface.agent import Agent


class CategoricalDQN(Agent, nn.Module):
    def __init__(self,
                 obs_spec,
                 act_spec,
                 mlp_hidden=128,
                 double_q=True,
                 q_lr=1e-4,
                 gamma=0.99,
                 tau=5e-3,
                 num_atoms=51,
                 v_min=0.,
                 v_max=100.,
                 epsilon_greedy_steps=1000,
                 ):
        assert num_atoms > 1, 'The number of atoms must be greater than 1'
        super(CategoricalDQN, self).__init__()
        self.mlp_hidden = mlp_hidden
        self.tau = tau
        self.gamma = gamma
        self.double_q = double_q
        self.obs_spec = obs_spec
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.act_dim = act_spec.n
        self.q_network = self._create_q_network()
        self.target_q_network = copy.deepcopy(self.q_network)
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=q_lr)
        self.epsilon_greedy_scheduler = rln.schedulers.LinearSchedule(schedule_timesteps=epsilon_greedy_steps,
                                                                      final_p=0.1,
                                                                      initial_p=1.0)
        self.support = nn.Parameter(torch.linspace(self.v_min, self.v_max, self.num_atoms), requires_grad=False)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        self.to(ptu.device)

    def _create_q_network(self):
        out_activation = lambda x: torch.reshape(x, shape=(-1, self.act_dim, self.num_atoms))
        model = rlu.nn.functional.build_mlp(input_dim=self.obs_spec.shape[0],
                                            output_dim=self.act_dim * self.num_atoms,
                                            mlp_hidden=self.mlp_hidden,
                                            out_activation=out_activation)
        return rlu.nn.values.CategoricalQModule(model=model)

    def log_tabular(self):
        self.logger.log_tabular('QVals', with_min_and_max=True)
        self.logger.log_tabular('LossQ', average_only=True)

    def update_target(self):
        rlu.functional.soft_update(self.target_q_network, self.q_network, self.tau)

    def compute_target_values(self, next_obs, rew, done):
        with torch.no_grad():
            batch_size = next_obs.shape[0]
            target_action_compute_q = self.q_network if self.double_q else self.target_q_network
            target_logits_action = target_action_compute_q(next_obs)  # (None, act_dim, num_atoms)
            target_q_values = torch.sum(target_logits_action * self.support[None, None, :], dim=-1)  # (None, act_dim)
            target_actions = torch.argmax(target_q_values, dim=-1)  # (None,)
            if self.double_q:
                target_logits = self.target_q_network(next_obs)
            else:
                target_logits = target_logits_action
            target_logits = target_logits[np.arange(batch_size), target_actions]  # (None, num_atoms)
            # atom values
            target_q_atoms = rew[:, None] + self.gamma * (1. - done[:, None]) * self.support[None, :]
            target_q_atoms = torch.clamp(target_q_atoms, min=self.v_min, max=self.v_max)  # (None, num_atoms)
            atom_distribution = 1. - torch.abs(target_q_atoms[:, :, None] - self.support[None, None, :]) / self.delta_z
            atom_distribution = torch.clamp(atom_distribution, min=0., max=1.)  # (None, j, i)
            probability = torch.sum(atom_distribution * target_logits[:, :, None], dim=1)  # (None, num_atoms)
            return probability

    def _update_nets(self, obs, act, next_obs, rew, done):
        target_q_values = self.compute_target_values(next_obs, rew, done)  # (None, num_atoms)
        self.q_optimizer.zero_grad()
        q_values = self.q_network(obs, act, log_prob=True)  # (None, num_atoms)
        cross_entropy = -torch.sum(target_q_values * q_values, dim=-1)
        loss = torch.mean(cross_entropy, dim=0)
        loss.backward()
        self.q_optimizer.step()
        with torch.no_grad():
            q_values = torch.sum(torch.exp(q_values) * self.support[None, :], dim=-1)
        info = dict(
            QVals=q_values,
            LossQ=loss
        )
        return info

    def train_on_batch(self, data, **kwargs):
        obs = data['obs']
        act = data['act']
        next_obs = data['next_obs']
        done = data['done']
        rew = data['rew']
        update_target = data['update_target']
        obs = torch.as_tensor(obs).pin_memory().to(ptu.device, non_blocking=True)
        act = torch.as_tensor(act).pin_memory().to(ptu.device, non_blocking=True)
        next_obs = torch.as_tensor(next_obs).pin_memory().to(ptu.device, non_blocking=True)
        done = torch.as_tensor(done).pin_memory().to(ptu.device, non_blocking=True)
        rew = torch.as_tensor(rew).pin_memory().to(ptu.device, non_blocking=True)
        info = self._update_nets(obs, act, next_obs, rew, done)
        if update_target:
            self.update_target()

        self.logger.store(**info)

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
        obs = torch.as_tensor(obs, device=ptu.device)
        with torch.no_grad():
            target_logits_action = self.q_network(obs)  # (None, act_dim, num_atoms)
            target_q_values = torch.sum(target_logits_action * self.support[None, None, :], dim=-1)  # (None, act_dim)
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
                                agent_cls=CategoricalDQN,
                                agent_kwargs=agent_kwargs,
                                seed=seed,
                                logger_path=logger_path
                                )


if __name__ == '__main__':
    ptu.set_device('cuda')
    rl_infra.runner.run_func_as_main(Runner.main)
