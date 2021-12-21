import copy
from typing import Callable

import numpy as np
import torch.nn as nn
import torch.optim

import rlutils.infra as rl_infra
import rlutils.np as rln
import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
from rlutils.interface.agent import OffPolicyAgent


def gather_q_values(q_values, action):
    return q_values.gather(1, action.unsqueeze(1)).squeeze(1)


class DQN(OffPolicyAgent, nn.Module):
    def __init__(self,
                 obs_spec,
                 act_spec,
                 mlp_hidden=128,
                 double_q=True,
                 q_lr=1e-3,
                 gamma=0.99,
                 tau=5e-3,
                 huber_delta=None,
                 epsilon_greedy_steps=1000,
                 ):
        super(DQN, self).__init__()
        self.tau = tau
        self.gamma = gamma
        self.double_q = double_q
        self.q_lr = q_lr
        self.obs_spec = obs_spec
        self.act_dim = act_spec.n
        self.mlp_hidden = mlp_hidden
        self.epsilon_greedy_steps = epsilon_greedy_steps
        self.q_network = self._create_q_network()
        self.target_q_network = copy.deepcopy(self.q_network).to(ptu.device)
        rlu.nn.functional.freeze(self.target_q_network)
        self.reset_optimizer()
        # define loss function
        self.loss_fn = torch.nn.MSELoss() if huber_delta is None else torch.nn.HuberLoss(delta=huber_delta)
        self.epsilon_greedy_scheduler = self._create_epsilon_greedy_scheduler()

    def reset_optimizer(self):
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.q_lr)

    def _create_q_network(self):
        if len(self.obs_spec.shape) == 1:  # 1D observation
            q_network = rlu.nn.build_mlp(input_dim=self.obs_spec.shape[0], output_dim=self.act_dim,
                                         mlp_hidden=self.mlp_hidden, num_layers=3).to(ptu.device)
        else:
            raise NotImplementedError
        return q_network

    def _create_epsilon_greedy_scheduler(self):
        return rln.schedulers.LinearSchedule(schedule_timesteps=self.epsilon_greedy_steps,
                                             final_p=0.1,
                                             initial_p=0.1)

    def log_tabular(self):
        self.logger.log_tabular('QVals', with_min_and_max=True)
        self.logger.log_tabular('LossQ', average_only=True)

    def update_target(self):
        rlu.functional.soft_update(self.target_q_network, self.q_network, self.tau)

    def sync_target(self):
        rlu.functional.hard_update(self.target_q_network, self.q_network)

    def compute_target_values(self, next_obs, rew, done):
        with torch.no_grad():
            if self.double_q:
                target_actions = torch.argmax(self.q_network(next_obs), dim=-1)  # (None,)
                target_q_values = self.target_q_network(next_obs)  # (None, act_dim)
                target_q_values = gather_q_values(target_q_values, target_actions)
            else:
                target_q_values = self.target_q_network(next_obs)
                target_q_values = torch.max(target_q_values, dim=-1)[0]
            target_q_values = rew + self.gamma * (1. - done) * target_q_values
            return target_q_values

    def _update_nets(self, obs, act, next_obs, rew, done):
        target_q_values = self.compute_target_values(next_obs, rew, done)
        self.q_optimizer.zero_grad()
        q_values = self.q_network(obs)
        q_values = gather_q_values(q_values, act)
        loss = self.loss_fn(q_values, target_q_values)
        loss.backward()
        self.q_optimizer.step()
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
            q_values = self.q_network(obs)
            return torch.argmax(q_values, dim=-1).cpu().numpy()


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
                                agent_cls=DQN,
                                agent_kwargs=agent_kwargs,
                                seed=seed,
                                logger_path=logger_path
                                )


if __name__ == '__main__':
    ptu.set_device('cuda')
    rl_infra.runner.run_func_as_main(Runner.main)
