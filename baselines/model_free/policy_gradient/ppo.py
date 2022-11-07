"""
Proximal Policy Optimization
"""

import torch

import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
from rlutils.interface.agent import Agent

from rlutils.replay_buffers import UniformReplayBuffer


class PPOAgent(torch.nn.Module, Agent):
    def __init__(self, env, actor_critic=lambda env: rlu.nn.MLPActorCriticSeparate(env=env),
                 lr=1e-3, clip_ratio=0.2,
                 entropy_coef=0.001, value_coef=1.0, target_kl=0.05,
                 train_iters=80, device=None,
                 ):
        """
        Args:
            policy_net: The policy net must implement following methods:
                - forward: takes obs and return action_distribution and value
                - forward_action: takes obs and return action_distribution
                - forward_value: takes obs and return value.
            The advantage is that we can save computation if we only need to fetch parts of the graph. Also, we can
            implement policy and value in both shared and non-shared way.
            learning_rate:
            lam:
            clip_param:
            entropy_coef:
            target_kl:
            max_grad_norm:
        """
        Agent.__init__(self, env)
        torch.nn.Module.__init__(self)
        self.actor_critic = actor_critic(env=env)
        self.optimizer = torch.optim.Adam(lr=lr, params=self.actor_critic.parameters())

        self.target_kl = target_kl
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.train_iters = train_iters

        self.logger = None
        self.device = device

        self.to(self.device)

    def log_tabular(self):
        self.logger.log_tabular('PolicyLoss', average_only=True)
        self.logger.log_tabular('ValueLoss', average_only=True)
        self.logger.log_tabular('Entropy', average_only=True)
        self.logger.log_tabular('AvgKL', average_only=True)
        self.logger.log_tabular('StopIter', average_only=True)

    def get_value(self, obs):
        obs = torch.as_tensor(obs, device=self.device)
        return ptu.to_numpy(self.actor_critic.get_value(obs))

    def act_batch_explore(self, obs, global_steps=None):
        obs = torch.as_tensor(obs, device=self.device)
        pi_distribution, value = self.actor_critic(obs)
        pi_action = pi_distribution.sample()
        log_prob = pi_distribution.log_prob(pi_action)
        return ptu.to_numpy(pi_action), ptu.to_numpy(log_prob), ptu.to_numpy(value)

    def act_batch_test(self, obs):
        obs = torch.as_tensor(obs, device=self.device)
        pi_distribution = self.actor_critic.get_pi_distribution(obs)
        return ptu.to_numpy(pi_distribution.sample())

    def _update_policy_step(self, obs, act, adv, logp, ret):
        self.optimizer.zero_grad()

        pi_distribution, value = self.actor_critic(obs)
        entropy = torch.mean(pi_distribution.entropy())
        log_prob = pi_distribution.log_prob(act)
        negative_approx_kl = log_prob - logp
        approx_kl_mean = torch.mean(-negative_approx_kl)

        ratio = torch.exp(negative_approx_kl)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, min=1.0 - self.clip_ratio, max=1.0 + self.clip_ratio) * adv
        policy_loss = -torch.mean(torch.minimum(surr1, surr2))

        value_loss = torch.nn.functional.mse_loss(value, ret)
        loss = policy_loss - entropy * self.entropy_coef + value_loss * self.value_coef

        loss.backward()
        self.optimizer.step()

        info = dict(
            PolicyLoss=policy_loss,
            Entropy=entropy,
            AvgKL=approx_kl_mean,
            ValueLoss=value_loss
        )
        return info

    def train_on_batch(self, data):
        i = 0
        info = {}

        dataset = UniformReplayBuffer.from_dataset(data)
        # create dataset
        for i in range(self.train_iters):
            data = dataset.sample(batch_size=64)
            data = ptu.convert_dict_to_tensor(data, device=self.device)
            info = self._update_policy_step(**data)
            if info['AvgKL'] > 1.5 * self.target_kl:
                self.logger.log(f'Early stopping at step {i} due to reaching max kl.')
                break

        self.logger.store(StopIter=i)
        self.logger.store(**info)


if __name__ == '__main__':
    from baselines.model_free.trainer import run_onpolicy
    from rlutils.infra.runner import run_func_as_main

    make_agent_fn = lambda env: PPOAgent(env, device=ptu.get_cuda_device())
    run_func_as_main(run_onpolicy, passed_args={
        'make_agent_fn': make_agent_fn,
        'backend': 'torch'
    })
