"""
Synchronous advantage actor critic
"""

"""
Proximal Policy Optimization
"""

import torch

import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
from rlutils.interface.agent import Agent


class A2CAgent(torch.nn.Module, Agent):
    def __init__(self, env, actor_critic=lambda env: rlu.nn.MLPActorCriticSeparate(env=env),
                 lr=1e-3, entropy_coef=0., value_coef=1.0, max_grad_norm=0.5, device=None,
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
        self.value_normalizer = rlu.preprocessing.StandardScaler(input_shape=())

        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.device = device

        self.to(self.device)

    def log_tabular(self):
        self.logger.log_tabular('PolicyLoss', average_only=True)
        self.logger.log_tabular('ValueLoss', average_only=True)
        self.logger.log_tabular('Entropy', average_only=True)

    def get_value(self, obs):
        obs = torch.as_tensor(obs, device=self.device)
        with torch.no_grad():
            value = self.actor_critic.get_value(obs)
            value = self.value_normalizer(value, inverse=True)
            return ptu.to_numpy(value)

    def act_batch_explore(self, obs, global_steps=None):
        obs = torch.as_tensor(obs, device=self.device)
        pi_distribution, value = self.actor_critic(obs)
        value = self.value_normalizer(value, inverse=True)
        pi_action = pi_distribution.sample()
        log_prob = pi_distribution.log_prob(pi_action)
        return ptu.to_numpy(pi_action), ptu.to_numpy(log_prob), ptu.to_numpy(value)

    def act_batch_test(self, obs):
        obs = torch.as_tensor(obs, device=self.device)
        pi_distribution = self.actor_critic.get_pi_distribution(obs)
        return ptu.to_numpy(pi_distribution.mode)

    def _update_policy_step(self, obs, act, adv, logp, ret):
        self.value_normalizer.adapt(ret)
        with torch.no_grad():
            ret_normalized = self.value_normalizer(ret)

        self.optimizer.zero_grad()

        pi_distribution, value = self.actor_critic(obs)
        entropy = torch.mean(pi_distribution.entropy())
        log_prob = pi_distribution.log_prob(act)
        policy_loss = -torch.mean(adv * log_prob)

        value_loss = torch.nn.functional.mse_loss(value, ret_normalized)
        loss = policy_loss - entropy * self.entropy_coef + value_loss * self.value_coef

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

        self.optimizer.step()

        info = dict(
            PolicyLoss=policy_loss,
            Entropy=entropy,
            ValueLoss=value_loss
        )
        return info

    def train_on_batch(self, data):
        data = ptu.convert_dict_to_tensor(data, device=self.device)
        info = self._update_policy_step(**data)
        self.logger.store(**info)


if __name__ == '__main__':
    from baselines.model_free.trainer import run_onpolicy
    from rlutils.infra.runner import run_func_as_main

    make_agent_fn = lambda env: A2CAgent(env, device=ptu.get_cuda_device())
    run_func_as_main(run_onpolicy, passed_args={
        'make_agent_fn': make_agent_fn,
        'backend': 'torch'
    })
