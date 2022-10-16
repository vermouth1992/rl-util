"""
Proximal Policy Optimization
"""

import numpy as np
import rlutils.pytorch as rlu
import torch

from rlutils.interface.agent import Agent


class PPOAgent(torch.nn.Module, Agent):
    def __init__(self, env,
                 lr=1e-3, clip_ratio=0.2,
                 entropy_coef=0.001, target_kl=0.05,
                 train_iters=80,
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
        super(PPOAgent, self).__init__()
        self.policy_net = rlu.nn.MLPActorCriticSeparate(env=env)
        self.optimizer = torch.optim.Adam(lr=lr, params=self.policy_net.parameters())

        self.target_kl = target_kl
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.train_pi_iters = train_pi_iters
        self.train_vf_iters = train_vf_iters

        self.logger = None

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('PolicyLoss', average_only=True)
        self.logger.log_tabular('ValueLoss', average_only=True)
        self.logger.log_tabular('Entropy', average_only=True)
        self.logger.log_tabular('AvgKL', average_only=True)
        self.logger.log_tabular('StopIter', average_only=True)

    def get_pi_distribution(self, obs, deterministic=tf.convert_to_tensor(False)):
        return self.policy_net((obs, deterministic))[-1]

    def call(self, inputs, training=None, mask=None):
        pi_distribution = self.get_pi_distribution(inputs)
        pi_action = pi_distribution.sample()
        return pi_action

    @tf.function
    def act_batch_tf(self, obs):
        pi_distribution = self.get_pi_distribution(obs)
        pi_action = pi_distribution.sample()
        log_prob = pi_distribution.log_prob(pi_action)
        v = self.value_net(obs)
        return pi_action, log_prob, v

    def act_batch(self, obs):
        pi_action, log_prob, v = self.act_batch_tf(tf.convert_to_tensor(obs))
        return pi_action.numpy(), log_prob.numpy(), v.numpy()

    def act_batch_explore(self, obs, global_steps):
        pass

    def act_batch_test(self, obs):
        pass

    def _update_policy_step(self, obs, act, adv, old_log_prob):
        distribution = self.get_pi_distribution(obs)
        entropy = torch.mean(distribution.entropy())
        log_prob = distribution.log_prob(act)
        negative_approx_kl = log_prob - old_log_prob
        approx_kl_mean = torch.mean(-negative_approx_kl)

        ratio = torch.exp(negative_approx_kl)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, min=1.0 - self.clip_ratio, max=1.0 + self.clip_ratio) * adv
        policy_loss = -torch.mean(torch.minimum(surr1, surr2))

        loss = policy_loss - entropy * self.entropy_coef

        info = dict(
            PolicyLoss=policy_loss,
            Entropy=entropy,
            AvgKL=approx_kl_mean,
        )
        return info

    def train_on_batch(self, obs, act, ret, adv, logp):
        for i in range(self.train_pi_iters):
            info = self._update_policy_step(obs, act, adv, logp)
            if info['AvgKL'] > 1.5 * self.target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break

        self.logger.store(StopIter=i)

        for i in range(self.train_vf_iters):
            loss = self.value_net.train_on_batch(x=obs, y=ret)

        # only record the final result
        info['ValueLoss'] = loss
        self.logger.store(**rlu.functional.to_numpy_or_python_type(info))
