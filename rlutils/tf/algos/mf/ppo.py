"""
Proximal Policy Optimization
"""

import numpy as np
import rlutils.tf as rlu
import tensorflow as tf
from rlutils.infra.runner import TFOnPolicyRunner


class PPOAgent(tf.keras.Model):
    def __init__(self, obs_spec, act_spec, mlp_hidden=64,
                 pi_lr=1e-3, vf_lr=1e-3, clip_ratio=0.2,
                 entropy_coef=0.001, target_kl=0.05,
                 train_pi_iters=80, train_vf_iters=80
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
        obs_dim = obs_spec.shape[0]
        if act_spec.dtype == np.int32 or act_spec.dtype == np.int64:
            self.policy_net = rlu.nn.CategoricalActor(obs_dim=obs_dim, act_dim=act_spec.n, mlp_hidden=mlp_hidden)
        else:
            self.policy_net = rlu.nn.NormalActor(obs_dim=obs_dim, act_dim=act_spec.shape[0], mlp_hidden=mlp_hidden)
        self.pi_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_lr)
        self.v_optimizer = tf.keras.optimizers.Adam(learning_rate=vf_lr)
        self.value_net = rlu.nn.build_mlp(input_dim=obs_dim, output_dim=1, squeeze=True, mlp_hidden=mlp_hidden)
        self.value_net.compile(optimizer=self.v_optimizer, loss='mse')

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

    @tf.function
    def _update_policy_step(self, obs, act, adv, old_log_prob):
        print(f'Tracing _update_policy_step with obs={obs}')
        with tf.GradientTape() as tape:
            distribution = self.get_pi_distribution(obs)
            entropy = tf.reduce_mean(distribution.entropy())
            log_prob = distribution.log_prob(act)
            negative_approx_kl = log_prob - old_log_prob
            approx_kl_mean = tf.reduce_mean(-negative_approx_kl)

            ratio = tf.exp(negative_approx_kl)
            surr1 = ratio * adv
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            loss = policy_loss - entropy * self.entropy_coef

        gradients = tape.gradient(loss, self.policy_net.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(gradients, self.policy_net.trainable_variables))

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


class Runner(TFOnPolicyRunner):
    @classmethod
    def main(cls, env_name, mlp_hidden=256, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3,
             train_pi_iters=80, train_vf_iters=80,
             target_kl=0.05, entropy_coef=1e-3, **kwargs):
        agent_kwargs = dict(
            mlp_hidden=mlp_hidden,
            pi_lr=pi_lr, vf_lr=vf_lr, clip_ratio=clip_ratio,
            entropy_coef=entropy_coef, target_kl=target_kl,
            train_pi_iters=train_pi_iters, train_vf_iters=train_vf_iters
        )
        super(Runner, cls).main(
            env_name=env_name,
            agent_cls=PPOAgent,
            agent_kwargs=agent_kwargs,
            **kwargs
        )
