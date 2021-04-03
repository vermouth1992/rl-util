"""
Implement soft actor critic agent. To make fair comparison with td3, we
1. Add target policy net
2. Delay policy update by 2
3. Use the same hyper-parameter as TD3
"""

import rlutils.tf as rlu
import tensorflow as tf
from rlutils.infra.runner import TFOffPolicyRunner, run_func_as_main


class SACAgent(tf.keras.Model):
    def __init__(self,
                 obs_spec,
                 act_spec,
                 num_ensembles=2,
                 policy_mlp_hidden=256,
                 policy_lr=3e-4,
                 q_mlp_hidden=256,
                 q_lr=3e-4,
                 alpha=1.0,
                 alpha_lr=1e-3,
                 tau=5e-3,
                 gamma=0.99,
                 target_entropy=None,
                 auto_alpha=True,
                 exploration_bonus=True,
                 target_policy=False,
                 ):
        super(SACAgent, self).__init__()
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.act_dim = self.act_spec.shape[0]
        if len(self.obs_spec.shape) == 1:  # 1D observation
            self.obs_dim = self.obs_spec.shape[0]
            self.policy_net = rlu.nn.SquashedGaussianMLPActor(self.obs_dim, self.act_dim, policy_mlp_hidden)
            if target_policy:
                print('Use target policy for SAC')
                self.target_policy_net = rlu.nn.SquashedGaussianMLPActor(self.obs_dim, self.act_dim, policy_mlp_hidden)
            else:
                print('No target policy for SAC')
                self.target_policy_net = None
            self.q_network = rlu.nn.EnsembleMinQNet(self.obs_dim, self.act_dim, q_mlp_hidden,
                                                    num_ensembles=num_ensembles)
            self.target_q_network = rlu.nn.EnsembleMinQNet(self.obs_dim, self.act_dim, q_mlp_hidden,
                                                           num_ensembles=num_ensembles)
        else:
            raise NotImplementedError
        rlu.functional.hard_update(self.target_q_network, self.q_network)
        if self.target_policy_net is not None:
            rlu.functional.hard_update(self.target_policy_net, self.policy_net)

        self.policy_optimizer = tf.keras.optimizers.Adam(lr=policy_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(lr=q_lr)

        self.log_alpha = rlu.nn.LagrangeLayer(initial_value=alpha)
        self.alpha_optimizer = tf.keras.optimizers.Adam(lr=alpha_lr)
        self.target_entropy = -self.act_dim if target_entropy is None else target_entropy
        self.auto_alpha = auto_alpha
        self.exploration_bonus = exploration_bonus

        self.tau = tau
        self.gamma = gamma

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        for i in range(self.q_network.num_ensembles):
            self.logger.log_tabular(f'Q{i + 1}Vals', with_min_and_max=True)
        self.logger.log_tabular('LogPi', average_only=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('TDError', average_only=True)
        self.logger.log_tabular('Alpha', average_only=True)
        self.logger.log_tabular('LossAlpha', average_only=True)

    @tf.function
    def update_target_policy(self):
        rlu.functional.soft_update(self.target_policy_net, self.policy_net, self.tau)

    @tf.function
    def update_target_q(self):
        rlu.functional.soft_update(self.target_q_network, self.q_network, self.tau)

    def _compute_next_obs_q(self, next_obs):
        alpha = self.log_alpha()
        if self.target_policy_net is not None:
            next_action, next_action_log_prob, _, _ = self.target_policy_net((next_obs, tf.constant(False)))
        else:
            next_action, next_action_log_prob, _, _ = self.policy_net((next_obs, tf.constant(False)))
        next_q_values = self.target_q_network((next_obs, next_action, tf.constant(True)))
        if self.exploration_bonus:
            print('Tracing exploration bonus')
            next_q_values = next_q_values - alpha * next_action_log_prob
        return next_q_values

    @tf.function
    def _update_q_nets(self, obs, act, next_obs, done, rew, weights=None):
        # compute target Q values
        next_q_values = self._compute_next_obs_q(next_obs)
        q_target = rlu.functional.compute_target_value(rew, self.gamma, done, next_q_values)
        q_target_ensemble = rlu.functional.expand_ensemble_dim(q_target, num_ensembles=self.q_network.num_ensembles)
        # q loss
        with tf.GradientTape() as q_tape:
            q_values = self.q_network((obs, act, tf.constant(False)))  # (num_ensembles, None)
            q_values_loss = 0.5 * tf.square(q_target_ensemble - q_values)
            # apply importance weights if needed
            if weights is not None:
                weights = rlu.functional.expand_ensemble_dim(weights, num_ensembles=self.q_network.num_ensembles)
                q_values_loss = q_values_loss * weights
            q_values_loss = tf.reduce_mean(q_values_loss, axis=-1)
            # (num_ensembles, None)
            q_values_loss = tf.reduce_sum(q_values_loss, axis=0)

        q_gradients = q_tape.gradient(q_values_loss, self.q_network.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_gradients, self.q_network.trainable_variables))

        self.update_target_q()

        td_error = tf.abs(tf.reduce_min(q_values, axis=0) - q_target)

        info = dict(
            LossQ=q_values_loss,
            TDError=td_error,
        )
        for i in range(self.q_network.num_ensembles):
            info[f'Q{i + 1}Vals'] = q_values[i]
        return info

    @tf.function
    def _update_actor(self, obs, weights=None):
        alpha = self.log_alpha()
        # policy loss
        with tf.GradientTape() as policy_tape:
            action, log_prob, _, _ = self.policy_net((obs, tf.constant(False)))
            q_values_pi_min = self.q_network((obs, action, tf.constant(True)))
            policy_loss = log_prob * alpha - q_values_pi_min
            if weights is not None:
                policy_loss = policy_loss * weights
            policy_loss = tf.reduce_mean(policy_loss, axis=0)
        policy_gradients = policy_tape.gradient(policy_loss, self.policy_net.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy_net.trainable_variables))

        # log alpha
        if self.auto_alpha:
            with tf.GradientTape() as alpha_tape:
                alpha = self.log_alpha()
                alpha_loss = alpha * (tf.stop_gradient(log_prob) + self.target_entropy)
                if weights is not None:
                    alpha_loss = alpha_loss * weights
                alpha_loss = -tf.reduce_mean(alpha_loss, axis=0)
            alpha_gradient = alpha_tape.gradient(alpha_loss, self.log_alpha.trainable_variables)
            self.alpha_optimizer.apply_gradients(zip(alpha_gradient, self.log_alpha.trainable_variables))
        else:
            alpha_loss = 0.

        if self.target_policy_net is not None:
            self.update_target_policy()

        info = dict(
            LogPi=log_prob,
            Alpha=alpha,
            LossAlpha=alpha_loss,
            LossPi=policy_loss,
        )
        return info

    def train_on_batch(self, data, **kwargs):
        update_target = data.pop('update_target')
        obs = data['obs']
        info = self._update_q_nets(**data)
        if update_target:
            actor_info = self._update_actor(obs)
            info.update(actor_info)
        self.logger.store(**rlu.functional.to_numpy_or_python_type(info))
        return info

    @tf.function
    def act_batch_explore_tf(self, obs):
        print(f'Tracing sac act_batch with obs {obs}')
        pi_final = self.policy_net((obs, tf.constant(False)))[0]
        return pi_final

    @tf.function
    def act_batch_test_tf(self, obs):
        pi_final = self.policy_net((obs, tf.constant(True)))[0]
        return pi_final

    @tf.function
    def act_batch_test_tf_v2(self, obs):
        n = 20
        batch_size = tf.shape(obs)[0]
        obs = tf.tile(obs, (n, 1))
        action = self.policy_net((obs, tf.constant(False)))[0]
        q_values_pi_min = self.q_network((obs, action), training=True)[0, :]
        action = tf.reshape(action, shape=(n, batch_size, self.act_dim))
        idx = tf.argmax(tf.reshape(q_values_pi_min, shape=(n, batch_size)), axis=0,
                        output_type=tf.int32)  # (batch_size)
        idx = tf.stack([idx, tf.range(batch_size)], axis=-1)
        pi_final = tf.gather_nd(action, idx)
        return pi_final

    def act_batch_test(self, obs):
        return self.act_batch_test_tf(tf.convert_to_tensor(obs)).numpy()

    def act_batch_explore(self, obs):
        return self.act_batch_explore_tf(tf.convert_to_tensor(obs)).numpy()


class Runner(TFOffPolicyRunner):
    @classmethod
    def main(cls,
             env_name,
             epochs=100,
             # sac args
             policy_mlp_hidden=256,
             policy_lr=3e-4,
             q_mlp_hidden=256,
             q_lr=3e-4,
             policy_delay=1,
             alpha=0.2,
             tau=5e-3,
             gamma=0.99,
             seed=1,
             target_policy=False,
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
            target_entropy=None,
            target_policy=target_policy
        )

        super(Runner, cls).main(
            env_name=env_name,
            epochs=epochs,
            agent_cls=SACAgent,
            agent_kwargs=agent_kwargs,
            policy_delay=policy_delay,
            seed=seed,
            logger_path=logger_path,
            **kwargs
        )


if __name__ == '__main__':
    run_func_as_main(Runner.main)
