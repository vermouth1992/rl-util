"""
Twin Delayed DDPG. https://arxiv.org/abs/1802.09477.
To obtain DDPG, set target smooth to zero and Q network ensembles to 1.
"""

import rlutils.tf as rlu
import tensorflow as tf
from rlutils.infra.runner import TFOffPolicyRunner


class TD3Agent(tf.keras.Model):
    def __init__(self,
                 obs_spec,
                 act_spec,
                 num_q_ensembles=2,
                 policy_mlp_hidden=256,
                 policy_lr=3e-4,
                 q_mlp_hidden=256,
                 q_lr=3e-4,
                 tau=5e-3,
                 gamma=0.99,
                 actor_noise=0.1,
                 target_noise=0.2,
                 noise_clip=0.5,
                 out_activation='tanh'
                 ):
        super(TD3Agent, self).__init__()
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.act_dim = self.act_spec.shape[0]
        self.act_lim = act_spec.high[0]
        self.actor_noise = actor_noise * self.act_lim
        self.target_noise = target_noise * self.act_lim
        self.noise_clip = noise_clip * self.act_lim
        self.tau = tau
        self.gamma = gamma
        if out_activation == 'sin':
            out_activation = tf.sin
        elif out_activation == 'tanh':
            out_activation = tf.tanh
        else:
            raise ValueError('Unknown output activation function')
        if len(self.obs_spec.shape) == 1:  # 1D observation
            self.obs_dim = self.obs_spec.shape[0]
            self.policy_net = rlu.nn.DeterministicMLPActor(ob_dim=self.obs_dim, ac_dim=self.act_dim,
                                                           mlp_hidden=policy_mlp_hidden,
                                                           out_activation=out_activation)
            self.target_policy_net = rlu.nn.DeterministicMLPActor(ob_dim=self.obs_dim, ac_dim=self.act_dim,
                                                                  mlp_hidden=policy_mlp_hidden,
                                                                  out_activation=out_activation)
            rlu.functional.hard_update(self.target_policy_net, self.policy_net)
            self.q_network = rlu.nn.EnsembleMinQNet(self.obs_dim, self.act_dim, q_mlp_hidden,
                                                    num_ensembles=num_q_ensembles)
            self.target_q_network = rlu.nn.EnsembleMinQNet(self.obs_dim, self.act_dim, q_mlp_hidden,
                                                           num_ensembles=num_q_ensembles)
            rlu.functional.hard_update(self.target_q_network, self.q_network)
        else:
            raise NotImplementedError

        self.policy_optimizer = tf.keras.optimizers.Adam(lr=policy_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(lr=q_lr)

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        for i in range(self.q_network.num_ensembles):
            self.logger.log_tabular(f'Q{i + 1}Vals', with_min_and_max=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)

    @tf.function
    def update_target_q(self):
        rlu.functional.soft_update(self.target_q_network, self.q_network, self.tau)

    @tf.function
    def update_target_policy(self):
        rlu.functional.soft_update(self.target_policy_net, self.policy_net, self.tau)

    def _compute_next_obs_q(self, next_obs):
        next_action = self.target_policy_net(next_obs)
        # Target policy smoothing
        if self.target_noise > 0.:
            epsilon = tf.random.normal(shape=[tf.shape(next_obs)[0], self.act_dim]) * self.target_noise
            epsilon = tf.clip_by_value(epsilon, -self.noise_clip, self.noise_clip)
            next_action = next_action + epsilon
            next_action = tf.clip_by_value(next_action, -self.act_lim, self.act_lim)
        next_q_value = self.target_q_network((next_obs, next_action, tf.constant(True)))
        return next_q_value

    @tf.function
    def _update_q_nets(self, obs, act, next_obs, done, rew):
        print(f'Tracing _update_nets with obs={obs}, actions={act}')
        # compute target q
        next_q_value = self._compute_next_obs_q(next_obs)
        q_target = rlu.functional.compute_target_value(rew, self.gamma, done, next_q_value)
        # q loss
        with tf.GradientTape() as q_tape:
            q_tape.watch(self.q_network.trainable_variables)
            q_values = self.q_network((obs, act, tf.constant(False)))  # (num_ensembles, None)
            q_values_loss = 0.5 * tf.square(tf.expand_dims(q_target, axis=0) - q_values)
            # (num_ensembles, None)
            q_values_loss = tf.reduce_sum(q_values_loss, axis=0)  # (None,)
            # apply importance weights
            q_values_loss = tf.reduce_mean(q_values_loss)
        q_gradients = q_tape.gradient(q_values_loss, self.q_network.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_gradients, self.q_network.trainable_variables))

        self.update_target_q()

        info = dict(
            LossQ=q_values_loss,
        )
        for i in range(self.q_network.num_ensembles):
            info[f'Q{i + 1}Vals'] = q_values[i]
        return info

    @tf.function
    def _update_actor(self, obs):
        print(f'Tracing _update_actor with obs={obs}')
        # policy loss
        with tf.GradientTape() as policy_tape:
            policy_tape.watch(self.policy_net.trainable_variables)
            a = self.policy_net(obs)
            q = self.q_network((obs, a, tf.constant(True)))
            policy_loss = -tf.reduce_mean(q, axis=0)
        policy_gradients = policy_tape.gradient(policy_loss, self.policy_net.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy_net.trainable_variables))

        self.update_target_policy()

        info = dict(
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

    @tf.function
    def act_batch_test_tf(self, obs):
        return self.policy_net(obs)

    @tf.function
    def act_batch_explore_tf(self, obs):
        print('Tracing act_batch_explore')
        pi_final = self.policy_net(obs)
        noise = tf.random.normal(shape=[tf.shape(obs)[0], self.act_dim], dtype=tf.float32) * self.actor_noise
        pi_final_noise = pi_final + noise
        pi_final_noise = tf.clip_by_value(pi_final_noise, -self.act_lim, self.act_lim)
        return pi_final_noise

    def act_batch_test(self, obs):
        return self.act_batch_test_tf(tf.convert_to_tensor(obs)).numpy()

    def act_batch_explore(self, obs):
        return self.act_batch_explore_tf(tf.convert_to_tensor(obs)).numpy()


class Runner(TFOffPolicyRunner):
    @classmethod
    def main(cls,
             env_name,
             epochs=200,
             num_q_ensembles=2,
             policy_mlp_hidden=256,
             policy_lr=1e-3,
             q_mlp_hidden=256,
             q_lr=1e-3,
             actor_noise=0.1,
             target_noise=0.2,
             noise_clip=0.5,
             out_activation='sin',
             tau=5e-3,
             gamma=0.99,
             seed=1,
             logger_path: str = None,
             **kwargs
             ):
        agent_kwargs = dict(
            num_q_ensembles=num_q_ensembles,
            policy_mlp_hidden=policy_mlp_hidden,
            policy_lr=policy_lr,
            q_mlp_hidden=q_mlp_hidden,
            q_lr=q_lr,
            tau=tau,
            gamma=gamma,
            actor_noise=actor_noise,
            target_noise=target_noise,
            noise_clip=noise_clip,
            out_activation=out_activation,
        )

        super(Runner, cls).main(env_name=env_name,
                                epochs=epochs,
                                policy_delay=2,
                                agent_cls=TD3Agent,
                                agent_kwargs=agent_kwargs,
                                seed=seed,
                                logger_path=logger_path,
                                **kwargs)
