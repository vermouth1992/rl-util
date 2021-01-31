"""
Twin Delayed DDPG. https://arxiv.org/abs/1802.09477.
To obtain DDPG, set target smooth to zero and Q network ensembles to 1.
"""

import numpy as np
import tensorflow as tf
from rlutils.runner import OffPolicyRunner, TFRunner, run_func_as_main
from rlutils.tf.exploration import OrnsteinUhlenbeckActionNoise
from rlutils.tf.functional import soft_update, hard_update, compute_target_value, to_numpy_or_python_type
from rlutils.tf.nn.functional import build_mlp


class DDPGAgent(tf.keras.Model):
    def __init__(self,
                 obs_spec,
                 act_spec,
                 policy_mlp_hidden=128,
                 policy_lr=3e-4,
                 q_mlp_hidden=256,
                 q_lr=3e-4,
                 tau=5e-3,
                 gamma=0.99,
                 ):
        super(DDPGAgent, self).__init__()
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.act_dim = self.act_spec.shape[0]
        self.act_lim = 1.
        if len(self.obs_spec.shape) == 1:  # 1D observation
            obs_dim = self.obs_spec.shape[0]
        else:
            raise NotImplementedError
        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(shape=self.act_dim))
        self.policy_net = build_mlp(obs_dim, self.act_dim, mlp_hidden=policy_mlp_hidden, num_layers=3,
                                    out_activation='tanh')
        self.target_policy_net = build_mlp(obs_dim, self.act_dim, mlp_hidden=policy_mlp_hidden, num_layers=3,
                                           out_activation='tanh')
        hard_update(self.target_policy_net, self.policy_net)
        self.q_network = build_mlp(obs_dim + self.act_dim, output_dim=1, mlp_hidden=q_mlp_hidden, num_layers=3,
                                   squeeze=True)
        self.target_q_network = build_mlp(obs_dim + self.act_dim, output_dim=1, mlp_hidden=q_mlp_hidden, num_layers=3,
                                          squeeze=True)
        hard_update(self.target_q_network, self.q_network)
        self.policy_optimizer = tf.keras.optimizers.Adam(lr=policy_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(lr=q_lr)
        self.tau = tau
        self.gamma = gamma

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('QVals', with_min_and_max=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)

    @tf.function
    def update_target(self):
        soft_update(self.target_q_network, self.q_network, self.tau)
        soft_update(self.target_policy_net, self.policy_net, self.tau)

    def _compute_next_obs_q(self, next_obs):
        next_action = self.target_policy_net(next_obs)
        next_q_value = self.target_q_network(tf.concat((next_obs, next_action), axis=-1))
        return next_q_value

    @tf.function
    def _update_nets(self, obs, actions, next_obs, done, reward):
        print(f'Tracing _update_nets with obs={obs}, actions={actions}')
        # compute target q
        next_q_value = self._compute_next_obs_q(next_obs)
        q_target = compute_target_value(reward, self.gamma, done, next_q_value)
        # q loss
        with tf.GradientTape() as q_tape:
            q_values = self.q_network(tf.concat((obs, actions), axis=-1))  # (None)
            q_values_loss = 0.5 * tf.square(q_target - q_values)
            q_values_loss = tf.reduce_mean(q_values_loss)
        q_gradients = q_tape.gradient(q_values_loss, self.q_network.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_gradients, self.q_network.trainable_variables))

        info = dict(
            QVals=q_values,
            LossQ=q_values_loss,
        )
        return info

    @tf.function
    def _update_actor(self, obs):
        print(f'Tracing _update_actor with obs={obs}')
        # policy loss
        with tf.GradientTape() as policy_tape:
            a = self.policy_net(obs)
            q = self.q_network(tf.concat((obs, a), axis=-1))
            policy_loss = -tf.reduce_mean(q, axis=0)
        policy_gradients = policy_tape.gradient(policy_loss, self.policy_net.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy_net.trainable_variables))
        info = dict(
            LossPi=policy_loss,
        )
        return info

    def update(self, obs, act, next_obs, done, rew, update_target=True):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        act = tf.convert_to_tensor(act, dtype=tf.float32)
        next_obs = tf.convert_to_tensor(next_obs, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        rew = tf.convert_to_tensor(rew, dtype=tf.float32)

        info = self._update_nets(obs, act, next_obs, done, rew)

        if update_target:
            actor_info = self._update_actor(obs)
            info.update(actor_info)
            self.update_target()

        self.logger.store(**to_numpy_or_python_type(info))

    @tf.function
    def act_batch(self, obs, deterministic):
        print(f'Tracing td3 act_batch with obs {obs}')
        pi_final = self.policy_net(obs)
        if deterministic:
            return pi_final
        else:
            # noise = self.actor_noise()
            noise = tf.random.normal(shape=[tf.shape(obs)[0], self.act_dim], dtype=tf.float32) * 0.1
            pi_final = pi_final + noise
            pi_final = tf.clip_by_value(pi_final, -self.act_lim, self.act_lim)
            return pi_final


class DDPGRunner(OffPolicyRunner, TFRunner):
    def get_action_batch_test(self, obs):
        return self.agent.act_batch(tf.convert_to_tensor(obs, dtype=tf.float32),
                                    tf.convert_to_tensor(True)).numpy()

    def get_action_batch_explore(self, obs):
        return self.agent.act_batch(self.o, deterministic=tf.convert_to_tensor(False)).numpy()


def ddpg(env_name,
         env_fn=None,
         steps_per_epoch=5000,
         epochs=200,
         start_steps=10000,
         update_after=4000,
         update_every=1,
         update_per_step=1,
         batch_size=256,
         num_parallel_env=1,
         num_test_episodes=20,
         seed=1,
         # sac args
         nn_size=256,
         learning_rate=1e-3,
         tau=5e-3,
         gamma=0.99,
         # replay
         replay_size=int(1e6),
         logger_path='data'
         ):
    config = locals()

    runner = DDPGRunner(seed=seed, steps_per_epoch=steps_per_epoch // num_parallel_env, epochs=epochs,
                        exp_name=None, logger_path=logger_path)
    runner.setup_env(env_name=env_name, num_parallel_env=num_parallel_env, env_fn=env_fn,
                     asynchronous=False, num_test_episodes=num_test_episodes)
    runner.setup_seed(seed)
    runner.setup_logger(config=config)
    runner.setup_agent(policy_mlp_hidden=nn_size,
                       policy_lr=learning_rate,
                       q_mlp_hidden=nn_size,
                       q_lr=learning_rate,
                       tau=tau,
                       gamma=gamma)
    runner.setup_extra(start_steps=start_steps,
                       update_after=update_after,
                       update_every=update_every,
                       update_per_step=update_per_step,
                       policy_delay=1
                       )
    runner.setup_replay_buffer(replay_size=replay_size,
                               batch_size=batch_size)

    runner.run()


if __name__ == '__main__':
    run_func_as_main(ddpg)
