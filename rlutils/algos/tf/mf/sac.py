"""
Implement soft actor critic agent here
"""

import time

import numpy as np
import tensorflow as tf
from rlutils.np import DataSpec
from rlutils.replay_buffers import PyUniformParallelEnvReplayBuffer
from rlutils.runner import TFRunner, run_func_as_main
from rlutils.tf.functional import soft_update, hard_update, compute_target_value, to_numpy_or_python_type
from rlutils.tf.nn import LagrangeLayer, SquashedGaussianMLPActor, EnsembleMinQNet, CenteredBetaMLPActor


class SACAgent(tf.keras.Model):
    def __init__(self,
                 obs_spec,
                 act_spec,
                 policy_type='beta',
                 policy_mlp_hidden=128,
                 policy_lr=3e-4,
                 q_mlp_hidden=256,
                 q_lr=3e-4,
                 alpha=1.0,
                 alpha_lr=1e-3,
                 tau=5e-3,
                 gamma=0.99,
                 target_entropy=None,
                 ):
        super(SACAgent, self).__init__()
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.act_dim = self.act_spec.shape[0]
        if len(self.obs_spec.shape) == 1:  # 1D observation
            self.obs_dim = self.obs_spec.shape[0]
            if policy_type == 'gaussian':
                self.policy_net = SquashedGaussianMLPActor(self.obs_dim, self.act_dim, policy_mlp_hidden)
            elif policy_type == 'beta':
                self.policy_net = CenteredBetaMLPActor(self.obs_dim, self.act_dim, policy_mlp_hidden)
            else:
                raise NotImplementedError
            self.q_network = EnsembleMinQNet(self.obs_dim, self.act_dim, q_mlp_hidden)
            self.target_q_network = EnsembleMinQNet(self.obs_dim, self.act_dim, q_mlp_hidden)
        else:
            raise NotImplementedError
        hard_update(self.target_q_network, self.q_network)

        self.policy_optimizer = tf.keras.optimizers.Adam(lr=policy_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(lr=q_lr)

        self.log_alpha = LagrangeLayer(initial_value=alpha)
        self.alpha_optimizer = tf.keras.optimizers.Adam(lr=alpha_lr)
        self.target_entropy = -self.act_dim if target_entropy is None else target_entropy

        self.tau = tau
        self.gamma = gamma

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('Q1Vals', with_min_and_max=True)
        self.logger.log_tabular('Q2Vals', with_min_and_max=True)
        self.logger.log_tabular('LogPi', average_only=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('Alpha', average_only=True)
        self.logger.log_tabular('LossAlpha', average_only=True)

    @tf.function
    def update_target(self):
        soft_update(self.target_q_network, self.q_network, self.tau)

    def _compute_next_obs_q(self, next_obs):
        alpha = self.log_alpha()
        next_action, next_action_log_prob, _, _ = self.policy_net((next_obs, False))
        next_q_values = self.target_q_network((next_obs, next_action), training=False) - alpha * next_action_log_prob
        return next_q_values

    @tf.function
    def _update_nets(self, obs, actions, next_obs, done, reward):
        """ Sample a mini-batch from replay buffer and update the network

        Args:
            obs: (batch_size, ob_dim)
            actions: (batch_size, action_dim)
            next_obs: (batch_size, ob_dim)
            done: (batch_size,)
            reward: (batch_size,)

        Returns: None

        """
        alpha = self.log_alpha()

        # compute target Q values
        next_q_values = self._compute_next_obs_q(next_obs)
        q_target = compute_target_value(reward, self.gamma, done, next_q_values)

        # q loss
        with tf.GradientTape() as q_tape:
            q_values = self.q_network((obs, actions), training=True)  # (num_ensembles, None)
            q_values_loss = 0.5 * tf.square(tf.expand_dims(q_target, axis=0) - q_values)
            # (num_ensembles, None)
            q_values_loss = tf.reduce_sum(q_values_loss, axis=0)  # (None,)
            # apply importance weights
            q_values_loss = tf.reduce_mean(q_values_loss)
        q_gradients = q_tape.gradient(q_values_loss, self.q_network.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_gradients, self.q_network.trainable_variables))

        # policy loss
        with tf.GradientTape() as policy_tape:
            action, log_prob, _, _ = self.policy_net((obs, False))
            q_values_pi_min = self.q_network((obs, action), training=False)
            policy_loss = tf.reduce_mean(log_prob * alpha - q_values_pi_min)
        policy_gradients = policy_tape.gradient(policy_loss, self.policy_net.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy_net.trainable_variables))

        with tf.GradientTape() as alpha_tape:
            alpha = self.log_alpha()
            alpha_loss = -tf.reduce_mean(alpha * (log_prob + self.target_entropy))
        alpha_gradient = alpha_tape.gradient(alpha_loss, self.log_alpha.trainable_variables)
        self.alpha_optimizer.apply_gradients(zip(alpha_gradient, self.log_alpha.trainable_variables))

        info = dict(
            Q1Vals=q_values[0],
            Q2Vals=q_values[1],
            LogPi=log_prob,
            Alpha=alpha,
            LossQ=q_values_loss,
            LossAlpha=alpha_loss,
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
        self.logger.store(**to_numpy_or_python_type(info))

        if update_target:
            self.update_target()

    @tf.function
    def act_batch(self, obs, deterministic):
        print(f'Tracing sac act_batch with obs {obs}')
        pi_final = self.policy_net((obs, deterministic))[0]
        return pi_final

    @tf.function
    def act_batch_test(self, obs):
        n = 20
        batch_size = tf.shape(obs)[0]
        obs = tf.tile(obs, (n, 1))
        action = self.policy_net((obs, False))[0]
        q_values_pi_min = self.q_network((obs, action), training=True)[0, :]
        action = tf.reshape(action, shape=(n, batch_size, self.act_dim))
        idx = tf.argmax(tf.reshape(q_values_pi_min, shape=(n, batch_size)), axis=0,
                        output_type=tf.int32)  # (batch_size)
        idx = tf.stack([idx, tf.range(batch_size)], axis=-1)
        pi_final = tf.gather_nd(action, idx)
        return pi_final


class SACRunner(TFRunner):
    def get_action_batch(self, o):
        return self.agent.act_batch_test(tf.convert_to_tensor(o, dtype=tf.float32)).numpy()

    def setup_replay_buffer(self,
                            replay_size,
                            batch_size):
        data_spec = {
            'obs': DataSpec(shape=self.env.single_observation_space.shape,
                            dtype=np.float32),
            'act': DataSpec(shape=self.env.single_action_space.shape,
                            dtype=np.float32),
            'next_obs': DataSpec(shape=self.env.single_observation_space.shape,
                                 dtype=np.float32),
            'rew': DataSpec(shape=None, dtype=np.float32),
            'done': DataSpec(shape=None, dtype=np.float32)
        }
        self.replay_buffer = PyUniformParallelEnvReplayBuffer(data_spec=data_spec,
                                                              capacity=replay_size,
                                                              batch_size=batch_size,
                                                              num_parallel_env=self.num_parallel_env)

    def setup_agent(self,
                    policy_mlp_hidden=128,
                    policy_lr=3e-4,
                    q_mlp_hidden=256,
                    q_lr=3e-4,
                    alpha=1.0,
                    alpha_lr=1e-3,
                    tau=5e-3,
                    gamma=0.99,
                    target_entropy=None,
                    ):
        obs_spec = tf.TensorSpec(shape=self.env.single_observation_space.shape,
                                 dtype=tf.float32)
        act_spec = tf.TensorSpec(shape=self.env.single_action_space.shape,
                                 dtype=tf.float32)
        self.agent = SACAgent(obs_spec=obs_spec, act_spec=act_spec,
                              policy_mlp_hidden=policy_mlp_hidden,
                              policy_lr=policy_lr,
                              q_mlp_hidden=q_mlp_hidden,
                              q_lr=q_lr,
                              alpha=alpha,
                              alpha_lr=alpha_lr,
                              tau=tau,
                              gamma=gamma,
                              target_entropy=target_entropy,
                              )
        self.agent.set_logger(self.logger)

    def setup_extra(self,
                    start_steps,
                    max_ep_len,
                    update_after,
                    update_every,
                    update_per_step):
        self.start_steps = start_steps
        self.max_ep_len = max_ep_len
        self.update_after = update_after
        self.update_every = update_every
        self.update_per_step = update_per_step

    def run_one_step(self, t):
        global_env_steps = self.global_step * self.num_parallel_env
        if global_env_steps >= self.start_steps:
            a = self.agent.act_batch(self.o, deterministic=tf.convert_to_tensor(False)).numpy()
            assert not np.any(np.isnan(a)), f'NAN action: {a}'
        else:
            a = self.env.action_space.sample()

        # Step the env
        o2, r, d, _ = self.env.step(a)
        self.ep_ret += r
        self.ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        true_d = np.logical_and(d, self.ep_len != self.max_ep_len)

        # Store experience to replay buffer
        self.replay_buffer.add(data={
            'obs': self.o,
            'act': a,
            'rew': r,
            'next_obs': o2,
            'done': true_d
        })

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        self.o = o2

        # End of trajectory handling
        if np.any(d):
            self.logger.store(EpRet=self.ep_ret[d], EpLen=self.ep_len[d])
            self.ep_ret[d] = 0
            self.ep_len[d] = 0
            self.o = self.env.reset_done()

        # Update handling
        if global_env_steps >= self.update_after and global_env_steps % self.update_every == 0:
            for j in range(self.update_every * self.update_per_step):
                batch = self.replay_buffer.sample()
                self.agent.update(**batch, update_target=True)

    def on_epoch_end(self, epoch):
        self.test_agent()

        # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TestEpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', self.global_step * self.num_parallel_env)
        self.agent.log_tabular()
        self.logger.log_tabular('Time', time.time() - self.start_time)
        self.logger.dump_tabular()

    def on_train_begin(self):
        self.start_time = time.time()
        self.o = self.env.reset()
        self.ep_ret = np.zeros(shape=self.num_parallel_env)
        self.ep_len = np.zeros(shape=self.num_parallel_env, dtype=np.int64)


def sac(env_name,
        max_ep_len=1000,
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
        learning_rate=3e-4,
        alpha=0.2,
        tau=5e-3,
        gamma=0.99,
        # replay
        replay_size=int(1e6),
        logger_path='data'
        ):
    config = locals()

    runner = SACRunner(seed=seed, steps_per_epoch=steps_per_epoch // num_parallel_env, epochs=epochs,
                       exp_name=None, logger_path=logger_path)
    runner.setup_env(env_name=env_name, num_parallel_env=num_parallel_env, frame_stack=None, wrappers=None,
                     asynchronous=False, num_test_episodes=num_test_episodes)
    runner.setup_logger(config=config)
    runner.setup_agent(policy_mlp_hidden=nn_size,
                       policy_lr=learning_rate,
                       q_mlp_hidden=nn_size,
                       q_lr=learning_rate,
                       alpha=alpha,
                       alpha_lr=learning_rate,
                       tau=tau,
                       gamma=gamma,
                       target_entropy=None)
    runner.setup_extra(start_steps=start_steps,
                       max_ep_len=max_ep_len,
                       update_after=update_after,
                       update_every=update_every,
                       update_per_step=update_per_step)
    runner.setup_replay_buffer(replay_size=replay_size,
                               batch_size=batch_size)

    runner.run()


if __name__ == '__main__':
    run_func_as_main(sac)
