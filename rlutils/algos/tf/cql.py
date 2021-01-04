"""
Implement soft actor critic agent here
"""

import time

import d4rl
import gym
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from rlutils.replay_buffers import PyUniformParallelEnvReplayBuffer
from rlutils.runner import TFRunner, run_func_as_main
from rlutils.tf.distributions import apply_squash_log_prob
from rlutils.tf.functional import soft_update, hard_update, compute_target_value, to_numpy_or_python_type
from rlutils.tf.nn import LagrangeLayer, SquashedGaussianMLPActor, EnsembleMinQNet, CenteredBetaMLPActor

EPS = 1e-6


class CQLAgent(tf.keras.Model):
    def __init__(self,
                 obs_spec,
                 act_spec,
                 policy_type='gaussian',
                 policy_mlp_hidden=128,
                 policy_lr=3e-4,
                 q_mlp_hidden=256,
                 q_lr=3e-4,
                 alpha=1.0,
                 alpha_lr=1e-3,
                 tau=5e-3,
                 gamma=0.99,
                 num_samples=10,
                 cql_threshold=-1,
                 target_entropy=None,
                 ):
        super(CQLAgent, self).__init__()
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.num_samples = num_samples
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
        self.log_cql = LagrangeLayer(initial_value=1.0)
        self.alpha_optimizer = tf.keras.optimizers.Adam(lr=alpha_lr)
        self.cql_alpha_optimizer = tf.keras.optimizers.Adam(lr=alpha_lr)
        self.target_entropy = -self.act_dim if target_entropy is None else target_entropy
        self.cql_threshold = cql_threshold
        self.min_q_weight = 5.

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
        self.logger.log_tabular('AlphaCQL', average_only=True)
        self.logger.log_tabular('AlphaCQLLoss', average_only=True)
        self.logger.log_tabular('DeltaQ1', with_min_and_max=True)
        self.logger.log_tabular('DeltaQ2', with_min_and_max=True)

    def update_target(self):
        soft_update(self.target_q_network, self.q_network, self.tau)

    def _compute_next_obs_q(self, next_obs):
        batch_size = tf.shape(next_obs)[0]
        next_obs = tf.tile(next_obs, multiples=(self.num_samples, 1))  # (None * n, obs_dim)
        next_action, next_action_log_prob, _, _ = self.policy_net((next_obs, False))
        next_q_values = self.target_q_network((next_obs, next_action), training=False)
        next_q_values = tf.reshape(next_q_values, shape=(self.num_samples, batch_size))
        next_q_values = tf.reduce_max(next_q_values, axis=0)  # max backup
        return next_q_values

    def _compute_raw_action(self, actions):
        raw_action = tf.atanh(tf.clip_by_value(actions, -1. + EPS, 1. - EPS))
        return raw_action

    @tf.function
    def _update_nets(self, obs, actions, next_obs, done, reward, behavior_cloning=False):
        """ Sample a mini-batch from replay buffer and update the network

        Args:
            obs: (batch_size, ob_dim)
            actions: (batch_size, action_dim)
            next_obs: (batch_size, ob_dim)
            done: (batch_size,)
            reward: (batch_size,)

        Returns: None

        """
        batch_size = tf.shape(obs)[0]
        alpha = self.log_alpha()

        # compute target Q values
        next_q_values = self._compute_next_obs_q(next_obs)
        q_target = compute_target_value(reward, self.gamma, done, next_q_values)

        # generate additional actions for CQL
        random_actions = tf.random.uniform(shape=[batch_size * self.num_samples, self.act_dim],
                                           minval=-1., maxval=1., dtype=tf.float32)
        log_prob_random = -np.log(2.)  # uniform distribution from [-1, 1], prob=0.5

        raw_pi_distribution = self.policy_net((obs, False))[-1]
        raw_pi_actions = raw_pi_distribution.sample(self.num_samples)  # (n, None, act_dim)
        pi_actions = tf.tanh(raw_pi_actions)
        pi_log_prob = apply_squash_log_prob(raw_log_prob=raw_pi_distribution.log_prob(raw_pi_actions),
                                            x=raw_pi_actions)  # (n, None)

        pi_actions = tf.reshape(pi_actions, shape=(self.num_samples * batch_size, self.act_dim))
        pi_log_prob = tf.reshape(pi_log_prob, shape=(self.num_samples * batch_size,))
        pi_log_prob = tf.expand_dims(pi_log_prob, axis=0)  # (1, n * None)

        # q loss
        with tf.GradientTape() as q_tape:
            q_values = self.q_network((obs, actions), training=True)  # (num_ensembles, None)
            q_values_loss = 0.5 * tf.square(tf.expand_dims(q_target, axis=0) - q_values)  # (num_ensembles, None)

            # CQL loss logsumexp(Q(s_i, a)) - Q(s_i, a_i). Importance sampling
            obs_tile = tf.tile(obs, multiples=(self.num_samples, 1))  # (n * None, obs_dim)
            q_random = self.q_network((obs_tile, random_actions),
                                      training=True) - log_prob_random  # (num_ensembles, n * None)
            q_pi = self.q_network((obs_tile, pi_actions), training=True) - pi_log_prob  # (num_ensembles, n * None)

            q_random = tf.reshape(q_random, shape=(self.q_network.num_ensembles, self.num_samples, batch_size))
            q_pi = tf.reshape(q_pi, shape=(self.q_network.num_ensembles, self.num_samples, batch_size))
            q = tf.concat((q_random, q_pi), axis=1)  # (num_ensembles, 2n, None)
            q = tf.math.reduce_logsumexp(q, axis=1)  # (num_ensembles, None)

            # the out-of-distribution Q should not be greater than in-distribution Q by threshold
            delta_q = q - q_values - self.cql_threshold  # (num_ensembles, None)

            alpha_cql = self.log_cql()
            q_values_loss = tf.reduce_mean(tf.reduce_sum(q_values_loss + alpha_cql * delta_q, axis=0), axis=0)

        q_gradients = q_tape.gradient(q_values_loss, self.q_network.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_gradients, self.q_network.trainable_variables))

        with tf.GradientTape() as cql_tape:
            alpha_cql = self.log_cql()
            alpha_cql_loss = -tf.reduce_mean(tf.reduce_sum(alpha_cql * delta_q, axis=0), axis=0)
        alpha_cql_gradients = cql_tape.gradient(alpha_cql_loss, self.log_cql.trainable_variables)
        self.cql_alpha_optimizer.apply_gradients(zip(alpha_cql_gradients, self.log_cql.trainable_variables))

        # policy loss
        with tf.GradientTape() as policy_tape:
            if behavior_cloning:
                _, log_prob, _, pi_distribution = self.policy_net((obs, False))
                raw_action = self._compute_raw_action(actions)
                policy_loss = tf.reduce_mean(-pi_distribution.log_prob(raw_action), axis=0)
            else:
                action, log_prob, _, _ = self.policy_net((obs, False))
                q_values_pi_min = self.q_network((obs, action), training=False)
                policy_loss = tf.reduce_mean(log_prob * alpha - q_values_pi_min, axis=0)
        policy_gradients = policy_tape.gradient(policy_loss, self.policy_net.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy_net.trainable_variables))

        if not behavior_cloning:
            with tf.GradientTape() as alpha_tape:
                alpha = self.log_alpha()
                alpha_loss = -tf.reduce_mean(alpha * (log_prob + self.target_entropy))
            alpha_gradient = alpha_tape.gradient(alpha_loss, self.log_alpha.trainable_variables)
            self.alpha_optimizer.apply_gradients(zip(alpha_gradient, self.log_alpha.trainable_variables))
        else:
            alpha_loss = 0.

        info = dict(
            Q1Vals=q_values[0],
            Q2Vals=q_values[1],
            LogPi=log_prob,
            Alpha=alpha,
            LossQ=q_values_loss,
            LossAlpha=alpha_loss,
            LossPi=policy_loss,
            AlphaCQL=alpha_cql,
            AlphaCQLLoss=alpha_cql_loss,
            DeltaQ1=delta_q[0],
            DeltaQ2=delta_q[1],
        )
        return info

    def update(self, obs, act, next_obs, done, rew, update_target=True, behavior_cloning=False):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        act = tf.convert_to_tensor(act, dtype=tf.float32)
        next_obs = tf.convert_to_tensor(next_obs, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        rew = tf.convert_to_tensor(rew, dtype=tf.float32)

        info = self._update_nets(obs, act, next_obs, done, rew, behavior_cloning)
        self.logger.store(**to_numpy_or_python_type(info))

        if update_target:
            self.update_target()

    @tf.function
    def act_batch(self, obs, deterministic):
        print(f'Tracing sac act_batch with obs {obs}')
        # batch_size = tf.shape(obs)[0]
        # obs = tf.tile(obs, (self.num_samples, 1))
        # action = self.policy_net((obs, False))[0]
        # q_values_pi_min = self.q_network((obs, action), training=True)[0, :]
        # action = tf.reshape(action, shape=(self.num_samples, batch_size, self.act_dim))
        # idx = tf.argmax(tf.reshape(q_values_pi_min, shape=(self.num_samples, batch_size)), axis=0,
        #                 output_type=tf.int32)  # (batch_size)
        # idx = tf.stack([idx, tf.range(batch_size)], axis=-1)
        # pi_final = tf.gather_nd(action, idx)
        pi_final = self.policy_net((obs, deterministic))[0]
        return pi_final


class CQLRunner(TFRunner):
    def get_action_batch(self, o, deterministic=False):
        return self.agent.act_batch(tf.convert_to_tensor(o, dtype=tf.float32),
                                    deterministic).numpy()

    def test_agent(self):
        o, d, ep_ret, ep_len = self.env.reset(), np.zeros(shape=self.num_test_episodes, dtype=np.bool), \
                               np.zeros(shape=self.num_test_episodes), \
                               np.zeros(shape=self.num_test_episodes, dtype=np.int64)
        t = tqdm(total=1, desc='Testing')
        while not np.all(d):
            a = self.get_action_batch(o, deterministic=True)
            o, r, d_, _ = self.env.step(a)
            ep_ret = r * (1 - d) + ep_ret
            ep_len = np.ones(shape=self.num_test_episodes, dtype=np.int64) * (1 - d) + ep_len
            d = np.logical_or(d, d_)
        t.update(1)
        t.close()
        normalized_ep_ret = self.dummy_env.get_normalized_score(ep_ret) * 100
        self.logger.store(TestEpRet=ep_ret, NormalizedTestEpRet=normalized_ep_ret, TestEpLen=ep_len)

    def setup_replay_buffer(self, batch_size):
        obs_dim = self.env.single_observation_space.shape[0]
        act_dim = self.env.single_action_space.shape[0]
        self.dummy_env = gym.make(self.env_name)
        dataset = d4rl.qlearning_dataset(env=self.dummy_env)
        # modify keys
        dataset['obs'] = dataset['observations']
        dataset['act'] = dataset['actions']
        dataset['next_obs'] = dataset['next_observations']
        dataset['rew'] = dataset['rewards']
        dataset['done'] = dataset['terminals']
        replay_size = dataset['obs'].shape[0]
        print(f'Dataset size: {replay_size}')
        self.replay_buffer = PyUniformParallelEnvReplayBuffer(obs_dim=obs_dim,
                                                              act_dim=act_dim,
                                                              act_dtype=np.float32,
                                                              capacity=replay_size,
                                                              batch_size=batch_size,
                                                              num_parallel_env=1)
        self.replay_buffer.load(dataset)

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
        self.agent = CQLAgent(obs_spec=obs_spec, act_spec=act_spec,
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
                    start_steps
                    ):
        self.start_steps = start_steps

    def run_one_step(self, t):
        batch = self.replay_buffer.sample()
        self.agent.update(**batch, update_target=True, behavior_cloning=self.global_step <= self.start_steps)

    def on_epoch_end(self, epoch):
        self.test_agent()

        # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        self.logger.log_tabular('TestEpLen', average_only=True)
        self.logger.log_tabular('NormalizedTestEpRet', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', self.global_step * self.num_parallel_env)
        self.agent.log_tabular()
        self.logger.log_tabular('Time', time.time() - self.start_time)
        self.logger.dump_tabular()

    def on_train_begin(self):
        self.start_time = time.time()


def cql(env_name,
        max_ep_len=1000,
        steps_per_epoch=1000,
        epochs=1000,
        start_steps=100000,
        batch_size=256,
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
        ):
    config = locals()

    runner = CQLRunner(seed=seed, steps_per_epoch=steps_per_epoch, epochs=epochs,
                       exp_name=None, logger_path='data')
    runner.setup_env(env_name=env_name, num_parallel_env=num_test_episodes, frame_stack=None, wrappers=None,
                     asynchronous=False, num_test_episodes=None)
    runner.setup_logger(config=config)
    runner.setup_agent(policy_mlp_hidden=nn_size,
                       policy_lr=learning_rate,
                       q_mlp_hidden=nn_size,
                       q_lr=learning_rate,
                       alpha=alpha,
                       alpha_lr=1e-3,
                       tau=tau,
                       gamma=gamma,
                       target_entropy=None)
    runner.setup_extra(start_steps=start_steps)
    runner.setup_replay_buffer(batch_size=batch_size)

    runner.run()


if __name__ == '__main__':
    run_func_as_main(cql)
