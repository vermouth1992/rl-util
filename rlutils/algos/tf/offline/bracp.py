"""
Implement soft actor critic agent here.
1. Full pipeline running
2. Restart from behavior policy
3. Restart from Q_b
"""

import os
import time

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from rlutils.future.optimizer import get_adam_optimizer
from rlutils.generative_models.vae import EnsembleBehaviorPolicy
from rlutils.logx import EpochLogger
from rlutils.np.functional import clip_arctanh
from rlutils.replay_buffers import PyUniformParallelEnvReplayBuffer
from rlutils.runner import TFRunner
from rlutils.tf.functional import soft_update, hard_update, to_numpy_or_python_type, clip_atanh
from rlutils.tf.nn import SquashedGaussianMLPActor, EnsembleMinQNet, LagrangeLayer
from rlutils.tf.nn.functional import build_mlp
from tqdm.auto import tqdm, trange

tfd = tfp.distributions


class BRACPAgent(tf.keras.Model):
    def __init__(self,
                 ob_dim,
                 ac_dim,
                 num_ensembles=5,
                 behavior_mlp_hidden=256,
                 behavior_lr=1e-4,
                 policy_mlp_hidden=128,
                 q_mlp_hidden=256,
                 q_lr=3e-4,
                 alpha_lr=1e-2,
                 alpha=1.0,
                 alpha_mlp_hidden=64,
                 tau=5e-3,
                 gamma=0.99,
                 target_entropy=None,
                 huber_delta=None,
                 alpha_update='nn',
                 gp_type='hard',
                 reg_type='kl',
                 sigma=10,
                 n=5,
                 gp_weight=0.1,
                 entropy_reg=True,
                 kl_backup=False,
                 ):
        super(BRACPAgent, self).__init__()
        self.reg_type = reg_type
        assert self.reg_type in ['kl', 'mmd']

        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.q_mlp_hidden = q_mlp_hidden
        self.behavior_policy = EnsembleBehaviorPolicy(num_ensembles=num_ensembles,
                                                      obs_dim=self.ob_dim, act_dim=self.ac_dim,
                                                      mlp_hidden=behavior_mlp_hidden)
        self.behavior_lr = behavior_lr
        # maybe overwrite later
        self.policy_net = SquashedGaussianMLPActor(ob_dim, ac_dim, policy_mlp_hidden)
        self.target_policy_net = SquashedGaussianMLPActor(ob_dim, ac_dim, policy_mlp_hidden)
        hard_update(self.target_policy_net, self.policy_net)
        self.q_network = EnsembleMinQNet(ob_dim, ac_dim, q_mlp_hidden)
        self.q_network.compile(optimizer=get_adam_optimizer(q_lr))
        self.target_q_network = EnsembleMinQNet(ob_dim, ac_dim, q_mlp_hidden)
        hard_update(self.target_q_network, self.q_network)

        self.log_alpha = LagrangeLayer(initial_value=alpha)
        self.log_alpha.compile(optimizer=get_adam_optimizer(1e-3))
        self.log_beta = LagrangeLayer(initial_value=alpha)
        self.log_beta.compile(optimizer=get_adam_optimizer(1e-3))
        self.alpha_net = build_mlp(input_dim=self.ob_dim, output_dim=1, mlp_hidden=alpha_mlp_hidden,
                                   squeeze=True, out_activation='softplus')
        self.alpha_net.compile(optimizer=get_adam_optimizer(alpha_lr))
        self.target_entropy = -ac_dim if target_entropy is None else target_entropy
        self.huber_delta = huber_delta
        self.alpha_update = alpha_update

        self.tau = tau
        self.gamma = gamma

        self.kl_n = 5
        self.n = n
        self.max_q_backup = True
        self.entropy_reg = entropy_reg
        self.kl_backup = kl_backup
        self.gradient_clipping = False
        self.gp_weight = gp_weight
        self.sensitivity = 1.0
        self.gp_type = gp_type
        assert self.gp_type in ['hard', 'sigmoid', 'none', 'softplus']
        self.sigma = sigma

        # delta should set according to the KL between initial policy and behavior policy
        self.delta_behavior = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32)
        self.delta_gp = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32)

    def get_alpha(self, obs):
        if self.alpha_update == 'nn':
            return self.alpha_net(obs)
        elif self.alpha_update == 'global':
            return self.log_alpha(obs)
        elif self.alpha_update == 'fixed':
            return self.log_alpha(obs)
        else:
            raise NotImplementedError

    def call(self, inputs, training=None, mask=None):
        obs, deterministic = inputs
        pi_final = self.policy_net((obs, deterministic))[0]
        return pi_final

    def set_delta_behavior(self, delta_behavior):
        EpochLogger.log(f'Setting behavior hard KL to {delta_behavior:.4f}')
        self.delta_behavior.assign(delta_behavior)

    def set_delta_gp(self, delta_gp):
        EpochLogger.log(f'Setting delta GP to {delta_gp:.4f}')
        self.delta_gp.assign(delta_gp)

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

        self.logger.log_tabular('KL', with_min_and_max=True)
        self.logger.log_tabular('ViolationRatio', average_only=True)
        self.logger.log_tabular('Beta', average_only=True)
        self.logger.log_tabular('BetaLoss', average_only=True)
        self.logger.log_tabular('BehaviorLoss', average_only=True)
        self.logger.log_tabular('GP', average_only=True)

    def update_target(self):
        soft_update(self.target_q_network, self.q_network, self.tau)
        soft_update(self.target_policy_net, self.policy_net, self.tau)

    @tf.function
    def compute_pi_pib_distance(self, obs):
        if self.reg_type == 'kl':
            _, log_prob, raw_action, pi_distribution = self.policy_net((obs, False))
            loss = self._compute_kl_behavior_v2(obs, raw_action, pi_distribution)
        elif self.reg_type == 'mmd':
            batch_size = tf.shape(obs)[0]
            obs = tf.tile(obs, (self.n, 1))
            _, log_prob, raw_action, pi_distribution = self.policy_net((obs, False))
            loss = self._compute_mmd(obs, raw_action, pi_distribution)
            log_prob = tf.reduce_mean(tf.reshape(log_prob, shape=(self.n, batch_size)), axis=0)
        else:
            raise NotImplementedError
        return loss, log_prob

    def mmd_loss_laplacian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Laplacian kernel for support matching"""
        # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
        # (n, None, ac_dim)
        diff_x_x = tf.expand_dims(samples1, axis=0) - tf.expand_dims(samples1, axis=1)  # (n, n, None, ac_dim)
        diff_x_x = tf.reduce_mean(tf.exp(-tf.reduce_sum(tf.abs(diff_x_x), axis=-1) / (2.0 * sigma)), axis=(0, 1))

        diff_x_y = tf.expand_dims(samples1, axis=0) - tf.expand_dims(samples2, axis=1)
        diff_x_y = tf.reduce_mean(tf.exp(-tf.reduce_sum(tf.abs(diff_x_y), axis=-1) / (2.0 * sigma)), axis=(0, 1))

        diff_y_y = tf.expand_dims(samples2, axis=0) - tf.expand_dims(samples2, axis=1)  # (n, n, None, ac_dim)
        diff_y_y = tf.reduce_mean(tf.exp(-tf.reduce_sum(tf.abs(diff_y_y), axis=-1) / (2.0 * sigma)), axis=(0, 1))
        overall_loss = tf.sqrt(diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6)  # (None,)
        return overall_loss

    def _compute_mmd(self, obs, raw_action, pi_distribution):
        # obs: (n * None, obs_dim), raw_actions: (n * None, ac_dim)
        batch_size = tf.shape(obs)[0] // self.n
        num_ensembles = self.behavior_policy.num_ensembles
        samples_pi = raw_action
        samples_pi = self.behavior_policy.expand_ensemble_dim(samples_pi)  # (num_ensembles, n * None, act_dim)
        samples_pi = tf.reshape(samples_pi, shape=(num_ensembles, self.n, batch_size, self.ac_dim))
        samples_pi = tf.transpose(samples_pi, perm=[1, 0, 2, 3])  # (n, ensembles, None, act_dim)
        samples_pi = tf.reshape(samples_pi, shape=(self.n, num_ensembles * batch_size, self.ac_dim))

        obs = self.behavior_policy.expand_ensemble_dim(obs)  # (num_ensembles, n * None, act_dim)
        samples_pi_b = self.behavior_policy.sample(obs, full_path=False)  # (num_ensembles, n * batch_size, d)
        samples_pi_b = tf.reshape(samples_pi_b, shape=(num_ensembles, self.n, batch_size, self.ac_dim))
        samples_pi_b = tf.transpose(samples_pi_b, perm=[1, 0, 2, 3])  # (n, ensembles, None, act_dim)
        samples_pi_b = tf.reshape(samples_pi_b, shape=(self.n, num_ensembles * batch_size, self.ac_dim))

        samples_pi = tf.tanh(samples_pi)
        samples_pi_b = tf.tanh(samples_pi_b)
        mmd_loss = self.mmd_loss_laplacian(samples_pi, samples_pi_b, sigma=self.sigma)
        mmd_loss = tf.reshape(mmd_loss, shape=(num_ensembles, batch_size, self.ac_dim))
        mmd_loss = tf.reduce_mean(mmd_loss, axis=0)
        return mmd_loss

    def _compute_kl_behavior_v2(self, obs, raw_action, pi_distribution):
        n = self.kl_n
        batch_size = tf.shape(obs)[0]
        pi_distribution = tfd.Independent(distribution=tfd.Normal(
            loc=tf.tile(pi_distribution.distribution.loc, (n, 1)),
            scale=tf.tile(pi_distribution.distribution.scale, (n, 1))
        ), reinterpreted_batch_ndims=1)  # (n * batch_size)

        # compute KLD upper bound
        x, cond = raw_action, obs
        print(f'Tracing call_n with x={x}, cond={cond}')
        x = self.behavior_policy.expand_ensemble_dim(x)  # (num_ensembles, None, act_dim)
        cond = self.behavior_policy.expand_ensemble_dim(cond)  # (num_ensembles, None, obs_dim)
        posterior = self.behavior_policy.encode_distribution(inputs=(x, cond))
        encode_sample = posterior.sample(n)  # (n, num_ensembles, None, z_dim)
        encode_sample = tf.transpose(encode_sample, perm=[1, 0, 2, 3])  # (num_ensembles, n, None, z_dim)
        encode_sample = tf.reshape(encode_sample, shape=(self.behavior_policy.num_ensembles,
                                                         n * batch_size,
                                                         self.behavior_policy.latent_dim))
        cond = tf.tile(cond, multiples=(1, n, 1))  # (num_ensembles, n * None, obs_dim)
        beta_distribution = self.behavior_policy.decode_distribution(z=(encode_sample, cond))
        posterior_kld = tfd.kl_divergence(posterior, self.behavior_policy.prior)  # (num_ensembles, None,)
        posterior_kld = tf.tile(posterior_kld, multiples=(1, n,))
        kl_loss = tfd.kl_divergence(pi_distribution, beta_distribution)  # (ensembles, n * None)
        final_kl_loss = kl_loss + posterior_kld  # (ensembles, None * n)
        final_kl_loss = tf.reshape(final_kl_loss, shape=(self.behavior_policy.num_ensembles, n, batch_size))
        final_kl_loss = tf.reduce_mean(final_kl_loss, axis=[0, 1])  # average both latent and ensemble dimension
        return final_kl_loss

    @tf.function
    def update_actor_first_order(self, obs):
        # TODO: maybe we just follow behavior policy and keep a minimum entropy instead of the optimal one.
        # policy loss
        with tf.GradientTape() as policy_tape, tf.GradientTape() as alpha_tape, tf.GradientTape() as beta_tape:
            """ Compute the loss function of the policy that maximizes the Q function """
            print(f'Tracing _compute_surrogate_loss_pi with obs={obs}')

            policy_tape.watch(self.policy_net.trainable_variables)
            alpha_tape.watch(self.alpha_net.trainable_variables)
            beta_tape.watch(self.log_beta.trainable_variables)

            batch_size = tf.shape(obs)[0]
            alpha = self.get_alpha(obs)  # (None, act_dim)
            beta = self.log_beta(obs)

            obs = tf.tile(obs, (self.n, 1))

            # policy loss
            action, log_prob, raw_action, pi_distribution = self.policy_net((obs, False))
            log_prob = tf.reduce_mean(tf.reshape(log_prob, shape=(self.n, batch_size)), axis=0)
            q_values_pi_min = self.q_network((obs, action), training=False)
            q_values_pi_min = tf.reduce_mean(tf.reshape(q_values_pi_min, shape=(self.n, batch_size)), axis=0)
            # add KL divergence penalty, high variance?
            if self.reg_type == 'kl':
                kl_loss = self._compute_kl_behavior_v2(obs, raw_action, pi_distribution)  # (None, act_dim)
                kl_loss = tf.reduce_mean(tf.reshape(kl_loss, shape=(self.n, batch_size)), axis=0)
            elif self.reg_type == 'mmd':
                kl_loss = self._compute_mmd(obs, raw_action, pi_distribution)
            else:
                raise NotImplementedError

            delta = kl_loss - self.delta_behavior
            penalty = delta * alpha  # (None, act_dim)

            if self.reg_type == 'kl':
                if self.entropy_reg:
                    policy_loss = tf.reduce_mean(- q_values_pi_min + penalty - beta * log_prob, axis=0)
                else:
                    policy_loss = tf.reduce_mean(- q_values_pi_min + penalty, axis=0)
            elif self.reg_type in ['mmd']:
                if self.entropy_reg:
                    policy_loss = tf.reduce_mean(- q_values_pi_min + penalty + beta * log_prob, axis=0)
                else:
                    policy_loss = tf.reduce_mean(- q_values_pi_min + penalty, axis=0)
            else:
                raise NotImplementedError

            # alpha loss
            alpha_loss = -tf.reduce_mean(penalty, axis=0)
            # beta loss
            if self.reg_type == 'kl':
                beta_loss = tf.reduce_mean(beta * (log_prob + self.target_entropy))
            elif self.reg_type in ['mmd']:
                beta_loss = -tf.reduce_mean(beta * (log_prob + self.target_entropy))
            else:
                raise NotImplementedError

        policy_gradients = policy_tape.gradient(policy_loss, self.policy_net.trainable_variables)
        self.policy_net.optimizer.apply_gradients(zip(policy_gradients, self.policy_net.trainable_variables))

        if self.alpha_update == 'nn':
            alpha_gradient = alpha_tape.gradient(alpha_loss, self.alpha_net.trainable_variables)
            self.alpha_net.optimizer.apply_gradients(zip(alpha_gradient, self.alpha_net.trainable_variables))
        elif self.alpha_update == 'global':
            alpha_gradient = alpha_tape.gradient(alpha_loss, self.log_alpha.trainable_variables)
            self.log_alpha.optimizer.apply_gradients(zip(alpha_gradient, self.log_alpha.trainable_variables))
        else:
            raise NotImplementedError

        if self.entropy_reg:
            beta_gradient = beta_tape.gradient(beta_loss, self.log_beta.trainable_variables)
            self.log_beta.optimizer.apply_gradients(zip(beta_gradient, self.log_beta.trainable_variables))

        info = dict(
            LossPi=policy_loss,
            KL=kl_loss,
            ViolationRatio=tf.reduce_mean(tf.cast(delta > 0., dtype=tf.float32), axis=-1),
            Alpha=alpha,
            LossAlpha=alpha_loss,
            Beta=beta,
            BetaLoss=beta_loss,
            LogPi=log_prob,
        )

        return info

    @tf.function
    def update_actor_cloning(self, obs):
        """ Minimize KL(pi, pi_b) """
        with tf.GradientTape() as policy_tape, tf.GradientTape() as beta_tape:
            policy_tape.watch(self.policy_net.trainable_variables)
            beta_tape.watch(self.log_beta.trainable_variables)
            beta = self.log_beta(obs)
            loss, log_prob = self.compute_pi_pib_distance(obs)
            if self.entropy_reg:
                if self.reg_type == 'kl':
                    policy_loss = tf.reduce_mean(loss - beta * log_prob, axis=0)
                elif self.reg_type == 'mmd':
                    policy_loss = tf.reduce_mean(loss + beta * log_prob, axis=0)
                else:
                    raise NotImplementedError
            else:
                policy_loss = tf.reduce_mean(loss, axis=0)
            beta_loss = tf.reduce_mean(beta * (log_prob + self.target_entropy), axis=0)

        policy_gradients = policy_tape.gradient(policy_loss, self.policy_net.trainable_variables)
        self.policy_net.optimizer.apply_gradients(zip(policy_gradients, self.policy_net.trainable_variables))

        if self.entropy_reg:
            beta_gradient = beta_tape.gradient(beta_loss, self.log_beta.trainable_variables)
            self.log_beta.optimizer.apply_gradients(zip(beta_gradient,
                                                        self.log_beta.trainable_variables))

        info = dict(
            KL=loss,
            LogPi=log_prob,
        )
        return info

    def _compute_target_q(self, next_obs, reward, done):
        batch_size = tf.shape(next_obs)[0]
        alpha = self.get_alpha(next_obs)
        if self.max_q_backup is True:
            next_obs = tf.tile(next_obs, multiples=(self.n, 1))
        next_action, next_action_log_prob, next_raw_action, pi_distribution = self.target_policy_net((next_obs, False))
        target_q_values = self.target_q_network((next_obs, next_action), training=False)
        if self.max_q_backup is True:
            target_q_values = tf.reduce_mean(tf.reshape(target_q_values, shape=(self.n, batch_size)), axis=0)
            if self.kl_backup is True:
                kl_loss = self._compute_kl_behavior_v2(next_obs, next_raw_action, pi_distribution)  # (None, act_dim)
                kl_loss = tf.reduce_mean(tf.reshape(kl_loss, shape=(self.n, batch_size)), axis=0)
                target_q_values = target_q_values - alpha * (kl_loss - self.delta_behavior)
        else:
            if self.kl_backup is True:
                kl_loss = self._compute_kl_behavior_v2(next_obs, next_raw_action, pi_distribution)  # (None, act_dim)
                target_q_values = target_q_values - alpha * tf.minimum(kl_loss, self.max_kl_backup)

        q_target = reward + self.gamma * (1.0 - done) * target_q_values
        return q_target

    def _compute_q_net_gp(self, obs):
        batch_size = tf.shape(obs)[0]
        if self.reg_type == 'kl':
            action, log_prob, raw_action, pi_distribution = self.policy_net((obs, False))
            kl = self._compute_kl_behavior_v2(obs, raw_action, pi_distribution)  # (None,)
        elif self.reg_type == 'mmd':
            obs = tf.tile(obs, (self.n, 1))
            action, log_prob, raw_action, pi_distribution = self.policy_net((obs, False))
            kl = self._compute_mmd(obs, raw_action, pi_distribution)
        else:
            raise NotImplementedError

        with tf.GradientTape() as inner_tape:
            inner_tape.watch(action)
            q_values = self.q_network((obs, action), training=False)  # (num_ensembles, None)
        input_gradient = inner_tape.gradient(q_values, action)  # (None, act_dim)
        penalty = tf.norm(input_gradient, axis=-1)  # (None,)
        if self.reg_type == 'mmd':
            penalty = tf.reshape(penalty, shape=(self.n, batch_size))
            penalty = tf.reduce_mean(penalty, axis=0)
        # TODO: consider using soft constraints instead of hard clip
        if self.gp_type == 'hard':
            penalty = penalty * tf.cast((kl - self.delta_gp) > 0, dtype=tf.float32)
        elif self.gp_type == 'sigmoid':
            penalty = penalty * tf.nn.sigmoid((kl - self.delta_gp) * self.sensitivity)
        elif self.gp_type == 'softplus':
            penalty = penalty * tf.nn.softplus((kl - self.delta_gp) * self.sensitivity)
        elif self.gp_type == 'none':
            penalty = tf.zeros(shape=[1], dtype=tf.float32)
        else:
            raise NotImplementedError
        penalty = tf.reduce_mean(penalty, axis=0) * self.gp_weight
        return penalty

    def _update_q_nets(self, obs, actions, q_target):
        # q loss
        with tf.GradientTape() as q_tape:
            q_values = self.q_network((obs, actions), training=True)  # (num_ensembles, None)
            q_values_loss = 0.5 * tf.square(tf.expand_dims(q_target, axis=0) - q_values)
            # (num_ensembles, None)
            q_values_loss = tf.reduce_sum(q_values_loss, axis=0)  # (None,)
            # apply importance weights
            q_values_loss = tf.reduce_mean(q_values_loss)

            gp = self._compute_q_net_gp(obs)
            loss = q_values_loss + gp

        q_gradients = q_tape.gradient(loss, self.q_network.trainable_variables)
        self.q_network.optimizer.apply_gradients(zip(q_gradients, self.q_network.trainable_variables))

        info = dict(
            Q1Vals=q_values[0],
            Q2Vals=q_values[1],
            LossQ=q_values_loss,
            GP=gp,
        )
        return info

    @tf.function
    def update_q_nets(self, obs, actions, next_obs, done, reward):
        """Normal SAC update"""
        q_target = self._compute_target_q(next_obs, reward, done)
        return self._update_q_nets(obs, actions, q_target)

    @tf.function
    def _update(self, obs, act, obs2, done, rew):
        raw_act = clip_atanh(act)
        behavior_loss = self.behavior_policy.train_on_batch(x=(raw_act, obs))['loss']
        info = self.update_q_nets(obs, act, obs2, done, rew)
        actor_info = self.update_actor_first_order(obs)
        self.update_target()
        # we only update alpha when policy is updated
        info.update(actor_info)
        info['BehaviorLoss'] = behavior_loss
        return info

    def update(self, replay_buffer: PyUniformParallelEnvReplayBuffer):
        # TODO: use different batches to update q and actor to break correlation
        data = replay_buffer.sample()
        info = self._update(**data)
        self.logger.store(**to_numpy_or_python_type(info))

    @tf.function
    def act_batch(self, obs, deterministic=True):
        print(f'Tracing act_batch with obs {obs}')
        if deterministic:
            pi_final, log_prob, raw_action, pi_distribution = self.policy_net((obs, deterministic))
        else:
            n = 20
            batch_size = tf.shape(obs)[0]
            obs = tf.tile(obs, (n, 1))
            action = self.policy_net((obs, False))[0]
            q_values_pi_min = self.q_network((obs, action), training=True)[0, :]
            action = tf.reshape(action, shape=(n, batch_size, self.ac_dim))
            idx = tf.argmax(tf.reshape(q_values_pi_min, shape=(n, batch_size)), axis=0,
                            output_type=tf.int32)  # (batch_size)
            idx = tf.stack([idx, tf.range(batch_size)], axis=-1)
            pi_final = tf.gather_nd(action, idx)
        return pi_final


class BRACPRunner(TFRunner):
    def get_action_batch(self, o, deterministic=False):
        return self.agent.act_batch(tf.convert_to_tensor(o, dtype=tf.float32),
                                    deterministic).numpy()

    def test_agent(self, agent, name, logger=None):
        o, d, ep_ret, ep_len = self.env.reset(), np.zeros(shape=self.num_test_episodes, dtype=np.bool), \
                               np.zeros(shape=self.num_test_episodes), np.zeros(shape=self.num_test_episodes,
                                                                                dtype=np.int64)
        t = tqdm(total=1, desc=f'Testing {name}')
        while not np.all(d):
            a = agent.act_batch(tf.convert_to_tensor(o, dtype=tf.float32),
                                tf.convert_to_tensor(False)).numpy()
            assert not np.any(np.isnan(a)), f'nan action: {a}'
            o, r, d_, _ = self.env.step(a)
            ep_ret = r * (1 - d) + ep_ret
            ep_len = np.ones(shape=self.num_test_episodes, dtype=np.int64) * (1 - d) + ep_len
            d = np.logical_or(d, d_)
        t.update(1)
        t.close()
        normalized_ep_ret = self.dummy_env.get_normalized_score(ep_ret) * 100

        if logger is not None:
            logger.store(TestEpRet=ep_ret, NormalizedTestEpRet=normalized_ep_ret, TestEpLen=ep_len)
        else:
            print(f'EpRet: {np.mean(ep_ret):.2f}, TestEpLen: {np.mean(ep_len):.2f}')

    def setup_replay_buffer(self,
                            batch_size,
                            reward_scale=True):
        import d4rl
        def rescale(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x))

        self.dummy_env = gym.make(self.env_name)
        dataset = d4rl.qlearning_dataset(env=self.dummy_env)

        if reward_scale:
            EpochLogger.log('Using reward scale', color='red')
            dataset['rewards'] = rescale(dataset['rewards'])
        # modify keys
        dataset['obs'] = dataset.pop('observations')
        dataset['act'] = dataset.pop('actions')
        dataset['obs2'] = dataset.pop('next_observations')
        dataset['rew'] = dataset.pop('rewards')
        dataset['done'] = dataset.pop('terminals').astype(np.float32)
        replay_size = dataset['obs'].shape[0]
        self.logger.log(f'Dataset size: {replay_size}')
        self.replay_buffer = PyUniformParallelEnvReplayBuffer.from_data_dict(
            data=dataset,
            batch_size=batch_size
        )

    def setup_agent(self,
                    num_ensembles,
                    behavior_mlp_hidden,
                    behavior_lr,
                    policy_mlp_hidden,
                    q_mlp_hidden,
                    alpha_mlp_hidden,
                    policy_lr,
                    q_lr,
                    alpha_lr,
                    alpha,
                    tau,
                    gamma,
                    target_entropy,
                    huber_delta,
                    gp_type,
                    alpha_update,
                    policy_behavior_lr,
                    reg_type,
                    sigma,
                    n,
                    gp_weight,
                    entropy_reg,
                    kl_backup
                    ):
        obs_dim = self.env.single_observation_space.shape[-1]
        act_dim = self.env.single_action_space.shape[-1]
        self.policy_lr = policy_lr
        self.policy_behavior_lr = policy_behavior_lr
        self.agent = BRACPAgent(ob_dim=obs_dim, ac_dim=act_dim, num_ensembles=num_ensembles,
                                behavior_mlp_hidden=behavior_mlp_hidden,
                                behavior_lr=behavior_lr,
                                policy_mlp_hidden=policy_mlp_hidden, q_mlp_hidden=q_mlp_hidden,
                                alpha_mlp_hidden=alpha_mlp_hidden,
                                q_lr=q_lr, alpha_lr=alpha_lr, alpha=alpha, tau=tau, gamma=gamma,
                                target_entropy=target_entropy, huber_delta=huber_delta, gp_type=gp_type,
                                alpha_update=alpha_update,
                                reg_type=reg_type, sigma=sigma, n=n, gp_weight=gp_weight,
                                entropy_reg=entropy_reg, kl_backup=kl_backup)
        self.agent.set_logger(self.logger)
        self.behavior_filepath = os.path.join(self.logger.output_dir, 'behavior.ckpt')
        self.policy_behavior_filepath = os.path.join(self.logger.output_dir,
                                                     f'policy_behavior_{target_entropy}_{reg_type}.ckpt')
        self.log_beta_behavior_filepath = os.path.join(self.logger.output_dir,
                                                       f'policy_behavior_log_beta_{target_entropy}_{reg_type}.ckpt')
        self.final_filepath = os.path.join(self.logger.output_dir, 'agent_final.ckpt')

    def setup_extra(self,
                    pretrain_epochs,
                    save_freq,
                    max_kl,
                    force_pretrain_behavior,
                    force_pretrain_cloning
                    ):
        self.pretrain_epochs = pretrain_epochs
        self.save_freq = save_freq
        self.max_kl = max_kl
        self.force_pretrain_behavior = force_pretrain_behavior
        self.force_pretrain_cloning = force_pretrain_cloning

    def run_one_step(self, t):
        self.agent.update(self.replay_buffer)

    def on_epoch_end(self, epoch):
        self.test_agent(agent=self.agent, name='policy', logger=self.logger)

        # set delta_gp
        kl_stats = self.logger.get_stats('KL')
        self.agent.set_delta_gp(kl_stats[0] + kl_stats[1])  # mean + std

        # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        self.logger.log_tabular('NormalizedTestEpRet', average_only=True)
        self.logger.log_tabular('TestEpLen', average_only=True)
        self.agent.log_tabular()
        self.logger.log_tabular('Time', time.time() - self.start_time)
        self.logger.dump_tabular()

        if self.save_freq is not None and (epoch + 1) % self.save_freq == 0:
            self.agent.save_weights(filepath=os.path.join(self.logger.output_dir, f'agent_final_{epoch + 1}.ckpt'))

    def on_train_begin(self):
        self.agent.policy_net.optimizer = get_adam_optimizer(lr=self.policy_behavior_lr)
        interval = self.pretrain_epochs * self.steps_per_epoch // 5
        behavior_lr = self.agent.behavior_lr
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[interval, interval * 2, interval * 3, interval * 4],
            values=[behavior_lr, 0.5 * behavior_lr, 0.1 * behavior_lr, 0.05 * behavior_lr, 0.01 * behavior_lr])
        self.agent.behavior_policy.optimizer = get_adam_optimizer(lr=lr_schedule)
        try:
            if self.force_pretrain_behavior or self.force_pretrain_cloning:
                raise ValueError
            self.agent.behavior_policy.load_weights(filepath=self.behavior_filepath).assert_consumed()
            self.agent.policy_net.load_weights(filepath=self.policy_behavior_filepath).assert_consumed()
            self.agent.log_beta.load_weights(filepath=self.log_beta_behavior_filepath).assert_consumed()
            self.logger.log(f'Loading behavior from {self.behavior_filepath}')
        except:
            self.pretrain(self.pretrain_epochs)
            self.agent.behavior_policy.save_weights(filepath=self.behavior_filepath)
            self.agent.policy_net.save_weights(filepath=self.policy_behavior_filepath)
            self.agent.log_beta.save_weights(filepath=self.log_beta_behavior_filepath)

        hard_update(self.agent.target_policy_net, self.agent.policy_net)
        self.agent.set_delta_behavior(self.max_kl)
        self.agent.set_delta_gp(self.max_kl)
        # reset policy net learning rate
        self.agent.policy_net.optimizer = get_adam_optimizer(lr=self.policy_lr)

        # test behavior policy
        self.test_agent(self.agent.behavior_policy, name='vae policy')
        self.test_agent(self.agent, name='behavior cloning')
        # compute the current KL between pi and pi_b
        obs_dataset = tf.data.Dataset.from_tensor_slices((self.replay_buffer.get()['obs'],)).batch(1000)
        distance = []
        for obs, in obs_dataset:
            distance.append(self.agent.compute_pi_pib_distance(obs)[0])
        distance = tf.reduce_mean(tf.concat(distance, axis=0)).numpy()
        self.logger.log(f'The average distance between pi and pi_b is {distance:.4f}')
        # set max_kl heuristically if it is None.
        self.start_time = time.time()

    def on_train_end(self):
        self.agent.save_weights(filepath=self.final_filepath)

    def pretrain(self, epochs):
        EpochLogger.log(f'Training behavior policy for {self.env_name}')
        t = trange(epochs)
        for epoch in t:
            loss, val_loss, kl, log_pi = [], [], [], []
            for _ in trange(self.steps_per_epoch, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
                # update q_b, pi_0, pi_b
                data = self.replay_buffer.sample()
                obs = data['obs']
                if self.force_pretrain_behavior:
                    raw_act = clip_arctanh(data['act'])
                    behavior_loss = self.agent.behavior_policy.train_on_batch(x=(raw_act, obs), return_dict=True)[
                        'loss']
                else:
                    behavior_loss = 0.
                if epoch > epochs // 2:
                    actor_info = self.agent.update_actor_cloning(obs)
                    kl.append(actor_info['KL'])
                    log_pi.append(actor_info['LogPi'])
                else:
                    kl.append(0.)
                    log_pi.append(0.)
                loss.append(behavior_loss)
            loss = tf.reduce_mean(loss).numpy()
            kl = tf.reduce_mean(kl).numpy()
            log_pi = tf.reduce_mean(log_pi).numpy()
            t.set_description(desc=f'Loss: {loss:.2f}, KL: {kl:.2f}, LogPi: {log_pi:.2f}')


def bracp(env_name,
          steps_per_epoch=2000,
          pretrain_epochs=250,
          force_pretrain_behavior=False,
          force_pretrain_cloning=False,
          epochs=500,
          batch_size=100,

          num_test_episodes=20,
          seed=1,
          # agent args
          policy_mlp_hidden=256,
          q_mlp_hidden=256,
          policy_lr=5e-6,
          policy_behavior_lr=3e-4,
          q_lr=3e-4,
          alpha_lr=1e-5,
          alpha=10.0,
          alpha_mlp_hidden=256,
          tau=1e-3,
          gamma=0.99,
          huber_delta=None,
          target_entropy=None,
          max_kl=None,
          alpha_update='nn',
          gp_type='hard',
          reg_type='kl',
          sigma=10,
          n=5,
          gp_weight=0.1,
          entropy_reg=True,
          kl_backup=False,
          # behavior policy
          num_ensembles=5,
          behavior_mlp_hidden=256,
          behavior_lr=1e-3,
          # others
          reward_scale=True,
          save_freq=None,
          ):
    config = locals()

    runner = BRACPRunner(seed=seed, steps_per_epoch=steps_per_epoch, epochs=epochs,
                         exp_name=None, logger_path='data')
    runner.setup_env(env_name=env_name, num_parallel_env=num_test_episodes, frame_stack=None, wrappers=None,
                     asynchronous=False, num_test_episodes=None)
    runner.setup_logger(config=config)
    runner.setup_agent(num_ensembles=num_ensembles,
                       behavior_mlp_hidden=behavior_mlp_hidden,
                       behavior_lr=behavior_lr,
                       policy_mlp_hidden=policy_mlp_hidden, q_mlp_hidden=q_mlp_hidden,
                       alpha_mlp_hidden=alpha_mlp_hidden,
                       policy_lr=policy_lr, q_lr=q_lr, alpha_lr=alpha_lr, alpha=alpha, tau=tau, gamma=gamma,
                       target_entropy=target_entropy, huber_delta=huber_delta, gp_type=gp_type,
                       alpha_update=alpha_update, policy_behavior_lr=policy_behavior_lr,
                       reg_type=reg_type, sigma=sigma, n=n, gp_weight=gp_weight,
                       entropy_reg=entropy_reg, kl_backup=kl_backup)
    runner.setup_extra(pretrain_epochs=pretrain_epochs,
                       save_freq=save_freq,
                       max_kl=max_kl,
                       force_pretrain_behavior=force_pretrain_behavior,
                       force_pretrain_cloning=force_pretrain_cloning)
    runner.setup_replay_buffer(batch_size=batch_size,
                               reward_scale=reward_scale)

    runner.run()


if __name__ == '__main__':
    default_hyperparameters = {
        'd4rl:walker2d-medium-v0': dict(
            max_kl=2.2,
            target_entropy=-9,
        ),
        'd4rl:walker2d-medium-expert-v0': dict(
            max_kl=5,
            target_entropy=-12,
        ),
        'd4rl:walker2d-medium-replay-v0': dict(
            max_kl=4.0,
            target_entropy=-6,
        ),
        'd4rl:walker2d-random-v0': dict(
            max_kl=0.1,
            target_entropy=6,
        ),
        'd4rl:walker2d-expert-v0': dict(
            max_kl=4.8,
            target_entropy=-12,
        ),
        'd4rl:hopper-medium-expert-v0': dict(
            max_kl=2.6,
            target_entropy=-6,
        ),
        'd4rl:hopper-medium-v0': dict(
            max_kl=2.4,
            target_entropy=-6,
        ),
        'd4rl:hopper-random-v0': dict(
            max_kl=3.0,
            target_entropy=-3,
        ),
        'd4rl:hopper-medium-replay-v0': dict(
            max_kl=3.0,
            target_entropy=-3,
        ),
        'd4rl:hopper-expert-v0': dict(
            max_kl=2.6,
            target_entropy=-6,
        ),
        'd4rl:halfcheetah-medium-expert-v0': dict(
            max_kl=11.5,
            target_entropy=-24,
        ),
        'd4rl:halfcheetah-medium-v0': dict(
            max_kl=4,
            target_entropy=-12,
        ),
        'd4rl:halfcheetah-medium-replay-v0': dict(
            max_kl=6.0,
            target_entropy=-12,
        ),
        'd4rl:halfcheetah-random-v0': dict(
            max_kl=9,
            target_entropy=-3,
        ),
        'd4rl:door-human-v0': dict(
            max_kl=0.1,
            target_entropy=-60,
            reg_type='mmd',
            alpha_lr=1e-7,
            policy_lr=5e-8,
            sigma=60,
        ),
        'd4rl:pen-human-v0': dict(
            max_kl=0.06,
            target_entropy=-200,
            reg_type='mmd',
            alpha_lr=1e-7,
            policy_lr=5e-8,
            sigma=60,
        ),
        'd4rl:hammer-human-v0': dict(
            max_kl=0.1,
            target_entropy=-60,
            reg_type='mmd',
            alpha_lr=1e-7,
            policy_lr=5e-8,
            sigma=60,
        ),
        'd4rl:relocate-human-v0': dict(
            max_kl=0.1,
            target_entropy=-60,
            reg_type='mmd',
            alpha_lr=1e-7,
            policy_lr=5e-8,
            sigma=60,
        )
    }

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--force_pretrain_behavior', action='store_true')
    parser.add_argument('--force_pretrain_cloning', action='store_true')
    parser.add_argument('--seed', type=int, default=1)

    args = vars(parser.parse_args())
    env_name = args['env_name']

    config = default_hyperparameters[env_name]
    config.update(args)

    bracp(**config)
