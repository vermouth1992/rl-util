"""
Implement soft actor critic agent here
"""

import pybullet_envs

import os
import time

import gym
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from rlutils.utils import set_tf_allow_growth
from rlutils import soft_update_tf, hard_update_tf
from rlutils.distributions import make_independent_normal_from_params
from rlutils.nn import build_mlp, LagrangeLayer
from rlutils.runner import BaseRunner
from rlutils.replay_buffers import ReverbTransitionReplayBuffer

set_tf_allow_growth()

import tensorflow_probability as tfp

tfd = tfp.distributions


class SquashedGaussianMLPActor(tf.keras.Model):
    def __init__(self, ob_dim, ac_dim, mlp_hidden):
        super(SquashedGaussianMLPActor, self).__init__()
        self.net = build_mlp(ob_dim, ac_dim * 2, mlp_hidden)
        self.ac_dim = ac_dim
        self.pi_dist_layer = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: make_independent_normal_from_params(t, min_log_scale=-10, max_log_scale=5.))
        self.call = tf.function(func=self.call, input_signature=[
            (tf.TensorSpec(shape=[None, ob_dim], dtype=tf.float32),
             tf.TensorSpec(shape=(), dtype=tf.bool))
        ])

    def call(self, inputs):
        inputs, deterministic = inputs
        # print(f'Tracing call with inputs={inputs}, deterministic={deterministic}')
        params = self.net(inputs)
        pi_distribution = self.pi_dist_layer(params)
        pi_action = tf.cond(pred=deterministic, true_fn=lambda: pi_distribution.mean(),
                            false_fn=lambda: pi_distribution.sample())
        logp_pi = pi_distribution.log_prob(pi_action)
        logp_pi -= tf.reduce_sum(2. * (tf.math.log(2.) - pi_action - tf.math.softplus(-2. * pi_action)), axis=-1)
        pi_action_final = tf.tanh(pi_action)
        return pi_action_final, logp_pi, pi_action, pi_distribution


class EnsembleQNet(tf.keras.Model):
    def __init__(self, ob_dim, ac_dim, mlp_hidden, num_ensembles=2):
        super(EnsembleQNet, self).__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.mlp_hidden = mlp_hidden
        self.num_ensembles = num_ensembles
        self.q_net = build_mlp(input_dim=self.ob_dim + self.ac_dim,
                               output_dim=1,
                               mlp_hidden=self.mlp_hidden,
                               num_ensembles=self.num_ensembles,
                               num_layers=3,
                               squeeze=True)
        self.build(input_shape=[(None, ob_dim), (None, ac_dim)])

    def get_config(self):
        config = super(EnsembleQNet, self).get_config()
        config.update({
            'ob_dim': self.ob_dim,
            'ac_dim': self.ac_dim,
            'mlp_hidden': self.mlp_hidden,
            'num_ensembles': self.num_ensembles
        })
        return config

    def call(self, inputs, training=None, mask=None):
        obs, act = inputs
        inputs = tf.concat((obs, act), axis=-1)
        inputs = tf.tile(tf.expand_dims(inputs, axis=0), (self.num_ensembles, 1, 1))
        q = self.q_net(inputs)  # (num_ensembles, None)
        if training:
            return q
        else:
            return tf.reduce_min(q, axis=0)


class SACAgent(tf.keras.Model):
    def __init__(self,
                 obs_spec,
                 act_spec,
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
        act_dim = self.act_spec.shape[0]
        if len(self.obs_spec.shape[0]) == 1:  # 1D observation
            obs_dim = self.obs_spec.shape[0]
            self.policy_net = SquashedGaussianMLPActor(obs_dim, act_dim, policy_mlp_hidden)
            self.q_network = EnsembleQNet(obs_dim, act_dim, q_mlp_hidden)
            self.target_q_network = EnsembleQNet(obs_dim, act_dim, q_mlp_hidden)
        else:
            raise NotImplementedError
        hard_update_tf(self.target_q_network, self.q_network)

        self.policy_optimizer = tf.keras.optimizers.Adam(lr=policy_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(lr=q_lr)

        self.log_alpha = LagrangeLayer(initial_value=alpha)
        self.alpha_optimizer = tf.keras.optimizers.Adam(lr=alpha_lr)
        self.target_entropy = -act_dim if target_entropy is None else target_entropy

        self.tau = tau
        self.gamma = gamma

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('Q1Vals', with_min_and_max=False)
        self.logger.log_tabular('Q2Vals', with_min_and_max=False)
        self.logger.log_tabular('LogPi', average_only=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('Alpha', average_only=True)
        self.logger.log_tabular('LossAlpha', average_only=True)

    def update_target(self):
        soft_update_tf(self.target_q_network, self.q_network, self.tau)

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

        next_action, next_action_log_prob, _, _ = self.policy_net((next_obs, False))
        target_q_values = self.target_q_network((next_obs, next_action), training=False) - alpha * next_action_log_prob
        q_target = reward + self.gamma * (1.0 - done) * target_q_values

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

    def update(self, obs, act, obs2, done, rew, update_target=True):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        act = tf.convert_to_tensor(act, dtype=tf.float32)
        obs2 = tf.convert_to_tensor(obs2, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        rew = tf.convert_to_tensor(rew, dtype=tf.float32)

        info = self._update_nets(obs, act, obs2, done, rew)
        for key, item in info.items():
            info[key] = item.numpy()
        self.logger.store(**info)

        if update_target:
            self.update_target()

    @tf.function
    def act_batch(self, obs, deterministic):
        print(f'Tracing sac act_batch with obs {obs}')
        pi_final = self.policy_net((obs, deterministic))[0]
        return pi_final


class SACRunner(BaseRunner):
    def setup_extra(self,
                    start_steps,
                    save_freq):
        self.start_steps = start_steps
        self.save_freq = save_freq

    def setup_replay_buffer(self,
                            replay_size,
                            batch_size,
                            gamma,
                            update_horizon,
                            frame_stack):
        obs_spec = tf.TensorSpec(shape=self.env.single_observation_space.shape, dtype=tf.float32)
        act_spec = tf.TensorSpec(shape=self.env.single_action_space.shape, dtype=tf.float32)
        self.replay_buffer = ReverbTransitionReplayBuffer(num_parallel_env=self.num_parallel_env,
                                                          obs_spec=obs_spec,
                                                          act_spec=act_spec,
                                                          replay_capacity=replay_size,
                                                          batch_size=batch_size,
                                                          gamma=gamma,
                                                          update_horizon=update_horizon,
                                                          frame_stack=frame_stack)

    def setup_agent(self,
                    policy_mlp_hidden,
                    q_mlp_hidden,
                    policy_lr,
                    q_lr,
                    alpha,
                    alpha_lr,
                    tau,
                    gamma
                    ):
        obs_spec = tf.TensorSpec(shape=self.env.single_observation_space.shape, dtype=tf.float32)
        act_spec = tf.TensorSpec(shape=self.env.single_action_space.shape, dtype=tf.float32)
        self.agent = SACAgent(obs_spec=obs_spec, act_spec=act_spec, policy_mlp_hidden=policy_mlp_hidden,
                              q_mlp_hidden=q_mlp_hidden, policy_lr=policy_lr, q_lr=q_lr,
                              alpha=alpha, alpha_lr=alpha_lr, tau=tau, gamma=gamma)
        self.agent.set_logger(self.logger)

    def run_one_step(self, global_step):
        pass

    def on_end_epoch(self):
        pass

    def test_agent(self):
        o, d, ep_ret, ep_len = self.test_env.reset(), np.zeros(shape=self.num_test_episodes, dtype=np.bool), np.zeros(
            shape=self.num_test_episodes), np.zeros(shape=self.num_test_episodes, dtype=np.int32)
        t = tqdm(total=1, desc='Testing')
        while not np.all(d):
            o = tf.convert_to_tensor(o, dtype=tf.float32)
            a = self.agent.act_batch(o, True).numpy()
            o, r, d_, _ = self.test_env.step(a)
            ep_ret = r * (1 - d) + ep_ret
            ep_len = np.ones(shape=self.num_test_episodes, dtype=np.int32) * (1 - d) + ep_len
            d = np.logical_or(d, d_)
        t.update(1)
        t.close()
        self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)


def sac(env_name,
        env_fn=None,
        max_ep_len=1000,
        steps_per_epoch=5000,
        epochs=200,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        update_per_step=1,
        batch_size=256,
        num_test_episodes=20,
        logger_kwargs=dict(),
        seed=1,
        # sac args
        nn_size=256,
        learning_rate=3e-4,
        alpha=0.2,
        tau=5e-3,
        gamma=0.99,
        # replay
        replay_size=int(1e6),
        save_freq=10,
        ):
    agent_args = dict(

    )
    extra_args = dict(

    )

    runner = SACRunner(env_name=env_name, seed=seed, steps_per_epoch=steps_per_epoch,
                       epochs=epochs, num_parallel_env=5, num_test_episodes=num_test_episodes,
                       asynchronous=False, max_ep_len=max_ep_len, exp_name=f'{env_name}_sac_test',
                       agent_args=agent_args, extra_args=extra_args)

    def get_action(o, deterministic=False):
        return agent.act(tf.convert_to_tensor(o, dtype=tf.float32), tf.convert_to_tensor(deterministic)).numpy()

    def get_action_batch(o, deterministic=False):
        return agent.act_batch(tf.convert_to_tensor(o, dtype=tf.float32), tf.convert_to_tensor(deterministic)).numpy()

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    bar = tqdm(total=steps_per_epoch)

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every * update_per_step):
                batch = replay_buffer.sample_batch(batch_size)
                agent.update(**batch, update_target=True)

        bar.update(1)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            bar.close()

            epoch = (t + 1) // steps_per_epoch

            if epoch % save_freq == 0:
                agent.save_weights(filepath=os.path.join(logger_kwargs['output_dir'], f'agent_final_{epoch}.ckpt'))

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            agent.log_tabular()
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

            if t < total_steps:
                bar = tqdm(total=steps_per_epoch)

    agent.save_weights(filepath=os.path.join(logger_kwargs['output_dir'], f'agent_final.ckpt'))


if __name__ == '__main__':
    import argparse
    from utils.run_utils import setup_logger_kwargs

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Hopper-v2')
    parser.add_argument('--seed', type=int, default=1)
    # agent arguments
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=5e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--nn_size', '-s', type=int, default=256)
    # training arguments
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--start_steps', type=int, default=1000)
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument('--steps_per_epoch', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_test_episodes', type=int, default=20)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--update_after', type=int, default=1000)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--update_per_step', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--gpu', action='store_true')

    args = vars(parser.parse_args())

    logger_kwargs = setup_logger_kwargs(exp_name=args['env_name'] + '_sac_test', data_dir='data', seed=args['seed'])

    use_gpu = args.pop('gpu')
    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    sac(**args, logger_kwargs=logger_kwargs)
