"""
Deep Q Network for low-dimensional observation space
"""

import time

import numpy as np
import tensorflow as tf

from rlutils.tf.utils import set_tf_allow_growth

set_tf_allow_growth()

from rlutils.runner import TFRunner
from rlutils.tf.nn.functional import hard_update, build_mlp
from rlutils.replay_buffers import PyUniformParallelEnvReplayBuffer, ReverbTransitionReplayBuffer
from rlutils.runner import run_func_as_main
from rlutils.schedulers import PiecewiseSchedule


def gather_q_values(q_values, actions):
    batch_size = tf.shape(actions)[0]
    idx = tf.stack([tf.range(batch_size), actions], axis=-1)  # (None, 2)
    q_values = tf.gather_nd(q_values, indices=idx)
    return q_values


class DQN(tf.keras.Model):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 mlp_hidden=128,
                 double_q=True,
                 q_lr=1e-4,
                 gamma=0.99,
                 huber_delta=None):
        super(DQN, self).__init__()
        self.q_network = build_mlp(obs_dim, act_dim, mlp_hidden=mlp_hidden, num_layers=3)
        self.target_q_network = build_mlp(obs_dim, act_dim, mlp_hidden=mlp_hidden, num_layers=3)
        self.q_optimizer = tf.keras.optimizers.Adam(lr=q_lr)
        self.epsilon = tf.Variable(initial_value=1.0, dtype=tf.float32)
        self.act_dim = act_dim
        self.double_q = double_q
        self.huber_delta = huber_delta
        self.gamma = gamma
        reduction = tf.keras.losses.Reduction.NONE  # Note: tensorflow uses reduce_mean at axis=-1 by default
        if huber_delta is None:
            self.loss_fn = tf.keras.losses.MeanSquaredError(reduction=reduction)
        else:
            self.loss_fn = tf.keras.losses.Huber(delta=huber_delta, reduction=reduction)
        hard_update(self.target_q_network, self.q_network)

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('QVals', with_min_and_max=True)
        self.logger.log_tabular('LossQ', average_only=True)

    def set_epsilon(self, epsilon):
        assert epsilon >= 0. and epsilon <= 1.
        self.epsilon.assign(epsilon)

    def update_target(self):
        hard_update(self.target_q_network, self.q_network)

    @tf.function
    def _update_nets(self, obs, act, next_obs, done, rew):
        print('Tracing _update_nets')
        # compute target Q values
        target_q_values = self.target_q_network(next_obs)
        if self.double_q:
            # select action using Q network instead of target Q network
            target_actions = tf.argmax(self.q_network(next_obs), axis=-1, output_type=tf.int32)
            target_q_values = gather_q_values(target_q_values, target_actions)
        else:
            target_q_values = tf.reduce_max(target_q_values, axis=-1)
        target_q_values = rew + self.gamma * (1. - done) * target_q_values
        with tf.GradientTape() as tape:
            q_values = gather_q_values(self.q_network(obs), act)  # (None,)
            loss = self.loss_fn(q_values, target_q_values)  # (None,)
        grad = tape.gradient(loss, self.q_network.trainable_variables)
        self.q_optimizer.apply_gradients(zip(grad, self.q_network.trainable_variables))
        info = dict(
            QVals=q_values,
            LossQ=loss
        )
        return info

    def update(self, obs, act, next_obs, rew, done):
        info = self._update_nets(obs, act, next_obs, done, rew)
        for key, item in info.items():
            info[key] = item.numpy()
        self.logger.store(**info)

    @tf.function
    def act_batch(self, obs, deterministic):
        """ Implement epsilon-greedy here """
        batch_size = tf.shape(obs)[0]
        epsilon = tf.random.uniform(shape=(batch_size,), minval=0., maxval=1., dtype=tf.float32)
        epsilon_indicator = tf.cast(epsilon > self.epsilon, dtype=tf.int32)  # (None,)
        random_actions = tf.random.uniform(shape=(batch_size,), minval=0, maxval=self.act_dim, dtype=tf.int32)
        deterministic_actions = tf.argmax(self.q_network(obs), axis=-1, output_type=tf.int32)
        epsilon_greedy_actions = tf.stack([random_actions, deterministic_actions], axis=-1)  # (None, 2)
        epsilon_greedy_actions = gather_q_values(epsilon_greedy_actions, epsilon_indicator)
        final_actions = tf.cond(deterministic, true_fn=lambda: deterministic_actions,
                                false_fn=lambda: epsilon_greedy_actions)
        return final_actions


class DQNRunner(TFRunner):
    def setup_replay_buffer(self,
                            num_parallel_env,
                            replay_capacity,
                            batch_size,
                            gamma,
                            update_horizon,
                            replay_type='py'
                            ):
        self.replay_type = replay_type
        if replay_type == 'reverb':
            self.replay_buffer = ReverbTransitionReplayBuffer(
                num_parallel_env=num_parallel_env,
                obs_spec=tf.TensorSpec(shape=self.env.single_observation_space.shape, dtype=tf.float32),
                act_spec=tf.TensorSpec(shape=(), dtype=tf.int32),
                replay_capacity=replay_capacity,
                batch_size=batch_size,
                gamma=gamma,
                update_horizon=update_horizon,
                frame_stack=1
            )
        else:
            self.replay_buffer = PyUniformParallelEnvReplayBuffer(
                obs_dim=self.env.single_observation_space.shape,
                act_dim=(),
                act_dtype=np.int32,
                capacity=replay_capacity,
                batch_size=batch_size,
                num_parallel_env=num_parallel_env,
            )

    def setup_agent(self,
                    mlp_hidden=128,
                    double_q=True,
                    q_lr=1e-4,
                    gamma=0.99,
                    huber_delta=1.0):
        self.agent = DQN(obs_dim=self.env.single_observation_space.shape[0],
                         act_dim=self.env.single_action_space.n,
                         mlp_hidden=mlp_hidden,
                         double_q=double_q,
                         q_lr=q_lr,
                         gamma=gamma,
                         huber_delta=huber_delta)
        self.agent.set_logger(self.logger)

    def setup_extra(self,
                    start_steps,
                    update_after,
                    update_every,
                    update_per_step,
                    target_update):
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.update_per_step = update_per_step
        self.target_update = target_update
        # epsilon scheduler
        self.epsilon_scheduler = PiecewiseSchedule(
            [
                (0, 1.0),
                (1e5, 0.1),
                (self.epochs * self.steps_per_epoch / 2, 0.01),
            ]
        )

    def get_action_batch(self, o, deterministic=False):
        return self.agent.act_batch(tf.convert_to_tensor(o, dtype=tf.float32),
                                    tf.convert_to_tensor(deterministic, dtype=tf.bool)).numpy()

    def run_one_step(self):
        global_env_steps = self.global_step * self.num_parallel_env
        if global_env_steps >= self.start_steps:
            a = self.get_action_batch(self.o, deterministic=False)
        else:
            a = self.env.action_space.sample()
            a = np.asarray(a, dtype=np.int32)

        # Step the env
        o2, r, d, info = self.env.step(a)
        self.ep_ret += r
        self.ep_len += 1

        timeouts = np.array([i.get('TimeLimit.truncated', False) for i in info], dtype=np.bool)
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        true_d = np.logical_and(d, np.logical_not(timeouts))

        # Store experience to replay buffer
        data = {
            'obs': self.o,
            'act': a,
            'rew': r,
            'done': true_d
        }

        if self.replay_type != 'reverb':
            data['next_obs'] = o2

        self.replay_buffer.add(data=data)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        self.o = o2

        # End of trajectory handling
        if np.any(d):
            self.logger.store(EpRet=self.ep_ret[d], EpLen=self.ep_len[d])
            self.ep_ret[d] = 0
            self.ep_len[d] = 0

        # Update handling
        if global_env_steps >= self.update_after and global_env_steps % self.update_every == 0:
            for j in range(int(self.update_every * self.update_per_step)):
                batch = self.replay_buffer.sample()
                self.agent.update(**batch)

        if global_env_steps % self.target_update == 0:
            self.agent.update_target()

    def on_train_begin(self):
        self.start_time = time.time()
        self.o = self.env.reset()
        self.ep_ret = np.zeros(shape=self.num_parallel_env)
        self.ep_len = np.zeros(shape=self.num_parallel_env, dtype=np.int64)

    def on_epoch_end(self, epoch):
        # schedule the learning rate and epsilon
        epsilon = self.epsilon_scheduler.value(self.global_step)
        self.logger.log(f'Setting epsilon to {epsilon:.4f}')
        self.agent.set_epsilon(epsilon=epsilon)
        # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', self.global_step * self.num_parallel_env)
        self.agent.log_tabular()
        self.logger.log_tabular('Time', time.time() - self.start_time)
        self.logger.dump_tabular()


def dqn(env_name,
        steps_per_epoch=1000,
        epochs=500,
        start_steps=2000,
        update_after=500,
        update_every=1,
        update_per_step=1,
        batch_size=256,
        num_parallel_env=1,
        seed=1,
        # sac args
        mlp_hidden=256,
        double_q=True,
        q_lr=1e-4,
        gamma=0.99,
        huber_delta=None,
        target_update=1000,
        # replay
        update_horizon=1,
        replay_size=int(1e6)
        ):
    config = locals()
    runner = DQNRunner(seed=seed, steps_per_epoch=steps_per_epoch, epochs=epochs,
                       exp_name=f'{env_name}_dqn_test', logger_path='data')
    runner.setup_env(env_name=env_name, num_parallel_env=num_parallel_env, asynchronous=False, num_test_episodes=None)
    runner.setup_seed(seed)
    runner.setup_logger(config=config)
    runner.setup_agent(mlp_hidden=mlp_hidden,
                       double_q=double_q,
                       q_lr=q_lr,
                       gamma=gamma,
                       huber_delta=huber_delta)
    runner.setup_extra(start_steps=start_steps,
                       update_after=update_after,
                       update_every=update_every,
                       update_per_step=update_per_step,
                       target_update=target_update)
    runner.setup_replay_buffer(num_parallel_env=num_parallel_env,
                               replay_capacity=replay_size,
                               batch_size=batch_size,
                               gamma=gamma,
                               update_horizon=update_horizon,
                               )
    runner.run()


if __name__ == '__main__':
    run_func_as_main(dqn)
