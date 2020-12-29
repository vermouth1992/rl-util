import time

import numpy as np
import tensorflow as tf
from rlutils.replay_buffers import PyUniformParallelEnvReplayBuffer
from rlutils.runner import TFRunner, run_func_as_main
from rlutils.tf.nn import EnsembleMinQNet
from rlutils.tf.nn.functional import soft_update, hard_update, build_mlp
from tqdm.auto import tqdm


class TD3Agent(tf.keras.Model):
    def __init__(self,
                 obs_spec,
                 act_spec,
                 policy_mlp_hidden=128,
                 policy_lr=3e-4,
                 q_mlp_hidden=256,
                 q_lr=3e-4,
                 tau=5e-3,
                 gamma=0.99,
                 actor_noise=0.1,
                 target_noise=0.2,
                 noise_clip=0.5
                 ):
        super(TD3Agent, self).__init__()
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.act_dim = self.act_spec.shape[0]
        self.act_lim = 1.
        if len(self.obs_spec.shape) == 1:  # 1D observation
            obs_dim = self.obs_spec.shape[0]
        else:
            raise NotImplementedError
        self.actor_noise = actor_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_net = build_mlp(obs_dim, self.act_dim, mlp_hidden=policy_mlp_hidden, num_layers=3,
                                    out_activation='tanh')
        self.target_policy_net = build_mlp(obs_dim, self.act_dim, mlp_hidden=policy_mlp_hidden, num_layers=3,
                                           out_activation='tanh')
        self.q_network = EnsembleMinQNet(obs_dim, self.act_dim, q_mlp_hidden)
        self.target_q_network = EnsembleMinQNet(obs_dim, self.act_dim, q_mlp_hidden)
        hard_update(self.target_q_network, self.q_network)
        hard_update(self.target_policy_net, self.policy_net)
        self.policy_optimizer = tf.keras.optimizers.Adam(lr=policy_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(lr=q_lr)
        self.tau = tau
        self.gamma = gamma

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('Q1Vals', with_min_and_max=False)
        self.logger.log_tabular('Q2Vals', with_min_and_max=False)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)

    @tf.function
    def update_target(self):
        soft_update(self.target_q_network, self.q_network, self.tau)
        soft_update(self.target_policy_net, self.policy_net, self.tau)

    @tf.function
    def _update_nets(self, obs, actions, next_obs, done, reward):
        print(f'Tracing _update_nets with obs={obs}, actions={actions}')
        # compute target q
        next_action = self.target_policy_net(next_obs)
        # Target policy smoothing
        epsilon = tf.random.normal(shape=[tf.shape(obs)[0], self.ac_dim]) * self.target_noise
        epsilon = tf.clip_by_value(epsilon, -self.noise_clip, self.noise_clip)
        next_action = next_action + epsilon
        next_action = tf.clip_by_value(next_action, -self.act_lim, self.act_lim)
        target_q_values = self.target_q_network((next_obs, next_action), training=False)
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

        info = dict(
            Q1Vals=q_values[0],
            Q2Vals=q_values[1],
            LossQ=q_values_loss,
        )
        return info

    @tf.function
    def _update_actor(self, obs):
        print(f'Tracing _update_actor with obs={obs}')
        # policy loss
        with tf.GradientTape() as policy_tape:
            a = self.policy_net(obs)
            q = self.q_network((obs, a))
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

        for key, item in info.items():
            info[key] = item.numpy()
        self.logger.store(**info)

    @tf.function
    def act_batch(self, obs, deterministic):
        print(f'Tracing td3 act_batch with obs {obs}')
        pi_final = self.policy_net(obs)
        if deterministic:
            return pi_final
        else:
            noise = tf.random.normal(shape=[tf.shape(obs)[0], self.act_dim], dtype=tf.float32) * self.actor_noise
            pi_final = pi_final + noise
            pi_final = tf.clip_by_value(pi_final, -self.act_lim, self.act_lim)
            return pi_final


class TD3Runner(TFRunner):
    def get_action_batch(self, o, deterministic=False):
        return self.agent.act_batch(tf.convert_to_tensor(o, dtype=tf.float32),
                                    deterministic).numpy()

    def test_agent(self):
        o, d, ep_ret, ep_len = self.test_env.reset(), np.zeros(shape=self.num_test_episodes, dtype=np.bool), \
                               np.zeros(shape=self.num_test_episodes), \
                               np.zeros(shape=self.num_test_episodes, dtype=np.int64)
        t = tqdm(total=1, desc='Testing')
        while not np.all(d):
            a = self.get_action_batch(o, deterministic=True)
            o, r, d_, _ = self.test_env.step(a)
            ep_ret = r * (1 - d) + ep_ret
            ep_len = np.ones(shape=self.num_test_episodes, dtype=np.int64) * (1 - d) + ep_len
            d = np.logical_or(d, d_)
        t.update(1)
        t.close()
        self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def setup_replay_buffer(self,
                            replay_size,
                            batch_size):
        obs_dim = self.env.single_observation_space.shape[0]
        act_dim = self.env.single_action_space.shape[0]
        self.replay_buffer = PyUniformParallelEnvReplayBuffer(obs_dim=obs_dim,
                                                              act_dim=act_dim,
                                                              act_dtype=np.float32,
                                                              capacity=replay_size,
                                                              batch_size=batch_size,
                                                              num_parallel_env=self.num_parallel_env)

    def setup_agent(self,
                    policy_mlp_hidden=128,
                    policy_lr=3e-4,
                    q_mlp_hidden=256,
                    q_lr=3e-4,
                    tau=5e-3,
                    gamma=0.99,
                    actor_noise=0.1,
                    target_noise=0.2,
                    noise_clip=0.5
                    ):
        obs_spec = tf.TensorSpec(shape=self.env.single_observation_space.shape,
                                 dtype=tf.float32)
        act_spec = tf.TensorSpec(shape=self.env.single_action_space.shape,
                                 dtype=tf.float32)
        self.agent = TD3Agent(obs_spec=obs_spec, act_spec=act_spec,
                              policy_mlp_hidden=policy_mlp_hidden,
                              policy_lr=policy_lr, q_mlp_hidden=q_mlp_hidden,
                              q_lr=q_lr, tau=tau, gamma=gamma,
                              actor_noise=actor_noise, target_noise=target_noise,
                              noise_clip=noise_clip)
        self.agent.set_logger(self.logger)

    def setup_extra(self,
                    start_steps,
                    max_ep_len,
                    update_after,
                    update_every,
                    update_per_step,
                    policy_delay):
        self.start_steps = start_steps
        self.max_ep_len = max_ep_len
        self.update_after = update_after
        self.update_every = update_every
        self.update_per_step = update_per_step
        self.policy_delay = policy_delay

    def run_one_step(self):
        global_env_steps = self.global_step * self.num_parallel_env
        if global_env_steps >= self.start_steps:
            a = self.get_action_batch(self.o, deterministic=False)
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
                self.agent.update(**batch, update_target=j % self.policy_delay == 0)

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


def td3(env_name,
        max_ep_len=1000,
        steps_per_epoch=5000,
        epochs=200,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        update_per_step=1,
        policy_delay=2,
        batch_size=256,
        num_parallel_env=1,
        num_test_episodes=20,
        seed=1,
        # sac args
        nn_size=256,
        learning_rate=1e-3,
        actor_noise=0.1,
        target_noise=0.2,
        noise_clip=0.5,
        tau=5e-3,
        gamma=0.99,
        # replay
        replay_size=int(1e6),
        ):
    config = locals()

    runner = TD3Runner(seed=seed, steps_per_epoch=steps_per_epoch // num_parallel_env, epochs=epochs,
                       exp_name=f'{env_name}_td3_test', logger_path='data')
    runner.setup_env(env_name=env_name, num_parallel_env=num_parallel_env, frame_stack=None, wrappers=None,
                     asynchronous=False, num_test_episodes=num_test_episodes)
    runner.setup_seed(seed)
    runner.setup_logger(config=config)
    runner.setup_agent(policy_mlp_hidden=nn_size,
                       policy_lr=learning_rate,
                       q_mlp_hidden=nn_size,
                       q_lr=learning_rate,
                       tau=tau,
                       gamma=gamma,
                       actor_noise=actor_noise,
                       target_noise=target_noise,
                       noise_clip=noise_clip)
    runner.setup_extra(start_steps=start_steps,
                       max_ep_len=max_ep_len,
                       update_after=update_after,
                       update_every=update_every,
                       update_per_step=update_per_step,
                       policy_delay=policy_delay)
    runner.setup_replay_buffer(replay_size=replay_size,
                               batch_size=batch_size)

    runner.run()


if __name__ == '__main__':
    run_func_as_main(td3)
