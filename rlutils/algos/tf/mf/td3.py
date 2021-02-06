"""
Twin Delayed DDPG. https://arxiv.org/abs/1802.09477.
To obtain DDPG, set target smooth to zero and Q network ensembles to 1.
"""

import tensorflow as tf
from rlutils.runner import OffPolicyRunner, TFRunner, run_func_as_main
from rlutils.tf.functional import soft_update, hard_update, compute_target_value
from rlutils.tf.nn import EnsembleMinQNet
from rlutils.tf.nn.functional import build_mlp


class TD3Agent(tf.keras.Model):
    def __init__(self,
                 obs_spec,
                 act_spec,
                 num_q_ensembles=2,
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
        self.actor_noise = actor_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.tau = tau
        self.gamma = gamma
        if len(self.obs_spec.shape) == 1:  # 1D observation
            self.obs_dim = self.obs_spec.shape[0]
            self.policy_net = build_mlp(self.obs_dim, self.act_dim, mlp_hidden=policy_mlp_hidden, num_layers=3,
                                        out_activation=tf.math.sin)
            self.target_policy_net = build_mlp(self.obs_dim, self.act_dim, mlp_hidden=policy_mlp_hidden,
                                               num_layers=3, out_activation=tf.math.sin)
            hard_update(self.target_policy_net, self.policy_net)
            self.q_network = EnsembleMinQNet(self.obs_dim, self.act_dim, q_mlp_hidden, num_ensembles=num_q_ensembles)
            self.target_q_network = EnsembleMinQNet(self.obs_dim, self.act_dim, q_mlp_hidden,
                                                    num_ensembles=num_q_ensembles)
            hard_update(self.target_q_network, self.q_network)
        else:
            raise NotImplementedError

        self.policy_optimizer = tf.keras.optimizers.Adam(lr=policy_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(lr=q_lr)

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('Q1Vals', with_min_and_max=True)
        self.logger.log_tabular('Q2Vals', with_min_and_max=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)

    @tf.function
    def update_target(self):
        soft_update(self.target_q_network, self.q_network, self.tau)
        soft_update(self.target_policy_net, self.policy_net, self.tau)

    def _compute_next_obs_q(self, next_obs):
        next_action = self.target_policy_net(next_obs)
        # Target policy smoothing
        epsilon = tf.random.normal(shape=[tf.shape(next_obs)[0], self.act_dim]) * self.target_noise
        epsilon = tf.clip_by_value(epsilon, -self.noise_clip, self.noise_clip)
        next_action = next_action + epsilon
        next_action = tf.clip_by_value(next_action, -self.act_lim, self.act_lim)
        next_q_value = self.target_q_network((next_obs, next_action), training=False)
        return next_q_value

    @tf.function
    def _update_q_nets(self, obs, actions, next_obs, done, reward):
        print(f'Tracing _update_nets with obs={obs}, actions={actions}')
        # compute target q
        next_q_value = self._compute_next_obs_q(next_obs)
        q_target = compute_target_value(reward, self.gamma, done, next_q_value)
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
            q = self.q_network((obs, a), training=False)
            policy_loss = -tf.reduce_mean(q, axis=0)
        policy_gradients = policy_tape.gradient(policy_loss, self.policy_net.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy_net.trainable_variables))
        info = dict(
            LossPi=policy_loss,
        )
        return info

    @tf.function
    def train_step(self, data):
        obs = data['obs']
        act = data['act']
        next_obs = data['next_obs']
        done = data['done']
        rew = data['rew']
        update_target = data['update_target']
        print(f'Tracing train_step with {update_target}')
        info = self._update_q_nets(obs, act, next_obs, done, rew)
        if update_target:
            actor_info = self._update_actor(obs)
            info.update(actor_info)
            self.update_target()
        return info

    @tf.function
    def act_batch_test(self, obs):
        return self.policy_net(obs)

    @tf.function
    def act_batch_explore(self, obs):
        print('Tracing act_batch_explore')
        pi_final = self.policy_net(obs)
        noise = tf.random.normal(shape=[tf.shape(obs)[0], self.act_dim], dtype=tf.float32) * self.actor_noise
        pi_final_noise = pi_final + noise
        pi_final_noise = tf.clip_by_value(pi_final_noise, -self.act_lim, self.act_lim)
        return pi_final_noise


class TD3Runner(OffPolicyRunner, TFRunner):
    def get_action_batch_test(self, obs):
        return self.agent.act_batch_test(tf.convert_to_tensor(obs, dtype=tf.float32)).numpy()

    def get_action_batch_explore(self, obs):
        return self.agent.act_batch_explore(tf.convert_to_tensor(obs, dtype=tf.float32)).numpy()


def td3(env_name,
        env_fn=None,
        steps_per_epoch=5000,
        epochs=200,
        start_steps=10000,
        update_after=4000,
        update_every=1,
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
        logger_path=None
        ):
    config = locals()

    runner = TD3Runner(seed=seed, steps_per_epoch=steps_per_epoch // num_parallel_env, epochs=epochs,
                       exp_name=None, logger_path=logger_path)
    runner.setup_env(env_name=env_name, num_parallel_env=num_parallel_env, env_fn=env_fn,
                     asynchronous=False, num_test_episodes=num_test_episodes)
    runner.setup_logger(config=config)

    agent_kwargs = dict(
        policy_mlp_hidden=nn_size,
        policy_lr=learning_rate,
        q_mlp_hidden=nn_size,
        q_lr=learning_rate,
        tau=tau,
        gamma=gamma,
        actor_noise=actor_noise,
        target_noise=target_noise,
        noise_clip=noise_clip
    )
    runner.setup_agent(agent_cls=TD3Agent, **agent_kwargs)
    runner.setup_extra(start_steps=start_steps,
                       update_after=update_after,
                       update_every=update_every,
                       update_per_step=update_per_step,
                       policy_delay=policy_delay)
    runner.setup_replay_buffer(replay_size=replay_size,
                               batch_size=batch_size)
    runner.run()


if __name__ == '__main__':
    run_func_as_main(td3)
