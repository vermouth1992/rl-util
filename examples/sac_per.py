"""
Difference from standard SAC
1. Updater should update the priority of the replay buffer
2. SAC agent update function should takes in weight as parameter
"""

import rlutils.infra as rl_infra
import rlutils.tf as rlu
import tensorflow as tf
import tensorflow_probability as tfp
from rlutils.algos.tf.mf.sac import SACAgent
from rlutils.np.schedulers import PiecewiseSchedule
from rlutils.replay_buffers import PyPrioritizedReplayBuffer

tfd = tfp.distributions


class PrioritizedUpdater(rl_infra.OffPolicyUpdater):
    def __init__(self, total_steps, **kwargs):
        super(PrioritizedUpdater, self).__init__(**kwargs)
        # TODO: beta scheduler
        self.beta_scheduler = PiecewiseSchedule(endpoints=[(0, 0.4), (total_steps, 1.0)])

    def update(self, global_step):
        if global_step % self.update_every == 0:
            for _ in range(self.update_per_step * self.update_every):
                batch, idx = self.replay_buffer.sample(beta=self.beta_scheduler.value(global_step))
                batch['update_target'] = ((self.policy_updates + 1) % self.policy_delay == 0)
                priorities = self.agent.train_on_batch(data=batch)
                self.policy_updates += 1
                if priorities is not None:
                    self.replay_buffer.update_priorities(idx, priorities=priorities,
                                                         min_priority=0.01,
                                                         # max_priority=10.
                                                         )


class PrioritizedSACAgent(SACAgent):
    def log_tabular(self):
        super(PrioritizedSACAgent, self).log_tabular()
        self.logger.log_tabular('Priorities', with_min_and_max=True)

    @tf.function
    def _update_actor(self, obs, weights=None):
        alpha = self.log_alpha()
        # policy loss
        with tf.GradientTape() as policy_tape:
            action, log_prob, _, old_pi_distribution = self.policy_net((obs, False))
            q_values_pi_min = self.q_network((obs, action), training=False)
            policy_loss = log_prob * alpha - q_values_pi_min
            if weights is not None:
                policy_loss = policy_loss * weights
            policy_loss = tf.reduce_mean(policy_loss, axis=0)
        policy_gradients = policy_tape.gradient(policy_loss, self.policy_net.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy_net.trainable_variables))

        # get new pi_distribution
        new_pi_distribution = self.policy_net((obs, False))[-1]
        kld = tfd.kl_divergence(new_pi_distribution, old_pi_distribution)

        # log alpha
        if self.auto_alpha:
            with tf.GradientTape() as alpha_tape:
                alpha = self.log_alpha()
                alpha_loss = alpha * (log_prob + self.target_entropy)
                alpha_loss = -tf.reduce_mean(alpha_loss, axis=0)
            alpha_gradient = alpha_tape.gradient(alpha_loss, self.log_alpha.trainable_variables)
            self.alpha_optimizer.apply_gradients(zip(alpha_gradient, self.log_alpha.trainable_variables))
        else:
            alpha_loss = 0.

        info = dict(
            LogPi=log_prob,
            Alpha=alpha,
            LossAlpha=alpha_loss,
            LossPi=policy_loss,
            Priorities=kld
        )
        return info

    def train_step(self, data):
        obs = data['obs']
        act = data['act']
        next_obs = data['next_obs']
        done = data['done']
        rew = data['rew']
        update_target = data['update_target']
        weights = data['weights']
        info = self._update_q_nets(obs, act, next_obs, done, rew, weights=weights)
        if update_target:
            actor_info = self._update_actor(obs)
            info.update(actor_info)
            self.update_target()
        return info

    def train_on_batch(self, data, **kwargs):
        info = self.train_step(data=data)
        self.logger.store(**rlu.functional.to_numpy_or_python_type(info))
        new_priorities = info.get('Priorities', None)
        if new_priorities is not None:
            new_priorities = new_priorities.numpy()
        return new_priorities


class Runner(rl_infra.runner.TFOffPolicyRunner):
    def setup_replay_buffer(self,
                            replay_size,
                            batch_size):
        self.replay_buffer = PyPrioritizedReplayBuffer.from_vec_env(self.env, capacity=replay_size,
                                                                    batch_size=batch_size)

    def setup_updater(self, update_after, policy_delay, update_per_step, update_every):
        self.update_after = update_after
        self.updater = PrioritizedUpdater(agent=self.agent,
                                          replay_buffer=self.replay_buffer,
                                          policy_delay=policy_delay,
                                          update_per_step=update_per_step,
                                          update_every=update_every,
                                          total_steps=self.epochs * self.steps_per_epoch)

    @classmethod
    def main(cls,
             env_name,
             steps_per_epoch=10000,
             epochs=100,
             start_steps=10000,
             update_after=5000,
             update_every=50,
             update_per_step=1,
             policy_delay=1,
             batch_size=100,
             num_parallel_env=1,
             num_test_episodes=30,
             seed=1,
             # agent args
             policy_mlp_hidden=256,
             policy_lr=3e-4,
             q_mlp_hidden=256,
             q_lr=3e-4,
             alpha=0.2,
             tau=5e-3,
             gamma=0.99,
             # replay
             replay_size=int(1e6),
             logger_path='data/sac_per'
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
            target_entropy=None
        )

        super(Runner, cls).main(
            env_name=env_name,
            agent_cls=PrioritizedSACAgent,
            agent_kwargs=agent_kwargs,
            seed=seed,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            start_steps=start_steps,
            update_after=update_after,
            update_every=update_every,
            update_per_step=update_per_step,
            policy_delay=policy_delay,
            batch_size=batch_size,
            num_parallel_env=num_parallel_env,
            num_test_episodes=num_test_episodes,
            replay_size=replay_size,
            logger_path=logger_path
        )


if __name__ == '__main__':
    rl_infra.runner.run_func_as_main(Runner.main)
