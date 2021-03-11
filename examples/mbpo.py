"""
Deep reinforcement learning for DyanQ+ with priority sweeping
"""

from typing import Callable

import rlutils.infra as rl_infra
import rlutils.tf as rlu
import tensorflow as tf
from rlutils.algos.tf.mf.sac import SACAgent


class MBPOUpdater(rl_infra.OffPolicyUpdater):
    def __init__(self, total_steps, model_update_every, **kwargs):
        super(MBPOUpdater, self).__init__(**kwargs)
        self.model_update_every = model_update_every
        self.total_steps = total_steps

    def update(self, global_step):
        if global_step % self.model_update_every == 0:
            self.agent.update_model(self.replay_buffer.get())

        if global_step % self.update_every == 0:
            for _ in range(self.update_every):
                for _ in range(self.update_per_step):
                    batch = self.replay_buffer.sample()
                    batch['update_target'] = ((self.policy_updates + 1) % self.policy_delay == 0)
                    self.agent.train_on_batch(data=batch)
                    self.policy_updates += 1


class MBPO(tf.keras.Model):
    def __init__(self, obs_spec, act_spec,
                 model_mlp_hidden=512, model_lr=1e-4, model_num_ensembles=5, reward_fn=None, terminate_fn=None,
                 policy_mlp_hidden=256, policy_lr=3e-4):
        super(MBPO, self).__init__()
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.act_dim = self.act_spec.shape[0]
        if len(self.obs_spec.shape) == 1:  # 1D observation
            self.obs_dim = self.obs_spec.shape[0]
        else:
            raise NotImplementedError
        self.dynamics_model = rlu.nn.EnsembleDynamicsModel(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                                           mlp_hidden=model_mlp_hidden, num_layers=4,
                                                           num_ensembles=model_num_ensembles, lr=model_lr,
                                                           reward_fn=reward_fn, terminate_fn=terminate_fn)
        self.agent = SACAgent(obs_spec=obs_spec, act_spec=act_spec, policy_mlp_hidden=policy_mlp_hidden,
                              policy_lr=policy_lr, q_mlp_hidden=policy_mlp_hidden, q_lr=policy_lr, alpha=0.02,
                              alpha_lr=policy_lr, tau=5e-3, gamma=0.99, target_entropy=-(self.act_dim // 2),
                              auto_alpha=True)

    def set_logger(self, logger):
        self.dynamics_model.set_logger(logger)
        self.agent.set_logger(logger)

    def log_tabular(self):
        self.agent.log_tabular()
        self.dynamics_model.log_tabular()

    def update_model(self, data):
        self.dynamics_model.update(inputs=data, sample_weights=None, batch_size=512, num_epochs=100,
                                   patience=5, validation_split=0.1, shuffle=True)

    @tf.function
    def unroll_trajectory(self, obs):
        act = self.agent.act_batch_explore_tf(obs)
        next_obs, rew, done = self.dynamics_model.predict_on_batch_tf(obs, act)
        done = tf.cast(done, tf.float32)
        return dict(
            act=act,
            next_obs=next_obs,
            rew=rew,
            done=done
        )

    def train_on_batch(self, data, **kwargs):
        data = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x), data)
        obs = data['obs']
        trajectory = self.unroll_trajectory(obs)
        data.update(trajectory)
        return self.agent.train_on_batch(data=data)

    def act_batch_explore(self, obs):
        return self.agent.act_batch_explore(obs)

    def act_batch_test(self, obs):
        return self.agent.act_batch_test(obs)


class Runner(rl_infra.runner.TFOffPolicyRunner):
    def setup_updater(self, update_after, policy_delay, update_per_step, update_every):
        self.update_after = update_after
        self.updater = MBPOUpdater(total_steps=self.epochs * self.steps_per_epoch,
                                   model_update_every=self.steps_per_epoch // 4,
                                   agent=self.agent,
                                   replay_buffer=self.replay_buffer,
                                   policy_delay=policy_delay,
                                   update_per_step=update_per_step,
                                   update_every=update_every
                                   )

    def setup_agent(self, agent_cls, **kwargs):
        from rlutils.gym import static
        static_fn = static.get_static_fn(self.env_name)
        self.agent = agent_cls(obs_spec=self.env.single_observation_space,
                               act_spec=self.env.single_action_space,
                               reward_fn=None,
                               terminate_fn=static_fn.terminate_fn_tf_batch,
                               **kwargs)

    @classmethod
    def main(cls,
             env_name,
             env_fn: Callable = None,
             exp_name: str = None,
             steps_per_epoch=1000,
             epochs=100,
             start_steps=1000,
             update_after=750,
             update_every=1,
             update_per_step=5,
             policy_delay=1,
             batch_size=400,
             num_parallel_env=1,
             num_test_episodes=30,
             seed=1,
             # replay
             replay_size=int(1e6),
             logger_path: str = None
             ):
        config = locals()
        runner = cls(seed=seed, steps_per_epoch=steps_per_epoch, epochs=epochs, exp_name=None,
                     logger_path=logger_path)
        runner.setup_env(env_name=env_name, env_fn=env_fn, num_parallel_env=1, asynchronous=False,
                         num_test_episodes=num_test_episodes)
        agent_kwargs = dict(

        )
        runner.setup_agent(agent_cls=MBPO, **agent_kwargs)
        runner.setup_replay_buffer(replay_size=replay_size,
                                   batch_size=batch_size)
        runner.setup_sampler(start_steps=start_steps)
        runner.setup_tester(num_test_episodes=num_test_episodes)
        runner.setup_updater(update_after=update_after,
                             policy_delay=policy_delay,
                             update_per_step=update_per_step,
                             update_every=update_every)
        runner.setup_logger(config=config, tensorboard=False)
        runner.run()


if __name__ == '__main__':
    rl_infra.runner.run_func_as_main(Runner.main)
