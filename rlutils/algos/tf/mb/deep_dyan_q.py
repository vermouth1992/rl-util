"""
Deep reinforcement learning for DyanQ+ with priority sweeping
"""

import rlutils.infra as rl_infra
import rlutils.np as rln
import rlutils.tf as rlu
import tensorflow as tf
from rlutils.algos.tf.mf.sac import SACAgent


class DyanQUpdater(rl_infra.OffPolicyUpdater):
    def __init__(self, total_steps, model_update_every, **kwargs):
        super(DyanQUpdater, self).__init__(**kwargs)
        self.model_update_every = model_update_every
        self.total_steps = total_steps
        self.beta_scheduler = rln.schedulers.LinearSchedule(schedule_timesteps=total_steps,
                                                            final_p=1.0,
                                                            initial_p=0.4)

    def update(self, global_step):
        if global_step % self.update_every == 0:
            for _ in range(self.update_per_step * self.update_every):
                batch, idx = self.replay_buffer.sample(beta=self.beta_scheduler.value(global_step))
                batch['update_target'] = ((self.policy_updates + 1) % self.policy_delay == 0)
                priorities = self.agent.train_on_batch(data=batch)
                self.policy_updates += 1
                self.replay_buffer.update_priorities(idx=idx, priorities=priorities,
                                                     min_priority=1e-4,
                                                     max_priority=50)
        if global_step % self.model_update_every:
            self.agent.update_model(self.replay_buffer.get())


class DeepDyanQ(tf.keras.Model):
    def __init__(self, obs_spec, act_spec,
                 model_mlp_hidden=512, model_lr=1e-4, model_num_ensembles=5, reward_fn=None, terminate_fn=None,
                 policy_mlp_hidden=256, policy_lr=3e-4, behavior_mlp_hidden=256):
        super(DeepDyanQ, self).__init__()
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.act_dim = self.act_spec.shape[0]
        if len(self.obs_spec.shape) == 1:  # 1D observation
            self.obs_dim = self.obs_spec.shape[0]
        else:
            raise NotImplementedError
        self.dynamics_model = rlu.nn.EnsembleDynamicsModel(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                                           mlp_hidden=model_mlp_hidden,
                                                           num_ensembles=model_num_ensembles, lr=model_lr,
                                                           reward_fn=reward_fn, terminate_fn=terminate_fn)
        self.agent = SACAgent(obs_spec=obs_spec, act_spec=act_spec, policy_mlp_hidden=policy_mlp_hidden,
                              policy_lr=policy_lr, q_mlp_hidden=policy_mlp_hidden, q_lr=policy_lr, alpha=0.02,
                              alpha_lr=policy_lr, tau=5e-3, gamma=0.99, target_entropy=None, auto_alpha=True)
        self.behavior_policy = rlu.nn.BehaviorPolicy(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                                     mlp_hidden=behavior_mlp_hidden, beta=1.0)

    def set_logger(self, logger):
        self.dynamics_model.set_logger(logger)
        self.agent.set_logger(logger)

    def log_tabular(self):
        self.agent.log_tabular()
        self.dynamics_model.log_tabular()

    def update_model(self, data):
        self.dynamics_model.update(inputs=data, sample_weights=None, batch_size=256, num_epochs=100,
                                   patience=5, validation_split=0.1, shuffle=True)

    def train_on_batch(self, data, **kwargs):
        obs = data['obs']
        weights = data['weights']
        act = self.behavior_policy.act_batch(tf.convert_to_tensor(obs)).numpy()
        next_obs, rew, done = self.dynamics_model.predict_on_batch_tf(obs, act)
        data['act'] = act
        data['next_obs'] = next_obs
        data['rew'] = rew
        data['done'] = done
        data['weights'] = weights
        info = self.agent.train_on_batch(**data)
        self.logger.store(**rlu.functional.to_numpy_or_python_type(info))

    def act_batch_explore(self, obs):
        return self.agent.act_batch_explore(obs)

    def act_batch_test(self, obs):
        return self.agent.act_batch_test(obs)


class Runner(rl_infra.runner.TFOffPolicyRunner):
    def setup_updater(self, update_after, policy_delay, update_per_step, update_every):
        pass

    def setup_agent(self, agent_cls, **kwargs):
        from rlutils.gym import static
        static_fn = static.get_static_fn(self.env_name)
        self.agent = DeepDyanQ(obs_spec=self.env.single_observation_space,
                               act_spec=self.env.single_action_space,
                               reward_fn=None,
                               terminate_fn=static_fn.terminate_fn_tf_batch)

    @classmethod
    def main(cls,
             env_name,
             env_fn=None,
             exp_name=None,
             steps_per_epoch=10000,
             epochs=100,
             start_steps=10000,
             update_after=5000,
             update_every=1,
             update_per_step=1,
             policy_delay=1,
             batch_size=256,
             num_parallel_env=1,
             num_test_episodes=30,
             seed=1,
             # agent args
             agent_cls=None,
             agent_kwargs={},
             # replay
             replay_size=int(1e6),
             logger_path=None
             ):
        pass


if __name__ == '__main__':
    rl_infra.runner.run_func_as_main(Runner.main)
