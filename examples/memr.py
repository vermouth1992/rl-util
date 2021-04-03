"""
Code implement MEMR in https://arxiv.org/abs/2006.04802
"""

from typing import Callable
from typing import Dict

import gym
import numpy as np
import rlutils.algos.tf.mf.sac as sac
import rlutils.infra as rl_infra
import rlutils.np as rln
import rlutils.replay_buffers as replay_buffers
import rlutils.tf as rlu
import tensorflow as tf


class PyPrioritizedReplayBuffer(replay_buffers.PyPrioritizedReplayBuffer):
    def reset(self):
        super(PyPrioritizedReplayBuffer, self).reset()
        self.last_priority_ptr = 0

    def get_priority_uninitialized_data(self):
        idx = np.arange(self.last_priority_ptr, len(self))
        self.last_priority_ptr = len(self)
        return idx, self.__getitem__(idx)


class SegmentReplayBuffer(replay_buffers.PyReplayBuffer):
    """
    When adding, the input size must be equal to the segment size.
    When sampling, first sample a segment, then sample the data.
    """

    def __init__(self, segment_size, *args, **kwargs):
        self.segment_size = segment_size
        super(SegmentReplayBuffer, self).__init__(*args, **kwargs)

    def add(self, data: Dict[str, np.ndarray]):
        batch_size = list(data.values())[0].shape[0]
        assert batch_size == self.segment_size
        super(SegmentReplayBuffer, self).add(data=data)

    def sample(self):
        assert not self.is_empty()
        num_segment = len(self) // self.segment_size
        segment_idx = self.np_random.randint(num_segment)
        start = segment_idx * self.segment_size
        end = (segment_idx + 1) * self.segment_size
        idx_range = np.arange(start, end)
        idxs = self.np_random.choice(idx_range, size=self.batch_size)
        return self.__getitem__(idxs)


class MEMRUpdater(rl_infra.OffPolicyUpdater):
    def __init__(self, total_steps, model_update_every, model_replay_buffer, model_rollout_freq,
                 **kwargs):
        super(MEMRUpdater, self).__init__(**kwargs)
        self.total_steps = total_steps
        self.model_update_every = model_update_every
        self.model_replay_buffer = model_replay_buffer
        self.model_rollout_freq = model_rollout_freq
        self.beta_scheduler = rln.schedulers.LinearSchedule(total_steps, initial_p=0.4, final_p=1.0)
        self.update_scheduler = rln.schedulers.LinearSchedule(total_steps, initial_p=self.update_per_step,
                                                              final_p=self.update_per_step)

    def log_tabular(self):
        super(MEMRUpdater, self).log_tabular()
        self.logger.log_tabular('Priority', with_min_and_max=True)

    def update(self, global_step):
        if global_step % self.model_update_every == 0:
            self.agent.update_model(self.replay_buffer.get())

        if global_step % self.model_rollout_freq == 0:
            # update uninitialized priority
            idx, data = self.replay_buffer.get_priority_uninitialized_data()
            priorities = self.agent.compute_priority(obs=data['obs'])
            self.replay_buffer.update_priorities(idx, priorities)
            # sample obs and unroll transitions.
            data, idx = self.replay_buffer.sample(beta=self.beta_scheduler.value(global_step))
            transitions = self.agent.unroll_trajectory(data['obs'])
            transitions['weights'] = data['weights'] / np.mean(data['weights'])
            self.agent.behavior_policy.train_on_batch(x=transitions['obs'],
                                                      y=transitions['act'])
            # update the priority after update the behavior policy
            priorities = self.agent.compute_priority(obs=transitions['obs']).numpy()
            self.replay_buffer.update_priorities(idx, priorities)
            self.logger.store(Priority=priorities)
            # add transitions to the model_replay_buffer
            transitions = rlu.functional.to_numpy_or_python_type(transitions)
            self.model_replay_buffer.add(transitions)

        if global_step % self.update_every == 0:
            update_per_step = int(self.update_scheduler.value(global_step))
            for _ in range(self.update_every):
                for _ in range(update_per_step):
                    batch = self.model_replay_buffer.sample()
                    batch['update_target'] = ((self.policy_updates + 1) % self.policy_delay == 0)
                    self.agent.train_on_batch(data=batch)
                    self.policy_updates += 1


class SquashedGaussianMLPActor(rlu.nn.SquashedGaussianMLPActor):
    @tf.function
    def compute_log_prob_and_log_std(self, obs, raw_actions):
        pi_distribution = self((obs, tf.constant(False)))[-1]
        logp_pi = pi_distribution.log_prob(raw_actions)
        log_std = tf.reduce_mean(tf.math.log(pi_distribution.stddev()), axis=-1)
        return logp_pi / self.ac_dim, log_std


class MEMRAgent(tf.keras.Model):
    def __init__(self, obs_spec, act_spec,
                 model_mlp_hidden=512, model_lr=1e-3, model_num_ensembles=5, reward_fn=None, terminate_fn=None,
                 policy_mlp_hidden=256, policy_lr=3e-4):
        super(MEMRAgent, self).__init__()
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.act_dim = self.act_spec.shape[0]
        if len(self.obs_spec.shape) == 1:  # 1D observation
            self.obs_dim = self.obs_spec.shape[0]
        else:
            raise NotImplementedError
        self.dynamics_model = rlu.nn.EnsembleWorldModel(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                                        mlp_hidden=model_mlp_hidden, num_layers=4,
                                                        num_ensembles=model_num_ensembles, lr=model_lr,
                                                        reward_fn=reward_fn, terminate_fn=terminate_fn)
        self.agent = sac.SACAgent(obs_spec=obs_spec, act_spec=act_spec, policy_mlp_hidden=policy_mlp_hidden,
                                  policy_lr=policy_lr, q_mlp_hidden=policy_mlp_hidden, q_lr=policy_lr, alpha=1.0,
                                  alpha_lr=policy_lr, tau=5e-3, gamma=0.99, target_entropy=-(self.act_dim // 2),
                                  auto_alpha=True)
        self.behavior_policy = SquashedGaussianMLPActor(ob_dim=self.obs_dim, ac_dim=self.act_dim,
                                                        mlp_hidden=policy_mlp_hidden)
        self.behavior_policy.compile(optimizer=rlu.future.get_adam_optimizer(policy_lr))

    def set_logger(self, logger):
        self.dynamics_model.set_logger(logger)
        self.agent.set_logger(logger)

    def log_tabular(self):
        self.agent.log_tabular()
        self.dynamics_model.log_tabular()

    @tf.function
    def compute_priority(self, obs):
        raw_actions = self.agent.policy_net((obs, tf.constant(False)))[2]
        logp_pi, log_std = self.behavior_policy.compute_log_prob_and_log_std(obs, raw_actions)
        constant = 0.5 * np.log(2. * np.pi)
        priorities = -(logp_pi + log_std + constant)
        priorities = tf.clip_by_value(priorities, clip_value_min=1e-4, clip_value_max=50.)
        return priorities

    def update_model(self, data):
        self.dynamics_model.update(inputs=data, sample_weights=None, batch_size=512, num_epochs=100,
                                   patience=5, validation_split=0.1, shuffle=True)

    @tf.function
    def unroll_trajectory(self, obs):
        act = self.agent.act_batch_explore_tf(obs)
        next_obs, rew, done = self.dynamics_model.predict_on_batch_tf(obs, act)
        done = tf.cast(done, tf.float32)
        return dict(
            obs=obs,
            act=act,
            next_obs=next_obs,
            rew=rew,
            done=done
        )

    def train_on_batch(self, data, **kwargs):
        return self.agent.train_on_batch(data=data)

    def act_batch_explore(self, obs):
        return self.agent.act_batch_explore(obs)

    def act_batch_test(self, obs):
        return self.agent.act_batch_test(obs)


class Runner(rl_infra.runner.TFOffPolicyRunner):
    def on_epoch_end(self, epoch):
        self.tester.test_agent(get_action=lambda obs: self.agent.act_batch_test(obs),
                               name=self.agent.__class__.__name__,
                               num_test_episodes=self.num_test_episodes)
        # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.tester.log_tabular()
        self.sampler.log_tabular()
        self.updater.log_tabular()
        self.logger.log_tabular(key='EnvDatsetSize', val=len(self.replay_buffer))
        self.logger.log_tabular(key='ModelDatasetSize', val=len(self.model_replay_buffer))
        self.timer.log_tabular()
        self.logger.dump_tabular()

    def setup_updater(self, update_after, policy_delay, update_per_step, update_every, model_rollout_freq=None):
        self.update_after = update_after
        self.updater = MEMRUpdater(total_steps=self.epochs * self.steps_per_epoch,
                                   model_update_every=self.steps_per_epoch // 4,
                                   agent=self.agent,
                                   replay_buffer=self.replay_buffer,
                                   model_replay_buffer=self.model_replay_buffer,
                                   model_rollout_freq=model_rollout_freq,
                                   policy_delay=policy_delay,
                                   update_per_step=update_per_step,
                                   update_every=update_every
                                   )

    def setup_agent(self, **kwargs):
        from rlutils.gym import static
        static_fn = static.get_static_fn(self.env_name)
        self.agent = MEMRAgent(obs_spec=self.env.single_observation_space,
                               act_spec=self.env.single_action_space,
                               reward_fn=None,
                               terminate_fn=static_fn.terminate_fn_tf_batch,
                               **kwargs)

    def setup_replay_buffer(self, replay_size, batch_size, segment_size=None):
        segment_size = segment_size
        self.seeds_info['replay_buffer'] = self.seeder.generate_seed()
        self.seeds_info['model_replay_buffer'] = self.seeder.generate_seed()
        self.replay_buffer = PyPrioritizedReplayBuffer.from_vec_env(vec_env=self.env, capacity=replay_size,
                                                                    batch_size=segment_size, alpha=0.6,
                                                                    seed=self.seeds_info['replay_buffer'])
        data_spec = {
            'obs': self.env.single_observation_space,
            'act': self.env.single_action_space,
            'next_obs': self.env.single_observation_space,
            'rew': gym.spaces.Space(shape=None, dtype=np.float32),
            'done': gym.spaces.Space(shape=None, dtype=np.float32),
            'weights': gym.spaces.Space(shape=None, dtype=np.float32)
        }
        self.model_replay_buffer = SegmentReplayBuffer(data_spec=data_spec, capacity=replay_size,
                                                       batch_size=batch_size,
                                                       seed=self.seeds_info['model_replay_buffer'],
                                                       segment_size=segment_size)

    @classmethod
    def main(cls,
             env_name,
             env_fn: Callable = None,
             exp_name: str = None,
             steps_per_epoch=1000,
             epochs=100,
             start_steps=3000,
             update_after=750,
             update_every=50,
             update_per_step=10,
             policy_delay=1,
             batch_size=4000,
             model_rollout_freq=50,
             num_model_rollouts=400,
             num_parallel_env=1,
             num_test_episodes=30,
             seed=1,
             # agent
             model_mlp_hidden=512,
             model_lr=1e-3,
             model_num_ensembles=5,
             policy_mlp_hidden=256,
             policy_lr=3e-4,
             # replay
             replay_size=int(1e6),
             logger_path='data'
             ):
        config = locals()
        runner = cls(seed=seed, steps_per_epoch=steps_per_epoch, epochs=epochs, exp_name=None,
                     logger_path=logger_path)
        runner.setup_env(env_name=env_name, env_fn=env_fn, num_parallel_env=1, asynchronous=False,
                         num_test_episodes=num_test_episodes)
        agent_kwargs = dict(

        )
        runner.setup_agent(**agent_kwargs)
        runner.setup_replay_buffer(replay_size=replay_size,
                                   batch_size=batch_size,
                                   segment_size=num_model_rollouts * model_rollout_freq
                                   )
        runner.setup_sampler(start_steps=start_steps)
        runner.setup_tester(num_test_episodes=num_test_episodes)
        runner.setup_updater(update_after=update_after,
                             policy_delay=policy_delay,
                             update_per_step=update_per_step,
                             update_every=update_every,
                             model_rollout_freq=model_rollout_freq)
        runner.setup_logger(config=config, tensorboard=False)
        runner.run()


if __name__ == '__main__':
    rl_infra.runner.run_func_as_main(Runner.main)
