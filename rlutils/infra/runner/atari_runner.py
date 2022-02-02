import gym

import rlutils.gym as rl_gym
import rlutils.infra as rl_infra
from rlutils.replay_buffers import UniformMemoryEfficientPyDictReplayBuffer as ReplayBuffer
from .off_policy import OffPolicyRunner


class AtariRunner(OffPolicyRunner):
    def setup_replay_buffer(self,
                            replay_size,
                            **kwargs):
        self.seeds_info['replay_buffer'] = self.seeder.generate_seed()
        self.replay_buffer = ReplayBuffer.from_vec_env(vec_env=self.env, capacity=replay_size,
                                                       seed=self.seeds_info['replay_buffer'],
                                                       )

    def setup_tester(self, num_test_episodes, **kwargs):
        env_fn = self.env_fn
        self.env_fn = lambda: gym.wrappers.FrameStack(env_fn(), num_stack=4)
        super(AtariRunner, self).setup_tester(num_test_episodes)

    def setup_env(self,
                  env_name,
                  env_fn=None,
                  num_parallel_env=1,
                  asynchronous=False,
                  num_test_episodes=None):
        assert env_fn is None
        env_fn = rl_gym.utils.create_atari_env_fn(env_name)
        # we handle frame stack in the sampler
        super(AtariRunner, self).setup_env(env_name=env_name, env_fn=env_fn,
                                           num_parallel_env=num_parallel_env,
                                           asynchronous=asynchronous,
                                           num_test_episodes=num_test_episodes)

    def setup_sampler(self, start_steps, n_steps, gamma, **kwargs):
        self.start_steps = start_steps
        self.sampler = rl_infra.samplers.BatchFrameStackSampler(env=self.env, n_steps=n_steps, gamma=gamma)
