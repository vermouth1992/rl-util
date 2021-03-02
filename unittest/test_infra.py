"""
Test infrastructure
"""

import os
import unittest

import gym
import numpy as np
import rlutils.gym
import rlutils.infra as rl_infra
import rlutils.tf as rlu
import tensorflow as tf

os.environ['TF_DETERMINISTIC_OPS'] = '1'
from tqdm.auto import tqdm


class TestInfra(unittest.TestCase):
    def run_single_env(self, env, get_action):
        done = False
        obs = env.reset()
        ep_ret = np.array(0., dtype=np.float64)
        ep_len = 0
        while not done:
            a = get_action(obs)
            obs, r, done, info = env.step(a)
            ep_ret += r
            ep_len += 1
        return ep_ret, ep_len

    def test_parallel_env_tester(self):
        # do not test using Mujoco as it is indeed stochastic, especially the HalfCheetah environment.
        id = 'Pendulum-v0'
        num_envs = 5
        seed = 10
        num_test_episodes = 10
        env = rlutils.gym.vector.make(id, num_envs=num_envs, asynchronous=False)
        env.seed(seed)
        mlp = rlu.nn.build_mlp(input_dim=env.single_observation_space.shape[0],
                               output_dim=env.single_action_space.shape[0],
                               mlp_hidden=8,
                               out_activation='tanh')

        get_action = lambda obs: mlp(tf.convert_to_tensor(obs, dtype=tf.float32)).numpy()

        def get_action_single(obs):
            obs = np.expand_dims(obs, axis=0)
            return get_action(obs)[0]

        tester = rl_infra.Tester(test_env=env)
        ep_ret, ep_len = tester.test_agent(get_action=get_action, name='random', num_test_episodes=num_test_episodes)

        # ep_ret_1, ep_len_1 = tester.test_agent(get_action=get_action, name='random',
        #                                        num_test_episodes=num_test_episodes)
        # make 20 individual envs, run them independently
        env_lst = [gym.make(id) for _ in range(num_envs)]
        for i, e in enumerate(env_lst):
            e.seed(seed + i)

        single_ep_ret = []
        single_ep_len = []
        for _ in range(num_test_episodes // num_envs):
            for e in tqdm(env_lst):
                ret, len = self.run_single_env(e, get_action_single)
                single_ep_ret.append(ret)
                single_ep_len.append(len)
        single_ep_ret = np.stack(single_ep_ret)
        single_ep_len = np.stack(single_ep_len)

        np.testing.assert_almost_equal(ep_ret, single_ep_ret)
        np.testing.assert_almost_equal(ep_len, single_ep_len)


if __name__ == '__main__':
    unittest.main()
