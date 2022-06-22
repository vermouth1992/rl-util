"""
Test infrastructure
"""

import os
import unittest

import gym
import numpy as np
import rlutils.infra as rl_infra
import rlutils.pytorch as rlu
from gym.wrappers import RescaleAction

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
        for asynchronous in [True, False]:
            id = 'Pendulum-v0'
            num_envs = 5
            seed = 11
            num_test_episodes = 30
            env_fn = lambda: gym.make(id)
            env = env_fn()
            mlp = rlu.nn.build_mlp(input_dim=env.observation_space.shape[0],
                                   output_dim=env.action_space.shape[0],
                                   mlp_hidden=8,
                                   out_activation='tanh')

            get_action = lambda obs: mlp(tf.convert_to_tensor(obs, dtype=tf.float32)).numpy()

            def get_action_single(obs):
                obs = np.expand_dims(obs, axis=0)
                return get_action(obs)[0]

            tester = rl_infra.Tester(env_fn=env_fn, num_parallel_env=num_envs,
                                     asynchronous=asynchronous, seed=seed)
            ep_ret, ep_len = tester.test_agent(get_action=get_action, name='random',
                                               num_test_episodes=num_test_episodes)

            # ep_ret_1, ep_len_1 = tester.test_agent(get_action=get_action, name='random',
            #                                        num_test_episodes=num_test_episodes)
            # make 20 individual envs, run them independently
            env_lst = [RescaleAction(gym.make(id), -1., 1.) for _ in range(num_envs)]
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

            np.testing.assert_almost_equal(ep_ret, single_ep_ret, decimal=5)
            np.testing.assert_almost_equal(ep_len, single_ep_len)


if __name__ == '__main__':
    unittest.main()
