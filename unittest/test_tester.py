import unittest
import gym
import numpy as np
import torch

import rlutils.infra as rl_infra
import rlutils.pytorch as rlu


class TestTester(unittest.TestCase):
    def test_tester(self):
        for env_name in ['Hopper-v2', 'HalfCheetah-v2', 'Walker2d-v2', 'Ant-v2']:
            env_fn = lambda: gym.make(env_name)
            dummy_env = env_fn()
            network = rlu.nn.build_mlp(input_dim=dummy_env.observation_space.shape[0],
                                       output_dim=dummy_env.action_space.shape[0],
                                       mlp_hidden=8)
            tester = rl_infra.tester.Tester(env_fn=env_fn, num_parallel_env=10, seed=1)

            manual_env = [env_fn() for _ in range(10)]
            for i, env in enumerate(manual_env):
                env.seed(1 + i)

            def get_action_batch(obs):
                with torch.no_grad():
                    return network(torch.as_tensor(obs, dtype=torch.float32)).cpu().numpy()

            all_ep_ret, all_ep_len = tester.test_agent(get_action=get_action_batch, name='Random', num_test_episodes=30)

            manual_all_ep_ret = []
            manual_all_ep_len = []

            for _ in range(3):
                for env in manual_env:
                    done = False
                    obs = env.reset()
                    ep_ret = 0.
                    ep_len = 0
                    while not done:
                        with torch.no_grad():
                            obs = torch.as_tensor(obs, dtype=torch.float32)
                            obs = torch.unsqueeze(obs, dim=0)
                            act = network(obs).numpy()[0]

                        obs, rew, done, info = env.step(act)
                        ep_ret += rew
                        ep_len += 1

                    manual_all_ep_ret.append(ep_ret)
                    manual_all_ep_len.append(ep_len)

            manual_all_ep_ret = np.array(manual_all_ep_ret)
            manual_all_ep_len = np.array(manual_all_ep_len)

            np.testing.assert_almost_equal(all_ep_ret, manual_all_ep_ret, decimal=4)
            np.testing.assert_equal(all_ep_len, manual_all_ep_len)


if __name__ == '__main__':
    unittest.main()
