import unittest

import numpy as np
from rlutils import gym
from tqdm.auto import trange


class TestVectorEnv(unittest.TestCase):
    def test_vector_env(self):
        seed = 1
        sync_env = gym.vector.make('Pendulum-v0', asynchronous=True)
        async_env = gym.vector.make('Pendulum-v0', asynchronous=False)
        sync_env.seed(seed)
        async_env.seed(seed)
        sync_obs = sync_env.reset()
        async_obs = async_env.reset()
        for _ in trange(1000):
            np.testing.assert_almost_equal(sync_obs, async_obs)
            action = sync_env.action_space.sample()
            sync_obs, sync_rew, sync_done, _ = sync_env.step(action)
            async_obs, async_rew, async_done, _ = async_env.step(action)
            np.testing.assert_almost_equal(sync_rew, async_rew, decimal=5)
            np.testing.assert_almost_equal(sync_done, async_done)
            if np.any(sync_done):
                sync_obs = sync_env.reset_done()
                async_obs = async_env.reset_done()

    def test_vector_env_reset_obs(self):
        seed = 2
        sync_env = gym.vector.make('PendulumResetObs-v0', asynchronous=True)
        async_env = gym.vector.make('PendulumResetObs-v0', asynchronous=False)
        sync_env.seed(seed)
        async_env.seed(seed)
        sync_obs = sync_env.reset()
        async_obs = async_env.reset()
        for _ in trange(1000):
            np.testing.assert_almost_equal(sync_obs, async_obs)
            action = sync_env.action_space.sample()
            sync_obs, sync_rew, sync_done, _ = sync_env.step(action)
            async_obs, async_rew, async_done, _ = async_env.step(action)
            np.testing.assert_almost_equal(sync_rew, async_rew, decimal=5)
            np.testing.assert_almost_equal(sync_done, async_done)
            if np.any(sync_done):
                obs = sync_env.observation_space.sample()
                sync_obs = sync_env.reset_obs(obs, mask=sync_done)
                async_obs = async_env.reset_obs(obs, mask=sync_done)


if __name__ == '__main__':
    unittest.main()
