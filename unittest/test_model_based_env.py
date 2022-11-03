import gym
import torch
import numpy as np
from tqdm.auto import trange

import rlutils.gym

if __name__ == '__main__':
    env_fn = lambda: gym.make('rlutils.gym:HopperMB-v4')

    dummy_env = env_fn()
    env = rlutils.gym.utils.create_vector_env(env_fn=env_fn, num_parallel_env=3)

    done = False
    obs, info = env.reset()
    for _ in trange(2000):
        action = env.action_space.sample()
        o2, reward, terminate, truncate, infos = env.step(action)

        d = np.logical_or(terminate, truncate)
        if np.any(d):
            next_obs = np.copy(o2)
            terminal_obs = np.asarray(infos['final_observation'][d].tolist())
            next_obs[d] = terminal_obs
        else:
            next_obs = o2

        terminate_model = dummy_env.terminate_fn_numpy_batch(obs, action, next_obs)
        reward_model = dummy_env.reward_fn_numpy_batch(obs, action, next_obs)

        terminate_model_torch = dummy_env.terminate_fn_torch_batch(torch.as_tensor(obs),
                                                                   torch.as_tensor(action),
                                                                   torch.as_tensor(next_obs))
        reward_model_torch = dummy_env.reward_fn_torch_batch(torch.as_tensor(obs),
                                                             torch.as_tensor(action),
                                                             torch.as_tensor(next_obs))

        assert np.all(terminate == terminate_model)
        assert np.all(terminate == terminate_model_torch.numpy())

        assert np.allclose(reward, reward_model)
        assert np.allclose(reward, reward_model_torch.numpy())

        obs = o2
