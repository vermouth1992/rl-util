import gym
import numpy as np
import torch

if __name__ == '__main__':
    env = gym.make('rlutils.gym:HopperMB-v4')

    done = False
    obs, info = env.reset()
    while not done:
        action = env.action_space.sample()
        next_obs, reward, terminate, truncated, info = env.step(action)

        terminate_model = env.terminate_fn_numpy_batch(np.expand_dims(obs, axis=0),
                                                       np.expand_dims(action, axis=0),
                                                       np.expand_dims(next_obs, axis=0))
        reward_model = env.reward_fn_numpy_batch(np.expand_dims(obs, axis=0),
                                                 np.expand_dims(action, axis=0),
                                                 np.expand_dims(next_obs, axis=0))

        terminate_model_torch = env.terminate_fn_torch_batch(torch.as_tensor(np.expand_dims(obs, axis=0)),
                                                             torch.as_tensor(np.expand_dims(action, axis=0)),
                                                             torch.as_tensor(np.expand_dims(next_obs, axis=0)))
        reward_model_torch = env.reward_fn_torch_batch(torch.as_tensor(np.expand_dims(obs, axis=0)),
                                                       torch.as_tensor(np.expand_dims(action, axis=0)),
                                                       torch.as_tensor(np.expand_dims(next_obs, axis=0)))

        print(terminate, terminate_model, terminate_model_torch)
        print(reward - reward_model, torch.as_tensor(reward) - reward_model_torch)

        obs = next_obs
        done = terminate or truncated
