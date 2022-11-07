import os
from typing import Callable

import gym
import numpy as np
import torch
from tqdm.auto import trange

import rlutils.gym
import rlutils.infra as rl_infra
import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
from rlutils.logx import setup_logger_kwargs, EpochLogger
from rlutils.replay_buffers import UniformReplayBuffer


class RandomShooting(object):
    def __init__(self, env, dynamics_model, horizon=10, num_action_sequence=400):
        self.env = env
        self.dynamics_model = dynamics_model
        self.horizon = horizon
        self.num_action_sequence = num_action_sequence
        self.device = self.dynamics_model.device

    def plan(self, obs):
        """

        Args:
            obs: shape (None, obs_dim)

        Returns:

        """
        batch_size = obs.shape[0]
        # generate random action sequences
        action_sequence_size = (batch_size, self.num_action_sequence, self.horizon, self.env.action_space.shape[0])
        # (None, num_action_sequence, horizon, act_dim)
        action_sequence = rlu.distributions.uniform(size=action_sequence_size, low=-1., high=1.)
        action_sequence = action_sequence.view(batch_size * self.num_action_sequence, self.horizon,
                                               self.env.action_space.shape[0])
        obs = torch.as_tensor(obs, device=self.device)
        obs = obs.unsqueeze(dim=1).repeat(1, self.num_action_sequence, 1)  # (None, num_action_sequence, obs_dim)
        obs = obs.view(batch_size * self.num_action_sequence, self.env.observation_space.shape[0])

        rewards = torch.zeros(size=(batch_size * self.num_action_sequence))
        already_terminate = torch.zeros(size=(batch_size * self.num_action_sequence), dtype=torch.bool)
        for t in range(self.horizon):
            act = action_sequence[:, t]
            next_obs = self.dynamics_model.predict(obs, act)
            terminate = self.env.terminate_fn_torch_batch(obs, act, next_obs)
            reward = self.env.reward_fn_torch_batch(obs, act, next_obs)
            rewards[t] = reward * torch.logical_not(already_terminate)

            already_terminate = torch.logical_or(terminate, already_terminate)
            obs = next_obs

        # find the action sequence with maximum rewards
        action_sequence = action_sequence.view(batch_size, self.num_action_sequence, self.horizon,
                                               self.env.action_space.shape[0])
        rewards = rewards.view(batch_size, self.num_action_sequence)
        indices = torch.argmax(rewards, dim=-1)  # (None,)

        # (None, act_dim)
        optimal_action_sequence = action_sequence[torch.arange(batch_size, device=self.device), indices, 0]
        return ptu.to_numpy(optimal_action_sequence)


def pets(env_name,
         env_fn: Callable = None,
         exp_name: str = None,
         epochs=100,
         steps_per_epoch=1000,
         model_training_batch_size=256,
         model_training_epochs=100,
         model_training_freq=250,
         train_model_after=1000,
         seed=1,
         logger_path: str = None,
         backend='torch'
         ):
    device = ptu.get_cuda_device()
    logger_path = os.path.abspath(logger_path)

    config = locals()

    # setup seed
    seeder = rl_infra.Seeder(seed=seed, backend=backend)
    seeder.setup_global_seed()

    # setup logger
    if exp_name is None:
        exp_name = f'{env_name}-MBPO_test'
    logger_kwargs = setup_logger_kwargs(exp_name=exp_name, data_dir=logger_path, seed=seed)
    logger = EpochLogger(**logger_kwargs, tensorboard=False)
    logger.save_config(config)

    timer = rl_infra.StopWatch()

    if env_fn is None:
        env_fn = lambda: gym.make(env_name)

    env_fn = rlutils.gym.utils.wrap_env_fn(env_fn, truncate_obs_dtype=True, normalize_action_space=True)

    dummy_env = env_fn()

    env = rlutils.gym.utils.create_vector_env(env_fn=env_fn,
                                              num_parallel_env=1,
                                              asynchronous=False,
                                              action_space_seed=seeder.generate_seed())

    real_replay_size = epochs * steps_per_epoch
    real_replay_buffer = UniformReplayBuffer.from_env(env=dummy_env, capacity=real_replay_size,
                                                      seed=seeder.generate_seed(),
                                                      memory_efficient=False)
    dynamics_model = rlu.nn.MLPDynamics(env=dummy_env, num_ensembles=5, device=device)
    sampler = rl_infra.samplers.BatchSampler(env=env, n_steps=1, gamma=0.99,
                                             seed=seeder.generate_seed())

    planner = RandomShooting(dummy_env, dynamics_model, horizon=10, num_action_sequence=100)

    dynamics_model.set_logger(logger=logger)
    timer.set_logger(logger=logger)
    sampler.set_logger(logger=logger)

    global_step = 0
    timer.start()
    sampler.reset()

    sampler.sample(num_steps=train_model_after,
                   collect_fn=lambda o: np.asarray(env.action_space.sample()),
                   replay_buffer=real_replay_buffer)
    data = real_replay_buffer.storage.get()
    dynamics_model.fit(data=data, num_epochs=model_training_epochs, batch_size=model_training_batch_size)

    for epoch in range(1, epochs + 1):
        for t in trange(steps_per_epoch, desc=f'Epoch {epoch}/{epochs}'):
            sampler.sample(num_steps=1,
                           collect_fn=lambda obs: agent.act_batch_explore(obs, global_step),
                           replay_buffer=real_replay_buffer)

            global_step += 1

            if global_step % model_training_freq == 0:
                data = real_replay_buffer.storage.get()
                dynamics_model.fit(data=data, num_epochs=model_training_epochs, batch_size=model_training_batch_size)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.dump_tabular()


if __name__ == '__main__':
    rl_infra.runner.run_func_as_main(pets)
