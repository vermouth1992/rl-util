"""
Due to the high computation requirement, we employ the following process:
- A process that collects data and performs learning. This controls the ratio between data collection and policy updates
- A process that samples data and performs dynamics training
- A process that samples data to perform rollouts
"""
import os.path
from typing import Callable

import gym
import numpy as np
import torch.optim
import torch.utils.data
from tqdm.auto import trange

import rlutils.gym
import rlutils.infra as rl_infra
import rlutils.np as rln
import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
from model_free.actor_critic.sac import SACAgent
from rlutils.logx import setup_logger_kwargs, EpochLogger
from rlutils.replay_buffers import UniformReplayBuffer


def generate_model_rollouts(env, agent, dynamics_model, real_dataset, rollout_length, model_rollout_batch_size):
    data = dict(
        obs=[],
        act=[],
        next_obs=[],
        rew=[],
        done=[]
    )

    obs = real_dataset.sample(model_rollout_batch_size)['obs']
    obs = torch.as_tensor(obs, device=agent.device)
    for _ in range(rollout_length):
        act = agent.act_batch_torch(obs, deterministic=False)
        next_obs = dynamics_model.predict(obs, act)
        terminate = env.terminate_fn_torch_batch(obs, act, next_obs)
        reward = env.reward_fn_torch_batch(obs, act, next_obs)

        data['obs'].append(obs)
        data['act'].append(act)
        data['next_obs'].append(next_obs)
        data['rew'].append(reward)
        data['done'].append(terminate)

        obs = next_obs.clone()  # the clone is critical.

        # if terminated, resample observations
        if torch.any(terminate):
            num_terminate = torch.sum(terminate).item()
            new_obs = real_dataset.sample(num_terminate)['obs']
            obs[terminate] = torch.as_tensor(new_obs, device=agent.device)

    for key, val in data.items():
        data[key] = ptu.to_numpy(torch.cat(val, dim=0))

    assert data['obs'].shape[0] == obs.shape[0] * rollout_length
    return data


def main(env_name,
         env_fn: Callable = None,
         exp_name: str = None,
         gamma=0.99,
         rollout_length=10,
         model_rollout_batch_size=400,
         model_training_batch_size=256,
         model_training_freq=250,
         model_training_epochs=100,
         agent_training_batch_size=256,
         agent_training_iterations=20,
         synthetic_replay_size=10000000,
         # runner args
         epochs=100,
         steps_per_epoch=1000,
         num_test_episodes=30,
         train_model_after=1000,
         seed=1,
         logger_path: str = None,
         backend='torch'):
    device = ptu.get_cuda_device()
    logger_path = os.path.abspath(logger_path)

    config = locals()

    # setup seed
    seeder = rl_infra.Seeder(seed=seed, backend=backend)
    seeder.setup_global_seed()

    # setup logger
    if exp_name is None:
        exp_name = f'{env_name}-MBPO_test'
    assert exp_name is not None, 'Call setup_env before setup_logger if exp passed by the contructor is None.'
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

    # replay buffer
    real_replay_size = epochs * steps_per_epoch + train_model_after
    real_replay_buffer = UniformReplayBuffer.from_env(env=dummy_env, capacity=real_replay_size,
                                                      seed=seeder.generate_seed(),
                                                      memory_efficient=False)
    synthetic_dataset = UniformReplayBuffer.from_env(env=dummy_env, capacity=synthetic_replay_size,
                                                     seed=seeder.generate_seed(),
                                                     memory_efficient=False)

    agent = SACAgent(env=dummy_env, target_entropy=-env.action_space.shape[0] // 2, device=device)
    dynamics_model = rlu.nn.MLPDynamics(env=dummy_env, device=device)
    sampler = rl_infra.samplers.BatchSampler(env=env, n_steps=1, gamma=gamma,
                                             seed=seeder.generate_seed())

    # setup tester
    tester = rl_infra.Tester(env_fn=env_fn, num_parallel_env=num_test_episodes,
                             asynchronous=False, seed=seeder.generate_seed())

    rollout_length_scheduler = rln.schedulers.PiecewiseSchedule(endpoints=[(1, 1), (20, 1), (100, 15)])

    dynamics_model.set_logger(logger=logger)
    timer.set_logger(logger=logger)
    agent.set_logger(logger=logger)
    sampler.set_logger(logger=logger)
    tester.set_logger(logger=logger)

    policy_updates = 0
    global_step = 0

    timer.start()
    sampler.reset()

    sampler.sample(num_steps=train_model_after,
                   collect_fn=lambda o: np.asarray(env.action_space.sample()),
                   replay_buffer=real_replay_buffer)
    data = real_replay_buffer.storage.get()
    dynamics_model.fit(data=data, num_epochs=model_training_epochs, batch_size=model_training_batch_size)

    for epoch in range(1, epochs + 1):
        # step 2: train dynamics model
        rollout_length = int(rollout_length_scheduler.value(epoch))
        print(f'Epoch {epoch}, rollout length: {rollout_length}')
        for t in trange(steps_per_epoch, desc=f'Epoch {epoch}/{epochs}'):
            # step 1: add data to true dataset
            sampler.sample(num_steps=1,
                           collect_fn=lambda obs: agent.act_batch_explore(obs, global_step),
                           replay_buffer=real_replay_buffer)

            # step 3: perform model rollouts
            synthetic_data = generate_model_rollouts(dummy_env, agent, dynamics_model, real_replay_buffer,
                                                     rollout_length, model_rollout_batch_size)
            synthetic_data['gamma'] = np.ones_like(synthetic_data['rew']) * gamma

            synthetic_dataset.add(synthetic_data)

            # step 4: perform policy training
            for _ in range(agent_training_iterations):
                data = synthetic_dataset.sample(agent_training_batch_size)
                agent.train_on_batch(data)
                policy_updates += 1

            global_step += 1

            if global_step % model_training_freq == 0:
                data = real_replay_buffer.storage.get()
                dynamics_model.fit(data=data, num_epochs=model_training_epochs, batch_size=model_training_batch_size)

        tester.test_agent(get_action=lambda obs: agent.act_batch_test(obs),
                          name=agent.__class__.__name__,
                          num_test_episodes=num_test_episodes)
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('PolicyUpdates', policy_updates)
        logger.dump_tabular()


if __name__ == '__main__':
    rl_infra.runner.run_func_as_main(main)
