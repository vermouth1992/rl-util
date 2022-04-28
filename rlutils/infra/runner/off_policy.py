import gym
import numpy as np

import rlutils.infra as rl_infra
from rlutils.replay_buffers import UniformReplayBuffer as ReplayBuffer
from rlutils.logx import EpochLogger, setup_logger_kwargs
import rlutils.gym

from tqdm.auto import trange

from typing import Callable


def run_offpolicy(env_name: str,
                  env_fn: Callable = None,
                  num_parallel_env=1,
                  asynchronous=False,
                  exp_name: str = None,
                  # agent
                  make_agent_fn: Callable = None,
                  # replay buffer
                  replay_size=1000000,
                  n_steps=1,
                  gamma=0.99,
                  # runner args
                  epochs=100,
                  steps_per_epoch=10000,
                  num_test_episodes=30,
                  start_steps=10000,
                  update_after=5000,
                  update_every=1,
                  update_per_step=1,
                  batch_size=256,
                  seed=1,
                  logger_path: str = None,
                  ):
    config = locals()

    # setup seed
    seeder = rl_infra.Seeder(seed=seed)
    seeder.setup_np_global_seed()
    seeder.setup_random_global_seed()

    if env_fn is None:
        env_fn = lambda: gym.make(env_name)

    # agent
    agent = make_agent_fn(env_fn())

    # setup logger
    if exp_name is None:
        exp_name = f'{env_name}_{agent.__class__.__name__}_test'
    assert exp_name is not None, 'Call setup_env before setup_logger if exp passed by the contructor is None.'
    logger_kwargs = setup_logger_kwargs(exp_name=exp_name, data_dir=logger_path, seed=seed)
    logger = EpochLogger(**logger_kwargs, tensorboard=False)
    logger.save_config(config)

    timer = rl_infra.StopWatch()

    # environment
    env_fn = rlutils.gym.utils.wrap_env_fn(env_fn, truncate_obs_dtype=True, normalize_action_space=True)

    env = rlutils.gym.utils.create_vector_env(env_fn=env_fn,
                                              num_parallel_env=num_parallel_env,
                                              asynchronous=asynchronous)

    env.seed(seeder.generate_seed())
    env.action_space.seed(seeder.generate_seed())

    # replay buffer
    replay_buffer = ReplayBuffer.from_env(env=env, capacity=replay_size, is_vec_env=True,
                                          seed=seeder.generate_seed(),
                                          memory_efficient=False)

    # setup sampler
    sampler = rl_infra.samplers.BatchSampler(env=env, n_steps=n_steps, gamma=gamma)

    # setup updater
    updater = rl_infra.OffPolicyUpdater(agent=agent,
                                        replay_buffer=replay_buffer,
                                        update_per_step=update_per_step,
                                        update_every=update_every,
                                        update_after=update_after,
                                        batch_size=batch_size)

    # setup tester
    test_env_seed = seeder.generate_seed()

    tester = rl_infra.Tester(env_fn=env_fn, num_parallel_env=num_test_episodes,
                             asynchronous=asynchronous, seed=test_env_seed)

    # register log_tabular args
    timer.set_logger(logger=logger)
    agent.set_logger(logger=logger)
    sampler.set_logger(logger=logger)
    tester.set_logger(logger=logger)

    sampler.reset()
    timer.start()
    global_step = 0
    for epoch in range(1, epochs + 1):
        for t in trange(steps_per_epoch, desc=f'Epoch {epoch}/{epochs}'):
            if sampler.total_env_steps < start_steps:
                sampler.sample(num_steps=1,
                               collect_fn=lambda o: np.asarray(env.action_space.sample()),
                               replay_buffer=replay_buffer)
            else:
                sampler.sample(num_steps=1,
                               collect_fn=lambda obs: agent.act_batch_explore(obs, global_step),
                               replay_buffer=replay_buffer)
            # Update handling
            updater.update(global_step)
            global_step += 1

        tester.test_agent(get_action=lambda obs: agent.act_batch_test(obs),
                          name=agent.__class__.__name__,
                          num_test_episodes=num_test_episodes)
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.dump_tabular()
