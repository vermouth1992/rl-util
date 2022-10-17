"""
A generic template to run Atari Environments. We use LazyFrame to save memory.
If there are enough memory, you can pass env_fn to the standard offpolicy runner to gain vectorized
performance on observation array indexing.
"""

import gym
import numpy as np

import rlutils.infra as rl_infra
from rlutils.replay_buffers import UniformReplayBuffer as ReplayBuffer
from rlutils.logx import EpochLogger, setup_logger_kwargs
import rlutils.gym

from tqdm.auto import trange

from typing import Callable


def run_offpolicy_atari(env_name: str,
                        env_fn: Callable = None,
                        num_parallel_env=1,
                        asynchronous=True,
                        exp_name: str = None,
                        # agent
                        make_agent_fn: Callable = None,
                        # replay buffer
                        replay_size=1000000,
                        n_steps=1,
                        gamma=0.99,
                        num_stack=4,
                        # runner args
                        epochs=200,
                        steps_per_epoch=10000,
                        num_test_episodes=10,
                        test_random_prob=0.01,
                        start_steps=10000,
                        update_after=5000,
                        update_every=4,
                        update_per_step=0.25,
                        batch_size=64,
                        seed=1,
                        logger_path: str = None,
                        backend=None
                        ):
    assert rlutils.gym.utils.is_atari_env(env_name)

    config = locals()

    # setup seed
    seeder = rl_infra.Seeder(seed=seed, backend=backend)
    seeder.setup_np_global_seed()
    seeder.setup_random_global_seed()
    seeder.setup_backend_seed()

    # environment
    if env_fn is None:
        frame_skip = 4 if 'NoFrameskip' in env_name else 1
        env_fn = lambda: gym.wrappers.AtariPreprocessing(env=gym.make(env_name), frame_skip=frame_skip)

    # agent
    agent = make_agent_fn(gym.wrappers.FrameStack(env_fn(), num_stack=num_stack))

    # setup logger
    if exp_name is None:
        exp_name = f'{env_name.replace("/", "-")}_{agent.__class__.__name__}_test'
    logger_kwargs = setup_logger_kwargs(exp_name=exp_name, data_dir=logger_path, seed=seed)
    logger = EpochLogger(**logger_kwargs, tensorboard=False)
    logger.save_config(config)

    timer = rl_infra.StopWatch()

    # no frame stack in this environment. (parallel_envs, 84, 84)
    env = rlutils.gym.utils.create_vector_env(env_fn=env_fn,
                                              num_parallel_env=num_parallel_env,
                                              asynchronous=asynchronous)

    env.action_space.seed(seeder.generate_seed())

    # replay buffer
    replay_buffer = ReplayBuffer.from_env(env=env_fn(), capacity=replay_size,
                                          seed=seeder.generate_seed(),
                                          memory_efficient=True)

    # setup sampler
    sampler = rl_infra.samplers.BatchFrameStackSampler(env=env, n_steps=n_steps, gamma=gamma,
                                                       seed=seeder.generate_seed(),
                                                       num_stack=num_stack)

    # setup tester
    test_env_fn = lambda: rlutils.gym.wrappers.RandomAction(
        env=gym.wrappers.FrameStack(env=env_fn(), num_stack=num_stack), prob=test_random_prob)
    tester = rl_infra.Tester(env_fn=test_env_fn, num_parallel_env=num_test_episodes,
                             asynchronous=asynchronous, seed=seeder.generate_seed())

    # register log_tabular args
    timer.set_logger(logger=logger)
    agent.set_logger(logger=logger)
    sampler.set_logger(logger=logger)
    tester.set_logger(logger=logger)

    sampler.reset()
    timer.start()
    global_step = 0
    policy_updates = 0

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
            if global_step > update_after:
                if global_step % update_every == 0:
                    for _ in range(int(update_per_step * update_every)):
                        batch = replay_buffer.sample(batch_size)
                        agent.train_on_batch(data=batch)
                        policy_updates += 1

            global_step += 1

        tester.test_agent(get_action=lambda obs: agent.act_batch_test(obs),
                          name=agent.__class__.__name__,
                          num_test_episodes=num_test_episodes)
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('PolicyUpdates', policy_updates)
        logger.dump_tabular()
