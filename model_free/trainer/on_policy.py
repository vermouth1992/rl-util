from typing import Callable

import gym

import rlutils.gym
import rlutils.infra as rl_infra
from rlutils.logx import EpochLogger, setup_logger_kwargs
from rlutils.replay_buffers import GAEBuffer


def run_onpolicy(env_name,
                 env_fn: Callable = None,
                 exp_name: str = None,
                 seed=0,
                 gamma=0.99,
                 lam=0.97,
                 num_parallel_env=5,
                 asynchronous=False,
                 make_agent_fn: Callable = None,
                 batch_size=5000,
                 epochs=200,
                 logger_path: str = None,
                 backend=None
                 ):
    assert batch_size % num_parallel_env == 0
    num_steps_per_sample = batch_size // num_parallel_env

    config = locals()

    # setup seed
    seeder = rl_infra.Seeder(seed=seed, backend=backend)
    seeder.setup_global_seed()

    if env_fn is None:
        env_fn = lambda: gym.make(env_name)
        env_fn = rlutils.gym.utils.wrap_env_fn(env_fn, truncate_obs_dtype=True, normalize_action_space=True)

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
    env = rlutils.gym.utils.create_vector_env(env_fn=env_fn,
                                              num_parallel_env=num_parallel_env,
                                              asynchronous=asynchronous)
    env.action_space.seed(seeder.generate_seed())

    replay_buffer = GAEBuffer(env, length=num_steps_per_sample, gamma=gamma, lam=lam)

    sampler = rl_infra.samplers.TrajectorySampler(env=env)

    timer.set_logger(logger=logger)
    agent.set_logger(logger=logger)
    sampler.set_logger(logger=logger)

    sampler.reset()
    timer.start()

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}')
        sampler.sample(num_steps=num_steps_per_sample,
                       collect_fn=(agent.act_batch_explore, agent.get_value),
                       replay_buffer=replay_buffer)
        data = replay_buffer.get()
        agent.train_on_batch(data=data)

        logger.log_tabular('Epoch', epoch)
        logger.dump_tabular()
