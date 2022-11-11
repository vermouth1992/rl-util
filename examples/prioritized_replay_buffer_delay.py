"""
Benchmark delay between sampling and priority update in Prioritized Replay Buffer.
We directly copy the off-policy trainer
"""
import collections
from typing import Callable, Union

import gym
import numpy as np
from tqdm.auto import trange

import rlutils.gym
import rlutils.infra as rl_infra
from rlutils.logx import EpochLogger, setup_logger_kwargs
from rlutils.replay_buffers import PrioritizedReplayBuffer


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
                  priority_update_delay=10,
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
                  backend: Union[str, None] = None
                  ):
    config = locals()

    # setup seed
    seeder = rl_infra.Seeder(seed=seed, backend=backend)
    seeder.setup_global_seed()

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

    env.action_space.seed(seeder.generate_seed())

    # replay buffer
    replay_buffer = PrioritizedReplayBuffer.from_env(env=env_fn(), capacity=replay_size,
                                                     seed=seeder.generate_seed(),
                                                     memory_efficient=False)

    # setup sampler
    sampler = rl_infra.samplers.BatchSampler(env=env, n_steps=n_steps, gamma=gamma,
                                             seed=seeder.generate_seed())

    # setup tester
    tester = rl_infra.Tester(env_fn=env_fn, num_parallel_env=num_test_episodes,
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

    pre_sampled_buffer = collections.deque(maxlen=priority_update_delay)

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
                        while len(pre_sampled_buffer) != priority_update_delay:
                            batch = replay_buffer.sample(batch_size)
                            pre_sampled_buffer.append(batch)
                        transaction_id, data = pre_sampled_buffer.popleft()
                        info = agent.train_on_batch(data=data)
                        replay_buffer.update_priorities(transaction_id, ptu.to_numpy(info['TDError']))

                        policy_updates += 1

            global_step += 1

        tester.test_agent(get_action=lambda obs: agent.act_batch_test(obs),
                          name=agent.__class__.__name__,
                          num_test_episodes=num_test_episodes)
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('PolicyUpdates', policy_updates)
        logger.dump_tabular()


from rlutils.infra.runner import ExperimentGrid, run_func_as_main
from baselines.model_free.q_learning.dqn import DQN
from baselines.model_free.actor_critic.td3 import TD3Agent
import rlutils.pytorch.utils as ptu


class Benchmark(object):
    def run_cartpole(self):
        grid = ExperimentGrid()
        grid.add('env_name', vals='CartPole-v1', in_name=True)
        grid.add('make_agent_fn', vals=lambda env: DQN(env, device=ptu.get_cuda_device()))
        grid.add('steps_per_epoch', vals=2000)
        grid.add('start_steps', vals=2000)
        grid.add('update_after', vals=1000)
        grid.add('epochs', vals=100)
        grid.add('seed', vals=[1, 2, 3])
        grid.add('priority_update_delay', vals=[1, 10, 50, 200])
        grid.add('backend', vals='torch')

        grid.run(thunk=run_offpolicy, data_dir='data')

    def run_hopper(self):
        grid = ExperimentGrid()
        grid.add('env_name', vals='Hopper-v4', in_name=True)
        grid.add('make_agent_fn', vals=lambda env: TD3Agent(env, device=ptu.get_cuda_device()))
        grid.add('backend', vals='torch')
        grid.add('seed', vals=[1, 2, 3])
        grid.add('priority_update_delay', vals=[1, 10, 50, 200])

        grid.run(thunk=run_offpolicy, data_dir='data')

    def run_pong(self):
        pass


def main(task):
    benchmark = Benchmark()
    fn = eval(f'benchmark.run_{task}')
    fn()


if __name__ == '__main__':
    run_func_as_main(main)
