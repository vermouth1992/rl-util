"""
A generic template to run Atari Environments. We use LazyFrame to save memory.
If there are enough memory, you can pass env_fn to the standard offpolicy runner to gain vectorized
performance on observation array indexing.
"""

import collections
from typing import Callable

import gym
import numpy as np
from tqdm.auto import trange

import rlutils.gym
import rlutils.infra as rl_infra
import rlutils.pytorch.utils as ptu
from rlutils.logx import EpochLogger, setup_logger_kwargs
from rlutils.replay_buffers import PrioritizedReplayBuffer


def run_offpolicy_atari(env_name: str,
                        env_fn: Callable = None,
                        num_parallel_env=1,
                        asynchronous=True,
                        exp_name: str = None,
                        # agent
                        make_agent_fn: Callable = None,
                        # replay buffer
                        priority_update_delay=1,
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
    replay_buffer = PrioritizedReplayBuffer.from_env(env=env_fn(), capacity=replay_size,
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
import torch
import rlutils.pytorch as rlu
import rlutils.np as rln
from baselines.model_free.q_learning.dqn import DQN


class Benchmark(object):
    def run_pong(self):
        def make_q_net(env):
            net = rlu.nn.values.LazyAtariDuelQModule(action_dim=env.action_space.n)
            dummy_inputs = torch.randn(1, *env.observation_space.shape)
            net(dummy_inputs)
            print(net)
            return net

        epsilon_greedy_scheduler = rln.schedulers.LinearSchedule(schedule_timesteps=1000000,
                                                                 final_p=0.1,
                                                                 initial_p=1.0)

        make_agent_fn = lambda env: DQN(env=env,
                                        make_q_net=make_q_net,
                                        epsilon_greedy_scheduler=epsilon_greedy_scheduler,
                                        device=ptu.get_cuda_device())

        grid = ExperimentGrid()
        grid.add('env_name', vals='ALE/Pong-v5', in_name=True)
        grid.add('make_agent_fn', vals=make_agent_fn)
        grid.add('seed', vals=[1, 2, 3])
        grid.add('priority_update_delay', vals=[1, 10, 50, 200])
        grid.add('backend', vals='torch')

        grid.run(thunk=run_offpolicy_atari, data_dir='data')


def main(task):
    benchmark = Benchmark()
    fn = eval(f'benchmark.run_{task}')
    fn()


if __name__ == '__main__':
    run_func_as_main(main)
