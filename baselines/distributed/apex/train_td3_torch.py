import gym
import torch
from torch import nn

import rlutils.gym as rl_gym
import rlutils.infra as rl_infra
import rlutils.pytorch as rlu
import rlutils.replay_buffers as rb
from baselines.distributed.apex.trainer import run_apex
from baselines.model_free.actor_critic.td3 import TD3Agent


def run_apex_td3_torch(env_name,
                       exp_name: str = None,
                       asynchronous=False,
                       replay_capacity=1000000,
                       num_actors=2,
                       num_test_episodes=30,
                       local_buffer_capacity=1000,
                       env_steps_before_start=10000,
                       total_num_policy_updates=1000000,
                       weight_update_freq=50,
                       weight_push_freq=10,
                       logging_freq=10000,
                       batch_size=256,
                       gamma=0.99,
                       n_steps=1,
                       seed=1,
                       logger_path: str = None,
                       device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'

    if exp_name is None:
        exp_name = f'apex-td3-{exp_name}'

    config = locals()

    train_env_fn = rl_gym.utils.wrap_env_fn(env_fn=lambda: gym.make(env_name))
    dummy_env = train_env_fn()
    test_env_fn = train_env_fn

    create_vec_env_fn = lambda: rl_gym.utils.create_vector_env(env_fn=train_env_fn, num_parallel_env=1,
                                                               asynchronous=False,
                                                               action_space_seed=seed)
    make_sampler_fn = lambda: rl_infra.samplers.BatchSampler(env=create_vec_env_fn(), n_steps=n_steps, gamma=gamma)

    make_replay_fn = lambda: rb.PrioritizedReplayBuffer.from_env(env=dummy_env,
                                                                 capacity=replay_capacity,
                                                                 memory_efficient=False)
    make_local_buffer_fn = lambda: rb.UniformReplayBuffer.from_env(env=dummy_env, capacity=local_buffer_capacity,
                                                                   memory_efficient=False)

    make_tester_fn = lambda: rl_infra.tester.Tester(env_fn=test_env_fn, num_parallel_env=num_test_episodes,
                                                    asynchronous=asynchronous)

    actor_fn = lambda: TD3Agent(env=train_env_fn(), device='cpu', actor_noise=0.3)
    learner_fn = lambda: TD3Agent(env=train_env_fn(), device='cuda', actor_noise=0.3)

    def set_thread_fn(num_threads):
        import torch
        torch.set_num_threads(num_threads)

    def set_weights_fn(agent: nn.Module, weights):
        agent.load_state_dict(weights)

    def get_weights_fn(agent: nn.Module):
        return rlu.nn.functional.get_state_dict(agent)

    run_apex(
        make_actor_fn_lst=[actor_fn for _ in range(num_actors)],
        make_learner_fn=learner_fn,
        make_test_agent_fn=actor_fn,
        make_sampler_fn=make_sampler_fn,
        make_local_buffer_fn=make_local_buffer_fn,
        make_replay_fn=make_replay_fn,
        make_tester_fn=make_tester_fn,
        set_thread_fn=set_thread_fn,
        set_weights_fn=set_weights_fn,
        get_weights_fn=get_weights_fn,
        exp_name=exp_name,
        config=config,
        # actor args
        num_cpus_per_actor=1,
        weight_update_freq=weight_update_freq,
        # learner args
        num_cpus_per_learner=1,
        num_gpus_per_learner=1 if torch.cuda.is_available() else 0,
        batch_size=batch_size,
        weight_push_freq=weight_push_freq,
        update_after=env_steps_before_start,
        # logging
        total_num_policy_updates=total_num_policy_updates,
        logging_freq=logging_freq,
        num_test_episodes=num_test_episodes,
        num_cpus_tester=1,
        logger_path=logger_path,
        seed=seed
    )


if __name__ == '__main__':
    rl_infra.runner.run_func_as_main(run_apex_td3_torch)
