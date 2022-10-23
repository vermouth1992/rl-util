"""
PPO for Atari Games
"""

import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
from model_free.policy_gradient.ppo import PPOAgent
from model_free.trainer import run_onpolicy
from rlutils.infra.runner import run_func_as_main
import gym


def main(env_name):
    make_agent_fn = lambda env: PPOAgent(env, actor_critic=rlu.nn.actor_critic.AtariActorCriticShared,
                                         device=ptu.get_cuda_device(), value_coef=0.1)

    frame_skip = 4 if 'NoFrameskip' in env_name else 1
    env_fn = lambda: gym.wrappers.FrameStack(
        gym.wrappers.AtariPreprocessing(env=gym.make(env_name), frame_skip=frame_skip), num_stack=4)
    run_onpolicy(env_name=env_name,
                 env_fn=env_fn,
                 backend='torch',
                 make_agent_fn=make_agent_fn,
                 batch_size=20000,
                 num_parallel_env=10,
                 asynchronous=True)


if __name__ == '__main__':
    run_func_as_main(main)
