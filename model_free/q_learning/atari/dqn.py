import torch

import rlutils.infra as rl_infra
import rlutils.np as rln
import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
from mf.q_learning.dqn import DQN
from mf.trainer import run_offpolicy_atari

if __name__ == '__main__':
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
                                    test_random_prob=0.01,
                                    n_steps=3,
                                    device=ptu.get_cuda_device())

    rl_infra.runner.run_func_as_main(run_offpolicy_atari, passed_args={
        'make_agent_fn': make_agent_fn,
        'backend': 'torch',
        'n_steps': 3,
    })
