import rlutils.infra as rl_infra
import rlutils.np as rln
import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
from rlutils.baselines.pytorch.mf.q_learning.categorical_dqn import CategoricalDQN
from rlutils.baselines.trainer import run_offpolicy_atari

if __name__ == '__main__':
    def make_q_net(env, num_atoms):
        net = rlu.nn.values.CategoricalAtariQModule(frame_stack=env.observation_space.shape[0],
                                                    action_dim=env.action_space.n,
                                                    num_atoms=num_atoms)
        print(net)
        return net


    epsilon_greedy_scheduler = rln.schedulers.LinearSchedule(schedule_timesteps=1000000,
                                                             final_p=0.1,
                                                             initial_p=1.0)

    make_agent_fn = lambda env: CategoricalDQN(env=env,
                                               make_q_net=make_q_net,
                                               epsilon_greedy_scheduler=epsilon_greedy_scheduler,
                                               test_random_prob=0.01,
                                               device=ptu.get_cuda_device(),
                                               v_min=-10.,
                                               v_max=10.,
                                               n_steps=3)
    rl_infra.runner.run_func_as_main(run_offpolicy_atari, passed_args={
        'make_agent_fn': make_agent_fn,
        'backend': 'torch',
        'n_steps': 3,
    })
