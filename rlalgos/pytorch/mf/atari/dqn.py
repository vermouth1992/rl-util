import rlutils.infra as rl_infra
import rlutils.np as rln
import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
from rlalgos.pytorch.mf.dqn import DQN


class AtariDQN(DQN):
    def __init__(self,
                 env,
                 frame_stack=4,
                 **kwargs
                 ):
        assert env.observation_space.shape == (84, 84), 'The environment must be Atari Games with 84x84 input'
        self.frame_stack = frame_stack
        super(AtariDQN, self).__init__(env=env, **kwargs)

    def _create_q_network(self):
        return rlu.nn.values.AtariDuelQModule(frame_stack=self.frame_stack, action_dim=self.act_dim)

    def _create_epsilon_greedy_scheduler(self):
        return rln.schedulers.LinearSchedule(schedule_timesteps=self.epsilon_greedy_steps,
                                             final_p=0.1,
                                             initial_p=1.0)


class Runner(rl_infra.runner.PytorchAtariRunner):
    @classmethod
    def main(cls,
             env_name,
             # agent args
             q_lr=1e-4,
             gamma=0.99,
             target_update_freq=2500,
             n_steps=1,
             **kwargs
             ):
        agent_kwargs = dict(
            q_lr=q_lr,
            gamma=gamma,
            target_update_freq=target_update_freq,
            epsilon_greedy_steps=1000000,
            n_steps=n_steps,
            device=ptu.get_device()
        )

        super(Runner, cls).main(env_name=env_name,
                                agent_cls=AtariDQN,
                                agent_kwargs=agent_kwargs,
                                batch_size=32,
                                update_every=4,
                                update_per_step=0.25,
                                num_test_episodes=10,
                                n_steps=n_steps,
                                **kwargs
                                )
