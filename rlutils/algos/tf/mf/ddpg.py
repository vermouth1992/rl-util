"""
Twin Delayed DDPG. https://arxiv.org/abs/1802.09477.
To obtain DDPG, set target smooth to zero and Q network ensembles to 1.
"""

from rlutils.runner import run_func_as_main

from .td3 import TD3Agent, Runner as TD3Runner


class DDPGAgent(TD3Agent):
    def __init__(self,
                 obs_spec,
                 act_spec,
                 policy_mlp_hidden=128,
                 policy_lr=3e-4,
                 q_mlp_hidden=256,
                 q_lr=3e-4,
                 tau=5e-3,
                 gamma=0.99,
                 ):
        super(DDPGAgent, self).__init__(obs_spec=obs_spec,
                                        act_spec=act_spec,
                                        num_q_ensembles=1,
                                        policy_mlp_hidden=policy_mlp_hidden,
                                        policy_lr=policy_lr,
                                        q_mlp_hidden=q_mlp_hidden,
                                        q_lr=q_lr,
                                        tau=tau,
                                        gamma=gamma,
                                        actor_noise=0.1,
                                        target_noise=0,
                                        noise_clip=0)


class Runner(TD3Runner):
    @staticmethod
    def main(env_name,
             env_fn=None,
             steps_per_epoch=5000,
             epochs=200,
             start_steps=10000,
             update_after=4000,
             update_every=1,
             update_per_step=1,
             batch_size=256,
             num_parallel_env=1,
             num_test_episodes=20,
             seed=1,
             # sac args
             nn_size=256,
             learning_rate=1e-3,
             tau=5e-3,
             gamma=0.99,
             # replay
             replay_size=int(1e6),
             logger_path=None
             ):
        config = locals()

        runner = Runner(seed=seed, steps_per_epoch=steps_per_epoch // num_parallel_env, epochs=epochs,
                        exp_name=None, logger_path=logger_path)
        runner.setup_env(env_name=env_name, num_parallel_env=num_parallel_env, env_fn=env_fn,
                         asynchronous=False, num_test_episodes=num_test_episodes)
        runner.setup_logger(config=config)

        agent_kwargs = dict(
            policy_mlp_hidden=nn_size,
            policy_lr=learning_rate,
            q_mlp_hidden=nn_size,
            q_lr=learning_rate,
            tau=tau,
            gamma=gamma
        )

        runner.setup_agent(agent_cls=DDPGAgent, **agent_kwargs)
        runner.setup_extra(start_steps=start_steps,
                           update_after=update_after,
                           update_every=update_every,
                           update_per_step=update_per_step,
                           policy_delay=1
                           )
        runner.setup_replay_buffer(replay_size=replay_size,
                                   batch_size=batch_size)

        runner.run()


if __name__ == '__main__':
    run_func_as_main(Runner.main)
