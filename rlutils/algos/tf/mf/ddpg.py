"""
Twin Delayed DDPG. https://arxiv.org/abs/1802.09477.
To obtain DDPG, set target smooth to zero and Q network ensembles to 1.
"""

from rlutils.infra.runner import run_func_as_main

from .td3 import TD3Agent, TFOffPolicyRunner


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


class Runner(TFOffPolicyRunner):
    @classmethod
    def main(cls,
             env_name,
             policy_mlp_hidden=256,
             policy_lr=1e-3,
             q_mlp_hidden=256,
             q_lr=1e-3,
             actor_noise=0.1,
             target_noise=0.2,
             noise_clip=0.5,
             tau=5e-3,
             gamma=0.99,
             **kwargs
             ):
        agent_kwargs = dict(
            policy_mlp_hidden=policy_mlp_hidden,
            policy_lr=policy_lr,
            q_mlp_hidden=q_mlp_hidden,
            q_lr=q_lr,
            tau=tau,
            gamma=gamma,
            actor_noise=actor_noise,
            target_noise=target_noise,
            noise_clip=noise_clip
        )

        super(Runner, cls).main(env_name=env_name,
                                policy_delay=2,
                                agent_cls=DDPGAgent,
                                agent_kwargs=agent_kwargs,
                                **kwargs)


if __name__ == '__main__':
    run_func_as_main(Runner.main)
