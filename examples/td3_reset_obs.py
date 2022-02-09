"""
TD3 if we can reset to any state in the environment. How it can help improve the sample efficiency
"""

import numpy as np
import rlutils.infra as rl_infra
import rlutils.np as rln
import tensorflow as tf
from rlalgos.tf.mf import td3


class TD3Agent(td3.TD3Agent):
    @tf.function
    def compute_q_values_tf(self, obs):
        act = self.policy_net(obs)
        return self.q_network((obs, act, tf.constant(True)))

    def compute_q_values(self, obs):
        return self.compute_q_values_tf(tf.convert_to_tensor(obs)).numpy()


class BatchSamplerResetObs(rl_infra.samplers.BatchSampler):
    def __init__(self, agent, **kwargs):
        self.agent = agent
        super(BatchSamplerResetObs, self).__init__(**kwargs)

    def reset(self):
        self._global_env_step_reset = 0
        self._global_env_step_reset_obs = 0
        self.reset_obs = False
        super(BatchSamplerResetObs, self).reset()

    def log_tabular(self):
        super(BatchSamplerResetObs, self).log_tabular()
        self.logger.log_tabular('TotalEnvInteractsResetDone', self._global_env_step_reset)
        self.logger.log_tabular('TotalEnvInteractsResetObs', self._global_env_step_reset_obs)


    def sample(self, num_steps, collect_fn, replay_buffer):
        for _ in range(num_steps):
            a = collect_fn(self.o)
            assert not np.any(np.isnan(a)), f'NAN action: {a}'
            # Step the env
            o2, r, d, infos = self.env.step(a)
            self.ep_ret += r
            self.ep_len += 1

            timeouts = rln.gather_dict_key(infos=infos, key='TimeLimit.truncated', default=False, dtype=np.bool)
            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            true_d = np.logical_and(d, np.logical_not(timeouts))

            # Store experience to replay buffer
            replay_buffer.add(dict(
                obs=self.o,
                act=a,
                rew=r,
                next_obs=o2,
                done=true_d
            ))

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            self.o = o2

            # End of trajectory handling
            if np.any(d):
                self.logger.store(EpRet=self.ep_ret[d], EpLen=self.ep_len[d])
                self.ep_ret[d] = 0
                self.ep_len[d] = 0
                # instead of reset to the initial state, reset to the state which has the highest
                # q value from a random batch
                if self._global_env_step_reset_obs > self._global_env_step_reset:
                    self.reset_obs = False
                    self.o = self.env.reset_done()
                else:
                    self.reset_obs = True
                    batch = replay_buffer.sample()
                    q_values = self.agent.compute_q_values(batch['obs'])
                    argmax = np.argsort(q_values, axis=0)[::-1][:self.env.num_envs]
                    self.o = self.env.reset_obs(batch['obs'][argmax], mask=d)

            self._global_env_step += self.env.num_envs
            if self.reset_obs:
                self._global_env_step_reset_obs += self.env.num_envs
            else:
                self._global_env_step_reset += self.env.num_envs


class Runner(rl_infra.runner.TFOffPolicyRunner):
    def setup_sampler(self, start_steps):
        self.start_steps = start_steps
        self.sampler = BatchSamplerResetObs(env=self.env, agent=self.agent)

    @classmethod
    def main(cls,
             env_name,
             epochs=300,
             policy_mlp_hidden=256,
             policy_lr=1e-3,
             q_mlp_hidden=256,
             q_lr=1e-3,
             actor_noise=0.1,
             target_noise=0.2,
             noise_clip=0.5,
             out_activation='sin',
             tau=5e-3,
             gamma=0.99,
             seed=1,
             logger_path: str = None,
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
            noise_clip=noise_clip,
            out_activation=out_activation,
        )

        super(Runner, cls).main(env_name=env_name,
                                epochs=epochs,
                                policy_delay=2,
                                agent_cls=TD3Agent,
                                agent_kwargs=agent_kwargs,
                                seed=seed,
                                logger_path=logger_path,
                                **kwargs)


if __name__ == '__main__':
    rl_infra.runner.run_func_as_main(Runner.main)
