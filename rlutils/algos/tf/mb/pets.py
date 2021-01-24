import time

import numpy as np
import tensorflow as tf
from rlutils.np import DataSpec
from rlutils.replay_buffers import PyUniformParallelEnvReplayBuffer
from rlutils.runner import TFRunner
from rlutils.tf.nn.models import EnsembleDynamicsModel


class PETSAgent(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, mlp_hidden=128, num_ensembles=5, lr=1e-3,
                 horizon=10, num_particles=5):
        super(PETSAgent, self).__init__()
        self.dynamics_model = EnsembleDynamicsModel(obs_dim=obs_dim, act_dim=act_dim, mlp_hidden=mlp_hidden,
                                                    num_ensembles=num_ensembles, lr=lr, reward_fn=None,
                                                    terminate_fn=None)
        self.inference_model = self.dynamics_model.build_ts_model(horizon=horizon, num_particles=num_particles)

    def set_logger(self, logger):
        self.logger = logger
        self.dynamics_model.set_logger(logger=logger)

    def log_tabular(self):
        self.dynamics_model.log_tabular()

    def update_model(self):
        pass

    def act_batch(self, obs):
        pass


class PETSRunner(TFRunner):
    def get_action_batch(self, obs):
        pass

    def setup_replay_buffer(self,
                            replay_size,
                            batch_size):
        data_spec = {
            'obs': DataSpec(shape=self.env.single_observation_space.shape,
                            dtype=np.float32),
            'act': DataSpec(shape=self.env.single_action_space.shape,
                            dtype=np.float32),
            'next_obs': DataSpec(shape=self.env.single_observation_space.shape,
                                 dtype=np.float32),
            'rew': DataSpec(shape=None, dtype=np.float32),
            'done': DataSpec(shape=None, dtype=np.float32)
        }
        self.replay_buffer = PyUniformParallelEnvReplayBuffer(data_spec=data_spec,
                                                              capacity=replay_size,
                                                              batch_size=batch_size,
                                                              num_parallel_env=self.num_parallel_env)

    def setup_agent(self, **kwargs):
        self.agent = PETSAgent()

    def setup_extra(self,
                    start_steps):
        self.start_steps = start_steps

    def run_one_step(self, t):
        global_env_steps = self.global_step * self.num_parallel_env
        if global_env_steps >= self.start_steps:
            a = self.agent.act_batch(self.o, deterministic=tf.convert_to_tensor(False)).numpy()
            assert not np.any(np.isnan(a)), f'NAN action: {a}'
        else:
            a = self.env.action_space.sample()

        # Step the env
        o2, r, d, _ = self.env.step(a)
        self.ep_ret += r
        self.ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        true_d = np.logical_and(d, self.ep_len != self.max_ep_len)

        # Store experience to replay buffer
        self.replay_buffer.add(data={
            'obs': self.o,
            'act': a,
            'rew': r,
            'next_obs': o2,
            'done': true_d
        })

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        self.o = o2

        # End of trajectory handling
        if np.any(d):
            self.logger.store(EpRet=self.ep_ret[d], EpLen=self.ep_len[d])
            self.ep_ret[d] = 0
            self.ep_len[d] = 0
            self.o = self.env.reset_done()

    def on_epoch_end(self, epoch):

        # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TestEpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', self.global_step * self.num_parallel_env)
        self.agent.log_tabular()
        self.logger.log_tabular('Time', time.time() - self.start_time)
        self.logger.dump_tabular()

    def on_train_begin(self):
        self.start_time = time.time()
        self.o = self.env.reset()
        self.ep_ret = np.zeros(shape=self.num_parallel_env)
        self.ep_len = np.zeros(shape=self.num_parallel_env, dtype=np.int64)


def pets():
    pass


if __name__ == '__main__':
    pass
