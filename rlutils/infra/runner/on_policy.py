from .base import BaseRunner
from rlutils.replay_buffers import GAEBuffer
import rlutils.infra as rl_infra


class OnPolicyRunner(BaseRunner):
    def setup_logger(self, config, tensorboard=False):
        super(OnPolicyRunner, self).setup_logger(config=config, tensorboard=tensorboard)
        self.sampler.set_logger(self.logger)
        self.updater.set_logger(self.logger)

    def setup_replay_buffer(self, max_length, gamma, lam):
        self.replay_buffer = GAEBuffer.from_vec_env(self.env, max_length=max_length, gamma=gamma, lam=lam)

    def setup_sampler(self, num_steps):
        self.num_steps = num_steps
        self.sampler = rl_infra.samplers.TrajectorySampler(env=self.env)

    def setup_updater(self):
        self.updater = rl_infra.OnPolicyUpdater(agent=self.agent, replay_buffer=self.replay_buffer)

    def run_one_step(self, t):
        self.sampler.sample(num_steps=self.num_steps,
                            collect_fn=(self.agent.act_batch, self.agent.value_net.predict),
                            replay_buffer=self.replay_buffer)
        self.updater.update(self.global_step)

    def on_epoch_end(self, epoch):
        self.logger.log_tabular('Epoch', epoch)
        self.logger.dump_tabular()

    def on_train_begin(self):
        self.sampler.reset()
        self.updater.reset()
        self.timer.start()

    @classmethod
    def main(cls, env_name, env_fn=None, seed=0, num_parallel_envs=5, agent_cls=None, agent_kwargs={},
             batch_size=5000, epochs=200, gamma=0.99, lam=0.97, logger_path: str = None):
        # Instantiate environment
        assert batch_size % num_parallel_envs == 0

        num_steps_per_sample = batch_size // num_parallel_envs

        config = locals()
        runner = cls(seed=seed, steps_per_epoch=1,
                     epochs=epochs, exp_name=None, logger_path=logger_path)
        runner.setup_env(env_name=env_name, env_fn=env_fn, num_parallel_env=num_parallel_envs,
                         asynchronous=False, num_test_episodes=None)
        runner.setup_agent(agent_cls=agent_cls, **agent_kwargs)
        runner.setup_replay_buffer(max_length=num_steps_per_sample, gamma=gamma, lam=lam)
        runner.setup_sampler(num_steps=num_steps_per_sample)
        runner.setup_updater()
        runner.setup_logger(config)

        runner.run()
