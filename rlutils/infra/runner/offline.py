from rlutils.replay_buffers import PyUniformReplayBuffer as ReplayBuffer
from .off_policy import OffPolicyRunner
import numpy as np
from rlutils.logx import EpochLogger
import pprint


def create_d4rl_dataset(env):
    try:
        import d4rl
        dataset = d4rl.qlearning_dataset(env=env)
        dataset['obs'] = dataset.pop('observations').astype(np.float32)
        dataset['act'] = dataset.pop('actions').astype(np.float32)
        dataset['next_obs'] = dataset.pop('next_observations').astype(np.float32)
        dataset['rew'] = dataset.pop('rewards').astype(np.float32)
        dataset['done'] = dataset.pop('terminals').astype(np.float32)
        return dataset
    except:
        print('d4rl is not installed and dataset is None')
        raise


class OfflineRunner(OffPolicyRunner):
    def setup_logger(self, config, tensorboard=False):
        super(OffPolicyRunner, self).setup_logger(config=config, tensorboard=tensorboard)
        self.tester.set_logger(self.logger)
        self.updater.set_logger(self.logger)

    def on_train_begin(self):
        self.updater.reset()
        self.timer.start()

    def run_one_step(self, t):
        self.updater.update(self.global_step)

    def setup_replay_buffer(self,
                            batch_size,
                            dataset=None,
                            reward_scale=True):
        def rescale(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x))

        if dataset is None:
            dummy_env = self.env_fn()
            dataset = create_d4rl_dataset(dummy_env)
            del dummy_env

        if reward_scale:
            EpochLogger.log('Using reward scale', color='red')
            self.agent.reward_scale_factor = np.max(dataset['rew'] - np.min(dataset['rew']))
            EpochLogger.log(f'The scale factor is {self.agent.reward_scale_factor:.2f}')
            dataset['rew'] = rescale(dataset['rew'])

        replay_size = dataset['obs'].shape[0]
        EpochLogger.log(f'Dataset size: {replay_size}')
        self.replay_buffer = ReplayBuffer.from_data_dict(
            data=dataset,
            batch_size=batch_size
        )

    @classmethod
    def main(cls,
             env_name,
             env_fn=None,
             exp_name=None,
             steps_per_epoch=10000,
             epochs=100,
             update_every=1,
             update_per_step=1,
             policy_delay=1,
             batch_size=256,
             num_test_episodes=30,
             seed=1,
             # agent args
             agent_cls=None,
             agent_kwargs={},
             # replay
             dataset=None,
             logger_path=None,
             **kwargs
             ):
        config = locals()

        runner = cls(seed=seed, steps_per_epoch=steps_per_epoch, epochs=epochs,
                     exp_name=exp_name, logger_path=logger_path)
        runner.setup_env(env_name=env_name, env_fn=env_fn, num_parallel_env=-1,
                         asynchronous=False, num_test_episodes=num_test_episodes)
        runner.setup_agent(agent_cls=agent_cls, **agent_kwargs)
        runner.setup_replay_buffer(dataset=dataset,
                                   batch_size=batch_size)
        runner.setup_tester(num_test_episodes=num_test_episodes)
        runner.setup_updater(update_after=-1,
                             policy_delay=policy_delay,
                             update_per_step=update_per_step,
                             update_every=update_every)
        runner.setup_logger(config=config, tensorboard=False)

        pprint.pprint(runner.seeds_info)

        runner.run()
