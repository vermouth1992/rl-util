from rlutils.replay_buffers import PyUniformReplayBuffer as ReplayBuffer
from .off_policy import OffPolicyRunner
import numpy as np
from rlutils.logx import EpochLogger


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
        print('d4rl is not installed')
        raise


class OfflineRunner(OffPolicyRunner):
    def run_one_step(self, t):
        self.updater.update(self.global_step)

    def setup_replay_buffer(self,
                            batch_size,
                            dataset=None,
                            reward_scale=True):
        def rescale(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x))

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
