import numpy as np
from rlutils.gym.vector import VectorEnv
from tqdm.auto import tqdm


class Tester(object):
    """
    A tester is bound to a single environment. It can be used to test different agents.
    """

    def __init__(self, test_env: VectorEnv):
        self.test_env = test_env
        self.logger = None

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        assert self.logger is not None
        self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        self.logger.log_tabular('TestEpLen', average_only=True)

    def test_agent(self, get_action, name, num_test_episodes):
        assert num_test_episodes % self.test_env.num_envs == 0
        num_iterations = num_test_episodes // self.test_env.num_envs
        t = tqdm(total=num_test_episodes, desc=f'Testing {name}')
        for _ in range(num_iterations):
            o = self.test_env.reset()
            d = np.zeros(shape=self.test_env.num_envs, dtype=np.bool_)
            ep_ret = np.zeros(shape=self.test_env.num_envs)
            ep_len = np.zeros(shape=self.test_env.num_envs, dtype=np.int64)
            while not np.all(d):
                a = get_action(o)
                o, r, d_, _ = self.test_env.step(a, mask=d)
                ep_ret = r * (1 - d) + ep_ret
                ep_len = np.ones(shape=self.test_env.num_envs, dtype=np.int64) * (1 - d) + ep_len
                prev_d = d.copy()
                d = np.logical_or(d, d_)
                newly_finished = np.sum(d) - np.sum(prev_d)
                if newly_finished > 0:
                    t.update(newly_finished)
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        t.close()
