import time

import numpy as np
from tqdm.auto import tqdm

import rlutils
from rlutils.gym.vector import VectorEnv
from rlutils.interface.logging import LogUser


class Tester(LogUser):
    """
    A tester is bound to a single environment. It can be used to test different agents.
    """

    def __init__(self, env_fn, num_parallel_env, asynchronous=False, seed=None):
        super(Tester, self).__init__()
        self.env_fn = env_fn
        self.test_env = rlutils.gym.utils.create_vector_env(env_fn=self.env_fn,
                                                            num_parallel_env=num_parallel_env,
                                                            asynchronous=asynchronous)
        self.seed = seed

    def log_tabular(self):
        assert self.logger is not None
        self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        self.logger.log_tabular('TestEpLen', average_only=True)

    def watch_agent(self, get_action, render=True, num_episodes=1):
        env = self.env_fn()
        for i in range(1, num_episodes + 1):
            done = False
            obs = env.reset()
            total_reward = 0.
            ep_len = 0
            while not done:
                if render:
                    env.render()
                a = get_action(np.expand_dims(obs, axis=0).astype(np.float32))[0]
                obs, reward, done, info = env.step(a)
                total_reward += reward
                ep_len += 1
                time.sleep(0.01)
            print(f'Episode: {i}. Total Reward: {total_reward:.2f}. Episode Length: {ep_len}')

    def test_agent(self, get_action, name, num_test_episodes, max_episode_length=None, timeout=None):
        assert num_test_episodes % self.test_env.num_envs == 0
        num_iterations = num_test_episodes // self.test_env.num_envs
        t = tqdm(total=num_test_episodes, desc=f'Testing {name}')

        all_ep_ret = []
        all_ep_len = []

        start = time.time()
        already_timeout = False

        for _ in range(num_iterations):
            o, _ = self.test_env.reset(seed=self.seed)  # keep evaluating the same random obs
            d = np.zeros(shape=self.test_env.num_envs, dtype=np.bool_)
            ep_ret = np.zeros(shape=self.test_env.num_envs, dtype=np.float64)
            ep_len = np.zeros(shape=self.test_env.num_envs, dtype=np.int64)
            steps = 0
            batch_action = None
            while not np.all(d):
                a = get_action(o[np.logical_not(d)])  # only inference on valid observations

                # init batch action
                if batch_action is None:
                    batch_action = np.zeros_like(a)

                batch_action[np.logical_not(d)] = a
                assert isinstance(a, np.ndarray), f'Action a must be np.ndarray. Got {type(a)}'
                o, r, terminate, truncate, _ = self.test_env.step(batch_action)

                # done happens either it is truely terminated or is truncated due to time limits
                d_ = np.logical_or(terminate, truncate)

                ep_ret = r * (1 - d) + ep_ret
                ep_len = np.ones(shape=self.test_env.num_envs, dtype=np.int64) * (1 - d) + ep_len
                prev_d = d.copy()
                d = np.logical_or(d, d_)
                newly_finished = np.sum(d) - np.sum(prev_d)
                if newly_finished > 0:
                    t.update(newly_finished)
                steps += 1
                if max_episode_length is not None and steps >= max_episode_length:
                    break

                elapsed = time.time() - start
                if timeout is not None and elapsed > timeout:
                    already_timeout = True
                    break

            if np.any(d):
                if self.logger is not None:
                    self.logger.store(TestEpRet=ep_ret[d], TestEpLen=ep_len[d])
                all_ep_ret.append(ep_ret[d])
                all_ep_len.append(ep_len[d])

            if already_timeout:
                break

        t.close()

        if len(all_ep_ret) > 0:
            return np.concatenate(all_ep_ret, axis=0), np.concatenate(all_ep_len, axis=0)
        else:
            return None, None
