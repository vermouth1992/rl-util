import unittest

import gym.spaces
import numpy as np
from rlutils import replay_buffers


class TestReplayBuffer(unittest.TestCase):
    def test_py_uniform(self):
        replay = replay_buffers.PyUniformReplayBuffer(data_spec={'data': gym.spaces.Space(shape=None, dtype=np.int32)},
                                                      capacity=5,
                                                      batch_size=2,
                                                      seed=1)
        for i in range(6):
            replay.add(dict(
                data=np.array([i, i + 1], dtype=np.int32)
            ))

        assert len(replay) == 5
        assert replay.is_full()
        np.testing.assert_array_equal(replay.get()['data'], np.array([5, 6, 4, 4, 5]))

        samples1 = replay.sample()
        replay.set_seed(1)
        samples2 = replay.sample()
        np.testing.assert_array_equal(samples1['data'], samples2['data'])

    def test_py_priority(self):
        capacity = 100
        replay = replay_buffers.DictPrioritizedReplayBuffer(
            data_spec={'data': gym.spaces.Space(shape=None, dtype=np.int32)},
            capacity=capacity,
            batch_size=capacity,
            seed=1,
            alpha=1.0)
        for i in range(capacity):
            replay.add(data={
                'data': np.array([i], dtype=np.int32)
            }, priority=np.log1p(i + 1))

        num_iters = 100000

        probability = np.log1p(np.arange(1, capacity + 1))
        probability /= np.sum(probability)

        idxes_count = np.zeros(shape=[capacity], dtype=np.int32)
        for _ in range(num_iters):
            _, idx = replay.sample(beta=1.0)
            unique, counts = np.unique(idx, return_counts=True)
            idxes_count[unique] += counts

        sample_prob = idxes_count / np.sum(idxes_count)
        np.testing.assert_allclose(sample_prob, probability, rtol=1e-2)


if __name__ == '__main__':
    unittest.main()
