import unittest

import gym.spaces
import numpy as np
from rlutils.replay_buffers import PyUniformReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def test_py_uniform(self):
        replay = PyUniformReplayBuffer(data_spec={'data': gym.spaces.Space(shape=None, dtype=np.int32)},
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


if __name__ == '__main__':
    unittest.main()
