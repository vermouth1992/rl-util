"""
Implement Ape-X. Parallel actors collect data. Compute the priority and store in a local buffer. When the local buffer
is full, it pushes the data into a Queue. The replay buffer
"""

import numpy as np
import torch.multiprocessing as mp
import torch.nn as nn

import rlutils.gym as rlu_gym
from rlutils.interface.agent import Agent


class Actor(object):
    def __init__(self, env_fn, agent_fn, make_local_buffer,
                 stats_send_freq=100,
                 terminal_check_freq=100,
                 weight_sync_freq=1000):
        self.env = env_fn()
        self.agent = agent_fn()
        self.local_buffer = make_local_buffer()
        self.terminal_check_freq = terminal_check_freq
        self.weight_sync_freq = weight_sync_freq
        self.stats_send_freq = stats_send_freq
        assert isinstance(self.agent, Agent)
        assert isinstance(self.agent, nn.Module)

    def is_terminate(self) -> bool:
        raise NotImplementedError

    def fetch_weights(self):
        raise NotImplementedError

    def send_stats(self, stats):
        raise NotImplementedError

    def send_data(self):
        raise NotImplementedError

    def run(self):
        local_steps = 0
        obs = self.env.reset()
        ep_ret_lst = []
        ep_len_lst = []
        ep_ret = np.zeros(shape=self.env.num_envs, dtype=np.float32)
        ep_len = np.zeros(shape=self.env.num_envs, dtype=np.int64)
        self.fetch_weights()
        while True:
            local_steps += 1

            if local_steps % self.terminal_check_freq == 0:
                if self.is_terminate():
                    break

            act = self.agent.act_batch_explore(obs, local_steps)
            next_obs, r, d, infos = self.env.step(act)

            ep_ret += r
            ep_len += 1

            true_done = rlu_gym.utils.get_true_done_from_infos(d, infos)
            self.local_buffer.add(dict(
                obs=obs,
                act=act,
                next_obs=next_obs,
                rew=r,
                done=true_done
            ))

            obs = next_obs

            if np.any(d):
                ep_ret_lst.extend(ep_ret[d].to_list())
                ep_len_lst.extend(ep_len[d].to_list())
                ep_ret[d] = 0
                ep_len[d] = 0
                obs = self.env.reset_done()

            if self.local_buffer.full():
                self.send_data()
                self.local_buffer.clear()

            if local_steps % self.stats_send_freq == 0:
                self.send_stats(dict(
                    EpRet=np.array(ep_ret_lst),
                    EpLen=np.array(ep_len_lst)
                ))
                ep_ret_lst.clear()
                ep_len_lst.clear()

            if local_steps % self.weight_sync_freq == 0:
                self.fetch_weights()


class SingleMachineActor(Actor):
    """
    data communication is performed in a single machine using multiprocess
    """

    def __init__(self,
                 terminal_event: mp.Event,
                 data_queue: mp.SimpleQueue,
                 stats_queue: mp.SimpleQueue,
                 weight_queue: mp.SimpleQueue,
                 **kwargs):
        super(SingleMachineActor, self).__init__(**kwargs)
        self.terminal_event = terminal_event
        self.data_queue = data_queue
        self.stats_queue = stats_queue
        self.weight_queue = weight_queue

    def is_terminate(self) -> bool:
        return self.terminal_event.is_set()

    def send_data(self):
        data = self.local_buffer.get()
        self.data_queue.put(data)

    def send_stats(self, stats):
        self.stats_queue.put(stats)

    def fetch_weights(self):
        weights = self.weight_queue.get()
        self.agent.load_state_dict(weights)
        self.weight_queue.put(weights)
