"""
Implement Ape-X. Parallel actors collect data. Compute the priority and store in a local buffer. When the local buffer
is full, it pushes the data into a Queue. The replay buffer
"""

import torch.nn as nn

import rlutils.gym as rlu_gym


class Actor(nn.Module):
    def __init__(self, env_fn, agent_fn, make_local_buffer,
                 ):
        super(Actor, self).__init__()
        self.env_fn = env_fn
        self.agent = agent_fn()
        self.local_buffer = make_local_buffer()

    def is_terminate(self) -> bool:
        raise NotImplementedError

    def reset(self):
        self.local_steps = 0
        self.env = self.env_fn()

    def run(self):
        self.reset()
        obs = self.env.reset()
        while True:
            if self.is_terminate():
                break

            act = self.agent.act_batch_explore(self.obs, self.local_steps)
            next_obs, rew, done, infos = self.env.step(act)

            true_done = rlu_gym.utils.get_true_done_from_infos(done, infos)
            self.local_buffer.add(dict(
                obs=obs,
                act=act,
                next_obs=next_obs,
                rew=rew,
                done=true_done
            ))

            if done:
                # log
                pass
