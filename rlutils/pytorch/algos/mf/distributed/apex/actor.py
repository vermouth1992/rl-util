"""
Implement Ape-X. Parallel actors collect data. Compute the priority and store in a local buffer. When the local buffer
is full, it pushes the data into a Queue. The replay buffer
"""

import rlutils.gym as rlu_gym
from rlutils.interface.agent import Agent


class Actor(object):
    def __init__(self, env_fn, agent_fn, make_local_buffer):
        super(Actor, self).__init__()
        self.env = env_fn()
        self.agent = agent_fn()
        self.local_buffer = make_local_buffer()
        assert isinstance(self.agent, Agent)

    def is_terminate(self) -> bool:
        raise NotImplementedError

    def reset(self):
        self.local_steps = 0

    def run(self):
        self.reset()
        obs = self.env.reset()
        while True:
            if self.is_terminate():
                break

            act = self.agent.act_batch_explore(obs, self.local_steps)
            next_obs, rew, done, infos = self.env.step(act)

            true_done = rlu_gym.utils.get_true_done_from_infos(done, infos)
            self.local_buffer.add(dict(
                obs=obs,
                act=act,
                next_obs=next_obs,
                rew=rew,
                done=true_done
            ))

            obs = next_obs

            if self.local_buffer.full():
                pass

            if done:
                # log
                pass


class ProcessActor(Actor):
    """
    data communication is performed in a single machine using multiprocess
    """
    def __init__(self,  **kwargs):
        super(ProcessActor, self).__init__(**kwargs)