import torch.nn as nn

import rlutils.pytorch as rlu
from rlutils.interface.agent import Agent


class AtariDQN(Agent, nn.Module):
    def __init__(self,
                 act_spec,
                 frame_stack=4,
                 double_q=True,
                 ):
        super(AtariDQN, self).__init__()
        self.q_network = rlu.nn.values.AtariDuelQModule(frame_history_len=4, )
