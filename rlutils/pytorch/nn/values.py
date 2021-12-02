import torch
import torch.nn as nn

import rlutils.pytorch as rlu


class EnsembleMinQNet(nn.Module):
    def __init__(self, ob_dim, ac_dim, mlp_hidden, num_ensembles=2, num_layers=3):
        super(EnsembleMinQNet, self).__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.mlp_hidden = mlp_hidden
        self.num_ensembles = num_ensembles
        self.q_net = rlu.nn.build_mlp(input_dim=self.ob_dim + self.ac_dim,
                                      output_dim=1,
                                      mlp_hidden=self.mlp_hidden,
                                      num_ensembles=self.num_ensembles,
                                      num_layers=num_layers,
                                      squeeze=True)

    def forward(self, inputs, training=None):
        assert training is not None
        obs, act = inputs
        inputs = torch.cat((obs, act), dim=-1)
        inputs = torch.unsqueeze(inputs, dim=0)  # (1, None, obs_dim + act_dim)
        inputs = inputs.repeat(self.num_ensembles, 1, 1)
        q = self.q_net(inputs)  # (num_ensembles, None)
        if training:
            return q
        else:
            return torch.min(q, dim=0)[0]


class AtariDuelQModule(nn.Module):
    def __init__(self, frame_stack, action_dim):
        super(AtariDuelQModule, self).__init__()
        self.model = nn.Sequential(
            *rlu.nn.functional.conv2d_bn_activation_block(frame_stack, 32, kernel_size=8, stride=4, padding=4,
                                                          normalize=False),
            *rlu.nn.functional.conv2d_bn_activation_block(32, 64, kernel_size=4, stride=2, padding=2, normalize=False),
            *rlu.nn.functional.conv2d_bn_activation_block(64, 64, kernel_size=3, stride=1, padding=1, normalize=False),
            nn.Flatten(),
            nn.Linear(12 * 12 * 64, 512),
            nn.ReLU()
        )
        self.adv_fc = nn.Linear(512, action_dim)
        self.value_fc = nn.Linear(512, 1)

    def forward(self, state: torch.Tensor, action=None):
        batch_size = state.shape[0]
        state = state.to(torch.float32)
        state = (state - 127.5) / 127.5
        state = self.model(state)
        value = self.value_fc(state)
        adv = self.adv_fc(state)
        adv = adv - torch.mean(adv, dim=-1, keepdim=True)
        out = value + adv
        if action is not None:
            out = out[torch.arange(batch_size), action]
        return out


class CategoricalQModule(nn.Module):
    def __init__(self, model):
        super(CategoricalQModule, self).__init__()
        self.model = model

    def forward(self, state: torch.Tensor, action=None, log_prob=False):
        batch_size = state.shape[0]
        q_value = self.model(state)
        if action is not None:
            q_value = q_value[torch.arange(batch_size), action]
        if log_prob:
            return torch.log_softmax(q_value, dim=-1)
        else:
            return torch.softmax(q_value, dim=-1)


class CategoricalAtariQModule(CategoricalQModule):
    def __init__(self, frame_stack, action_dim, num_atoms):
        model = nn.Sequential(
            *rlu.nn.functional.conv2d_bn_activation_block(frame_stack, 32, kernel_size=8, stride=4, padding=4,
                                                          normalize=False),
            *rlu.nn.functional.conv2d_bn_activation_block(32, 64, kernel_size=4, stride=2, padding=2, normalize=False),
            *rlu.nn.functional.conv2d_bn_activation_block(64, 64, kernel_size=3, stride=1, padding=1, normalize=False),
            nn.Flatten(),
            nn.Linear(12 * 12 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * num_atoms),
            rlu.nn.LambdaLayer(function=lambda x: torch.reshape(x, (-1, action_dim, num_atoms))),
        )
        super(CategoricalAtariQModule, self).__init__(model=model)

    def forward(self, state: torch.Tensor, action=None, log_prob=False):
        state = state.to(torch.float32)
        state = (state - 127.5) / 127.5
        return super(CategoricalAtariQModule, self).forward(state=state, action=action, log_prob=log_prob)
