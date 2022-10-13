import gym.spaces
import torch
from torch import nn

import rlutils.pytorch as rlu


class NormalActor(nn.Module):
    def __init__(self, make_net):
        super(NormalActor, self).__init__()
        self.net = make_net()
        self.log_std = nn.Parameter(data=torch.randn(), requires_grad=True)

    def forward(self, obs):
        mean = self.net(obs)
        return rlu.distributions.make_independent_normal(loc=mean, scale=nn.functional.softplus(self.log_std), ndims=1)


class CategoricalActor(nn.Module):
    def __init__(self, make_net):
        super(CategoricalActor, self).__init__()
        self.net = make_net()

    def forward(self, obs):
        logits = self.net(obs)
        return torch.distributions.Categorical(logits=logits)


class MLPActorCriticSeparate(nn.Module):
    def __init__(self, env, policy_mlp_hidden=64, value_mlp_hidden=256):
        super(MLPActorCriticSeparate, self).__init__()

        make_make_net = lambda act_dim: lambda: rlu.nn.build_mlp(input_dim=env.observation_space.shape[0],
                                                                 output_dim=act_dim,
                                                                 mlp_hidden=policy_mlp_hidden)

        if isinstance(env.action_space, gym.spaces.Box):
            act_dim = env.action_space.shape[0]
            self.policy_net = NormalActor(make_net=make_make_net(act_dim))
        elif isinstance(env.action_space, gym.spaces.Discrete):
            act_dim = env.action_space.n
            self.policy_net = CategoricalActor(make_net=make_make_net(act_dim))
        else:
            raise NotImplementedError

        self.value_net = rlu.nn.build_mlp(input_dim=env.observation_space.shape[0],
                                          output_dim=1,
                                          squeeze=True,
                                          mlp_hidden=value_mlp_hidden)

    def forward(self, obs):
        pi_distribution = self.policy_net(obs)
        value = self.value_net(obs)
        return pi_distribution, value

    def get_policy_parameters(self):
        return self.policy_net.parameters()

    def get_value_parameters(self):
        return self.value_net.parameters()


class MLPActorCriticShared(nn.Module):
    pass


class AtariActorCriticShared(nn.Module):
    pass
