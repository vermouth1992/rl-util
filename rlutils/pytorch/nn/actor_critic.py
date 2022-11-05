import gym.spaces
import torch
from torch import nn

import rlutils.pytorch as rlu


class NormalActor(nn.Module):
    def __init__(self, make_net):
        super(NormalActor, self).__init__()
        self.net = make_net()
        self.log_std = nn.Parameter(data=torch.randn(size=(), dtype=torch.float32), requires_grad=True)

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


class ActorCritic(nn.Module):
    def __init__(self, env):
        super(ActorCritic, self).__init__()
        self.env = env

    def forward(self, obs):
        raise NotImplementedError

    def get_pi_distribution(self, obs):
        raise NotImplementedError

    def get_value(self, obs):
        raise NotImplementedError


class MLPActorCriticSeparate(ActorCritic):
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

    def get_pi_distribution(self, obs):
        return self.policy_net(obs)

    def get_value(self, obs):
        with torch.no_grad():
            value = self.value_net(obs)
            return value


class MLPActorCriticShared(nn.Module):
    pass


class AtariActorCriticShared(ActorCritic):
    def __init__(self, env):
        super(AtariActorCriticShared, self).__init__(env=env)
        self.model = nn.Sequential(
            rlu.nn.LambdaLayer(function=lambda state: (state.to(torch.float32) - 127.5) / 127.5),
            nn.LazyConv2d(out_channels=32, kernel_size=8, stride=4, padding=4),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=64, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU()
        )
        action_dim = env.action_space.n
        self.actor_fc = nn.Sequential(
            nn.LazyLinear(action_dim),
            rlu.nn.LambdaLayer(function=lambda logits: torch.distributions.Categorical(logits=logits))
        )
        self.value_fc = nn.Sequential(
            nn.LazyLinear(1),
            rlu.nn.SqueezeLayer(dim=-1)
        )

        # build
        obs = torch.randint(low=0, high=255, size=(1, *env.observation_space.shape), dtype=torch.uint8)
        self(obs)

    def forward(self, obs):
        feature = self.model(obs)
        pi_distribution = self.actor_fc(feature)
        value = self.value_fc(feature)
        return pi_distribution, value

    def get_pi_distribution(self, obs):
        feature = self.model(obs)
        pi_distribution = self.actor_fc(feature)
        return pi_distribution

    def get_value(self, obs):
        feature = self.model(obs)
        value = self.value_fc(feature)
        return value