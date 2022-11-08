import gym.spaces
import torch
from torch import nn

import rlutils.pytorch as rlu


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
        super(MLPActorCriticSeparate, self).__init__(env=env)

        if isinstance(env.action_space, gym.spaces.Box):
            act_dim = env.action_space.shape[0]
            self.output_net = nn.Sequential(
                nn.Tanh(),
                rlu.distributions.IndependentNormalWithFixedVar(var_shape=(act_dim,),
                                                                reinterpreted_batch_ndims=1)
            )
        elif isinstance(env.action_space, gym.spaces.Discrete):
            act_dim = env.action_space.n
            self.output_net = rlu.nn.LambdaLayer(function=lambda logits: torch.distributions.Categorical(logits=logits))
        else:
            raise NotImplementedError

        self.policy_net = rlu.nn.build_mlp(input_dim=env.observation_space.shape[0],
                                           output_dim=act_dim,
                                           mlp_hidden=policy_mlp_hidden)

        self.value_net = rlu.nn.build_mlp(input_dim=env.observation_space.shape[0],
                                          output_dim=1,
                                          squeeze=True,
                                          mlp_hidden=value_mlp_hidden)

    def forward(self, obs):
        pi_distribution_params = self.policy_net(obs)
        pi_distribution = self.output_net(pi_distribution_params)
        value = self.value_net(obs)
        return pi_distribution, value

    def get_pi_distribution(self, obs):
        pi_distribution_params = self.policy_net(obs)
        return self.output_net(pi_distribution_params)

    def get_value(self, obs):
        value = self.value_net(obs)
        return value


class MLPActorCriticShared(ActorCritic):
    def __init__(self, env, mlp_hidden=128):
        super(MLPActorCriticShared, self).__init__(env=env)
        if isinstance(env.action_space, gym.spaces.Box):
            act_dim = env.action_space.shape[0]
            self.output_net = nn.Sequential(
                nn.Tanh(),
                rlu.distributions.IndependentNormalWithFixedVar(var_shape=(act_dim,),
                                                                reinterpreted_batch_ndims=1)
            )
        elif isinstance(env.action_space, gym.spaces.Discrete):
            act_dim = env.action_space.n
            self.output_net = rlu.nn.LambdaLayer(function=lambda logits: torch.distributions.Categorical(logits=logits))
        else:
            raise NotImplementedError

        self.features = rlu.nn.functional.build_mlp(input_dim=env.observation_space.shape[0],
                                                    output_dim=mlp_hidden,
                                                    mlp_hidden=mlp_hidden,
                                                    num_layers=2,
                                                    out_activation='relu')
        self.policy_header = nn.Linear(in_features=mlp_hidden, out_features=act_dim)
        self.value_header = nn.Sequential(
            nn.Linear(in_features=mlp_hidden, out_features=1),
            rlu.nn.SqueezeLayer(dim=-1)
        )

    def forward(self, obs):
        feature = self.features(obs)
        pi_distribution_params = self.policy_header(feature)
        pi_distribution = self.output_net(pi_distribution_params)
        value = self.value_header(feature)
        return pi_distribution, value

    def get_pi_distribution(self, obs):
        feature = self.features(obs)
        pi_distribution_params = self.policy_header(feature)
        pi_distribution = self.output_net(pi_distribution_params)
        return pi_distribution

    def get_value(self, obs):
        feature = self.features(obs)
        value = self.value_header(feature)
        return value


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
