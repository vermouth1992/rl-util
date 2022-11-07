import copy

import gym
import numpy as np
import torch.nn as nn
import torch.optim
from tqdm.auto import trange

import rlutils.gym
import rlutils.infra as rl_infra
import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
from rlutils.interface.agent import Agent
from rlutils.logx import EpochLogger, setup_logger_kwargs
from rlutils.replay_buffers import UniformReplayBuffer


class CQLDiscreteAgent(Agent, nn.Module):
    def __init__(self,
                 env,
                 make_q_net=lambda env: rlu.nn.build_mlp(input_dim=env.observation_space.shape[0],
                                                         output_dim=env.action_space.n,
                                                         mlp_hidden=256, num_layers=3),
                 q_lr=3e-4,
                 alpha_cql=1.,
                 alpha_cql_lr=1e-3,
                 tau=5e-3,
                 gamma=0.99,
                 cql_threshold=-5.,
                 device=None
                 ):
        nn.Module.__init__(self)
        Agent.__init__(self, env=env)
        self.q_network = make_q_net(self.env)
        self.target_q_network = copy.deepcopy(self.q_network)
        rlu.functional.hard_update(self.target_q_network, self.q_network)

        self.q_optimizer = torch.optim.Adam(params=self.q_network.parameters(), lr=q_lr)

        self.log_cql = rlu.nn.LagrangeLayer(initial_value=alpha_cql)
        self.cql_alpha_optimizer = torch.optim.Adam(params=self.log_cql.parameters(), lr=alpha_cql_lr)
        self.cql_threshold = cql_threshold

        self.tau = tau
        self.gamma = gamma

        self.device = device
        self.to(self.device)

    def log_tabular(self):
        self.logger.log_tabular('QVals', with_min_and_max=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('AlphaCQL', average_only=True)
        self.logger.log_tabular('AlphaCQLLoss', average_only=True)
        self.logger.log_tabular('DeltaCQL', with_min_and_max=True)

    def update_target(self):
        rlu.functional.soft_update(self.target_q_network, self.q_network, self.tau)

    def _compute_next_obs_q(self, next_obs):
        """ Max backup """
        with torch.no_grad():
            target_q_values = self.target_q_network(next_obs)  # (None, act_dim)
            if self.double_q:
                target_actions = torch.argmax(self.q_network(next_obs), dim=-1)  # (None,)
                target_q_values = rlu.functional.gather_q_values(target_q_values, target_actions)
            else:
                target_q_values = torch.max(target_q_values, dim=-1)[0]
            return target_q_values

    def train_nets_cql_pytorch(self, obs, act, next_obs, rew, done):
        # update
        with torch.no_grad():
            alpha_cql = self.log_cql()
            next_q_values = self._compute_next_obs_q(next_obs)
            target_q_values = rew + self.gamma * (1. - done) * next_q_values

        # q loss
        self.q_optimizer.zero_grad()
        q_values_all = self.q_network(obs)  # (None, act_dim)
        q_values = rlu.functional.gather_q_values(q_values_all, act)  # (None,)

        mse_q_values_loss = torch.nn.functional.mse_loss(q_values, target_q_values)

        cql_q_values = torch.logsumexp(q_values_all, dim=-1)  # (None,)

        cql_threshold = torch.mean(cql_q_values - q_values.detach(), dim=0)

        q_loss = mse_q_values_loss + alpha_cql * cql_threshold
        q_loss.backward()
        self.q_optimizer.step()

        # update alpha_cql
        self.cql_alpha_optimizer.zero_grad()
        alpha_cql = self.log_cql()
        delta_cql = cql_threshold - self.cql_threshold
        alpha_cql_loss = -alpha_cql * delta_cql.detach()
        alpha_cql_loss.backward()
        self.cql_alpha_optimizer.step()

        info = dict(
            QVals=q_values,
            LossQ=mse_q_values_loss,
            AlphaCQL=alpha_cql,
            AlphaCQLLoss=alpha_cql_loss,
            DeltaCQL=cql_threshold,
        )
        return info

    def train_on_batch(self, data):
        data = ptu.convert_dict_to_tensor(data, self.device)
        info = self.train_nets_cql_pytorch(**data)
        self.update_target()
        self.logger.store(**info)

    def act_batch_explore(self, obs, global_steps):
        raise NotImplementedError

    def act_batch_test(self, obs):
        obs = torch.as_tensor(obs).to(self.device)
        return self.act_batch_test_pytorch(obs).cpu().numpy()

    def act_batch_test_pytorch(self, obs):
        with torch.no_grad():
            q_values = self.q_network(obs)
            actions = torch.argmax(q_values, dim=-1).cpu().numpy()
            return actions


from gym.wrappers import LazyFrames
from collections import deque


def create_qlearning_dataset_atari(env, num_stack=4):
    dataset = env.get_dataset()
    obs_deque = deque(maxlen=num_stack)

    for _ in range(num_stack):
        obs_deque.append(dataset['observations'][0])

    N = dataset['rewards'].shape[0]
    output_dataset = {
        'observations': np.zeros(shape=(N - 1,), dtype=np.object),
        'actions': np.zeros(shape=(N - 1,), dtype=np.int64),
        'next_observations': np.zeros(shape=(N - 1,), dtype=np.object),
        'rewards': np.zeros(shape=(N - 1,), dtype=np.float32),
        'terminals': np.zeros(shape=(N - 1,), dtype=np.float32),
    }

    for i in range(N - 1):
        output_dataset['observations'][i] = LazyFrames(frames=list(obs_deque))
        output_dataset['actions'][i] = dataset['actions'][i]
        output_dataset['rewards'][i] = dataset['rewards'][i]
        output_dataset['terminals'][i] = dataset['terminals'][i]

        obs_deque.append(dataset['observations'][i + 1])

        output_dataset['next_observations'][i] = LazyFrames(frames=list(obs_deque))

        if dataset['terminals'][i]:
            for _ in range(num_stack):
                obs_deque.append(dataset['observations'][i + 1])

        return output_dataset


def run_atari_cql(env_name: str,
                  exp_name: str = None,
                  asynchronous=False,
                  # agent
                  policy_mlp_hidden=256,
                  policy_lr=3e-5,
                  q_mlp_hidden=256,
                  q_lr=3e-4,
                  alpha=0.2,
                  tau=5e-3,
                  gamma=0.99,
                  cql_threshold=-5.,
                  # runner args
                  epochs=250,
                  steps_per_epoch=4000,
                  num_test_episodes=30,
                  batch_size=64,
                  seed=1,
                  logger_path: str = None
                  ):
    config = locals()

    # setup seed
    seeder = rl_infra.Seeder(seed=seed, backend='torch')
    seeder.setup_global_seed()

    # environment
    env_fn = lambda: gym.make(env_name)
    env_fn = rlutils.gym.utils.wrap_env_fn(env_fn, truncate_obs_dtype=True, normalize_action_space=True)

    # agent
    env = env_fn()

    import d4rl
    dataset = d4rl.qlearning_dataset(env)
    dataset['obs'] = dataset.pop('observations').astype(np.float32)
    dataset['act'] = dataset.pop('actions').astype(np.float32)
    dataset['next_obs'] = dataset.pop('next_observations').astype(np.float32)
    dataset['rew'] = dataset.pop('rewards').astype(np.float32)
    dataset['done'] = dataset.pop('terminals').astype(np.float32)

    def make_q_net(env):
        net = rlu.nn.values.LazyAtariDuelQModule(action_dim=env.action_space.n)
        dummy_inputs = torch.randn(1, *env.observation_space.shape)
        net(dummy_inputs)
        print(net)
        return net

    agent = CQLDiscreteAgent(env=env, make_q_net=make_q_net, q_lr=q_lr,
                             tau=tau, gamma=gamma, cql_threshold=cql_threshold,
                             device=ptu.get_cuda_device())

    # setup logger
    if exp_name is None:
        exp_name = f'{env_name}_{agent.__class__.__name__}_test'
    assert exp_name is not None, 'Call setup_env before setup_logger if exp passed by the contructor is None.'
    logger_kwargs = setup_logger_kwargs(exp_name=exp_name, data_dir=logger_path, seed=seed)
    logger = EpochLogger(**logger_kwargs, tensorboard=False)
    logger.save_config(config)

    timer = rl_infra.StopWatch()

    # replay buffer
    replay_buffer = UniformReplayBuffer.from_dataset(dataset=dataset, seed=seeder.generate_seed())

    # setup tester
    tester = rl_infra.Tester(env_fn=env_fn, num_parallel_env=num_test_episodes,
                             asynchronous=asynchronous, seed=seeder.generate_seed())

    # register log_tabular args
    timer.set_logger(logger=logger)
    agent.set_logger(logger=logger)
    tester.set_logger(logger=logger)

    timer.start()
    policy_updates = 0

    for epoch in range(1, epochs + 1):
        for t in trange(steps_per_epoch, desc=f'Epoch {epoch}/{epochs}'):
            # Update handling
            batch = replay_buffer.sample(batch_size)
            agent.train_on_batch(data=batch)
            policy_updates += 1

        tester.test_agent(get_action=lambda obs: agent.act_batch_test(obs),
                          name=agent.__class__.__name__,
                          num_test_episodes=num_test_episodes)
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('PolicyUpdates', policy_updates)
        logger.dump_tabular()


if __name__ == '__main__':
    from rlutils.infra.runner import run_func_as_main

    run_func_as_main(func=run_d4rl_cql)
