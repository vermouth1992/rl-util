"""
Due to the high computation requirement, we employ the following process:
- A process that collects data and performs learning. This controls the ratio between data collection and policy updates
- A process that samples data and performs dynamics training
- A process that samples data to perform rollouts
"""

from typing import Callable

import gym
import numpy as np
import torch.optim
import torch.utils.data
from torch import nn
from tqdm.auto import trange

import rlutils.gym
import rlutils.infra as rl_infra
import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
from model_free.actor_critic.sac import SACAgent
from rlutils.interface.logging import LogUser
from rlutils.logx import setup_logger_kwargs, EpochLogger
from rlutils.replay_buffers import UniformReplayBuffer


class StandardScaler(nn.Module):
    def __init__(self, input_shape):
        super(StandardScaler, self).__init__()
        self.mean = nn.Parameter(data=torch.zeros(size=[1] + list(input_shape), dtype=torch.float32),
                                 requires_grad=False)
        self.std = nn.Parameter(data=torch.ones(size=[1] + list(input_shape), dtype=torch.float32),
                                requires_grad=False)

    def adapt(self, data):
        self.mean.data = torch.mean(data, dim=0, keepdim=True)
        self.std.data = torch.std(data, dim=0, keepdim=True)

    def forward(self, data, inverse=False):
        if inverse:
            return data * self.std + self.mean
        else:
            return (data - self.mean) / self.std


class MLPDynamics(LogUser, nn.Module):
    def __init__(self, env, num_ensembles=5, device=None):
        LogUser.__init__(self)
        nn.Module.__init__(self)
        obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        out_activation = lambda params: rlu.distributions.make_independent_normal_from_params(params,
                                                                                              min_log_scale=-2,
                                                                                              max_log_scale=1.0)
        self.num_ensembles = num_ensembles
        self.model = rlu.nn.build_mlp(input_dim=obs_dim + self.act_dim, output_dim=obs_dim * 2,
                                      mlp_hidden=512, num_ensembles=self.num_ensembles, num_layers=4, batch_norm=True,
                                      out_activation=out_activation)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.device = device

        self.obs_scalar = StandardScaler(input_shape=(obs_dim,))
        self.act_scalar = StandardScaler(input_shape=(self.act_dim,))
        self.delta_obs_scalar = StandardScaler(input_shape=(obs_dim,))

        self.to(device)

    def log_tabular(self):
        self.logger.log_tabular(key='TrainLoss', average_only=True)
        self.logger.log_tabular(key='ValLoss', average_only=True)

    def fit(self, data, num_epochs=10, batch_size=256, validation_split=0.1):
        obs = torch.as_tensor(data['obs'], device=self.device)
        act = torch.as_tensor(data['act'], device=self.device)
        next_obs = torch.as_tensor(data['next_obs'], device=self.device)
        # step 0: calculate statistics
        delta_obs = next_obs - obs

        self.obs_scalar.adapt(obs)
        self.act_scalar.adapt(act)
        self.delta_obs_scalar.adapt(delta_obs)

        obs_normalized = self.obs_scalar(obs)
        act_normalized = self.act_scalar(act)
        delta_obs_normalized = self.delta_obs_scalar(torch.as_tensor(delta_obs, device=self.device))

        # step 1: make a dataloader
        dataset = torch.utils.data.TensorDataset(obs_normalized, act_normalized, delta_obs_normalized)

        # split into train and test
        lengths = [1. - validation_split, validation_split]
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=lengths)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                       drop_last=False)
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                                                     drop_last=False)

        # step 2: training
        for _ in trange(num_epochs, desc='Training dynamics'):
            for obs_batch, act_batch, delta_obs_batch in train_dataloader:
                self.optimizer.zero_grad()
                inputs = torch.cat([obs_batch, act_batch], dim=-1)
                inputs = torch.unsqueeze(inputs, dim=0)  # (1, None, obs_dim + act_dim)
                inputs = inputs.repeat(self.num_ensembles, 1, 1)
                delta_obs_batch_hat_distribution = self.model(inputs)  # (num_ensembles, None, obs_dim)
                log_prob = delta_obs_batch_hat_distribution.log_prob(delta_obs_batch)  # (num_ensembles, None)
                log_prob = torch.sum(log_prob, dim=0)
                loss = -torch.mean(log_prob, dim=0)
                loss.backward()
                self.optimizer.step()

                self.logger.store(TrainLoss=loss.detach() / self.num_ensembles / self.act_dim)

        # step 3: validation
        self.model.eval()
        with torch.no_grad():
            for obs_batch, act_batch, delta_obs_batch in val_dataloader:
                inputs = torch.cat([obs_batch, act_batch], dim=-1)
                inputs = torch.unsqueeze(inputs, dim=0)  # (1, None, obs_dim + act_dim)
                inputs = inputs.repeat(self.num_ensembles, 1, 1)
                delta_obs_batch_hat_distribution = self.model(inputs)  # (num_ensembles, None, obs_dim)
                log_prob = delta_obs_batch_hat_distribution.log_prob(delta_obs_batch)  # (num_ensembles, None)
                log_prob = torch.sum(log_prob, dim=0)
                loss = -torch.mean(log_prob, dim=0)

                self.logger.store(ValLoss=loss.detach() / self.num_ensembles / self.act_dim)

        self.model.train()

    def predict(self, obs, act):
        with torch.no_grad():
            self.model.eval()
            batch_size = obs.shape[0]
            obs = torch.as_tensor(obs, device=self.device)
            act = torch.as_tensor(act, device=self.device)
            obs_normalized = self.obs_scalar(obs)
            act_normalized = self.act_scalar(act)

            inputs = torch.cat([obs_normalized, act_normalized], dim=-1)
            inputs = torch.unsqueeze(inputs, dim=0)  # (1, None, obs_dim + act_dim)
            inputs = inputs.repeat(self.num_ensembles, 1, 1)

            delta_obs_normalized_distribution = self.model(inputs)
            delta_obs_normalized = delta_obs_normalized_distribution.sample()  # (num_ensembles, None, obs_dim)

            # randomly choose one from ensembles
            delta_obs_normalized = delta_obs_normalized[
                torch.randint(self.num_ensembles, size=(batch_size,), device=self.device),
                torch.arange(batch_size, device=self.device)]  # (None, obs_dim)
            delta_obs = self.delta_obs_scalar(delta_obs_normalized, inverse=True)
            next_obs = delta_obs + obs

            self.model.train()
            return next_obs


def generate_model_rollouts(env, agent, dynamics_model, real_dataset, rollout_length, model_rollout_batch_size):
    data = dict(
        obs=[],
        act=[],
        next_obs=[],
        rew=[],
        done=[]
    )

    obs = real_dataset.sample(model_rollout_batch_size)['obs']
    obs = torch.as_tensor(obs, device=agent.device)
    for _ in range(rollout_length):
        act = agent.act_batch_torch(obs, deterministic=False)
        next_obs = dynamics_model.predict(obs, act)
        terminate = env.terminate_fn_torch_batch(obs, act, next_obs)
        reward = env.reward_fn_torch_batch(obs, act, next_obs)

        data['obs'].append(obs)
        data['act'].append(act)
        data['next_obs'].append(next_obs)
        data['rew'].append(reward)
        data['done'].append(terminate)

        obs = next_obs

        # if terminated, resample observations
        if torch.any(terminate):
            num_terminate = torch.sum(terminate).item()
            new_obs = real_dataset.sample(num_terminate)['obs']
            obs[terminate] = torch.as_tensor(new_obs, device=agent.device)

    for key, val in data.items():
        data[key] = ptu.to_numpy(torch.cat(val, dim=0))

    assert data['obs'].shape[0] == obs.shape[0] * rollout_length
    return data


def main(env_name,
         env_fn: Callable = None,
         exp_name: str = None,
         gamma=0.99,
         rollout_length=10,
         model_rollout_batch_size=400,
         model_training_iterations=500,
         model_training_batch_size=256,
         agent_training_batch_size=256,
         agent_training_iterations=20,
         synthetic_replay_size=1000000,
         # runner args
         epochs=200,
         steps_per_epoch=500,
         num_test_episodes=30,
         train_model_after=1000,
         seed=1,
         logger_path: str = None,
         backend='torch'):
    device = ptu.get_cuda_device()

    config = locals()

    # setup seed
    seeder = rl_infra.Seeder(seed=seed, backend=backend)
    seeder.setup_global_seed()

    # setup logger
    if exp_name is None:
        exp_name = f'{env_name}-MBPO_test'
    assert exp_name is not None, 'Call setup_env before setup_logger if exp passed by the contructor is None.'
    logger_kwargs = setup_logger_kwargs(exp_name=exp_name, data_dir=logger_path, seed=seed)
    logger = EpochLogger(**logger_kwargs, tensorboard=False)
    logger.save_config(config)

    timer = rl_infra.StopWatch()

    if env_fn is None:
        env_fn = lambda: gym.make(env_name)

    env_fn = rlutils.gym.utils.wrap_env_fn(env_fn, truncate_obs_dtype=True, normalize_action_space=True)

    dummy_env = env_fn()

    env = rlutils.gym.utils.create_vector_env(env_fn=env_fn,
                                              num_parallel_env=1,
                                              asynchronous=False)

    env.action_space.seed(seeder.generate_seed())

    # replay buffer
    real_replay_size = epochs * steps_per_epoch + train_model_after
    real_replay_buffer = UniformReplayBuffer.from_env(env=dummy_env, capacity=real_replay_size,
                                                      seed=seeder.generate_seed(),
                                                      memory_efficient=False)
    synthetic_dataset = UniformReplayBuffer.from_env(env=dummy_env, capacity=synthetic_replay_size,
                                                     seed=seeder.generate_seed(),
                                                     memory_efficient=False)

    agent = SACAgent(env=dummy_env, target_entropy=-env.action_space.shape[0] // 2, device=device)
    dynamics_model = MLPDynamics(env=dummy_env, device=device)
    sampler = rl_infra.samplers.BatchSampler(env=env, n_steps=1, gamma=gamma,
                                             seed=seeder.generate_seed())

    # setup tester
    tester = rl_infra.Tester(env_fn=env_fn, num_parallel_env=num_test_episodes,
                             asynchronous=False, seed=seeder.generate_seed())

    dynamics_model.set_logger(logger=logger)
    timer.set_logger(logger=logger)
    agent.set_logger(logger=logger)
    sampler.set_logger(logger=logger)
    tester.set_logger(logger=logger)

    policy_updates = 0
    global_step = 0

    timer.start()
    sampler.reset()

    sampler.sample(num_steps=train_model_after,
                   collect_fn=lambda o: np.asarray(env.action_space.sample()),
                   replay_buffer=real_replay_buffer)

    for epoch in range(1, epochs + 1):
        # step 2: train dynamics model
        data = real_replay_buffer.storage.get()
        num_epochs = max(model_training_iterations // (len(real_replay_buffer) // model_training_batch_size), 5)
        dynamics_model.fit(data=data, num_epochs=num_epochs, batch_size=model_training_batch_size)

        for t in trange(steps_per_epoch, desc=f'Epoch {epoch}/{epochs}'):
            # step 1: add data to true dataset
            sampler.sample(num_steps=1,
                           collect_fn=lambda obs: agent.act_batch_explore(obs, global_step),
                           replay_buffer=real_replay_buffer)

            # step 3: perform model rollouts
            synthetic_data = generate_model_rollouts(dummy_env, agent, dynamics_model, real_replay_buffer,
                                                     rollout_length, model_rollout_batch_size)
            synthetic_data['gamma'] = np.ones_like(synthetic_data['rew']) * gamma

            synthetic_dataset.add(synthetic_data)

            # step 4: perform policy training
            for _ in range(agent_training_iterations):
                data = synthetic_dataset.sample(agent_training_batch_size)
                agent.train_on_batch(data)
                policy_updates += 1

            global_step += 1

        tester.test_agent(get_action=lambda obs: agent.act_batch_test(obs),
                          name=agent.__class__.__name__,
                          num_test_episodes=num_test_episodes)
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('PolicyUpdates', policy_updates)
        logger.dump_tabular()


if __name__ == '__main__':
    rl_infra.runner.run_func_as_main(main)
