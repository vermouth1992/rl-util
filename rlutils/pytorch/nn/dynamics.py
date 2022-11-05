import numpy as np
import torch
import torch.utils.data
from torch import nn
from tqdm.auto import trange

import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
from rlutils.interface.logging import LogUser


class MLPDynamics(LogUser, nn.Module):
    def __init__(self, env, num_ensembles=5, device=None):
        LogUser.__init__(self)
        nn.Module.__init__(self)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        out_activation = lambda params: rlu.distributions.make_independent_normal_from_params(params,
                                                                                              min_log_scale=-5,
                                                                                              max_log_scale=1.0)
        self.num_ensembles = num_ensembles
        self.model = rlu.nn.build_mlp(input_dim=obs_dim + self.act_dim, output_dim=obs_dim * 2,
                                      mlp_hidden=512, num_ensembles=self.num_ensembles, num_layers=4, batch_norm=True,
                                      out_activation=out_activation)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.device = device

        self.obs_scalar = rlu.preprocessing.StandardScaler(input_shape=(obs_dim,))
        self.act_scalar = rlu.preprocessing.StandardScaler(input_shape=(act_dim,))
        self.delta_obs_scalar = rlu.preprocessing.StandardScaler(input_shape=(obs_dim,))

        self.to(device)

    def log_tabular(self):
        self.logger.log_tabular(key='TrainLoss', average_only=True)
        self.logger.log_tabular(key='ValLoss', average_only=True)

    def fit(self, data, num_epochs=10, batch_size=256, validation_split=0.1, patience=5):
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
                                                       drop_last=True)
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                                                     drop_last=False)

        val_loss_array = np.zeros(shape=(num_epochs,), dtype=np.float32)
        # step 2: training
        for i in trange(num_epochs, desc='Training dynamics'):
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

                self.logger.store(TrainLoss=loss.detach() / self.num_ensembles / self.obs_dim)

            # step 3: validation
            self.model.eval()

            val_loss = []
            with torch.no_grad():
                for obs_batch, act_batch, delta_obs_batch in val_dataloader:
                    inputs = torch.cat([obs_batch, act_batch], dim=-1)
                    inputs = torch.unsqueeze(inputs, dim=0)  # (1, None, obs_dim + act_dim)
                    inputs = inputs.repeat(self.num_ensembles, 1, 1)
                    delta_obs_batch_hat_distribution = self.model(inputs)  # (num_ensembles, None, obs_dim)
                    log_prob = delta_obs_batch_hat_distribution.log_prob(delta_obs_batch)  # (num_ensembles, None)
                    log_prob = torch.sum(log_prob, dim=0)  # (None,)
                    val_loss.append(-log_prob)

                val_loss = torch.cat(val_loss, dim=0)  # (None,)
                val_loss = torch.mean(val_loss)
                val_loss = ptu.to_numpy(val_loss)
                self.logger.store(ValLoss=val_loss / self.num_ensembles / self.obs_dim)

            val_loss_array[i] = val_loss

            self.model.train()

            # if patience
            if i >= patience:
                if np.all(val_loss_array[i - patience] <= val_loss_array[i - patience + 1: i + 1]):
                    # TODO: restore the best weights
                    break

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
