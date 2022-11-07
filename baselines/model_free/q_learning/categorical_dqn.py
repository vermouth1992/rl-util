import torch.nn as nn
import torch.optim

import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
from baselines.model_free.q_learning.dqn import DQN


class CategoricalDQN(DQN):
    def __init__(self, make_q_net=lambda env, num_atoms: rlu.nn.values.CategoricalQModule(
        rlu.nn.functional.build_mlp(input_dim=env.observation_space.shape[0],
                                    output_dim=env.action_space.n * num_atoms,
                                    mlp_hidden=256,
                                    out_activation=lambda x: torch.reshape(x, shape=(
                                            -1, env.action_space.n, num_atoms)))),
                 num_atoms=51, v_min=0., v_max=100.,
                 **kwargs):
        assert num_atoms > 1, 'The number of atoms must be greater than 1'
        super(CategoricalDQN, self).__init__(double_q=False,
                                             make_q_net=lambda env: make_q_net(env, num_atoms),
                                             **kwargs)
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = nn.Parameter(torch.linspace(self.v_min, self.v_max, self.num_atoms), requires_grad=False)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        if self.double_q:
            self.double_q = False
            print('Double q is set to False in CategoricalDQN')

        self.to(self.device)

    def compute_target_values(self, next_obs, rew, done, gamma):
        # double q doesn't perform very well.
        with torch.no_grad():
            batch_size = next_obs.shape[0]
            target_action_compute_q = self.q_network if self.double_q else self.target_q_network
            target_logits_action = target_action_compute_q(next_obs)  # (None, act_dim, num_atoms)
            target_q_values = torch.sum(target_logits_action * self.support[None, None, :], dim=-1)  # (None, act_dim)
            target_actions = torch.argmax(target_q_values, dim=-1)  # (None,)
            if self.double_q:
                target_logits = self.target_q_network(next_obs)
            else:
                target_logits = target_logits_action
            target_logits = target_logits[
                torch.arange(batch_size, device=ptu.device), target_actions]  # (None, num_atoms)
            # atom values
            target_q_atoms = rew[:, None] + gamma[:, None] * (1. - done[:, None]) * self.support[None, :]
            target_q_atoms = torch.clamp(target_q_atoms, min=self.v_min, max=self.v_max)  # (None, num_atoms)
            atom_distribution = 1. - torch.abs(target_q_atoms[:, :, None] - self.support[None, None, :]) / self.delta_z
            atom_distribution = torch.clamp(atom_distribution, min=0., max=1.)  # (None, j, i)
            probability = torch.sum(atom_distribution * target_logits[:, :, None], dim=1)  # (None, num_atoms)
            return probability

    def train_on_batch_torch(self, obs, act, next_obs, rew, done, gamma, weights=None):
        target_q_values = self.compute_target_values(next_obs, rew, done, gamma)  # (None, num_atoms)
        self.q_optimizer.zero_grad()
        q_values = self.q_network(obs, act, log_prob=True)  # (None, num_atoms)
        cross_entropy = -torch.sum(target_q_values * q_values, dim=-1)

        loss = cross_entropy
        if weights is not None:
            loss = loss * weights

        loss = torch.mean(loss, dim=0)
        loss.backward()
        self.q_optimizer.step()

        with torch.no_grad():
            q_values = torch.sum(torch.exp(q_values) * self.support[None, :], dim=-1)

        info = dict(
            QVals=q_values,
            LossQ=loss,
            TDError=cross_entropy
        )

        return info

    def act_batch_test(self, obs):
        obs = torch.as_tensor(obs, device=self.device)
        with torch.no_grad():
            target_logits_action = self.q_network(obs)  # (None, act_dim, num_atoms)
            target_q_values = torch.sum(target_logits_action * self.support[None, None, :], dim=-1)  # (None, act_dim)
            target_actions = torch.argmax(target_q_values, dim=-1)  # (None,)
            return target_actions.cpu().numpy()


if __name__ == '__main__':
    from baselines.model_free.trainer import run_offpolicy
    import rlutils.infra as rl_infra

    make_agent_fn = lambda env: CategoricalDQN(env=env, device=ptu.get_cuda_device())
    rl_infra.runner.run_func_as_main(run_offpolicy, passed_args={
        'make_agent_fn': make_agent_fn,
        'backend': 'torch'
    })
