import torch.optim

from rlutils.interface.agent import Agent
import rlutils.infra as rl_infra
import torch.nn as nn
import rlutils.pytorch as rlu
import rlutils.pytorch.utils as ptu
import copy
import pprint


class CQLAgent(Agent, nn.Module):
    def __init__(self,
                 obs_spec,
                 act_spec,
                 policy_mlp_hidden=128,
                 policy_lr=3e-4,
                 q_mlp_hidden=256,
                 q_lr=3e-4,
                 alpha=1.0,
                 alpha_lr=1e-3,
                 alpha_cql=1.,
                 alpha_cql_lr=1e-3,
                 tau=5e-3,
                 gamma=0.99,
                 num_samples=10,
                 cql_threshold=1.,
                 target_entropy=None,
                 ):
        super(CQLAgent, self).__init__()
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.num_samples = num_samples
        self.act_dim = self.act_spec.shape[0]
        if len(self.obs_spec.shape) == 1:  # 1D observation
            self.obs_dim = self.obs_spec.shape[0]
            self.policy_net = rlu.nn.SquashedGaussianMLPActor(self.obs_dim, self.act_dim, policy_mlp_hidden)
            self.target_policy_net = copy.deepcopy(self.policy_net)
            self.q_network = rlu.nn.EnsembleMinQNet(self.obs_dim, self.act_dim, q_mlp_hidden)
            self.target_q_network = copy.deepcopy(self.q_network)
        else:
            raise NotImplementedError
        rlu.functional.hard_update(self.target_q_network, self.q_network)

        self.policy_optimizer = torch.optim.Adam(params=self.policy_net.parameters(), lr=policy_lr)
        self.q_optimizer = torch.optim.Adam(params=self.q_network.parameters(), lr=q_lr)

        self.log_alpha = rlu.nn.LagrangeLayer(initial_value=alpha)
        self.log_cql = rlu.nn.LagrangeLayer(initial_value=alpha_cql)
        self.alpha_optimizer = torch.optim.Adam(params=self.log_alpha.parameters(), lr=alpha_lr)
        self.cql_alpha_optimizer = torch.optim.Adam(params=self.log_cql.parameters(), lr=alpha_cql_lr)

        self.target_entropy = -self.act_dim if target_entropy is None else target_entropy
        self.cql_threshold = cql_threshold

        self.tau = tau
        self.gamma = gamma

        self.max_backup = True

    def log_tabular(self):
        self.logger.log_tabular('Q1Vals', with_min_and_max=True)
        self.logger.log_tabular('Q2Vals', with_min_and_max=True)
        self.logger.log_tabular('LogPi', average_only=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('Alpha', average_only=True)
        self.logger.log_tabular('LossAlpha', average_only=True)
        self.logger.log_tabular('AlphaCQL', average_only=True)
        self.logger.log_tabular('AlphaCQLLoss', average_only=True)
        self.logger.log_tabular('DeltaCQL', with_min_and_max=True)

    def update_target(self):
        rlu.functional.soft_update(self.target_q_network, self.q_network, self.tau)
        rlu.functional.soft_update(self.target_policy_net, self.policy_net, self.tau)

    def _compute_next_obs_q(self, next_obs, max_backup=True):
        """ Max backup """
        with torch.no_grad():
            batch_size = next_obs.shape[0]
            next_obs = torch.tile(next_obs, (self.num_samples, 1))
            actions = self.policy_net.select_action((next_obs, False))
            q_values = self.target_q_network(inputs=(next_obs, actions), training=False)
            q_values = torch.reshape(q_values, (self.num_samples, batch_size))  # (num_samples, None)
            if max_backup:
                q_values = torch.max(q_values, dim=0)[0]
            else:
                q_values = torch.mean(q_values, dim=0)
            return q_values

    def _update_nets_behavior_cloning(self, obs, act, next_obs, rew, done):
        self.policy_optimizer.zero_grad()
        log_prob = self.policy_net.compute_log_prob((obs, act))
        loss = -torch.mean(log_prob, dim=0)
        loss.backward()
        self.policy_optimizer.step()
        info = dict(
            LossPi=loss
        )
        return info

    def _update_nets_cql(self, obs, act, next_obs, rew, done):
        # update
        with torch.no_grad():
            alpha = self.log_alpha()
            alpha_cql = self.log_cql()
            batch_size = obs.shape[0]
            next_q_values = self._compute_next_obs_q(next_obs, max_backup=self.max_backup)
            q_target = rlu.functional.compute_target_value(rew, self.gamma, done, next_q_values)

        # q loss
        self.q_optimizer.zero_grad()
        q_values = self.q_network((obs, act), training=True)
        mse_q_values_loss = 0.5 * torch.square(torch.unsqueeze(q_target, dim=0) - q_values)  # (num_ensembles, None)
        mse_q_values_loss = torch.mean(torch.sum(mse_q_values_loss, dim=0), dim=0)  # scalar

        # in-distribution q values is simply q_values
        # max_a Q(s,a)
        with torch.no_grad():
            obs_tile = torch.tile(obs, (self.num_samples, 1))
            actions = self.policy_net.select_action((obs_tile, False))  # (num_samples * None, act_dim)

        cql_q_values = self.q_network((obs_tile, actions), training=False)  # (num_samples * None)
        cql_q_values = torch.reshape(cql_q_values, shape=(self.num_samples, batch_size))
        cql_q_values = torch.max(cql_q_values, dim=0)[0]  # (None,)
        cql_threshold = torch.mean(cql_q_values - torch.min(q_values, dim=0)[0].detach(), dim=0)

        q_loss = mse_q_values_loss + alpha_cql * cql_threshold
        q_loss.backward()
        self.q_optimizer.step()

        # update alpha_cql
        self.cql_alpha_optimizer.zero_grad()
        alpha_cql = self.log_cql()
        delta_cql = cql_threshold - self.cql_threshold
        alpha_cql_loss = -alpha_cql * delta_cql
        alpha_cql_loss.backward()
        self.cql_alpha_optimizer.step()

        # update policy
        action, log_prob, _, _ = self.policy_net((obs, False))
        q_values_pi_min = self.q_network((obs, action), training=False)
        policy_loss = torch.mean(log_prob * alpha - q_values_pi_min)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha = self.log_alpha()
        alpha_loss = -torch.mean(alpha * (log_prob.detach() + self.target_entropy))
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        info = dict(
            Q1Vals=q_values[0],
            Q2Vals=q_values[1],
            LogPi=log_prob,
            Alpha=alpha,
            LossQ=mse_q_values_loss,
            LossAlpha=alpha_loss,
            LossPi=policy_loss,
            AlphaCQL=alpha_cql,
            AlphaCQLLoss=alpha_cql_loss,
            DeltaCQL=delta_cql,
        )
        return info

    def train_on_batch(self, data, **kwargs):
        update_target = data.pop('update_target')
        behavior_cloning = data.pop('behavior_cloning')
        data = {key: torch.as_tensor(value, device=ptu.device) for key, value in data.items()}
        if behavior_cloning:
            info = self._update_nets_behavior_cloning(**data)
        else:
            info = self._update_nets(**data)
        self.logger.store(**info)
        if update_target:
            self.update_target()

    def act_batch_test(self, obs):
        with torch.no_grad():
            batch_size = obs.shape[0]
            obs_tile = torch.tile(obs, (self.num_samples, 1))
            actions = self.policy_net.select_action((obs_tile, False))  # (num_samples * None, act_dim)
            q_values = self.q_network((obs_tile, actions), training=False)  # (num_samples * None)
            q_values = torch.reshape(q_values, shape=(self.num_samples, batch_size))
            max_idx = torch.max(q_values, dim=0)[1]  # (None)
            actions = torch.reshape(actions, shape=(
                self.num_samples, batch_size * self.act_dim))  # (num_samples, None * act_dim)
            actions = actions.gather(0, max_idx.unsqueeze(0)).squeeze(0)  # (None * act_dim)
            actions = torch.reshape(actions, shape=(batch_size, self.act_dim))
            return actions


class CQLUpdater(rl_infra.updater.OffPolicyUpdater):
    def __init__(self, agent, replay_buffer, policy_delay, update_per_step, update_every, behavior_cloning_steps):
        super(CQLUpdater, self).__init__(agent, replay_buffer, policy_delay, update_per_step, update_every)
        self.policy_delay = policy_delay
        self.update_per_step = update_per_step
        self.update_every = update_every
        self.behavior_cloning_steps = behavior_cloning_steps

    def update(self, global_step):
        if global_step % self.update_every == 0:
            for _ in range(int(self.update_per_step * self.update_every)):
                batch = self.replay_buffer.sample()
                batch['update_target'] = ((self.policy_updates + 1) % self.policy_delay == 0)
                if global_step < self.behavior_cloning_steps:
                    batch['behavior_cloning'] = True
                else:
                    batch['behavior_cloning'] = False
                self.agent.train_on_batch(data=batch)
                self.policy_updates += 1


class Runner(rl_infra.runner.OfflineRunner):
    def setup_updater(self, update_after, policy_delay, update_per_step, update_every, behavior_cloning_steps):
        self.updater = CQLUpdater(agent=self.agent,
                                  replay_buffer=self.replay_buffer,
                                  policy_delay=policy_delay,
                                  update_per_step=update_per_step,
                                  update_every=update_every,
                                  behavior_cloning_steps=behavior_cloning_steps)

    @classmethod
    def main(cls,
             env_name,
             env_fn=None,
             exp_name=None,
             steps_per_epoch=5000,
             epochs=200,
             behavior_cloning_epochs=5,
             update_every=1,
             update_per_step=1,
             policy_delay=1,
             batch_size=256,
             num_test_episodes=30,
             seed=1,
             # agent args
             # sac args
             policy_mlp_hidden=256,
             policy_lr=3e-4,
             q_mlp_hidden=256,
             q_lr=3e-4,
             alpha=0.2,
             tau=5e-3,
             gamma=0.99,
             cql_threshold=1.,
             # replay
             dataset=None,
             logger_path=None,
             **kwargs
             ):
        agent_kwargs = dict(
            policy_mlp_hidden=policy_mlp_hidden,
            policy_lr=policy_lr,
            q_mlp_hidden=q_mlp_hidden,
            q_lr=q_lr,
            alpha=alpha,
            alpha_lr=q_lr,
            alpha_cql=alpha,
            alpha_cql_lr=q_lr,
            tau=tau,
            gamma=gamma,
            target_entropy=None,
            cql_threshold=1.
        )

        config = locals()

        runner = cls(seed=seed, steps_per_epoch=steps_per_epoch, epochs=epochs,
                     exp_name=exp_name, logger_path=logger_path)
        runner.setup_env(env_name=env_name, env_fn=env_fn, num_parallel_env=-1,
                         asynchronous=False, num_test_episodes=num_test_episodes)
        runner.setup_agent(agent_cls=CQLAgent, **agent_kwargs)
        runner.setup_replay_buffer(dataset=dataset,
                                   batch_size=batch_size)
        runner.setup_tester(num_test_episodes=num_test_episodes)
        runner.setup_updater(update_after=-1,
                             policy_delay=policy_delay,
                             update_per_step=update_per_step,
                             update_every=update_every,
                             behavior_cloning_steps=behavior_cloning_epochs * steps_per_epoch)
        runner.setup_logger(config=config, tensorboard=False)

        pprint.pprint(runner.seeds_info)

        runner.run()
