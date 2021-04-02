import rlutils.tf as rlu
import tensorflow as tf
from rlutils.algos.tf.mf import td3
from rlutils.infra.runner import TFOffPolicyRunner, run_func_as_main


class TD3DynamicsQAgent(td3.TD3Agent):
    def __init__(self, *args, **kwargs):
        super(TD3DynamicsQAgent, self).__init__(*args, **kwargs)
        q_mlp_hidden = kwargs['q_mlp_hidden']
        num_q_ensembles = kwargs['num_q_ensembles']
        self.q_network = rlu.nn.EnsembleMinQNet(self.obs_dim, self.obs_dim, q_mlp_hidden,
                                                num_ensembles=num_q_ensembles)
        self.target_q_network = rlu.nn.EnsembleMinQNet(self.obs_dim, self.obs_dim, q_mlp_hidden,
                                                       num_ensembles=num_q_ensembles)
        rlu.functional.hard_update(self.target_q_network, self.q_network)
        self.dynamics = rlu.nn.build_mlp(input_dim=self.obs_dim + self.act_dim, output_dim=self.obs_dim * 2,
                                         mlp_hidden=256, dropout=0.1)
        self.dynamics.add(rlu.distributions.IndependentNormal(min_log_scale=-10, max_log_scale=2.))
        self.dynamics_optimizer = rlu.future.get_adam_optimizer(lr=1e-4)

    def log_tabular(self):
        super(TD3DynamicsQAgent, self).log_tabular()
        self.logger.log_tabular('DynamicsLoss')

    @tf.function
    def _update_dynamics(self, obs, act, next_obs):
        delta_obs = next_obs - obs
        with tf.GradientTape() as tape:
            out_dist = self.dynamics(inputs=tf.concat(values=[obs, act], axis=-1), training=True)
            loss = out_dist.log_prob(delta_obs)
            loss = -tf.reduce_mean(loss, axis=0)
        rlu.future.minimize(loss, tape, self.dynamics, optimizer=self.dynamics_optimizer)
        return dict(
            DynamicsLoss=loss / self.obs_dim
        )

    def predict_next_obs(self, obs, act):
        out_dist = self.dynamics(inputs=tf.concat(values=[obs, act], axis=-1), training=False)
        delta_obs = out_dist.sample()
        next_obs = obs + delta_obs
        return next_obs

    def _compute_next_obs_q(self, next_obs):
        next_action = self.target_policy_net(next_obs)
        next_next_obs = self.predict_next_obs(next_obs, next_action)
        next_q_value = self.target_q_network((next_obs, next_next_obs, tf.constant(True)))
        return next_q_value

    @tf.function
    def _update_q_nets(self, obs, act, next_obs, done, rew):
        print(f'Tracing _update_nets with obs={obs}, actions={act}')
        # compute target q
        next_q_value = self._compute_next_obs_q(next_obs)
        q_target = rlu.functional.compute_target_value(rew, self.gamma, done, next_q_value)
        # q loss
        with tf.GradientTape() as q_tape:
            q_tape.watch(self.q_network.trainable_variables)
            q_values = self.q_network((obs, next_obs, tf.constant(False)))  # (num_ensembles, None)
            q_values_loss = 0.5 * tf.square(tf.expand_dims(q_target, axis=0) - q_values)
            # (num_ensembles, None)
            q_values_loss = tf.reduce_sum(q_values_loss, axis=0)  # (None,)
            # apply importance weights
            q_values_loss = tf.reduce_mean(q_values_loss)
        q_gradients = q_tape.gradient(q_values_loss, self.q_network.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_gradients, self.q_network.trainable_variables))

        self.update_target_q()

        info = dict(
            LossQ=q_values_loss,
        )
        for i in range(self.q_network.num_ensembles):
            info[f'Q{i + 1}Vals'] = q_values[i]
        return info

    @tf.function
    def _update_actor(self, obs):
        print(f'Tracing _update_actor with obs={obs}')
        # policy loss
        with tf.GradientTape() as policy_tape:
            policy_tape.watch(self.policy_net.trainable_variables)
            act = self.policy_net(obs)
            next_obs = self.predict_next_obs(obs, act)
            q = self.q_network((obs, next_obs, tf.constant(True)))
            policy_loss = -tf.reduce_mean(q, axis=0)
        policy_gradients = policy_tape.gradient(policy_loss, self.policy_net.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy_net.trainable_variables))

        self.update_target_policy()

        info = dict(
            LossPi=policy_loss,
        )
        return info

    def train_on_batch(self, data, **kwargs):
        update_target = data.pop('update_target')
        obs = data['obs']
        act = data['act']
        next_obs = data['next_obs']
        info = self._update_q_nets(**data)
        if update_target:
            actor_info = self._update_actor(obs)
            info.update(actor_info)
            dynamics_info = self._update_dynamics(obs, act, next_obs)
            info.update(dynamics_info)
        self.logger.store(**rlu.functional.to_numpy_or_python_type(info))


class Runner(TFOffPolicyRunner):
    @classmethod
    def main(cls,
             env_name,
             epochs=200,
             num_q_ensembles=2,
             policy_mlp_hidden=256,
             policy_lr=1e-3,
             q_mlp_hidden=256,
             q_lr=1e-3,
             actor_noise=0.1,
             target_noise=0.2,
             noise_clip=0.5,
             out_activation='tanh',
             tau=5e-3,
             gamma=0.99,
             seed=1,
             logger_path: str = 'data',
             **kwargs
             ):
        agent_kwargs = dict(
            num_q_ensembles=num_q_ensembles,
            policy_mlp_hidden=policy_mlp_hidden,
            policy_lr=policy_lr,
            q_mlp_hidden=q_mlp_hidden,
            q_lr=q_lr,
            tau=tau,
            gamma=gamma,
            actor_noise=actor_noise,
            target_noise=target_noise,
            noise_clip=noise_clip,
            out_activation=out_activation,
        )

        super(Runner, cls).main(env_name=env_name,
                                epochs=epochs,
                                policy_delay=2,
                                agent_cls=TD3DynamicsQAgent,
                                agent_kwargs=agent_kwargs,
                                seed=seed,
                                logger_path=logger_path,
                                **kwargs)


if __name__ == '__main__':
    run_func_as_main(Runner.main)
