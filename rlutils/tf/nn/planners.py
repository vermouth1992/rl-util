import numpy as np


class Planner(object):
    def __init__(self, inference_model, horizon=10):
        self.inference_model = inference_model
        self.horizon = horizon

    def reset(self):
        pass

    def act_batch(self, obs):
        raise NotImplementedError


class RandomShooter(Planner):
    def __init__(self, inference_model, horizon=10, num_actions=4096):
        self.num_actions = num_actions
        super(RandomShooter, self).__init__(inference_model=inference_model,
                                            horizon=horizon)

    def act_batch(self, obs):
        """

        Args:
            obs (np.ndarray): (None, obs_dim)

        Returns:

        """

        obs = np.tile(np.expand_dims(obs, axis=0), (self.num_actions, 1)).astype(np.float32)
        act_seq = np.random.uniform(low=-1., high=1.,
                                    size=(self.num_actions, self.horizon, self.act_dim)).astype(np.float32)
        _, reward_seq, _ = self.model.predict_obs_seq(obs, act_seq)
        reward = np.sum(reward_seq, axis=-1)  # (None)
        best_index = np.argmax(reward)
        return act_seq[best_index][0]
