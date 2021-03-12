"""
Copied from https://github.com/openai/gym/blob/master/gym/vector/sync_vector_env.py.
Modified on done. Instead of directly reset, we add reset_done to reset after observing next_obs.
"""

import numpy as np
from gym.vector.utils import create_empty_array

from .vector_env import VectorEnv

__all__ = ['SyncVectorEnv']


class SyncVectorEnv(VectorEnv):
    """Vectorized environment that serially runs multiple environments.

    Parameters
    ----------
    env_fns : iterable of callable
        Functions that create the environments.

    observation_space : `gym.spaces.Space` instance, optional
        Observation space of a single environment. If `None`, then the
        observation space of the first environment is taken.

    action_space : `gym.spaces.Space` instance, optional
        Action space of a single environment. If `None`, then the action space
        of the first environment is taken.

    copy : bool (default: `True`)
        If `True`, then the `reset` and `step` methods return a copy of the
        observations.
    """

    def __init__(self, env_fns, observation_space=None, action_space=None,
                 copy=True):
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.copy = copy

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space
        super(SyncVectorEnv, self).__init__(num_envs=len(env_fns),
                                            observation_space=observation_space, action_space=action_space)

        self._check_observation_spaces()
        self.observations = create_empty_array(self.single_observation_space,
                                               n=self.num_envs, fn=np.zeros)
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._dones = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = None

    def seed(self, seeds=None):
        if seeds is None:
            seeds = [None for _ in range(self.num_envs)]
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]
        assert len(seeds) == self.num_envs

        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    """
    reset done
    """

    def reset_done_wait(self):
        for i, env in enumerate(self.envs):
            if self._dones[i]:
                self.observations[i] = env.reset()
        return np.copy(self.observations) if self.copy else self.observations

    """
    reset obs
    """

    def reset_obs_async(self, obs, mask=None):
        self._reset_obs = obs
        self._reset_mask = mask if mask is not None else np.ones(shape=(self.num_envs,), dtype=np.bool_)

    def reset_obs_wait(self, **kwargs):
        for i, env in enumerate(self.envs):
            if self._reset_mask[i]:
                self.observations[i] = env.reset_obs(self._reset_obs[i])
        return np.copy(self.observations) if self.copy else self.observations

    """
    reset
    """

    def reset_wait(self):
        for i, env in enumerate(self.envs):
            self.observations[i] = env.reset()
        return np.copy(self.observations) if self.copy else self.observations

    """
    step
    """

    def step_async(self, actions, mask=None):
        self._actions = actions
        self._mask = mask if mask is not None else np.ones(shape=(self.num_envs,), dtype=np.bool_)

    def step_wait(self):
        observations, infos = [], []
        for i, (env, action, mask) in enumerate(zip(self.envs, self._actions, self._mask)):
            if mask:
                self.observations[i], self._rewards[i], self._dones[i], info = env.step(action)
                infos.append(info)
            else:
                infos.append({})

        return (np.copy(self.observations) if self.copy else self.observations,
                np.copy(self._rewards), np.copy(self._dones), infos)

    def close_extras(self, **kwargs):
        [env.close() for env in self.envs]

    def _check_observation_spaces(self):
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                break
        else:
            return True
        raise RuntimeError('Some environments have an observation space '
                           'different from `{0}`. In order to batch observations, the '
                           'observation spaces from all environments must be '
                           'equal.'.format(self.single_observation_space))
