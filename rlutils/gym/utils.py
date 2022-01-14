import gym
import numpy as np

import rlutils.gym
import rlutils.np as rln


def verify_continuous_action_space(act_spec: gym.spaces.Box):
    assert np.max(act_spec.high) == np.min(act_spec.high), \
        f'Not all the values in high are the same. Got {act_spec.high}'
    assert np.max(act_spec.low) == np.min(act_spec.low), \
        f'Not all the values in low are the same. Got {act_spec.low}'
    assert act_spec.high[0] + act_spec.low[0] == 0., f'High is not equal to low'
    assert act_spec.high[0] == 1.0


def get_true_done_from_infos(done, infos):
    timeouts = rln.gather_dict_key(infos=infos, key='TimeLimit.truncated', default=False, dtype=np.bool)
    # Ignore the "done" signal if it comes from hitting the time
    # horizon (that is, when it's an artificial terminal signal
    # that isn't based on the agent's state)
    true_d = np.logical_and(done, np.logical_not(timeouts))
    return true_d


def wrap_env_fn(env_fn,
                truncate_obs_dtype=True,
                normalize_action_space=True):
    original_env_fn = env_fn
    dummy_env = original_env_fn()

    wrappers = []
    # convert to 32-bit observation and action space
    if isinstance(dummy_env.observation_space, gym.spaces.Box) and truncate_obs_dtype:
        if dummy_env.observation_space.dtype == np.float64:
            print('Truncating observation_space dtype from np.float64 to np.float32')
            fn = lambda env: rlutils.gym.wrappers.TransformObservationDtype(env, dtype=np.float32)
            wrappers.append(fn)
    else:
        raise NotImplementedError

    if isinstance(dummy_env.action_space, gym.spaces.Box) and normalize_action_space:
        act_lim = 1.
        high_all = np.all(dummy_env.action_space.high == act_lim)
        low_all = np.all(dummy_env.action_space.low == -act_lim)
        print(f'Original high: {dummy_env.action_space.high}, low: {dummy_env.action_space.low}')
        if not (high_all and low_all):
            print(f'Rescale action space to [-{act_lim}, {act_lim}]')
            fn = lambda env: gym.wrappers.RescaleAction(env, -act_lim, act_lim)
            wrappers.append(fn)

    def _make_env():
        env = original_env_fn()
        for wrapper in wrappers:
            env = wrapper(env)
        return env

    return _make_env


def create_vector_env(env_fn,
                      truncate_obs_dtype=True,
                      normalize_action_space=True,
                      num_parallel_env=1,
                      asynchronous=False,
                      seed=None,
                      action_space_seed=None
                      ):
    """
    Environment is either created by env_name or env_fn. In addition, we apply Rescale action wrappers to
    run Pendulum-v0 using env_name from commandline.
    Other complicated wrappers should be included in env_fn.
    """
    assert env_fn is not None
    _make_env = wrap_env_fn(env_fn, truncate_obs_dtype, normalize_action_space)
    VecEnv = rlutils.gym.vector.AsyncVectorEnv if asynchronous else rlutils.gym.vector.SyncVectorEnv

    env = VecEnv([_make_env for _ in range(num_parallel_env)])
    env.seed(seed)
    env.action_space.seed(action_space_seed)

    return env


def create_atari_env_fn(env_name):
    if 'NoFrameskip' not in env_name:
        frame_skip = 1
    else:
        frame_skip = 4
    env_fn = lambda: gym.wrappers.AtariPreprocessing(gym.make(env_name), frame_skip=frame_skip)
    return env_fn


def create_atari_vector_env(env_name, num_parallel_env=1, asynchronous=False, seed=None, action_space_seed=None):
    env_fn = create_atari_env_fn(env_name=env_name)
    env = rlutils.gym.utils.create_vector_env(env_fn=env_fn,
                                              normalize_action_space=True,
                                              num_parallel_env=num_parallel_env,
                                              asynchronous=asynchronous,
                                              seed=seed,
                                              action_space_seed=action_space_seed)
    return env
