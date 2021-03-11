import gym
import numpy as np
import rlutils.gym


def create_vector_env(env_fn=None,
                      normalize_action_space=True,
                      num_parallel_env=1,
                      asynchronous=False
                      ):
    """
    Environment is either created by env_name or env_fn. In addition, we apply Rescale action wrappers to
    run Pendulum-v0 using env_name from commandline.
    Other complicated wrappers should be included in env_fn.
    """
    assert env_fn is not None
    original_env_fn = env_fn
    dummy_env = original_env_fn()

    wrappers = []
    # convert to 32-bit observation and action space
    if isinstance(dummy_env.observation_space, gym.spaces.Box):
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
            fn = lambda env: gym.wrappers.RescaleAction(env, a=-act_lim, b=act_lim)
            wrappers.append(fn)

    def _make_env():
        env = original_env_fn()
        for wrapper in wrappers:
            env = wrapper(env)
        return env

    VecEnv = rlutils.gym.vector.AsyncVectorEnv if asynchronous else rlutils.gym.vector.SyncVectorEnv

    env = VecEnv([_make_env for _ in range(num_parallel_env)])

    return env
