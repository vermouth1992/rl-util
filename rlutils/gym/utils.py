import gym
import numpy as np

import rlutils.gym
import rlutils.np as rln

atari_games_lst = ['adventure', 'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
                   'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
                   'centipede', 'chopper_command', 'crazy_climber', 'defender', 'demon_attack', 'double_dunk',
                   'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar', 'hero',
                   'ice_hockey', 'jamesbond', 'journey_escape', 'kaboom', 'kangaroo', 'krull', 'kung_fu_master',
                   'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
                   'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing', 'solaris',
                   'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down', 'venture',
                   'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']


def get_atari_capitalize(name):
    name = name.split('_')
    name = list(map(lambda s: s.capitalize(), name))
    return ''.join(name)


def is_atari_env(env_name):
    for name in atari_games_lst:
        atari_name = get_atari_capitalize(name)
        if atari_name in env_name:
            return True
    return False


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
                truncate_act_dtype=True,
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

    if isinstance(dummy_env.action_space, gym.spaces.Box):
        if truncate_act_dtype:
            wrappers.append(lambda env: rlutils.gym.wrappers.TransformActionDtype(env, dtype=np.float32))

        if normalize_action_space:
            act_lim = 1.
            high_all = np.all(dummy_env.action_space.high == act_lim)
            low_all = np.all(dummy_env.action_space.low == -act_lim)
            if not (high_all and low_all):
                print(f'Original high: {dummy_env.action_space.high}, low: {dummy_env.action_space.low}')
                print(f'Rescale action space to [-{act_lim}, {act_lim}]')
                fn = lambda env: gym.wrappers.RescaleAction(env, -act_lim, act_lim)
                wrappers.append(fn)

    def _make_env():
        env = original_env_fn()
        for wrapper in wrappers:
            env = wrapper(env)
        return env

    return _make_env


def wrap_atari_env_fn(env_name, terminal_on_life_loss=True):
    if 'NoFrameskip' not in env_name:
        frame_skip = 1
    else:
        frame_skip = 4

    env_fn = lambda: gym.wrappers.AtariPreprocessing(gym.make(env_name), frame_skip=frame_skip,
                                                     terminal_on_life_loss=terminal_on_life_loss)

    dummy_env = gym.make(env_name)
    if terminal_on_life_loss and 'FIRE' in dummy_env.unwrapped.get_action_meanings():
        old_env_fn = env_fn
        env_fn = lambda: rlutils.gym.wrappers.atari.FireResetEnv(old_env_fn())

    return env_fn


def create_vector_env(env_fn,
                      num_parallel_env=1,
                      asynchronous=False,
                      action_space_seed=None
                      ):
    """
    Environment is either created by env_name or env_fn. In addition, we apply Rescale action wrappers to
    run Pendulum-v0 using env_name from commandline.
    Other complicated wrappers should be included in env_fn.
    """
    assert env_fn is not None
    # VecEnv = rlutils.gym.vector.AsyncVectorEnv if asynchronous else rlutils.gym.vector.SyncVectorEnv
    VecEnv = gym.vector.AsyncVectorEnv if asynchronous else gym.vector.SyncVectorEnv

    env = VecEnv([env_fn for _ in range(num_parallel_env)])
    env.action_space.seed(action_space_seed)

    return env
