from gym import register, make

from . import utils
from . import vector
from . import wrappers

__all__ = ['make']

register(
    id='AntTruncated-v2',
    entry_point='rlutils.gym.envs:AntEnvTruncated',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='HumanoidTruncated-v2',
    entry_point='rlutils.gym.envs:HumanoidEnvTruncated',
    max_episode_steps=1000,
)

register(
    id='PendulumResetObs-v0',
    entry_point='rlutils.gym.envs:PendulumEnvResetObs',
    max_episode_steps=200,
)

register(
    id='HopperResetObs-v2',
    entry_point='rlutils.gym.envs:HopperEnvResetObs',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='Walker2dResetObs-v2',
    entry_point='rlutils.gym.envs:Walker2dEnvResetObs',
    max_episode_steps=1000,
)

register(
    id='HalfCheetahResetObs-v2',
    entry_point='rlutils.gym.envs:HalfCheetahEnvResetObs',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='AntResetObs-v2',
    entry_point='rlutils.gym.envs:AntEnvResetObs',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
