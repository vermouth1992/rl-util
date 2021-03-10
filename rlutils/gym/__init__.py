from gym import register

from . import utils
from . import vector
from . import wrappers

register(
    id='AntTruncated-v2',
    entry_point='rlutils.gym.envs:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='HumanoidTruncated-v2',
    entry_point='rlutils.gym.envs:HumanoidEnv',
    max_episode_steps=1000,
)
