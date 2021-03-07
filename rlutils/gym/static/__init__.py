from .base import ModelBasedStaticFn

model_based_wrapper_dict = {}

from .classic import model_based_wrapper_dict as classic_dict
from .pybullet import model_based_wrapper_dict as pybullet_dict
from .mujoco import model_based_wrapper_dict as mujoco_dict

model_based_wrapper_dict.update(classic_dict)
model_based_wrapper_dict.update(pybullet_dict)
model_based_wrapper_dict.update(mujoco_dict)


def get_static_fn(env_name):
    return model_based_wrapper_dict[env_name]
