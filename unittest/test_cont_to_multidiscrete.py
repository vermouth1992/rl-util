import rlutils.gym as rlg
import gym

if __name__ == '__main__':
    env = gym.make('Hopper-v4')
    env = rlg.wrappers.ContinuousToMultiDiscrete(env, bins_per_dim=3)
    obs, _ = env.reset()

