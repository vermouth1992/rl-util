if __name__ == '__main__':
    import gym
    import pickle
    import numpy as np

    from gym.wrappers import AtariPreprocessing, LazyFrames
    from collections import deque

    from rlutils.replay_buffers.memory_efficient_py import PyMemoryEfficientReplayBuffer


    def make_env(env_name, **kwargs):
        def _make():
            env = gym.make(env_name, **kwargs)
            env = AtariPreprocessing(env)
            return env

        return _make


    num_envs = 2
    num_frames = 4

    env = gym.vector.AsyncVectorEnv(
        [make_env('BreakoutNoFrameskip-v4') for _ in range(num_envs)])

    replay_buffer = PyMemoryEfficientReplayBuffer.from_env(env, capacity=1000, batch_size=32)

    obs = env.reset()  # (2, ...)

    frames = [deque(maxlen=num_frames) for _ in range(num_envs)]

    for i in range(num_envs):
        for _ in range(num_frames):
            frames[i].append(obs[i])

    for i in range(1000):  # Burn-in
        # get current obs
        current_frame = np.array(frames)  # (num_env, num_frame, 84, 84)
        current_lazy_frame = [LazyFrames(list(f)) for f in frames]

        actions = env.action_space.sample()
        next_obs, rew, done, _ = env.step(actions)

        for j in range(num_envs):
            frames[j].append(next_obs[j])

        next_lazy_frame = [LazyFrames(list(f)) for f in frames]

        replay_buffer.add(data=dict(
            obs=current_lazy_frame,
            act=np.array(actions),
            next_obs=next_lazy_frame,
            done=done,
            rew=rew
        ))

    replay_buffer_lazy = list(replay_buffer.storage["obs"])
    replay_buffer_nolazy = list(map(np.asarray, replay_buffer.storage["obs"]))

    print(f'Size of replay buffer: {len(pickle.dumps(replay_buffer_lazy)) / 1000:.2f}kb')
    print(f'Size of replay buffer (not lazy): {len(pickle.dumps(replay_buffer_nolazy)) / 1000:.2f}kb')
    print()
    print(f'Size of observations: {np.asarray(obs).nbytes / 1000:.2f}kb')
    print(f'Shape of Observation: {np.asarray(obs).shape}')

    # Size of replay buffer: 106.39kb
    # Size of replay buffer (not lazy): 169.56kb

    # Size of observations: 84.67kb
    # Shape of Observation: (4, 3, 84, 84)
