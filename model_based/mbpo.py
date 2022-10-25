"""
Due to the high computation requirement, we employ the following process:
- A process that collects data and performs learning. This controls the ratio between data collection and policy updates
- A process that samples data and performs dynamics training
- A process that samples data to perform rollouts
"""


class PolicyTrainer(object):
    def __init__(self, env_fn, agent_fn, make_replay_fn,
                 start_steps, num_policy_updates_per_env, seed):
        self.env = env_fn()
        self.seed = seed
        self.start_steps = start_steps
        self.num_policy_updates_per_env = num_policy_updates_per_env
        self.agent = agent_fn()
        self.imagined_dataset = make_replay_fn()

    def run(self):
        obs, _ = self.env.reset(seed=self.seed)
        done = False
        num_env_interactions = 0
        while True:
            # step 1: performs an environmental interaction
            if num_env_interactions < self.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.agent.act_batch_explore(obs, global_steps=None)
            next_obs, reward, terminate, truncate, info = self.env.step(action)
            num_env_interactions += 1

            # add data to remote true dataset

            # step 2: perform rollouts

            # step 2: perform policy training
            pass


class DynamicsTrainer(object):
    def __init__(self, make_dataset_fn):
        pass

    def add_to_dataset(self):
        pass

    def run(self):
        pass


class Logger(object):
    pass


if __name__ == '__main__':
    pass
