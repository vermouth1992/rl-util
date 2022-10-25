"""
Due to the high computation requirement, we employ the following process:
- A process that collects data and performs learning. This controls the ratio between data collection and policy updates
- A process that samples data and performs dynamics training
- A process that samples data to perform rollouts
"""


class PolicyTrainer(object):
    def __init__(self):
        pass

    def run(self):
        while True:
            # step 1: performs an environmental interaction

            # step 1: samples a batch
            pass


class DynamicsTrainer(object):
    pass


class Rollout(object):
    pass


class Logger(object):
    pass


if __name__ == '__main__':
    pass
