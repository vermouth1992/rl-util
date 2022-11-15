"""
<<<<<<< HEAD:baselines/distributed/a3c.py
Asynchronous Advantage Actor Critic.
- The actor sends rollouts to the learner instead of gradients. This facilitates the implementation of impala
-Each local actor collects trajectories. Send the trajectory to the learner.
- A central learner performs policy updates and value updates and sends the updated weights.
"""


class Actor(object):
    def __init__(self):
        pass


class Learner(object):
    def __init__(self):
        pass
