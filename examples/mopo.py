"""
Train a dynamics model. Then, optimize the policy in a surrogate MDP defined by he model.
The key is how to define the uncertainty. Apart from standard inputs, we also pass uncertainty.
In MOPO, the uncertainty estimator is the maximum of the standard deviation of ensemble output Gaussian distributions.
"""

import tensorflow as tf
import rlutils.tf as rlu


