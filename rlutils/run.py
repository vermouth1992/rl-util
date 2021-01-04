from rlutils.algos.tf.a2c import a2c
from rlutils.algos.tf.a2c_q import a2c_q
from rlutils.algos.tf.ppo import ppo
from rlutils.algos.tf.sac import sac
from rlutils.algos.tf.td3 import td3
from rlutils.algos.tf.trpo import trpo
from rlutils.runner import get_argparser_from_func

__all__ = ['ppo', 'td3', 'trpo', 'sac', 'a2c', 'a2c_q']

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Running rl algorithms', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    algorithm_parsers = parser.add_subparsers(title='algorithm', help='algorithm specific parser', dest='algo')
    for algo in __all__:
        algo_parser = algorithm_parsers.add_parser(algo, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        get_argparser_from_func(eval(algo), algo_parser)

    kwargs = vars(parser.parse_args())
    algo = kwargs.pop('algo')
    eval(algo)(**kwargs)
