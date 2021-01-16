import d4rl
from rlutils.algos.tf.offline.bracp import bracp

__all__ = ['d4rl']

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--pretrain_behavior', action='store_true')
    parser.add_argument('--pretrain_cloning', action='store_true')
    parser.add_argument('--seed', type=int, nargs='*', default=[1])

    args = vars(parser.parse_args())
    env_name = args['env_name']
    # setup env specific arguments.
    seeds = args.pop('seed')

    print(f'Running {env_name} for seeds {seeds}')

    for seed in seeds:
        bracp(**args, seed=seed)
