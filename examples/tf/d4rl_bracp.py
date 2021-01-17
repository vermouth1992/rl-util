import glob
import json
import os
import pprint
import sys
import time

import d4rl
import gym
import numpy as np
import tensorflow as tf
from rlutils.algos.tf.offline.bracp import bracp, BRACPAgent
from rlutils.logx import EpochLogger
from tqdm.auto import tqdm

__all__ = ['d4rl']


def load_policy_and_env(filepath):
    print('\n\nLoading from %s.\n\n' % filepath)
    with open(os.path.join(filepath, 'config.json'), 'r') as f:
        config = json.load(f)

    dummy_env = gym.make(config['env_name'])
    filepath = os.path.join(filepath, 'agent_final.ckpt')

    obs_dim = dummy_env.observation_space.shape[-1]
    act_dim = dummy_env.action_space.shape[-1]
    agent = BRACPAgent(ob_dim=obs_dim, ac_dim=act_dim,
                       out_dist=config['actor_distribution'],
                       num_ensembles=config['num_ensembles'],
                       behavior_mlp_hidden=config['behavior_mlp_hidden'],
                       behavior_lr=1e-3,
                       policy_mlp_hidden=config['policy_mlp_hidden'], q_mlp_hidden=config['q_mlp_hidden'],
                       alpha_mlp_hidden=config['alpha_mlp_hidden'],
                       q_lr=1e-3, alpha_lr=1e-3, alpha=1, tau=None, gamma=None,
                       target_entropy=None, huber_delta=None, gp_type='none',
                       alpha_update='global',
                       reg_type='kl', sigma=None, n=None, gp_weight=1,
                       entropy_reg=None, kl_backup=None)
    agent.load_weights(filepath=filepath).expect_partial()  # no optimizer is defined
    del dummy_env

    return config['env_name'], agent


def test_agent(test_env, dummy_env, num_test_episodes, agent, name, logger=None):
    o, d, ep_ret, ep_len = test_env.reset(), np.zeros(shape=num_test_episodes, dtype=np.bool), \
                           np.zeros(shape=num_test_episodes), np.zeros(shape=num_test_episodes, dtype=np.int64)
    t = tqdm(total=1, desc=f'Testing {name}')
    while not np.all(d):
        a = agent.act_batch(tf.convert_to_tensor(o, dtype=tf.float32)).numpy()
        assert not np.any(np.isnan(a)), f'nan action: {a}'
        o, r, d_, _ = test_env.step(a)
        ep_ret = r * (1 - d) + ep_ret
        ep_len = np.ones(shape=num_test_episodes, dtype=np.int64) * (1 - d) + ep_len
        d = np.logical_or(d, d_)
    t.update(1)
    t.close()
    normalized_ep_ret = dummy_env.get_normalized_score(ep_ret) * 100
    if logger is not None:
        logger.store(TestEpRet=ep_ret, NormalizedTestEpRet=normalized_ep_ret, TestEpLen=ep_len)
    else:
        print(f'EpRet: {np.mean(ep_ret):.2f}, TestEpLen: {np.mean(ep_len):.2f}')


def run_policy(env_name, agent, num_episodes=1000, seed=0):
    num_test_episodes = 20
    assert num_episodes % num_test_episodes == 0, f"num_episodes must be multiplier of {num_test_episodes}"
    dummy_env = gym.make(env_name)
    test_env = gym.vector.make(env_name, num_envs=num_test_episodes, asynchronous=False)
    np.random.seed(seed)
    test_env.seed(np.random.randint(sys.maxsize))
    tf.random.set_seed(np.random.randint(sys.maxsize))
    logger = EpochLogger(output_dir=os.path.expanduser("~/tmp/experiments/%i" % int(time.time())))
    for _ in range(num_episodes // num_test_episodes):
        test_agent(test_env, dummy_env, num_test_episodes, agent, 'final policy', logger)

    mean_ep_ret = logger.get_stats('TestEpRet')[0]
    mean_normalized_ep_ret = logger.get_stats('NormalizedTestEpRet')[0]

    logger.log_tabular('TestEpRet', with_min_and_max=True)
    logger.log_tabular('NormalizedTestEpRet', with_min_and_max=True)
    logger.log_tabular('TestEpLen', average_only=True)
    logger.dump_tabular()

    return mean_ep_ret, mean_normalized_ep_ret


def test_policy(args):
    sub_folders = glob.glob(os.path.join(args['fpath'], './*/'))
    sub_folders = list(filter(lambda s: 'tensorboard' not in s, sub_folders))
    if len(sub_folders) == 0:
        env_name, agent = load_policy_and_env(args['fpath'])
        run_policy(env_name, agent, num_episodes=args['episodes'], seed=args['seed'])
    else:
        ret = []
        normalized_ret = []
        for sub_folder in sub_folders:
            env_name, agent = load_policy_and_env(sub_folder)
            mean_ep_ret, mean_normalized_ep_ret = run_policy(
                env_name, agent, num_episodes=args['episodes'], seed=args['seed'])
            ret.append(mean_ep_ret)
            normalized_ret.append(mean_normalized_ep_ret)
        print(f'Mean EpRet: {np.mean(ret):.2f}, Std EpRet: {np.std(ret):.2f}')
        print(f'Mean Normalized EpRet: {np.mean(normalized_ret):.2f}, '
              f'Std Normalized EpRet: {np.std(normalized_ret):.2f}')


def train_policy(args):
    env_name = args['env_name']
    # setup env specific arguments.
    seeds = args.pop('seed')
    print(f'Running {env_name} for seeds {seeds}')

    if 'medium-expert' in env_name:
        generalization_threshold = 0.1
        std_scale = 4.
    elif 'medium-replay' in env_name:
        generalization_threshold = 4.0
        std_scale = 4.
    elif 'medium' in env_name:
        generalization_threshold = 0.1
        std_scale = 4.
    elif 'random' in env_name:
        generalization_threshold = None
        std_scale = None
    elif 'human' in env_name:
        generalization_threshold = None
        std_scale = None
    else:
        raise ValueError(f'Unknown env_name {env_name}')

    for seed in seeds:
        print(f'Running {env_name} for seed {seed}')
        bracp(**args,
              std_scale=std_scale,
              generalization_threshold=generalization_threshold,
              seed=seed)

    for seed in seeds:
        bracp(**args, seed=seed)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='action')
    train_parser = subparser.add_parser(name='train')
    train_parser.add_argument('--env_name', type=str, required=True)
    train_parser.add_argument('--pretrain_behavior', action='store_true')
    train_parser.add_argument('--pretrain_cloning', action='store_true')
    train_parser.add_argument('--seed', type=int, nargs='*', default=[1])

    test_parser = subparser.add_parser(name='test')
    test_parser.add_argument('fpath', type=str)
    test_parser.add_argument('--episodes', '-n', type=int, default=100)
    test_parser.add_argument('--seed', type=int, default=0)
    test_parser.add_argument('--render', '-r', action='store_true')

    args = vars(parser.parse_args())

    pprint.pprint(args)

    action = args.pop('action')
    if action == 'train':
        train_policy(args)
    elif action == 'test':
        test_policy(args)
