from __future__ import absolute_import, division, print_function

import os
import datetime
import argparse
# import logging

import tqdm
import torch
import gym
from src.utils import suppress_stdout
from src import DATA_DIR

# logger = logging.getLogger(__name__)


# TODO: move to agents file?
class RandomAgent(object):

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs, reward, done):
        return self.action_space.sample()


# TODO: should rollout take fname? should log_interval be forced to divide n_rollouts? use (kw)args?
def rollout(env, agent, n_rollouts, log_interval, fname):
    """
    Collects the actions and resulting observations given an agent in an environment.
    Repeats the process n_rollouts times.
    """
    # TODO: numpy array instead of list?
    rollouts = {'actions': [], 'observations': []}

    # TODO: parallelize?
    for i in tqdm.tqdm(range(n_rollouts)):
        with suppress_stdout():  # Suppress track generation message for tqdm to work.
            obs = env.reset()
        done = False
        reward = 0
        while not done:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            rollouts['actions'].append(action)
            rollouts['observations'].append(obs)
        if (i+1) % log_interval == 0:
            save_rollouts(rollouts, fname, i+1)

    return save_rollouts(rollouts, fname, n_rollouts)


def save_rollouts(rollouts, fname, rollout_num):
    torch.save(rollouts, os.path.join(DATA_DIR, 'rollouts',
                                      fname + '_' + datetime.datetime.today().isoformat()) + '_' + str(rollout_num) + '.rollout')


def main():
    parser = argparse.ArgumentParser(description='Rollout of an agent in an environment')
    parser.add_argument('--env', nargs='?', default='CarRacing-v0', help='Environment to use')
    parser.add_argument('--agent', nargs='?', default='RandomAgent', help='Agent to run')
    parser.add_argument('--n_rollouts', nargs='?', default='10000', type=int, help='How many rollouts to perform')
    parser.add_argument('--log_interval', nargs='?', default='500', type=int, help='After how many rollouts to log')
    args = parser.parse_args()

    # TODO: reorganize?
    if args.env == 'CarRacing-v0':
        env = gym.make(args.env)
    else:
        raise NotImplementedError('Environment not supported: ' + args.env)

    if args.agent == 'RandomAgent':
        agent = RandomAgent(env.action_space)
    else:
        raise NotImplementedError('Agent not supported: ' + args.agent)

    if not os.path.exists(os.path.join(DATA_DIR, 'rollouts')):
        os.makedirs(os.path.join(DATA_DIR, 'rollouts'))

    fname = args.env + '_' + args.agent
    rollout(env, agent, args.n_rollouts, args.log_interval, fname)

if __name__ == '__main__':
    main()
