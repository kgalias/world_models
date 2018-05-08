from __future__ import absolute_import, division, print_function

import os
import datetime
import argparse
# import logging

import numpy as np
import gym
from src import DATA_DIR

# logger = logging.getLogger(__name__)


# TODO: move to agents file?
class RandomAgent(object):

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs, reward, done):
        return self.action_space.sample()


def rollout(env, agent, n_rollouts):
    """
    Collects the actions and resulting observations given an agent in an environment.
    Repeats the process n_rollouts times.
    """
    # TODO: parallelize? add tqdm?
    rollouts = []
    for i in range(n_rollouts):
        actions = []
        observations = []
        obs = env.reset()
        done = False
        reward = 0
        while not done:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            actions.append(action)
            observations.append(obs)
        rollouts.append({'actions': actions, 'observations': observations})
        if i % 500 == 0:
            np.save(os.path.join(DATA_DIR, 'rollouts',
                                 'temp_' + datetime.datetime.today().isoformat()),
                    rollouts)
    return rollouts


def main():
    parser = argparse.ArgumentParser(description='Rollout of an agent in an environment')
    parser.add_argument('--env', nargs='?', default='CarRacing-v0', help='Environment to use')
    parser.add_argument('--agent', nargs='?', default='RandomAgent', help='Agent to run')
    parser.add_argument('--n_rollouts', nargs='?', default='10000', type=int, help='How many rollouts to perform')
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

    # TODO: have consistent args type?
    np.save(os.path.join(DATA_DIR,
                         'rollouts',
                         args.env + '_' + args.agent + '_' + str(args.n_rollouts) + '_' +
                         datetime.datetime.today().isoformat()),
            rollout(env, agent, args.n_rollouts))

if __name__ == '__main__':
    main()
