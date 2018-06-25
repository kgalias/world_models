from __future__ import absolute_import, division, print_function

import os
import datetime
import argparse

import tqdm
import numpy as np

import gym
from src.utils import suppress_stdout, obs_to_array
from src import DATA_DIR


# TODO: move to agents file?
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs, reward, done):
        return self.action_space.sample()


def save_rollouts(env, agent, n_rollouts, dir_name):
    """
    Collects the actions and resulting observations given an agent in an environment.
    Repeats the process n_rollouts times.
    """

    # TODO: parallelize?
    for rollout_num in tqdm.tqdm(range(1, n_rollouts+1)):
        with suppress_stdout():  # Suppress track generation message for tqdm to work.
            obs = env.reset()
        env.env.viewer.window.dispatch_events()  # CarRacing-v0 is bugged and corrupts obs without this.
        done = False
        reward = 0
        rollout = {'actions': [], 'observations': [], 'rewards': [], 'dones': []}

        while not done:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            rollout['actions'].append(np.float32(action))
            rollout['observations'].append(obs_to_array(obs))  # Resize and save as uint8 to save space.
            rollout['rewards'].append(np.float32(reward))
            rollout['dones'].append(np.uint8(done))

        np.savez(os.path.join(DATA_DIR, 'rollouts', dir_name, str(rollout_num)),
                 **rollout)


def main():
    parser = argparse.ArgumentParser(description='Rollout of an agent in an environment')
    parser.add_argument('--env', nargs='?', default='CarRacing-v0',
                        help='Environment to use (default=CarRacing-v0)')
    parser.add_argument('--agent', nargs='?', default='RandomAgent',
                        help='Agent to run (default=RandomAgent)')
    parser.add_argument('--n_rollouts', nargs='?', default='10000', type=int,
                        help='How many rollouts to perform (default=10000)')
    args = parser.parse_args()

    if args.env == 'CarRacing-v0':
        env = gym.make(args.env)
    else:
        raise NotImplementedError('Environment not supported: ' + args.env)

    if args.agent == 'RandomAgent':
        agent = RandomAgent(env.action_space)
    else:
        raise NotImplementedError('Agent not supported: ' + args.agent)

    start_time = datetime.datetime.today().isoformat()
    dir_name = args.env + '_' + args.agent + '_' + start_time + '_' + args.n_rollouts
    os.makedirs(os.path.join(DATA_DIR, 'rollouts', dir_name))

    save_rollouts(env, agent, args.n_rollouts, dir_name)

if __name__ == '__main__':
    main()
