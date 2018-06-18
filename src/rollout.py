from __future__ import absolute_import, division, print_function

import os
import datetime
import argparse

import tqdm
import numpy as np
from PIL import Image
from torchvision.transforms import Resize
import gym
from src.utils import suppress_stdout
from src import DATA_DIR


# TODO: move to agents file?
class RandomAgent(object):

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs, reward, done):
        return self.action_space.sample()


# TODO: should rollout take fname? should log_interval be forced to divide n_rollouts? use (kw)args?
def save_rollouts(env, agent, n_rollouts, dir_name):
    """
    Collects the actions and resulting observations given an agent in an environment.
    Repeats the process n_rollouts times.
    """

    n_frames = 0
    # TODO: parallelize?
    for rollout_num in tqdm.tqdm(range(1, n_rollouts + 1)):
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
            rollout['observations'].append(np.array(Resize((64, 64))(Image.fromarray(obs)), dtype=np.uint8))  # Resize to save space.
            rollout['rewards'].append(np.float32(reward))
            rollout['dones'].append(np.uint8(done))
            n_frames += 1

        np.savez(os.path.join(DATA_DIR, 'rollouts', dir_name, str(rollout_num)),
                 **rollout)
    # keep track of total number of observations
    os.rename(os.path.join(DATA_DIR, 'rollouts', dir_name),
              os.path.join(DATA_DIR, 'rollouts', dir_name) + '_' + str(n_frames))


def main():
    # TODO: have consistent (with other files) argument descriptions.
    parser = argparse.ArgumentParser(description='Rollout of an agent in an environment')
    parser.add_argument('--env', nargs='?', default='CarRacing-v0',
                        help='Environment to use')
    parser.add_argument('--agent', nargs='?', default='RandomAgent',
                        help='Agent to run')
    parser.add_argument('--n_rollouts', nargs='?', default='10000', type=int,
                        help='How many rollouts to perform')
    # parser.add_argument('--save_action', action='store_true', default=False,
    #                     help='Saves actions as well as observations')
    # parser.add_argument('--log_interval', nargs='?', default='500', type=int,
    #                     help='After how many rollouts to log')
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
    dir_name = args.env + '_' + args.agent + '_' + start_time

    # if not os.path.exists(os.path.join(DATA_DIR, 'rollouts')):
    os.makedirs(os.path.join(DATA_DIR, 'rollouts', dir_name))

    save_rollouts(env, agent, args.n_rollouts, dir_name)

if __name__ == '__main__':
    main()
