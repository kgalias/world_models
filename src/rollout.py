from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import gym


# TODO: Move to agents file?
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
    # TODO: parallelize?
    rollouts = []
    for _ in range(n_rollouts):
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
    return rollouts


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', nargs='?', default='CarRacing-v0', help='Select the environment to run')
    parser.add_argument('--agent', nargs='?', default='RandomAgent', help='Select the agent to run')
    parser.add_argument('--n_rollouts', nargs='?', default='10000', type=int, help='Select the agent to run')
    args = parser.parse_args()

    env = gym.make(args.env)

    if args.agent == 'RandomAgent':
        agent = RandomAgent(env.action_space)
    else:
        raise NotImplementedError('Agent not supported: ' + args.agent)

    # TODO: Save to file?
    print(rollout(env, agent, args.n_rollouts))

if __name__ == "__main__":
    main()
