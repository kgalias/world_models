from __future__ import absolute_import, division, print_function

import argparse
import os
from functools import partial

import gym
import numpy as np
import torch
import torch.multiprocessing as mp

from src import DATA_DIR
from src.controller import ControllerAgent
from src.es import CMAES
from src.rnn import MDNRNN
from src.utils import suppress_stdout, param_count
from src.vae import VAE


def evaluate(env_name, vae, rnn, v_dim, action_dim, m_dim, controller_params, n_rollouts):
    if env_name == 'CarRacing-v0':
        env = gym.make(env_name)
    else:
        raise NotImplementedError('Environment not supported: ' + env_name)

    agent = ControllerAgent(vae, rnn, v_dim, action_dim, m_dim, controller_params)
    rewards = []
    # TODO: add support for memory. hidden_state is of shape (1, 1, m_dim)?
    for rollout_num in range(1, n_rollouts+1):
        ep_reward = 0
        with suppress_stdout():  # Suppress track generation message.
            obs = env.reset()
        env.env.viewer.window.dispatch_events()  # CarRacing-v0 is bugged and corrupts obs without this.
        done = False
        reward = 0

        while not done:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward

        rewards.append(ep_reward)

    return rewards


def main():
    # TODO: add option to use observation without memory.
    parser = argparse.ArgumentParser(description='Evolutionary training of controller')
    parser.add_argument('--env_name', nargs='?', default='CarRacing-v0',
                        help='Environment to use (default=CarRacing-v0)')
    parser.add_argument('--n_rollouts', type=int, default=1,
                        help='How many rollouts to perform when evaluating (default=1)')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs to train (default=10)')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Dimension of latent space (default=32)')
    parser.add_argument('--seq_len', type=int, default=10,
                        help='Length of sequences for learning (default=10)')
    parser.add_argument('--action_dim', type=int, default=3,
                        help='Dimension of action space (default=3)')
    parser.add_argument('--rnn_hidden_dim', type=int, default=256,
                        help='Dimension of RNN hidden state (default=256)')
    parser.add_argument('--n_gaussians', type=int, default=5,
                        help='Number of gaussians for the Mixture Density Network (default=5)')
    parser.add_argument('--pop_size', type=int, default=64,
                        help='Population size for evolutionary search (default=64)')
    parser.add_argument('--n_workers', type=int, default=16,
                        help='Number of workers for parallel processing (default=16)')
    parser.add_argument('--vae_fname',
                        help='VAE model file name')
    parser.add_argument('--rnn_fname',
                        help='RNN model file name')
    parser.add_argument('--dir_name',
                        help='Rollouts directory name')
    parser.add_argument('--log_interval', nargs='?', default='2', type=int,
                        help='After how many epochs to log')
    args = parser.parse_args()

    device = torch.device('cpu')

    # Load the VAE model from file.
    vae = VAE(latent_dim=args.latent_dim)
    vae.load_state_dict(torch.load(os.path.join(DATA_DIR, 'vae', args.vae_fname)))
    vae.to(device)

    # Load the MDNRNN model from file.
    mdnrnn = MDNRNN(action_dim=args.action_dim,
                    hidden_dim=args.rnn_hidden_dim,
                    latent_dim=args.latent_dim,
                    n_gaussians=args.n_gaussians)
    mdnrnn.load_state_dict(torch.load(os.path.join(DATA_DIR, 'rnn', args.rnn_fname)))
    mdnrnn.to(device)

    # Set up controller model.
    agent = ControllerAgent(vae=vae,
                            rnn=mdnrnn,
                            v_dim=args.latent_dim,
                            action_dim=args.action_dim,
                            m_dim=args.rnn_hidden_dim)

    # Set up evolutionary strategy optimizer.
    # weight_decay = 0.01
    es = CMAES(num_params=param_count(agent.controller),
               sigma_init=0.1,  # initial standard deviation
               popsize=args.pop_size)

    # Set up multiprocessing.
    pool = mp.Pool(processes=args.n_workers)

    while True:
        # Create a set of candidate specimens.
        specimens = es.ask()

        # Evaluate the fitness of candidate specimens.
        func = partial(evaluate,
                       env_name=args.env_name,
                       vae=vae,
                       rnn=mdnrnn,
                       v_dim=args.latent_dim,
                       action_dim=args.action_dim,
                       m_dim=args.rnn_hidden_dim,
                       n_rollouts=args.n_rollouts)
        fitness_list = np.array(pool.map(func, specimens))

        # give list of fitness results back to ES
        es.tell(fitness_list)

        # get best parameter, fitness from ES
        best_solution, best_fitness = es.result()

        if best_fitness > MY_REQUIRED_FITNESS:
            break


if __name__ == '__main__':
    main()
