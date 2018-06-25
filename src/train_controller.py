from __future__ import absolute_import, division, print_function

import os
import datetime
import argparse

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Queue

import gym
from estool.es import CMAES

from src.vae import VAE
from src.rnn import MDNRNN
from src.controller import Controller
from src.utils import suppress_stdout, obs_to_array, obs_array_to_tensor, param_count, updated_model
from src import DATA_DIR


class ControllerAgent(object):
    def __init__(self, controller_params, vae, v_dim, action_dim, m_dim):
        self.controller = Controller(v_dim, action_dim, m_dim)
        self.controller = updated_model(self.controller, controller_params)
        self.vae = vae
        self.vae.eval()

    def act(self, obs, reward, done, mem):
        obs = obs_to_array(obs)  # Convert to resized np.array.
        obs = obs[None, :]  # Add dummy dimension for batch size.
        obs = obs_array_to_tensor(obs)  # Transform to normalized PyTorch tensor with channels first.
        with torch.no_grad():
            enc_obs = self.vae.reparameterize(*self.vae.encode(obs))  # TODO: want sampling here?
        return self.controller(enc_obs, mem)


class Specimen(mp.Process):
    def __init__(self, env_name, controller_params, vae, v_dim, action_dim, m_dim, n_rollouts=1):
        super(Specimen, self).__init__()
        self.env_name = env_name
        self.controller_params = controller_params
        self.vae = vae
        self.v_dim = v_dim
        self.act_dim = action_dim
        self.m_dim = m_dim
        self.n_rollouts = n_rollouts

    # TODO: add support for memory.
    def run(self):
        if self.env_name == 'CarRacing-v0':
            env = gym.make(self.env_name)
        else:
            raise NotImplementedError('Environment not supported: ' + self.env_name)
        agent = ControllerAgent(self.controller_params, self.vae, self.v_dim, self.action_dim, self.m_dim)
        rewards = []
        for rollout_num in range(1, self.n_rollouts+1):
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
    parser = argparse.ArgumentParser(description='Evolutionary training of controller')
    parser.add_argument('--batchsize', type=int, default=128,
                        help='Input batch size for training (default=128)')
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
    # parser.add_argument('--cuda', action='store_true', default=False,
    #                     help='enables CUDA training')
    parser.add_argument('--vae_fname',
                        help='VAE model file name')
    parser.add_argument('--rnn_fname',
                        help='RNN model file name')
    parser.add_argument('--dir_name',
                        help='Rollouts directory name')
    parser.add_argument('--log_interval', nargs='?', default='2', type=int,
                        help='After how many epochs to log')
    args = parser.parse_args()

    # use_cuda = args.cuda and torch.cuda.is_available()
    # device = torch.device('cuda' if use_cuda else 'cpu')
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
    controller = Controller(v_dim=args.latent_dim, action_dim=args.action_dim, m_dim=args.rnn_hidden_dim).to(device)

    # __init__(self, num_params,  # number of model parameters
    #          sigma_init=0.10,  # initial standard deviation
    #          popsize=255,  # population size
    #          weight_decay=0.01):
    es = CMAES(num_params=param_count(controller),
               sigma_init=0.1,
               popsize=args.pop_size)

    result_queue = Queue()

    while True:

        # ask the ES to give us a set of candidate solutions
        solutions = es.ask()

        # create an array to hold the fitness results.
        fitness_list = np.zeros(args.pop_size)

        curr_processing = 0

        # evaluate the fitness for each given solution.
        for i in range(args.pop_size):
            fitness_list[i] = evaluate(solutions[i])

        # give list of fitness results back to ES
        es.tell(fitness_list)

        # get best parameter, fitness from ES
        best_solution, best_fitness = es.result()

        if best_fitness > MY_REQUIRED_FITNESS:
            break


if __name__ == '__main__':
    main()
