from __future__ import absolute_import, division, print_function

import os
import datetime
import argparse

import numpy as np
import torch
from torch.multiprocessing import Queue
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from estool.es import CMAES

from src.data import RolloutDataset
from src.vae import VAE, vae_loss
from src.rnn import MDNRNN, nll_gmm_loss
from src.controller import Controller
from src import DATA_DIR


def param_count(model):
    """Returns number of trainable parameters of given model."""
    n_params = 0
    for param in model.parameters():
        if param.requires_grad:  # only count the trainable parameters
            n_params += param.detach().cpu().numpy().flatten().shape[0]
    return n_params


def updated_model(model, params):
    """Sets model parameters with supplied array.

    Args:
        model: A PyTorch model.
        params: A numpy.array containing the appropriate number of parameters.

    Returns:
        The model with its weights set as the supplied array.
    """
    assert param_count(model) == len(params)

    dtype = next(model.parameters()).type()

    offset = 0

    for param in model.parameters():
        if param.requires_grad:
            param.data = torch.from_numpy(params[offset:offset + param.numel()]).view(param.size()).type(dtype)
            offset += param.numel()

    return model


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

    vae = VAE(latent_dim=args.latent_dim)
    vae.load_state_dict(torch.load(os.path.join(DATA_DIR, 'vae', args.vae_fname)))
    vae.to(device)

    mdnrnn = MDNRNN(action_dim=args.action_dim,
                    hidden_dim=args.rnn_hidden_dim,
                    latent_dim=args.latent_dim,
                    n_gaussians=args.n_gaussians).to(device)
    mdnrnn.load_state_dict(torch.load(os.path.join(DATA_DIR, 'rnn', args.rnn_fname)))
    mdnrnn.to(device)

    controller = Controller(v_dim=args.latent_dim, act_dim=args.action_dim, m_dim=args.rnn_hidden_dim).to(device)

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
