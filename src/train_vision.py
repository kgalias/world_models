from __future__ import absolute_import, division, print_function

import os
import datetime
import argparse
# import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from src.vae import VAE, vae_loss
from src import DATA_DIR


def main():
    # TODO: have consistent (with other file) argument descriptions
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--batchsize', type=int, default=128,
                        help='Input batch size for training (default=128)')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs to train (default=10)')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Dimension of latent space (default=32)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    device = torch.device('cuda' if args.cuda else 'cpu')

    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters())

    for i in range(args.n_epochs):
        pass

    # As the environment may give us observations as high dimensional pixel images,
    # we first resize each image to 64x64 pixels before and use this resized image
    # as the V Model's observation. Each pixel is stored as three floating point values
    # between 0 and 1 to represent each of the RGB channels.

if __name__ == '__main__':
    main()
