from __future__ import absolute_import, division, print_function

import os
import datetime
import argparse
# import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from torch.utils.data import DataLoader
from torchvision import transforms, utils

from src.obs_data import ObservationDataset
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
    parser.add_argument('--rollouts_fname',
                        help='Rollouts file name')
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # read in data, transform to preprocess data
    composed = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train = ObservationDataset(os.path.join(DATA_DIR, 'rollouts', args.rollouts_fname),
                               str(int(args.rollouts_path.split('.')[-2][-1]) * 1000),  # TODO: hack
                               transform=composed)
    train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True)

    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters())

    loss = vae_loss()

    for i in range(args.n_epochs):
        pass

    # As the environment may give us observations as high dimensional pixel images,
    # we first resize each image to 64x64 pixels before and use this resized image
    # as the V Model's observation. Each pixel is stored as three floating point values
    # between 0 and 1 to represent each of the RGB channels.

if __name__ == '__main__':
    main()
