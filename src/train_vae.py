from __future__ import absolute_import, division, print_function

import os
import argparse
# import logging

import torch
from torch import optim

from torch.utils.data import DataLoader
from torchvision import transforms

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
    # composed = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    shape = (int(args.rollouts_path.split('.')[-2][-1]) * 1000, 64, 64, 3)  # TODO: hack
    train = ObservationDataset(os.path.join(DATA_DIR, 'rollouts', args.rollouts_fname),
                               shape,
                               transform=transforms.ToTensor())
    train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True)

    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters())

    def train(epoch):
        vae.train()
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            output = vae(batch)
            loss = vae_loss(output, batch)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      epoch, batch_idx * len(batch), len(train_loader.dataset),
                      100. * batch_idx / len(train_loader), loss.item()))

    for i in range(1, args.n_epochs + 1):
        train(i)

if __name__ == '__main__':
    main()
