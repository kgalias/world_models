from __future__ import absolute_import, division, print_function

import os
import datetime
import argparse
# import logging

import tqdm

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

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
    # parser.add_argument('--cuda', action='store_true', default=False,
    #                     help='enables CUDA training')
    parser.add_argument('--gpu_id', nargs='?', default=None,
                        help='Which GPU to use (default: None)')
    parser.add_argument('--rollouts_fname',
                        help='Rollouts file name')
    parser.add_argument('--log_interval', nargs='?', default='2', type=int,
                        help='After how many epochs to log')
    args = parser.parse_args()

    # use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + args.gpu_id if args.gpu_id else 'cpu')

    # read in and preprocess data
    train = ObservationDataset(path_to_file=os.path.join(DATA_DIR, 'rollouts', args.rollouts_fname),
                               size=int(args.rollouts_fname.split('.')[-2].split('_')[-2]) * 1000,  # TODO: hack
                               transform=ToTensor())
    train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True)

    # set up model and optimizer
    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters())

    def train(epoch):
        vae.train()
        train_loss = 0
        # for batch_idx, batch in enumerate(tqdm.tqdm(train_loader)):
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = vae(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      epoch, batch_idx * len(batch), len(train_loader.dataset),
                      100. * batch_idx / len(train_loader),
                      loss.item() / len(batch)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))

    for i in range(1, args.n_epochs + 1):
        train(i)

    # save learned model
    if not os.path.exists(os.path.join(DATA_DIR, 'vae')):
        os.makedirs(os.path.join(DATA_DIR, 'vae'))

    torch.save(vae.state_dict(), os.path.join(DATA_DIR, 'vae', datetime.datetime.today().isoformat()))
    # the_model = TheModelClass(*args, **kwargs)
    # the_model.load_state_dict(torch.load(PATH))

if __name__ == '__main__':
    main()
