from __future__ import absolute_import, division, print_function

import os
import datetime
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from src.data import RolloutDataset
from src.vae import VAE
from src.rnn import MDNRNN, nll_gmm_loss
from src import DATA_DIR


def main():
    # TODO: have consistent (with other files) argument descriptions.
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--batchsize', type=int, default=128,
                        help='Input batch size for training (default=128)')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs to train (default=10)')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Dimension of latent space (default=32)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--vae_fname',
                        help='VAE model file name')
    parser.add_argument('--obs_fname',
                        help='Observation rollouts file name')
    parser.add_argument('--act_fname',
                        help='Action rollouts file name')
    parser.add_argument('--log_interval', nargs='?', default='2', type=int,
                        help='After how many epochs to log')
    args = parser.parse_args()

    # TODO: obs & act have to be synchronized, hence shuffle=False. Find different way. Set seed?
    obs_dataset = RolloutDataset(path_to_file=os.path.join(DATA_DIR, 'rollouts', args.obs_fname),
                                 size=int(args.obs_fname.split('.')[-2].split('_')[-2]) * 1000,
                                 transform=ToTensor())
    obs_loader = DataLoader(obs_dataset, batch_size=args.batchsize, shuffle=False)

    act_dataset = RolloutDataset(path_to_file=os.path.join(DATA_DIR, 'rollouts', args.act_fname),
                                 size=int(args.act_fname.split('.')[-2].split('_')[-2]) * 1000,
                                 image=False,
                                 transform=torch.Tensor)
    act_loader = DataLoader(act_dataset, batch_size=args.batchsize, shuffle=False)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    vae = VAE()
    vae.load_state_dict(torch.load(os.path.join(DATA_DIR, 'vae', args.vae_filename)))
    vae.to(device)

    mdnrnn = MDNRNN().to(device)
    optimizer = optim.Adam(mdnrnn.parameters())  # TODO: use different optimizer?

    def train(epoch):


if __name__ == '__main__':
    main()
