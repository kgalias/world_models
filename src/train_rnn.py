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

    # TODO: need temporal order and obs & act have to be synchronized, hence shuffle=False. Find different way. Set seed? Write custom sampler for DataLoader?
    # TODO: change to batch_size + 1, because we later use one less?
    obs_dataset = RolloutDataset(path_to_file=os.path.join(DATA_DIR, 'rollouts', args.obs_fname),
                                 size=int(args.obs_fname.split('.')[-2].split('_')[-2]) * 1000,
                                 transform=ToTensor())
    obs_loader = DataLoader(obs_dataset, batch_size=args.batchsize, shuffle=False)

    act_dataset = RolloutDataset(path_to_file=os.path.join(DATA_DIR, 'rollouts', args.act_fname),
                                 size=int(args.act_fname.split('.')[-2].split('_')[-2]) * 1000,
                                 image=False,
                                 transform=torch.Tensor)
    act_loader = DataLoader(act_dataset, batch_size=args.batchsize, shuffle=False)

    assert len(obs_dataset) == len(act_dataset)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    vae = VAE()
    vae.load_state_dict(torch.load(os.path.join(DATA_DIR, 'vae', args.vae_filename)))
    vae.to(device)

    mdnrnn = MDNRNN().to(device)
    optimizer = optim.Adam(mdnrnn.parameters())  # TODO: use different optimizer?

    def train(epoch):
        for batch_idx, (obs_batch, act_batch) in enumerate(zip(obs_loader, act_loader)):

            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            optimizer.zero_grad()

            # Encode obs using vae.
            z_batch = vae.reparameterize(*vae.encode(obs_batch))

            # Predict all but first encoded obs from all but last encoded obs and action.
            targets = z_batch[1:]
            z_batch = z_batch[:-1]
            act_batch = act_batch[:-1]

            # Add seq_len dimension.
            z_batch = z_batch.unsqueeze(0)
            act_batch = act_batch.unsqueeze(0)

            pi, mu, sigma, hidden = mdnrnn(act_batch, z_batch)
            # output = mdnrnn.mdn.sample(pi, mu, sigma)  # Need output for anything in training?

            loss = nll_gmm_loss(targets, pi, mu, sigma)
            loss.backward()

            # torch.nn.utils.clip_grad_norm(mdnrnn.parameters(), 0.25)

            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      epoch, batch_idx * len(obs_batch), len(obs_loader.dataset),
                      100. * batch_idx / len(obs_loader),
                      loss.item() / len(obs_batch)))
    # train
    for i in range(1, args.n_epochs + 1):
        train(i)

    # save learned model
    if not os.path.exists(os.path.join(DATA_DIR, 'rnn')):
        os.makedirs(os.path.join(DATA_DIR, 'rnn'))

    torch.save(mdnrnn.state_dict(), os.path.join(DATA_DIR, 'rnn',
                                                 datetime.datetime.today().isoformat() + str(args.n_epochs)))
    # To load model, do:
    # mdnrnn = MDNRNN()
    # mdnrnn.load_state_dict(torch.load(PATH))

if __name__ == '__main__':
    main()
