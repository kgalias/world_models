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
    parser.add_argument('--seq_len', type=int, default=10,
                        help='Length of sequences for learning (default=10)')
    parser.add_argument('--action_dim', type=int, default=3,
                        help='Dimension of action space (default=3)')
    parser.add_argument('--n_gaussians', type=int, default=5,
                        help='Number of gaussians for the Mixture Density Network (default=5)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--vae_fname',
                        help='VAE model file name')
    parser.add_argument('--obs_fname',
                        help='Observation rollouts file name')
    parser.add_argument('--act_fname',
                        help='Action rollouts file name')
    parser.add_argument('--obs_test_fname',
                        help='Observation rollouts test file name')
    parser.add_argument('--act_test_fname',
                        help='Action rollouts test file name')
    parser.add_argument('--log_interval', nargs='?', default='2', type=int,
                        help='After how many epochs to log')
    args = parser.parse_args()

    # TODO: do better?
    if not os.path.exists(os.path.join(DATA_DIR, 'rollouts', args.obs_fname)):
        print("File {} does not exist.".format(args.obs_fname))
        pass

    if not os.path.exists(os.path.join(DATA_DIR, 'rollouts', args.act_fname)):
        print("File {} does not exist.".format(args.act_fname))
        pass

    if not os.path.exists(os.path.join(DATA_DIR, 'rollouts', args.obs_test_fname)):
        print("File {} does not exist.".format(args.obs_test_fname))
        pass

    if not os.path.exists(os.path.join(DATA_DIR, 'rollouts', args.act_test_fname)):
        print("File {} does not exist.".format(args.act_test_fname))
        pass

    # TODO: is there a nicer way to keep observation and action data synchronized?
    # TODO: currently takes batchsize*seq_len samples and reshapes. Is there a nicer way to do this?
    obs_dataset = RolloutDataset(path_to_file=os.path.join(DATA_DIR, 'rollouts', args.obs_fname),
                                 size=int(args.obs_fname.split('.')[-2].split('_')[-2]) * 1000,
                                 transform=ToTensor())
    obs_loader = DataLoader(obs_dataset, batch_size=args.batchsize*args.seq_len, shuffle=False)

    act_dataset = RolloutDataset(path_to_file=os.path.join(DATA_DIR, 'rollouts', args.act_fname),
                                 size=int(args.act_fname.split('.')[-2].split('_')[-2]) * 1000,
                                 image=False,
                                 transform=torch.Tensor)
    act_loader = DataLoader(act_dataset, batch_size=args.batchsize*args.seq_len, shuffle=False)

    obs_test_dataset = RolloutDataset(path_to_file=os.path.join(DATA_DIR, 'rollouts', args.obs_test_fname),
                                      size=int(args.obs_test_fname.split('.')[-2].split('_')[-2]) * 1000,
                                      transform=ToTensor())
    obs_test_loader = DataLoader(obs_test_dataset, batch_size=args.batchsize*args.seq_len, shuffle=False)

    act_test_dataset = RolloutDataset(path_to_file=os.path.join(DATA_DIR, 'rollouts', args.act_test_fname),
                                      size=int(args.act_test_fname.split('.')[-2].split('_')[-2]) * 1000,
                                      image=False,
                                      transform=torch.Tensor)
    act_test_loader = DataLoader(act_test_dataset, batch_size=args.batchsize*args.seq_len, shuffle=False)

    assert len(obs_dataset) == len(act_dataset)
    assert len(obs_test_dataset) == len(act_test_dataset)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    vae = VAE()
    vae.load_state_dict(torch.load(os.path.join(DATA_DIR, 'vae', args.vae_fname)))
    vae.to(device)

    mdnrnn = MDNRNN(latent_dim=args.latent_dim, n_gaussians=args.n_gaussians).to(device)
    optimizer = optim.Adam(mdnrnn.parameters())  # TODO: use different optimizer?

    def train(epoch):
        mdnrnn.train()
        train_loss = 0
        for batch_idx, (obs_batch, act_batch) in enumerate(zip(obs_loader, act_loader)):

            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            optimizer.zero_grad()

            # Encode obs using VAE.
            z_batch = vae.reparameterize(*vae.encode(obs_batch))

            # Reshape into seq_len.
            z_batch = z_batch.view(args.seq_len, -1, args.latent_dim)
            act_batch = act_batch.view(args.seq_len, -1, args.action_dim)

            # Predict all but first encoded obs from all but last encoded obs and action.
            targets = z_batch[1:]
            z_batch = z_batch[:-1]
            act_batch = act_batch[:-1]

            # # Add seq_len dimension.
            # z_batch = z_batch.unsqueeze(0)
            # act_batch = act_batch.unsqueeze(0)

            pi, mu, sigma, hidden = mdnrnn(act_batch, z_batch)
            # output = mdnrnn.mdn.sample(pi, mu, sigma)  # Need output for anything in training?

            # Reshape for calculating the loss.
            pi = pi.view(args.seq_len-1, -1, args.n_gaussians)
            mu = mu.view(args.seq_len-1, -1, args.n_gaussians, args.latent_dim)
            sigma = sigma.view(args.seq_len-1, -1, args.n_gaussians, args.latent_dim)

            loss = nll_gmm_loss(targets, pi, mu, sigma)
            loss.backward()
            train_loss += loss.item()

            optimizer.step()
            # TODO: make output line up in columns.
            if batch_idx % args.log_interval == 0:
                print("Epoch: {0:} | Examples: {1:}/{2:} ({3:.0f}%) | Loss: {4:.2f}\t".format(
                      epoch, batch_idx * len(obs_batch), len(obs_loader.dataset),
                      100. * batch_idx / len(obs_loader),
                      loss.item() / len(obs_batch)))

        print('====> Average train loss: {:.4f}'.format(
              train_loss / len(obs_loader.dataset)))

    def test(epoch):
        mdnrnn.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, (obs_batch, act_batch) in enumerate(zip(obs_test_loader, act_test_loader)):
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)

                # Encode obs using VAE.
                z_batch = vae.reparameterize(*vae.encode(obs_batch))

                # Reshape into seq_len.
                z_batch = z_batch.view(args.seq_len, -1, args.latent_dim)
                act_batch = act_batch.view(args.seq_len, -1, args.action_dim)

                # Predict all but first encoded obs from all but last encoded obs and action.
                targets = z_batch[1:]
                z_batch = z_batch[:-1]
                act_batch = act_batch[:-1]

                # # Add seq_len dimension.
                # z_batch = z_batch.unsqueeze(0)
                # act_batch = act_batch.unsqueeze(0)

                pi, mu, sigma, hidden = mdnrnn(act_batch, z_batch)

                # Reshape for calculating the loss.
                pi = pi.view(args.seq_len-1, -1, args.n_gaussians)
                mu = mu.view(args.seq_len-1, -1, args.n_gaussians, args.latent_dim)
                sigma = sigma.view(args.seq_len-1, -1, args.n_gaussians, args.latent_dim)

                test_loss += nll_gmm_loss(targets, pi, mu, sigma).item()

        print('====> Average test loss: {:.4f}'.format(test_loss / len(obs_test_loader.dataset)))

    # train/test loop
    for i in range(1, args.n_epochs + 1):
        train(i)
        test(i)

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
