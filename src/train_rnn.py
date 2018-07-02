from __future__ import absolute_import, division, print_function

import os
import datetime
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader

from src.data import RolloutDataset
from src.vae import VAE
from src.rnn import MDNRNN, nll_gmm_loss
from src import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description='RNN')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Input batch size for training (default=100)')
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='Number of epochs to train (default=20)')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Dimension of latent space (default=32)')
    parser.add_argument('--seq_len', type=int, default=1000,
                        help='Length of sequences for learning (default=1000)')
    parser.add_argument('--action_dim', type=int, default=3,
                        help='Dimension of action space (default=3)')
    parser.add_argument('--rnn_hidden_dim', type=int, default=256,
                        help='Dimension of RNN hidden state (default=256)')
    parser.add_argument('--n_gaussians', type=int, default=5,
                        help='Number of gaussians for the Mixture Density Network (default=5)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for optimizer (default=1e-3)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value (default=1.0)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--vae_fname',
                        help='VAE model file name')
    parser.add_argument('--train_dir_name',
                        help='Rollouts directory name for training')
    parser.add_argument('--test_dir_name',
                        help='Rollouts directory name for testing')
    parser.add_argument('--log_interval', nargs='?', default='2', type=int,
                        help='After how many epochs to log')
    args = parser.parse_args()

    # TODO: is there a better way to do this?
    if not os.path.exists(os.path.join(DATA_DIR, 'rollouts', args.train_dir_name)):
        print("Folder {} does not exist.".format(args.train_dir_name))
        pass
    if not os.path.exists(os.path.join(DATA_DIR, 'rollouts', args.test_dir_name)):
        print("Folder {} does not exist.".format(args.test_dir_name))
        pass

    # Read in and prepare the data.
    train_dataset = RolloutDataset(path_to_dir=os.path.join(DATA_DIR, 'rollouts', args.train_dir_name),
                                   size=int(args.train_dir_name.split('_')[-1]))  # TODO: hack. fix?
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = RolloutDataset(path_to_dir=os.path.join(DATA_DIR, 'rollouts', args.test_dir_name),
                                  size=int(args.test_dir_name.split('_')[-1]))  # TODO: hack. fix?
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Use GPU if available.
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Load the VAE model from file.
    vae = VAE(latent_dim=args.latent_dim)
    vae.load_state_dict(torch.load(os.path.join(DATA_DIR, 'vae', args.vae_fname)))
    vae.to(device)

    # Set up the MDNRNN model and the optimizer.
    mdnrnn = MDNRNN(action_dim=args.action_dim,
                    hidden_dim=args.rnn_hidden_dim,
                    latent_dim=args.latent_dim,
                    n_gaussians=args.n_gaussians).to(device)
    optimizer = optim.Adam(params=mdnrnn.parameters(), lr=args.learning_rate)

    # Train procedure.
    def train(epoch):
        mdnrnn.train()
        train_loss = 0
        start_time = datetime.datetime.now()
        for batch_id, batch in enumerate(train_loader):
            obs_batch = batch['obs'].to(device)
            act_batch = batch['act'].to(device)

            optimizer.zero_grad()

            # Encode obs using VAE.
            vae_obs_batch = obs_batch.view((-1,) + obs_batch.size()[2:])  # Reshape for VAE.
            z_batch = vae.reparameterize(*vae.encode(vae_obs_batch))
            z_batch = z_batch.view(-1, args.seq_len, args.latent_dim)

            # Predict all but first encoded obs from all but last encoded obs and action.
            targets = z_batch[:, 1:]
            z_batch = z_batch[:, :-1]
            act_batch = act_batch[:, :-1]

            pi, mu, sigma, _ = mdnrnn(act_batch, z_batch)

            loss = nll_gmm_loss(targets, pi, mu, sigma)
            loss.backward()
            train_loss += loss.item()

            torch.nn.utils.clip_grad_value_(mdnrnn.parameters(), args.grad_clip)
            optimizer.step()

            if batch_id % args.log_interval == 0:
                print('Epoch: {0:}\t| Examples: {1:}/{2:} ({3:.0f}%)\t| Loss: {4:.2f}\t'.format(
                      epoch, (batch_id+1) * len(obs_batch), len(train_loader.dataset),
                      100. * (batch_id+1) / len(train_loader),
                      loss.item() / len(obs_batch)))

        duration = datetime.datetime.now() - start_time
        print('Epoch {} average train loss was {:.4f} after {}m{}s of training.'.format(
              epoch, train_loss / len(train_loader.dataset),
              *divmod(int(duration.total_seconds()), 60)))

    # Test procedure.
    def test(epoch):
        mdnrnn.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_id, batch in enumerate(test_loader):
                obs_batch = batch['obs'].to(device)
                act_batch = batch['act'].to(device)

                # Encode obs using VAE.
                vae_obs_batch = obs_batch.view((-1,) + obs_batch.size()[2:])  # Reshape for VAE.
                z_batch = vae.reparameterize(*vae.encode(vae_obs_batch))
                z_batch = z_batch.view(-1, args.seq_len, args.latent_dim)

                # Predict all but first encoded obs from all but last encoded obs and action.
                targets = z_batch[:, 1:]
                z_batch = z_batch[:, :-1]
                act_batch = act_batch[:, :-1]

                pi, mu, sigma, _ = mdnrnn(act_batch, z_batch)

                test_loss += nll_gmm_loss(targets, pi, mu, sigma).item()
            print('Epoch {} average test loss was {:.4f}.'.format(
                  epoch, test_loss / len(test_loader.dataset)))

    # Train/test loop.
    for i in range(1, args.n_epochs+1):
        train(i)
        test(i)

    # Save the learned model.
    if not os.path.exists(os.path.join(DATA_DIR, 'rnn')):
        os.makedirs(os.path.join(DATA_DIR, 'rnn'))

    torch.save(mdnrnn.state_dict(), os.path.join(DATA_DIR, 'rnn',
                                                 datetime.datetime.today().isoformat() + '_' + str(args.n_epochs)))
    # To load the model, do:
    # mdnrnn = MDNRNN()
    # mdnrnn.load_state_dict(torch.load(PATH))

if __name__ == '__main__':
    main()
