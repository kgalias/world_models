from __future__ import absolute_import, division, print_function

import os
import datetime
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader

from src.data import RolloutDataset
from src.vae import VAE, vae_loss
from src import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for training (default=100)')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of epochs to train (default=100)')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Dimension of latent space (default=32)')
    parser.add_argument('--episode_len', type=int, default=1000,
                        help='Length of rollout (default=1000)')
    parser.add_argument('--kl_bound', type=float, default=0.5,
                        help='Clamp KL loss by kl_bound*latent_dim from below (default=0.5)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for optimizer (default=1e-4)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training (default=False)')
    parser.add_argument('--dir_name',
                        help='Rollouts directory name')
    parser.add_argument('--log_interval', nargs='?', default='2', type=int,
                        help='After how many batches to log (default=2)')
    args = parser.parse_args()

    # Read in and prepare the data.
    dataset = RolloutDataset(path_to_dir=os.path.join(DATA_DIR, 'rollouts', args.dir_name),
                             size=int(args.dir_name.split('_')[-1]))  # TODO: hack. fix?
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Use GPU if available.
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Set up the model and the optimizer.
    vae = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(params=vae.parameters(), lr=args.learning_rate)

    # Training procedure.
    def train(epoch):
        vae.train()
        train_loss = 0
        start_time = datetime.datetime.now()

        # for rollout_id, rollout in enumerate(data_loader):
        #     n_batches = len(rollout.squeeze()['obs']) // args.batch_size
        #     for batch_id in range(n_batches):
        #         start, stop = args.batch_size * batch_id, args.batch_size * (batch_id + 1)
        #         batch = rollout.squeeze()['obs'][start:stop]
        #         batch = batch.to(device)
        #
        #         optimizer.zero_grad()
        #
        #         recon_batch, mu, logvar = vae(batch)
        #         rec_loss, kl_loss = vae_loss(recon_batch, batch, mu, logvar, kl_bound=args.kl_bound)
        #         loss = rec_loss + kl_loss
        #         loss.backward()
        #         train_loss += loss.item()
        #
        #         optimizer.step()
        #
        #         if batch_id % args.log_interval == 0:
        #             print(
        #                 'Epoch: {0:}\t| Examples: {1:} / {2:}({3:.0f}%)\t| Rec Loss: {4: .4f}\t| KL Loss: {5:.4f}'
        #                  .format(epoch, (batch_id + 1) * len(batch), len(data_loader.dataset),
        #                          100. * (batch_id + 1) / len(data_loader),
        #                          rec_loss.item() / len(batch),
        #                          kl_loss.item() / len(batch)))

        for batch_id, batch in enumerate(data_loader):
            batch = batch['obs']
            # Take a random observation from each rollout.
            batch = batch[torch.arange(args.batch_size, dtype=torch.long),
                          torch.randint(high=1000, size=(args.batch_size,), dtype=torch.long)]
            # TODO: use all obs from the rollout (from the randomized start)?
            batch = batch.to(device)

            optimizer.zero_grad()

            recon_batch, mu, logvar = vae(batch)
            rec_loss, kl_loss = vae_loss(recon_batch, batch, mu, logvar, kl_bound=args.kl_bound)
            loss = rec_loss + kl_loss
            loss.backward()
            train_loss += loss.item()

            optimizer.step()

            if batch_id % args.log_interval == 0:
                print('Epoch: {0:}\t| Examples: {1:} / {2:}({3:.0f}%)\t| Rec Loss: {4: .4f}\t| KL Loss: {5:.4f}'.format(
                      epoch, (batch_id+1) * len(batch), len(data_loader.dataset),
                      100. * (batch_id+1) / len(data_loader),
                      rec_loss.item() / len(batch),
                      kl_loss.item() / len(batch)))

        duration = datetime.datetime.now() - start_time
        print('Epoch {} average train loss was {:.4f} after {}m{}s of training.'.format(
              epoch, train_loss / len(data_loader.dataset),
              *divmod(int(duration.total_seconds()), 60)))

    # TODO: add test for VAE?

    # Train loop.
    for i in range(1, args.n_epochs + 1):
        train(i)

    # Save the learned model.
    if not os.path.exists(os.path.join(DATA_DIR, 'vae')):
        os.makedirs(os.path.join(DATA_DIR, 'vae'))

    torch.save(vae.state_dict(), os.path.join(DATA_DIR, 'vae',
                                              datetime.datetime.today().isoformat() + '_' + str(args.n_epochs)))
    # To load the model, do:
    # vae = VAE()
    # vae.load_state_dict(torch.load(PATH))

if __name__ == '__main__':
    main()
