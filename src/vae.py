from __future__ import absolute_import, division, print_function

import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)

        self.fc11 = nn.Linear(2*2*256, latent_dim)
        self.fc12 = nn.Linear(2*2*256, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim, 1024)

        self.conv5 = nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=2)
        self.conv6 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2)
        self.conv7 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2)
        self.conv8 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2)

    def encode(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv4 = F.relu(self.conv4(conv3)).view(-1, 2*2*256)  # Reshape for dense.

        return self.fc11(conv4), self.fc12(conv4)

    # TODO: make static?
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        fc2 = F.relu(self.fc2(z))[:, :, None, None]  # Add two dummy dimensions for conv.
        conv5 = F.relu(self.conv5(fc2))
        conv6 = F.relu(self.conv6(conv5))
        conv7 = F.relu(self.conv7(conv6))
        return F.sigmoid(self.conv8(conv7))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    l2 = F.mse_loss(recon_x, x, size_average=False)  # reconstruction loss
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL loss
    return l2, kl
