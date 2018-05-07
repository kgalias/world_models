from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import argparse
# import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import gym
from src import DATA_DIR


class VAE(nn.Module):
    def __init__(self, hidden_dim=32):
        super(VAE, self).__init__()

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size,
        # stride=1, padding=0, dilation=1, groups=1, bias=True)

        # Encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)

        self.fc11 = nn.Linear(2*2*256, hidden_dim)
        self.fc12 = nn.Linear(2*2*256, hidden_dim)

        # Decoder
        self.fc2 = nn.Linear(hidden_dim, 1024)

        # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
        # stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        self.conv5 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)  #
        self.conv6 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)  #
        self.conv7 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)  #
        self.conv8 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)  #

    def encode(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv4 = F.relu(self.conv4(conv3)).view(-1, 2*2*256)

        return self.fc11(conv4), self.fc12(conv4)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc2 = F.relu(self.fc2(z)).view(-1, )  # unsqueeze?
        conv5 = F.relu(self.conv5(fc2))
        conv6 = F.relu(self.conv6(conv5))
        conv7 = F.relu(self.conv7(conv6))
        return F.sigmoid(self.conv8(conv7))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def main():

    # TODO: have consistent (with other file) argument descriptions
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
