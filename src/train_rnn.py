from __future__ import absolute_import, division, print_function

import os
import datetime
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from src.data import RolloutDataset
from src.vae import VAE, vae_loss
from src.rnn import MDNRNN
from src import DATA_DIR


def main():
    pass

if __name__ == '__main__':
    main()
