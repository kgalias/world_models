from __future__ import absolute_import, division, print_function

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class RolloutDataset(Dataset):

    def __init__(self, path_to_dir, size):
        """
        Args:
            path_to_dir (string): Path to directory with rollouts.
            size (int): Number of observations in directory altogether.
            # transform (callable, optional): Optional transform to be applied on an observation.
        """
        self.path_to_dir = path_to_dir
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        rollout = np.load(os.path.join(self.path_to_dir, str(idx+1) + '.npz'))

        # Transform observations to normalized PyTorch tensors with channels first.
        observations = torch.from_numpy(rollout['observations'].transpose(0, 3, 1, 2)).float() / 255.

        return {'obs': observations, 'act': rollout['actions']}
