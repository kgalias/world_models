from __future__ import absolute_import, division, print_function

import os

from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class RolloutDataset(Dataset):

    def __init__(self, path_to_dir, size, transform=None):
        # """
        # Args:
        #     path_to_dir (string): Path to file.
        #     size (int): Number of observations in file.
        #     image (bool): Whether the date is comprised of images.
        #     transform (callable, optional): Optional transform to be applied on a sample.
        # """
        self.path_to_dir = path_to_dir
        self.size = size
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # TODO: currently assume that all rollouts have length 1000. fix?
        curr_idx = idx // 1000 + 1
        rollout = np.load(os.path.join(self.path_to_dir, str(curr_idx) + '.npz'))

        obs = rollout['observations'][curr_idx]
        action = rollout['actions'][curr_idx]
        reward = rollout['rewards'][curr_idx]
        done = rollout['dones'][curr_idx]

        obs = Image.fromarray(obs)

        if self.transform:
            obs = self.transform(obs)

        return {'obs': obs, 'action': action, 'reward': reward, 'done': done}
