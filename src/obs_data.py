from __future__ import absolute_import, division, print_function

from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class ObservationDataset(Dataset):

    def __init__(self, path_to_file, size, transform=None):
        """
        Args:
            path_to_file (string): Path to file.
            size (int): Number of observations in file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.size = size
        self.data = np.load(path_to_file, mmap_mode='r')
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = Image.fromarray(self.data[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample
