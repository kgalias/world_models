from __future__ import print_function, division

from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class ObservationDataset(Dataset):

    def __init__(self, path_to_file, shape, transform=None):
        """
        Args:
            path_to_file (string): Path to the file.
            shape (tuple): Shape of array in file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.size = shape[0]
        self.data = np.memmap(path_to_file, dtype='uint8', mode='r', shape=shape)
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = Image.fromarray(self.data[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample
