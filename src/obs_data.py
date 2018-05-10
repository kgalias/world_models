from __future__ import print_function, division
import os

from PIL import Image

import torch
import numpy as np
from torch.utils.data import Dataset


class ObservationDataset(Dataset):

    def __init__(self, path_to_file, size, transform=None):
        """
        Args:
            path_to_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.size = size
        self.data = torch.FloatStorage.from_file(path_to_file, True, size)
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = Image.fromarray(np.array(self.data['observations'][idx]))

#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
#         landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
