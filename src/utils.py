from __future__ import absolute_import, division, print_function

import os
import sys
import logging
from contextlib import contextmanager

from PIL import Image
import numpy as np
import torch
from torchvision.transforms import Resize


def setup_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def obs_to_array(obs):
    """"Resizes an observation and returns it as a np.array."""
    return np.array(Resize((64, 64))(Image.fromarray(obs)), dtype=np.uint8)


def obs_array_to_tensor(observations):
    """Transforms np.array of observations to normalized PyTorch tensor with channels first.

    Args:
        observations: A np.array of shape (batch_size, width, height, n_channels).

    Returns:
        A PyTorch tensor of shape (batch_size, n_channels, width, height).
    """
    return torch.from_numpy(observations.transpose(0, 3, 1, 2)).float() / 255.


def param_count(model):
    """Returns number of trainable parameters of given model."""
    n_params = 0
    for param in model.parameters():
        if param.requires_grad:  # Count only the trainable parameters.
            n_params += param.detach().cpu().numpy().flatten().shape[0]
    return n_params


def updated_model(model, params):
    """Sets trainable model parameters with supplied array.

    Args:
        model: A PyTorch model.
        params: A np.array containing the appropriate number of parameters.

    Returns:
        The model with its weights set as the supplied array.
    """
    assert param_count(model) == len(params)

    dtype = next(model.parameters()).type()

    offset = 0

    for param in model.parameters():
        if param.requires_grad:
            param.data = torch.from_numpy(params[offset:offset + param.numel()]).view(param.size()).type(dtype)
            offset += param.numel()

    return model
