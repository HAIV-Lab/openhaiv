import os
import os.path as osp
import errno
import random
import torch
import numpy as np


def mkdir_if_missing(dirname):
    """Create dirname if it is missing.

    Args:
        dirname (str): directory path
    """
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def auto_device(device):
    """Automatically set the device for the input tensor.

    Args:
        device (str | torch.device | None): device name or device object.
            If None, return torch.device('cuda') if available, 
            otherwise return torch.device('cpu').

    Returns:
        device (torch.device): device object
    """
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    return device


def set_random_seed(seed):
    """Set random seed for reproducibility.

    Args:
        seed (int): random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
