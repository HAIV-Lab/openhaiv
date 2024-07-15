import os
import errno
import os.path as osp
import torch


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
        torch.device: device object
    """
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    return device
