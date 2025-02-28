import torch
import torch.nn as nn
import torch.nn.functional as F 

from ncdia.utils import MODELS, Configs

@MODELS.register
class LwFNET(nn.Module):
    """
    LwFNET for incremental learning.
    """
    def __init__(
        self, 
        network,

    ):
    pass