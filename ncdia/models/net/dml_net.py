import copy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ncdia.models.net.der_net import SimpleLinear

from ncdia.utils import MODELS, Configs

@MODELS.register
class DMLNet(nn.Module):
    """Net for Decoupling Maxlogit.

    Args:
        network (Configs): Network configuration.
    
    """
    def __init__(
        self,
        network_C: Configs,
        network_N: Configs,
    ) -> None:
        super().__init__()
        self.args_C = network_C.cfg
        self.args_N = network_N.cfg
        self.network_C = MODELS.build(self.args_C)
        self.network_N = MODELS.build(self.args_N)
        self.fc = None
        self.out_features = None

    def forward(self, x):
        _ = self.network_N(x)
        self.out_features = self.network_N.out_features

        return self.network_C(x)