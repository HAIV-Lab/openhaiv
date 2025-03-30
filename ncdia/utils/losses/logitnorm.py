import torch.nn as nn
import torch
import torch.nn.functional as F
from ncdia.utils import LOSSES

@LOSSES.register
class LogitNormLoss(nn.Module):
    ''' LogitNormLoss
    Args:
        t (float): The temperature. Default: 1.0.
    '''

    def __init__(self, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.t = t

    def forward(self, input, target):
        norms = torch.norm(input, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(input, norms) / self.t
        return F.cross_entropy(logit_norm, target)
