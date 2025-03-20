import os
import json
import math
import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F 
from torch.optim.optimizer import Optimizer, required

from torch import Tensor
from typing import List, Optional

from ncdia.utils import LOSSES

@LOSSES.register
class CenterLoss(nn.Module):

    def __init__(self, device, num_classes=100, feat_dim=128):  #128 100
        super(CenterLoss, self).__init__()
        print(f"device: {device}, num_classes: {num_classes}, feat_dim: {feat_dim}")
        self.num_class = num_classes
        self.num_feature = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature).to(device))

    def forward(self, x, labels):
        # print(labels)
        center = self.centers[labels]
        # print(x.shape, center.shape)
        dist = (x-center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        return loss


