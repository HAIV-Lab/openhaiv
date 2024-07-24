import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import *

class LwFNET(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.encoder = resnet18(True, args, num_classes=self.args.network.net_lwf.moco_dim)
        self.num_features = 512

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pre_allocate = self.args.dataloader.num_classes
        self.fc = nn.Linear(self.num_features, self.pre_allocate, bias=False)
        nn.init.orthogonal_(self.fc.weight)

    def encode(self, x):
        x = self.encoder(x)[0]
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, x):
        x = self.encode(x)
        x1 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
        x = self.args.network.net_lwf.temperature * x
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def get_features(self, x):
        x, y = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x