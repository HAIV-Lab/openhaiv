import copy
import logging
import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from ncdia.utils import MODELS, Configs


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        #out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        out = x.mm(F.normalize(self.weight, dim=0))
        return out
    

@MODELS.register
class DMLNet(nn.Module):
    """Net for Decoupling Maxlogit.

    Args:
        network (Configs): Network configuration.
    
    """
    def __init__(
        self,
        network: Configs,
        checkpoint_C: str = None,
        checkpoint_N: str = None,
        loss: str = 'focal', # 'focal' or 'center' or 'dml'
        **kwargs
    ) -> None:
        super().__init__()
        self.args = network.cfg
        self.network_C = MODELS.build(copy.deepcopy(self.args))
        self.network_N = MODELS.build(copy.deepcopy(self.args))
        
        # replace fc layer
        in_features = self.network_C.fc.in_features
        out_features = self.network_C.fc.out_features
        self.network_C.fc = NormedLinear(in_features, out_features)
        in_features = self.network_N.fc.in_features
        out_features = self.network_N.fc.out_features
        self.network_N.fc = NormedLinear(in_features, out_features)
        
        # Load the state_dict

        if checkpoint_C:
            state_dict_C = torch.load(checkpoint_C)
            if 'state_dict' in state_dict_C:
                state_dict_C = state_dict_C['state_dict']
            # 过滤出 network_C 相关的权重
            network_C_state_dict = {k.replace('network_C.', ''): v for k, v in state_dict_C.items() if k.startswith('network_C.')}
            self.network_C.load_state_dict(network_C_state_dict)


        if checkpoint_N:
            state_dict_N = torch.load(checkpoint_N)
            if 'state_dict' in state_dict_N:
                state_dict_N = state_dict_N['state_dict']
            # 过滤出 network_N 相关的权重
            network_N_state_dict = {k.replace('network_N.', ''): v for k, v in state_dict_N.items() if k.startswith('network_N.')}
            self.network_N.load_state_dict(network_N_state_dict)



        self.fc = self.network_C.fc
        self.out_features = None
        self.loss = loss

    def feature_norm_forward(self, model, x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        self.features = x[:]
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.normalize(x, dim=1) * 40
        self.out_features = x[:]
        return x
    
    def parameters(self, recurse: bool = True):
        if self.loss == 'focal':
            return self.network_C.parameters(recurse=recurse)
        elif self.loss == 'center':
            return self.network_N.parameters(recurse=recurse)
        return itertools.chain(self.network_C.parameters(recurse=recurse), self.network_N.parameters(recurse=recurse))


    def forward(self, x):
        if self.loss == 'focal':
            return self.feature_norm_forward(self.network_C, x)
        elif self.loss == 'center':
            return self.feature_norm_forward(self.network_N, x)
        
        _ = self.network_N(x)
        self.out_features = self.network_N.out_features
        return self.network_C(x)
    