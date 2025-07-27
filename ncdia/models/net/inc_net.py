import copy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ncdia.models.net.der_net import SimpleLinear

from ncdia.utils import MODELS, Configs


@MODELS.register
class BaseNet(nn.Module):
    """BaseNet for incremental learning.

    Args:
        network (Configs): Network configuration.

    """

    def __init__(
        self,
        network: Configs,
        base_classes,
        num_classes,
        att_classes,
        net_alice,
        mode="ft_cos",
    ) -> None:
        super().__init__()
        self.args = network.cfg
        self.args["pretrained"] = True
        self.args["num_classes"] = 1000
        if "type" not in network:
            self.args["type"] = "resnet50"
        self.convnet = MODELS.build(self.args)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        self.convnet(x)
        return self.convnet.out_features

    def forward(self, x):
        x = self.convnet(x)
        features = self.convnet.out_features
        out = self.fc(features)
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        # out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


@MODELS.register
class IncrementalNet(BaseNet):
    """BaseNet for incremental learning.

    Args:
        network (Configs): Network configuration.

    """

    def __init__(
        self,
        network: Configs,
        base_classes,
        num_classes,
        att_classes,
        net_alice,
        mode="ft_cos",
    ) -> None:
        super().__init__(
            network, base_classes, num_classes, att_classes, net_alice, mode
        )
        self.update_fc(num_classes)

    def update_fc(self, nb_classes):
        fc = self.generate_fc(2048, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        x = self.convnet(x)
        features = self.convnet.out_features
        out = self.fc(features)
        # out.update(x)

        return out
