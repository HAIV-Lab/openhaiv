import copy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ncdia.models.net.der_net import SimpleLinear

from ncdia.utils import MODELS, Configs


@MODELS.register
class FOSTERNet(nn.Module):
    def __init__(
        self,
        network: Configs,
        base_classes,
        num_classes,
        net_alice,
        total_classes=None,
        pretrained=True,
        mode="ft_cos",
    ) -> None:
        super().__init__()
        self.args = network.cfg
        self.args["pretrained"] = pretrained
        self.args["num_classes"] = 1000
        self.network = network
        if "type" not in network:
            self.args["type"] = "resnet18"
        self.convnets = nn.ModuleList()
        self.fc = None
        self.fe_fc = None
        self.old_fc = None

        print("------------------Initialized A New FOSTER Net!------------------")
        if total_classes is None:
            self.update_fc(base_classes)
        else:
            self.update_fc(total_classes)

    @property
    def feature_dim(self):
        out_dim = 0
        for convnet in self.convnets:
            out_dim += convnet.fc.in_features
        return out_dim

    def extract_vector(self, x):
        features = []
        for convnet in self.convnets:
            convnet(x)
            features.append(convnet.features)
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = []
        for convnet in self.convnets:
            convnet(x)
            features.append(convnet.out_features)
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.convnets[-1].fc.in_features :])
        if self.old_fc is not None:
            old_logits = self.old_fc(features[:, : -self.convnets[-1].fc.in_features])
        else:
            old_logits = None
        return out, fe_logits, old_logits

    def update_fc(self, nb_classes):
        if "type" not in self.network:
            self.args["type"] = "resnet18"
        convnet_new = MODELS.build(self.args)
        self.convnets.append(convnet_new.cuda())
        out_dim = convnet_new.fc.in_features
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - out_dim] = weight.cuda()
            fc.bias.data[:nb_output] = bias.cuda()
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())
        self.old_fc = self.fc
        self.fc = fc
        self.fe_fc = self.generate_fc(out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim).cuda()
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def replace(self, model):
        self.convnets = model.convnets
        self.old_fc = model.old_fc
        self.fe_fc = model.fe_fc
        self.fc = model.fc

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        self.fc.weight.data[-increment:, :] *= gamma
